#!/usr/bin/env python3
"""AAPR loser-pool generation: mask + DiffAb sample + antigen-preserved PDB output.

Forks the sampling skeleton of ``scripts/diffab_ft/evaluate.py`` (``run_design``)
but:
  * Replaces the per-CDR loop with a single ``MaskMultipleCDRs`` call on
    H1+H2+H3 (the multi-CDR π_ref scope locked in
    ``docs/aapr_generation_context.md`` §3).
  * Drops AAR/RMSD computation — AAPR generates the loser pool; quality
    metrics belong to the downstream judge pipeline.
  * Writes a ``candidates.csv`` manifest with the §6 schema, one row per
    (GT × sample), so the judges can score the saved PDBs directly.

Antigen preservation:
  ``MaskMultipleCDRs`` sets ``generate_flag = True`` only on the CDR
  residues; antigen residues stay ``generate_flag = False``. DiffAb's
  reverse diffusion only denoises flagged residues, so antigen
  coordinates are preserved end-to-end. ``MergeChains()`` then folds the
  antigen and nanobody into a single residue list with chain-ID metadata
  that ``save_pdb()`` writes back out per chain. A canary script verifies
  this end-to-end on the first run.

Diversity:
  DiffAb has no temperature knob. Diversity comes from per-replicate RNG
  in the reverse-diffusion stochasticity (cf. ``docs/aapr_generation_context.md``
  §5). We sample K independent replicates per GT, all batched together,
  each with fresh RNG state.

Usage
-----
::

    # Canary (10 entries × K=8 = 80 candidates):
    python scripts/aapr/sample_candidates.py \\
        --checkpoint /projects/0/.../codesign_multicdrs.pt \\
        --config     configs/diffab_ft/vhh_ft.yml \\
        --split      test \\
        --num-samples 8 \\
        --max-entries 10 \\
        --output-dir data/aapr \\
        --run-name   canary_K8 \\
        --seed       42

    # Full run (test split × K=8):
    python scripts/aapr/sample_candidates.py \\
        --checkpoint runs/vhh_ft/seed42_v3/checkpoints/best_ema.pt \\
        --config     configs/diffab_ft/vhh_ft.yml \\
        --split      test \\
        --num-samples 8 \\
        --output-dir data/aapr \\
        --run-name   v3_seed42_K8_$(date +%Y%m%d) \\
        --seed       42
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
from tqdm.auto import tqdm

# ── Project paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "diffab"))

from diffab.datasets import get_dataset  # noqa: E402
from diffab.models import get_model  # noqa: E402
from diffab.modules.common.geometry import reconstruct_backbone_partially  # noqa: E402
from diffab.modules.common.so3 import so3vec_to_rotation  # noqa: E402
from diffab.utils.data import PaddingCollate, apply_patch_to_tensor  # noqa: E402
from diffab.utils.misc import load_config, seed_all  # noqa: E402
from diffab.utils.protein.writers import save_pdb  # noqa: E402
from diffab.utils.train import recursive_to  # noqa: E402
from diffab.utils.transforms import (  # noqa: E402
    Compose, MaskMultipleCDRs, MergeChains, PatchAroundAnchor,
)

# Register our vhh_andd dataset adapter (side-effect import).
import src.diffab_ft.datasets  # noqa: E402, F401

# Biopython index_to_one matches DiffAb's AA indexing.
from Bio.PDB import Polypeptide  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("aapr.sample")


CDRS_DEFAULT = ["H1", "H2", "H3"]
MASK_STRATEGY_LABEL = "MULTIPLE_CDRS_H1H2H3"


# ── Helpers ──────────────────────────────────────────────────────────────

def _aa_to_str(aa_tensor: torch.Tensor) -> str:
    """Render an AA-index tensor as a single-letter string."""
    return "".join(Polypeptide.index_to_one(int(a)) for a in aa_tensor.flatten())


def _load_checkpoint_into(
    model: torch.nn.Module, ckpt_path: Path, device: str,
) -> dict:
    """Best-effort state-dict loader. Handles both our wrapped checkpoints
    (``{"model": sd, "iteration": ..., ...}``) and luost26/DiffAb's
    upstream format (plain state-dict or EasyDict-wrapped).

    ``weights_only=False`` because luost26's checkpoint pickles its
    EasyDict training config; torch>=2.6's default refuses that. Source
    is trusted (HF hub + locally produced).
    """
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(ck, dict) and "model" in ck:
        sd = ck["model"]
        meta = {k: v for k, v in ck.items() if k != "model"}
    else:
        sd = ck
        meta = {}
    result = model.load_state_dict(sd, strict=False)
    if result.missing_keys or result.unexpected_keys:
        logger.warning(
            "Non-strict checkpoint load: %d missing, %d unexpected keys.",
            len(result.missing_keys), len(result.unexpected_keys),
        )
    return meta


def _load_antigen_manifest(manifest_path: Path) -> dict[str, dict[str, str]]:
    """Parse ``diffab_manifest.tsv`` into ``{pdb_id: {Hchain, antigen_chain}}``.

    Used to populate ``nanobody_chain_id`` and ``antigen_chain_ids`` in
    the AAPR candidate manifest. Comma-joined antigen chain strings are
    preserved as-is (DiffAb already uses this convention).
    """
    out: dict[str, dict[str, str]] = {}
    with open(manifest_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            out[row["pdb"]] = {
                "Hchain": row.get("Hchain", ""),
                "antigen_chain": row.get("antigen_chain", ""),
            }
    logger.info("Loaded antigen-chain manifest for %d entries from %s",
                len(out), manifest_path)
    return out


def _entry_to_pdb_id(entry_id: str) -> str:
    """DiffAb entry-id convention is ``<pdb>_<Hchain>``; return the pdb part."""
    return entry_id.split("_", 1)[0]


# ── Core sampler ─────────────────────────────────────────────────────────

@torch.no_grad()
def run_aapr_sampling(
    model,
    dataset,
    *,
    device: str,
    num_samples: int,
    output_dir: Path,
    antigen_map: dict[str, dict[str, str]],
    checkpoint_id: str,
    seed: int,
    max_entries: int | None,
) -> list[dict]:
    """For each GT entry, mask H1+H2+H3 and sample K candidates.

    Saves K PDBs per entry under ``output_dir/pdbs/<entry_id>/`` and
    returns one manifest row per (entry × sample).
    """
    model.eval()
    collate = PaddingCollate(eight=False)
    masking = Compose([
        MaskMultipleCDRs(selection=CDRS_DEFAULT, augmentation=False),
        MergeChains(),
    ])
    inference_tfm = Compose([PatchAroundAnchor()])

    rows: list[dict] = []
    n_entries = len(dataset) if max_entries is None else min(max_entries, len(dataset))
    sample_times: list[float] = []

    for idx in tqdm(range(n_entries), desc="GT entries"):
        entry_id = dataset.ids_in_split[idx]
        pdb_id = _entry_to_pdb_id(entry_id)
        meta_for_entry = antigen_map.get(pdb_id, {})

        # Load raw structure dict (un-transformed); apply mask + merge.
        structure = dataset.get_structure(entry_id)
        try:
            masked = masking(deepcopy(structure))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "entry %s: MaskMultipleCDRs failed (%s: %s); skipping.",
                entry_id, exc.__class__.__name__, exc,
            )
            continue

        # Patch around CDR anchors (DiffAb's local-frame transform).
        data = inference_tfm(deepcopy(masked))

        # Replicate K times and batch — each replicate gets independent
        # diffusion RNG inside model.sample().
        data_list = [data] * num_samples
        batch = collate([deepcopy(d) for d in data_list])
        batch = recursive_to(batch, device)

        # Reverse diffusion. traj[0] = final denoising step output.
        t0 = time.time()
        traj = model.sample(batch, sample_opt={
            "pbar": False,
            "sample_structure": True,
            "sample_sequence": True,
        })
        sample_times.append(time.time() - t0)

        v_final, t_final, s_final = traj[0]
        R_final = so3vec_to_rotation(v_final)

        # Reconstruct heavy-atom coords for the regenerated CDRs.
        pos_atom_new, mask_atom_new = reconstruct_backbone_partially(
            pos_ctx=batch["pos_heavyatom"],
            R_new=R_final,
            t_new=t_final,
            aa=s_final,
            chain_nb=batch["chain_nb"],
            res_nb=batch["res_nb"],
            mask_atoms=batch["mask_heavyatom"],
            mask_recons=batch["generate_flag"],
        )

        # The mask dict still has the un-patched (original) data; use it
        # as the template that we splice the regenerated CDR back into
        # via apply_patch_to_tensor.
        data_tmpl = masked
        patch_idx = data["patch_idx"]
        cdr_mask_residues = data_tmpl["generate_flag"].bool()

        entry_pdb_dir = output_dir / "pdbs" / entry_id
        entry_pdb_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples):
            aa_full = apply_patch_to_tensor(
                data_tmpl["aa"], s_final[i].cpu(), patch_idx,
            )
            pos_full = apply_patch_to_tensor(
                data_tmpl["pos_heavyatom"],
                pos_atom_new[i].cpu() + batch["origin"][i].view(1, 1, 3).cpu(),
                patch_idx,
            )
            mask_atom_full = apply_patch_to_tensor(
                data_tmpl["mask_heavyatom"], mask_atom_new[i].cpu(), patch_idx,
            )

            pdb_path = entry_pdb_dir / f"sample_{i:04d}.pdb"
            save_pdb({
                "chain_nb": data_tmpl["chain_nb"],
                "chain_id": data_tmpl["chain_id"],
                "resseq":   data_tmpl["resseq"],
                "icode":    data_tmpl["icode"],
                "aa":       aa_full,
                "mask_heavyatom": mask_atom_full,
                "pos_heavyatom":  pos_full,
            }, path=str(pdb_path))

            # Reconstruct the regenerated VHH sequence for the manifest.
            # Use the heavy-chain residues only (chain_id == Hchain).
            heavy_chain = meta_for_entry.get("Hchain", "")
            if heavy_chain:
                heavy_mask = [
                    cid == heavy_chain
                    for cid in data_tmpl["chain_id"]
                ]
                heavy_aa = aa_full[torch.tensor(heavy_mask, dtype=torch.bool)]
            else:
                # Fallback: assume the first chain is the nanobody.
                heavy_aa = aa_full

            rows.append({
                "candidate_id":      f"{entry_id}_s{i:04d}",
                "gt_complex_id":     entry_id,
                "sample_idx":        i,
                "raw_sequence":      _aa_to_str(heavy_aa),
                "complex_pdb_path":  str(pdb_path.resolve()),
                "nanobody_chain_id": heavy_chain,
                "antigen_chain_ids": meta_for_entry.get("antigen_chain", ""),
                "mask_strategy":     MASK_STRATEGY_LABEL,
                "cdrs_masked":       ",".join(CDRS_DEFAULT),
                "temperature":       "",  # DiffAb has no temperature knob
                "checkpoint_id":     checkpoint_id,
                "seed":              seed,
            })

    if sample_times:
        mean_t = sum(sample_times) / len(sample_times)
        logger.info(
            "Sampling wall-time: mean %.2fs/GT (K=%d replicates batched), "
            "%.2fs/replicate amortized, n=%d entries.",
            mean_t, num_samples, mean_t / num_samples, len(sample_times),
        )

    return rows


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Path to a DiffAb .pt checkpoint (raw or wrapped).")
    parser.add_argument("--config", required=True, type=Path,
                        help="YAML used during training (architecture + dataset).")
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test", "test_antigen_disjoint"],
                        help="GT split to sample from.")
    parser.add_argument("--num-samples", type=int, default=8,
                        help="K replicates per GT (default 8).")
    parser.add_argument("--max-entries", type=int, default=None,
                        help="Cap on GT entries (canary runs). Default: full split.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/aapr"),
                        help="Output root; the run subdir is created under here.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run subdirectory name. Default: "
                             "<ckpt_short>_seed<N>_K<K>_<YYYYMMDD>.")
    parser.add_argument("--manifest-tsv", type=Path,
                        default=Path("data/datasets/diffab_manifest.tsv"),
                        help="GT manifest for antigen-chain lookup.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        return 2
    if not args.config.exists():
        logger.error("Config not found: %s", args.config)
        return 2
    if not args.manifest_tsv.exists():
        logger.error("Manifest TSV not found: %s", args.manifest_tsv)
        return 2

    seed_all(args.seed)

    # Resolve run name + output paths.
    if args.run_name is None:
        ckpt_short = args.checkpoint.stem.split("_")[0]
        date_tag = time.strftime("%Y%m%d")
        args.run_name = f"{ckpt_short}_seed{args.seed}_K{args.num_samples}_{date_tag}"
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("AAPR run dir: %s", run_dir.resolve())

    # Antigen-chain lookup (manifest TSV).
    antigen_map = _load_antigen_manifest(args.manifest_tsv)

    # Dataset block: clone val and override split (mirrors evaluate.py
    # which does the same for test / test_antigen_disjoint).
    config, _ = load_config(str(args.config))
    base_block = config.dataset.val
    eval_block = deepcopy(base_block)
    eval_block.split = args.split
    logger.info("Loading split %r from %s", args.split, eval_block.manifest_path)
    dataset = get_dataset(eval_block)
    logger.info("Split %r: %d GT entries.", args.split, len(dataset))

    # Disable the dataset's configured transform — AAPR needs the raw
    # cached structure dict so we can apply MaskMultipleCDRs ourselves.
    original_transform = dataset.transform
    dataset.transform = None

    # Build model + load weights.
    logger.info("Constructing model and loading weights from %s", args.checkpoint)
    model = get_model(config.model).to(args.device)
    meta = _load_checkpoint_into(model, args.checkpoint, args.device)
    if "iteration" in meta:
        logger.info(
            "Checkpoint iter: %s | val_loss: %s",
            meta.get("iteration"), meta.get("val_loss"),
        )

    # Run AAPR sampling.
    try:
        rows = run_aapr_sampling(
            model, dataset,
            device=args.device,
            num_samples=args.num_samples,
            output_dir=run_dir,
            antigen_map=antigen_map,
            checkpoint_id=args.checkpoint.name,
            seed=args.seed,
            max_entries=args.max_entries,
        )
    finally:
        dataset.transform = original_transform

    # Write the candidate manifest.
    csv_path = run_dir / "candidates.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote %d manifest rows to %s", len(rows), csv_path)
    else:
        logger.warning("No candidates generated — candidates.csv NOT written.")

    # Run metadata for provenance.
    meta_out = {
        "run_name":     args.run_name,
        "checkpoint":   str(args.checkpoint),
        "checkpoint_id": args.checkpoint.name,
        "config":       str(args.config),
        "split":        args.split,
        "num_samples":  args.num_samples,
        "max_entries":  args.max_entries,
        "seed":         args.seed,
        "mask_strategy": MASK_STRATEGY_LABEL,
        "cdrs_masked":  CDRS_DEFAULT,
        "n_gt_entries": len(dataset) if args.max_entries is None else min(args.max_entries, len(dataset)),
        "n_candidates": len(rows),
        "ckpt_meta":    {
            k: (v if isinstance(v, (int, float, str, bool)) else str(v))
            for k, v in meta.items() if k not in ("config",)
        },
    }
    meta_path = run_dir / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)
    logger.info("Wrote run metadata to %s", meta_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
