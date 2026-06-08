#!/usr/bin/env python3
"""Brief 17 — Generate noise-matched decoy winners for all-channel DPO.

Background
----------
At iter 0 of DPO, the policy prefers ground-truth (X-ray crystal) winners
over AAPR-sampled losers by +2.28 on the rot+pos channels — a shortcut
the model is exploiting on the "real crystal vs DiffAb sample" axis
rather than biology. This script removes the shortcut by replacing each
GT-winner PDB with a *decoy*: the GT structure put through a partial
forward+reverse diffusion pass (t=10) so it carries DiffAb's noise
signature while keeping the GT's biological residues + approximate fold.

Implementation
--------------
For each unique ``gt_complex_id`` in the pair pool:

  1. Load the GT structure via the base dataset's LMDB.
  2. Apply ``Compose([MaskMultipleCDRs(H1+H2+H3), MergeChains(),
     PatchAroundAnchor])`` — the same mask AAPR used to produce the
     losers, so decoy and loser come out of the same input distribution.
  3. Run ``model.optimize(batch, opt_step=t_decoy, ...)`` which
     forward-diffuses the CDRs to t=t_decoy then reverse-denoises them
     back to t=0 using π_ref. (DiffAb's ``FullDPM.optimize`` is the
     canonical "partial reverse" wrapper.)
  4. Reconstruct heavy-atom coordinates from (R, t, s) at t=0 and splice
     them back into the full antibody+antigen template via
     ``apply_patch_to_tensor``.
  5. Save as a packed PDB (PyRosetta side-chain placement, restricted to
     the VHH chain — identical post-processing to AAPR losers so the two
     are file-format-comparable).

One PDB is emitted per unique GT, named ``{gt_id}__decoy_t{T}.pdb``.
A run-metadata JSON records the seed, t_decoy, and per-GT seeds for
audit / reproducibility.

Reproducibility
---------------
Each GT's RNG seed is derived from SHA-256 of its bare-PDB ID (NOT
Python's salted ``hash()``), so re-running the script always produces
byte-identical decoys for a given (GT, t_decoy) pair.

CLI
---
::

    python scripts/dpo/sample_decoy_winners.py \\
        --pi-ref-checkpoint runs/vhh_ft/seed42_jfix/checkpoints/best_ema.pt \\
        --config            configs/diffab_ft/vhh_ft.yml \\
        --pairs-parquet     data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs.parquet \\
        --output-dir        data/aapr/ftseed42_jfix_trainval_K8_20260525/decoys_t10/ \\
        --t-decoy 10 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
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

import src.diffab_ft.datasets  # noqa: E402, F401  — registry side effect

# Side-chain packing post-process — same call AAPR's sample_candidates.py
# uses. Decoy PDBs go through the identical post-process as losers so
# downstream parsers see the same heavy-atom content.
from src.biophysics_judge.pdb_utils import pack_missing_sidechains  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("dpo.decoy")


CDRS_DEFAULT = ["H1", "H2", "H3"]
MASK_STRATEGY_LABEL = "MULTIPLE_CDRS_H1H2H3"


# ── Helpers ──────────────────────────────────────────────────────────────

def _gt_seed(gt_id: str, base_seed: int = 42) -> int:
    """Deterministic 31-bit seed derived from gt_id.

    Python's built-in ``hash`` is salted per-interpreter (PEP 456), so we
    use SHA-256 for cross-run determinism. ``base_seed`` lets the caller
    re-roll the whole pool by changing one constant.
    """
    h = hashlib.sha256(gt_id.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) ^ int(base_seed)) & 0x7FFFFFFF


def _load_checkpoint_into(
    model: torch.nn.Module, ckpt_path: Path, device: str,
) -> dict:
    """Load π_ref weights into ``model``. Mirrors AAPR loader behaviour."""
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
    """Parse ``diffab_manifest.tsv`` → ``{pdb_id: {Hchain, antigen_chain}}``."""
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


# ── Core decoy generator ─────────────────────────────────────────────────

@torch.no_grad()
def generate_decoys(
    model,
    dataset,
    *,
    gt_ids: list[str],
    pdb_to_entry: dict[str, dict],
    device: str,
    t_decoy: int,
    output_dir: Path,
    antigen_map: dict[str, dict[str, str]],
    base_seed: int,
) -> list[dict]:
    """For each unique GT, partial-reverse-diffuse and save one PDB."""
    model.eval()
    collate = PaddingCollate(eight=False)
    masking = Compose([
        MaskMultipleCDRs(selection=CDRS_DEFAULT, augmentation=False),
        MergeChains(),
    ])
    inference_tfm = Compose([PatchAroundAnchor()])

    pdb_dir = output_dir / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "pdbs_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    sample_times: list[float] = []
    skipped: list[dict] = []

    for gt_id in tqdm(gt_ids, desc="GT entries"):
        entry = pdb_to_entry.get(gt_id)
        if entry is None:
            skipped.append({"gt_id": gt_id, "reason": "not in LMDB"})
            logger.warning("GT %s not in base_dataset LMDB; skipping.", gt_id)
            continue

        entry_id = entry["id"]
        heavy_chain_id = entry.get("H_chain") or antigen_map.get(gt_id, {}).get("Hchain", "") or None

        # Raw, un-transformed structure dict from LMDB. Deep-copy so the
        # in-place transforms below don't mutate the cached version.
        try:
            structure = deepcopy(dataset.get_structure(entry_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "GT %s (entry %s): get_structure failed (%s: %s); skipping.",
                gt_id, entry_id, exc.__class__.__name__, exc,
            )
            skipped.append({"gt_id": gt_id, "reason": f"get_structure: {exc}"})
            continue

        try:
            masked = masking(structure)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "GT %s: MaskMultipleCDRs failed (%s: %s); skipping.",
                gt_id, exc.__class__.__name__, exc,
            )
            skipped.append({"gt_id": gt_id, "reason": f"masking: {exc}"})
            continue

        data = inference_tfm(deepcopy(masked))

        # Single-replicate batch (batch size 1) — we only need one decoy
        # per GT. The batch gets per-GT-seeded RNG so reruns are
        # byte-deterministic.
        batch = collate([deepcopy(data)])
        batch = recursive_to(batch, device)

        seed_for_gt = _gt_seed(gt_id, base_seed=base_seed)
        torch.manual_seed(seed_for_gt)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed_for_gt)

        # Partial reverse — DiffAb's FullDPM.optimize() encodes the
        # batch (encoder is blinded to CDR structure/sequence per the
        # remove_structure=True/remove_sequence=True defaults), forward-
        # diffuses the CDRs to t=t_decoy, then reverse-denoises step by
        # step back to t=0 using π_ref's noise predictor.
        t0 = time.time()
        traj = model.optimize(batch, opt_step=int(t_decoy), optimize_opt={
            "pbar": False,
            "sample_structure": True,
            "sample_sequence": True,
        })
        sample_times.append(time.time() - t0)

        v_final, t_final, s_final = traj[0]
        R_final = so3vec_to_rotation(v_final)

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

        data_tmpl = masked
        patch_idx = data["patch_idx"]

        aa_full = apply_patch_to_tensor(
            data_tmpl["aa"], s_final[0].cpu(), patch_idx,
        )
        pos_full = apply_patch_to_tensor(
            data_tmpl["pos_heavyatom"],
            pos_atom_new[0].cpu() + batch["origin"][0].view(1, 1, 3).cpu(),
            patch_idx,
        )
        mask_atom_full = apply_patch_to_tensor(
            data_tmpl["mask_heavyatom"], mask_atom_new[0].cpu(), patch_idx,
        )

        raw_pdb_path = raw_dir / f"{gt_id}__decoy_t{int(t_decoy)}.raw.pdb"
        pdb_path = pdb_dir / f"{gt_id}__decoy_t{int(t_decoy)}.pdb"

        save_pdb({
            "chain_nb": data_tmpl["chain_nb"],
            "chain_id": data_tmpl["chain_id"],
            "resseq":   data_tmpl["resseq"],
            "icode":    data_tmpl["icode"],
            "aa":       aa_full,
            "mask_heavyatom": mask_atom_full,
            "pos_heavyatom":  pos_full,
        }, path=str(raw_pdb_path))

        # Side-chain packing — identical to AAPR. Restricted to the VHH
        # chain so antigen rotamers are preserved.
        try:
            pack_missing_sidechains(
                raw_pdb_path, pdb_path,
                chain_id=heavy_chain_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "GT %s: side-chain packing failed (%s: %s); falling back "
                "to raw backbone-only PDB.",
                gt_id, exc.__class__.__name__, exc,
            )
            import shutil
            shutil.copy(raw_pdb_path, pdb_path)

        rows.append({
            "gt_id":            gt_id,
            "entry_id":         entry_id,
            "decoy_pdb_path":   str(pdb_path.resolve()),
            "raw_pdb_path":     str(raw_pdb_path.resolve()),
            "t_decoy":          int(t_decoy),
            "seed_for_gt":      int(seed_for_gt),
            "base_seed":        int(base_seed),
            "heavy_chain_id":   heavy_chain_id or "",
            "antigen_chain_ids": antigen_map.get(gt_id, {}).get("antigen_chain", ""),
            "mask_strategy":    MASK_STRATEGY_LABEL,
            "cdrs_masked":      ",".join(CDRS_DEFAULT),
        })

    if sample_times:
        mean_t = sum(sample_times) / len(sample_times)
        logger.info(
            "Optimize() wall-time: mean %.2fs/GT, n=%d successful entries.",
            mean_t, len(sample_times),
        )
    if skipped:
        logger.warning(
            "Skipped %d/%d GTs; reasons: %s",
            len(skipped), len(gt_ids),
            {r["reason"] for r in skipped},
        )

    return rows


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pi-ref-checkpoint", required=True, type=Path,
                        help="π_ref .pt checkpoint (the policy whose "
                             "noise signature decoys should match).")
    parser.add_argument("--config", required=True, type=Path,
                        help="Diffab fine-tune YAML used to build the "
                             "base dataset (e.g. configs/diffab_ft/vhh_ft.yml).")
    parser.add_argument("--pairs-parquet", required=True, type=Path,
                        help="Pair parquet whose unique gt_complex_ids "
                             "drive the decoy pool.")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Output root. Decoy PDBs go to "
                             "<output-dir>/pdbs/, raw bb-only PDBs to "
                             "<output-dir>/pdbs_raw/.")
    parser.add_argument("--t-decoy", type=int, default=10,
                        help="Partial-reverse depth. The GT is forward-"
                             "diffused to t=t_decoy then reverse-denoised "
                             "back to t=0. Default: 10.")
    parser.add_argument("--manifest-tsv", type=Path,
                        default=PROJECT_ROOT / "data/datasets/diffab_manifest.tsv",
                        help="GT manifest for antigen-chain lookup.")
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "test"],
                        help="Base-dataset split to load. The split is "
                             "only used as a constructor argument; the "
                             "decoy pool iterates over pairs.parquet's "
                             "unique GTs via base_dataset.sabdab_entries "
                             "(which spans the whole LMDB). Default: train.")
    parser.add_argument("--max-entries", type=int, default=None,
                        help="Cap on unique GTs (canary runs). Default: all.")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Base seed for per-GT RNG derivation.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    for p in (args.pi_ref_checkpoint, args.config, args.pairs_parquet,
              args.manifest_tsv):
        if not p.exists():
            logger.error("Required input not found: %s", p)
            return 2

    seed_all(args.base_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Unique GT list — sorted for determinism.
    pairs_df = pd.read_parquet(args.pairs_parquet)
    gt_col = "gt_complex_id"
    if gt_col not in pairs_df.columns:
        logger.error(
            "pairs parquet %s missing required column %r; have %s",
            args.pairs_parquet, gt_col, list(pairs_df.columns),
        )
        return 3
    unique_gts = sorted(pairs_df[gt_col].astype(str).unique().tolist())
    n_pairs = len(pairs_df)
    if args.max_entries is not None:
        unique_gts = unique_gts[: int(args.max_entries)]
    logger.info(
        "Pair parquet %s: %d pairs over %d unique GTs (running on %d).",
        args.pairs_parquet, n_pairs,
        pairs_df[gt_col].nunique(), len(unique_gts),
    )

    # Antigen-chain lookup (TSV manifest).
    antigen_map = _load_antigen_manifest(args.manifest_tsv)

    # Base dataset — built with split='train' by default so the LMDB is
    # populated; we then look up every unique GT through sabdab_entries
    # + db_ids (which span the whole LMDB regardless of which split was
    # selected). Disable the dataset's configured transform so
    # get_structure returns raw cached dicts.
    config, _ = load_config(str(args.config))
    base_block = deepcopy(config.dataset.train)
    base_block.split = args.split
    logger.info("Loading base dataset (split=%r) from %s",
                args.split, getattr(base_block, "manifest_path", "<n/a>"))
    dataset = get_dataset(base_block)
    original_transform = dataset.transform
    dataset.transform = None
    try:
        # Build {bare_pdb -> entry} lookup, mirroring PairDataset's
        # filter. ``db_ids`` covers the full LMDB (all splits), so every
        # GT referenced by the pair parquet should be reachable.
        live_ids = set(dataset.db_ids or [])
        pdb_to_entry: dict[str, dict] = {}
        for entry in dataset.sabdab_entries:
            if entry["id"] not in live_ids:
                continue
            pdb_to_entry[entry["pdbcode"]] = entry
        missing = [g for g in unique_gts if g not in pdb_to_entry]
        if missing:
            logger.warning(
                "Pair parquet references %d GTs not in the LMDB "
                "(sample: %s). They will be skipped.",
                len(missing), missing[:5],
            )

        # Build model + load π_ref weights.
        logger.info("Constructing model from config.model and loading "
                    "weights from %s", args.pi_ref_checkpoint)
        model = get_model(config.model).to(args.device)
        meta = _load_checkpoint_into(model, args.pi_ref_checkpoint, args.device)
        if "iteration" in meta:
            logger.info("π_ref ckpt: iter=%s val_loss=%s",
                        meta.get("iteration"), meta.get("val_loss"))
        # Belt-and-braces: π_ref must not learn.
        for p in model.parameters():
            p.requires_grad_(False)

        # Verify diffusion horizon matches t_decoy upper bound. Decoy
        # depth above the model's training horizon is undefined.
        T = int(model.diffusion.num_steps)
        if args.t_decoy <= 0 or args.t_decoy > T:
            logger.error(
                "--t-decoy=%d outside valid range [1, %d] (model's "
                "diffusion num_steps).", args.t_decoy, T,
            )
            return 4

        rows = generate_decoys(
            model, dataset,
            gt_ids=unique_gts,
            pdb_to_entry=pdb_to_entry,
            device=args.device,
            t_decoy=int(args.t_decoy),
            output_dir=args.output_dir,
            antigen_map=antigen_map,
            base_seed=int(args.base_seed),
        )
    finally:
        dataset.transform = original_transform

    # Persist per-GT manifest + run metadata.
    if rows:
        manifest_path = args.output_dir / "decoys_manifest.csv"
        fieldnames = list(rows[0].keys())
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote %d manifest rows to %s", len(rows), manifest_path)
    else:
        logger.warning("No decoys produced — manifest NOT written.")

    meta_out = {
        "pi_ref_checkpoint": str(args.pi_ref_checkpoint),
        "config":            str(args.config),
        "pairs_parquet":     str(args.pairs_parquet),
        "output_dir":        str(args.output_dir),
        "t_decoy":           int(args.t_decoy),
        "base_seed":         int(args.base_seed),
        "split":             args.split,
        "max_entries":       args.max_entries,
        "n_unique_gts_in_parquet": int(pairs_df[gt_col].nunique()),
        "n_processed":       len(rows),
        "mask_strategy":     MASK_STRATEGY_LABEL,
        "cdrs_masked":       CDRS_DEFAULT,
    }
    meta_path = args.output_dir / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)
    logger.info("Wrote run metadata to %s", meta_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
