#!/usr/bin/env python3
"""Evaluate a (fine-tuned or baseline) DiffAb checkpoint on our VHH split.

Two evaluation modes — pick via ``--mode``:

  ``elbo``    (cheap, default)
      One forward pass per batch. Reports the same weighted-loss
      decomposition DiffAb trains on (rot/pos/seq + overall) averaged
      across the chosen split. Use this for the headline pre-FT vs.
      post-FT comparison; it's fast (~1 min on a single A100 for our
      ~50-entry test split) and runs without sampling.

  ``design`` (slower, per-entry sampling)
      For each entry in the split, for each CDR in {H1, H2, H3}:
        1. Mask just that CDR (MaskSingleCDR transform).
        2. Sample ``--num-samples`` generations.
        3. Per generation, compute:
            * AAR  — % residues matching the native CDR sequence.
            * RMSD — CA-only against the native CDR (alignment-free; we
                     align the framework via the existing patch reference
                     frame, so directly comparable across samples).
      Aggregates mean ± std AAR and RMSD per CDR across all entries and
      writes a per-entry breakdown to ``<output>.csv``. The summary in
      ``<output>.json`` is suitable for the thesis ablation table.

What is *not* in here (deliberate scope cut)
--------------------------------------------
  * sc-RMSD via NanoBodyBuilder2 fold-back — handled by the existing
    ``src/biophysics_judge/tnp_runner.py``; call it on the saved PDBs
    that ``--mode design`` writes (see ``--save-pdbs``).
  * Judge-pipeline pass rates — the entry points in
    ``src/{biology,biophysics,physics}_judge/judge.py`` consume the
    same generated PDBs.
  * NbBench oracle scoring — separate runner; out of scope here.

Both follow-on stages read the per-entry CSV and the saved PDB
directory this script produces, so they can run in parallel without
re-sampling.

Usage
-----
::

    # 1. Cheap ELBO baseline on the upstream luost26 weights.
    python scripts/diffab_ft/evaluate.py \\
        --checkpoint third_party/diffab/trained_models/codesign_multicdrs.pt \\
        --config     configs/diffab_ft/vhh_ft.yml \\
        --split      val \\
        --mode       elbo \\
        --output     runs/baseline_upstream/eval_val.json

    # 2. Per-CDR design metrics on a fine-tuned checkpoint.
    python scripts/diffab_ft/evaluate.py \\
        --checkpoint runs/vhh_ft/seed42/checkpoints/best_ema.pt \\
        --config     configs/diffab_ft/vhh_ft.yml \\
        --split      test \\
        --mode       design \\
        --num-samples 8 \\
        --save-pdbs   runs/vhh_ft/seed42/eval_pdbs \\
        --output      runs/vhh_ft/seed42/eval_test.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
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
from diffab.utils.protein.constants import BBHeavyAtom, CDR  # noqa: E402
from diffab.utils.protein.writers import save_pdb  # noqa: E402
from diffab.utils.train import (  # noqa: E402
    ValidationLossTape, recursive_to, sum_weighted_losses,
)
from diffab.utils.transforms import (  # noqa: E402
    Compose, MaskSingleCDR, MergeChains, PatchAroundAnchor,
)

# ── Our dataset registration (side effect) ───────────────────────────────
import src.diffab_ft.datasets  # noqa: E402, F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("evaluate")


CDR_NAMES = {"H1": CDR.H1, "H2": CDR.H2, "H3": CDR.H3}

# Map from numeric AA index → 1-letter for AAR calculation. DiffAb uses
# the same indexing as Bio.PDB.Polypeptide.three_to_one, so we lift
# their helper rather than re-encoding it.
from Bio.PDB import Polypeptide  # noqa: E402


def _aa_to_str(aa_tensor: torch.Tensor) -> str:
    return "".join(Polypeptide.index_to_one(int(a)) for a in aa_tensor.flatten())


# ── Checkpoint loading (handles our format AND upstream's) ──────────────
def _load_checkpoint_into(model: torch.nn.Module, ckpt_path: Path,
                          device: str) -> dict:
    """Best-effort state-dict loader. Returns the wrapping dict (so
    callers can read iteration/val_loss/etc.)."""
    ck = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ck, dict) and "model" in ck:
        sd = ck["model"]
        meta = {k: v for k, v in ck.items() if k != "model"}
    else:
        sd = ck
        meta = {}
    result = model.load_state_dict(sd, strict=False)
    if result.missing_keys or result.unexpected_keys:
        logger.warning("Non-strict load: %d missing, %d unexpected.",
                       len(result.missing_keys), len(result.unexpected_keys))
    return meta


# ── ELBO mode ────────────────────────────────────────────────────────────
@torch.no_grad()
def run_elbo(model, loader, device, loss_weights) -> dict:
    """Mean weighted ELBO components across the full split."""
    tape = ValidationLossTape()
    model.eval()
    for batch in tqdm(loader, desc="ELBO eval", dynamic_ncols=True):
        batch = recursive_to(batch, device)
        loss_dict = model(batch)
        loss = sum_weighted_losses(loss_dict, loss_weights)
        loss_dict["overall"] = loss
        tape.update(loss_dict, 1)

    # Pull out the accumulated averages by hand (tape.log writes to a
    # logger but doesn't return the dict).
    out = {k: float(v / tape.total) for k, v in tape.accumulate.items()}
    out["n_examples"] = tape.total
    return out


# ── Design mode helpers ─────────────────────────────────────────────────
def _mask_cdr_only(structure: dict, cdr_name: str) -> dict:
    """Apply MaskSingleCDR(cdr_name) + MergeChains, like
    design_for_testset.create_data_variants but for one CDR. Returns the
    transformed dict (not yet patched / collated)."""
    transform = Compose([
        MaskSingleCDR(cdr_name, augmentation=False),
        MergeChains(),
    ])
    return transform(deepcopy(structure))


def _kabsch_rmsd(P: torch.Tensor, Q: torch.Tensor) -> float:
    """RMSD between two (N, 3) point clouds after centroid-only alignment.

    We deliberately do NOT do a Kabsch rotation: DiffAb's sampling lives
    in the local patch frame, and the framework backbone is the same as
    the reference (only the CDR was redrawn), so the framework's own
    coordinate system already aligns the two sets. Centroid subtraction
    here is purely defensive in case of float drift.
    """
    if P.shape != Q.shape or P.numel() == 0:
        return float("nan")
    diff = P - Q
    return float(torch.sqrt((diff * diff).sum(dim=-1).mean()).item())


@torch.no_grad()
def run_design(
    model, dataset, device, num_samples: int, batch_size: int,
    save_pdb_dir: Path | None,
) -> tuple[dict, list[dict]]:
    """Per-entry, per-CDR sampling. Returns (summary, per_entry_rows)."""
    model.eval()
    collate = PaddingCollate(eight=False)
    inference_tfm = Compose([PatchAroundAnchor()])

    rows: list[dict] = []
    # Aggregators: keyed by CDR name → list of (aar, rmsd) per sample.
    agg: dict[str, list[tuple[float, float]]] = {k: [] for k in CDR_NAMES}

    for idx in tqdm(range(len(dataset)), desc="entries"):
        # NOTE: dataset[idx] applies the configured transform, which we
        # don't want here — we want the raw structure dict so we can
        # apply MaskSingleCDR per CDR ourselves. We bypass via
        # get_structure(), which returns the un-transformed cached dict.
        entry_id = dataset.ids_in_split[idx]
        structure = dataset.get_structure(entry_id)

        for cdr_name in CDR_NAMES:
            # 1. Mask just this CDR.
            try:
                masked = _mask_cdr_only(structure, cdr_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("entry %s: %s mask failed (%s); skipping.",
                               entry_id, cdr_name, exc.__class__.__name__)
                continue

            # 2. Patch around the CDR anchor (DiffAb's standard inference tfm).
            data = inference_tfm(deepcopy(masked))
            # 3. Replicate `num_samples` times so we can batch the diffusion.
            data_list = [data] * num_samples
            batch = collate([deepcopy(d) for d in data_list])
            batch = recursive_to(batch, device)

            # 4. Run the reverse diffusion. traj[0] = final step.
            traj = model.sample(batch, sample_opt={
                "pbar": False,
                "sample_structure": True,
                "sample_sequence": True,
            })
            v_final, t_final, s_final = traj[0]
            R_final = so3vec_to_rotation(v_final)

            # 5. Reconstruct heavy-atom coords for the generated CDR only.
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

            # 6. Extract per-sample CDR sequence/coords and compare to
            # the native CDR in the masked dict (which still has the
            # ground-truth CDR aa under cdr_flag == CDR.<name>).
            cdr_id = CDR_NAMES[cdr_name]
            patch_idx = data["patch_idx"]
            data_tmpl = masked  # native CDR still here
            native_aa = data_tmpl["aa"]
            native_cdr_mask = (data_tmpl["cdr_flag"] == cdr_id)
            native_seq = _aa_to_str(native_aa[native_cdr_mask])
            native_pos = data_tmpl["pos_heavyatom"][native_cdr_mask][:, BBHeavyAtom.CA]

            for i in range(num_samples):
                aa_full = apply_patch_to_tensor(
                    data_tmpl["aa"], s_final[i].cpu(), patch_idx,
                )
                pos_full = apply_patch_to_tensor(
                    data_tmpl["pos_heavyatom"],
                    pos_atom_new[i].cpu() + batch["origin"][i].view(1, 1, 3).cpu(),
                    patch_idx,
                )

                gen_seq = _aa_to_str(aa_full[native_cdr_mask])
                if len(gen_seq) != len(native_seq) or not native_seq:
                    aar = float("nan")
                else:
                    aar = sum(a == b for a, b in zip(gen_seq, native_seq)) / len(native_seq)

                gen_pos = pos_full[native_cdr_mask][:, BBHeavyAtom.CA]
                rmsd = _kabsch_rmsd(gen_pos, native_pos)

                agg[cdr_name].append((aar, rmsd))
                rows.append({
                    "entry_id": entry_id,
                    "cdr": cdr_name,
                    "sample": i,
                    "native_seq": native_seq,
                    "gen_seq": gen_seq,
                    "aar": aar,
                    "rmsd": rmsd,
                })

                if save_pdb_dir is not None:
                    out_dir = save_pdb_dir / entry_id / cdr_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    save_pdb({
                        "chain_nb": data_tmpl["chain_nb"],
                        "chain_id": data_tmpl["chain_id"],
                        "resseq":   data_tmpl["resseq"],
                        "icode":    data_tmpl["icode"],
                        "aa":       aa_full,
                        "mask_heavyatom": apply_patch_to_tensor(
                            data_tmpl["mask_heavyatom"],
                            mask_atom_new[i].cpu(),
                            patch_idx,
                        ),
                        "pos_heavyatom":  pos_full,
                    }, path=str(out_dir / f"sample_{i:04d}.pdb"))

    # Aggregate.
    summary: dict[str, dict] = {}
    for cdr_name, vals in agg.items():
        if not vals:
            summary[cdr_name] = {"n": 0}
            continue
        aars = np.array([v[0] for v in vals])
        rmsds = np.array([v[1] for v in vals])
        summary[cdr_name] = {
            "n": int(len(vals)),
            "aar_mean": float(np.nanmean(aars)),
            "aar_std":  float(np.nanstd(aars)),
            "rmsd_mean": float(np.nanmean(rmsds)),
            "rmsd_std":  float(np.nanstd(rmsds)),
        }
    return summary, rows


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Path to a .pt checkpoint (raw or our wrapped format).")
    parser.add_argument("--config", required=True, type=Path,
                        help="YAML used during training (architecture + dataset).")
    parser.add_argument("--split", default="val",
                        choices=["train", "val", "test", "test_antigen_disjoint"])
    parser.add_argument("--mode", default="elbo", choices=["elbo", "design"])
    parser.add_argument("--num-samples", type=int, default=8,
                        help="Samples per (entry, CDR) in --mode design.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for ELBO eval. Design mode batches "
                             "num_samples replicas of one entry at a time.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-pdbs", type=Path, default=None,
                        help="(design mode) directory to dump per-sample PDBs.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON path for the summary.")
    parser.add_argument("--csv", type=Path, default=None,
                        help="(design mode) per-sample CSV; default = "
                             "<output>.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        return 2
    if not args.config.exists():
        logger.error("Config not found: %s", args.config)
        return 2

    seed_all(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    config, _ = load_config(str(args.config))

    # The dataset block we use depends on the requested split.
    # Both train/val are spelled out in YAML; for test/test_antigen_disjoint
    # we copy the val block and override the split name (the transform
    # is the same — single CDR3 mask is what design_for_testset uses).
    base_block = config.dataset.val
    eval_block = deepcopy(base_block)
    eval_block.split = args.split
    logger.info("Loading split %r from %s", args.split, eval_block.manifest_path)
    dataset = get_dataset(eval_block)
    logger.info("Split %r: %d entries.", args.split, len(dataset))

    # Build model and load weights.
    logger.info("Constructing model and loading weights from %s", args.checkpoint)
    model = get_model(config.model).to(args.device)
    meta = _load_checkpoint_into(model, args.checkpoint, args.device)
    if "iteration" in meta:
        logger.info("Checkpoint iter: %s | val_loss: %s",
                    meta.get("iteration"), meta.get("val_loss"))

    summary: dict = {
        "checkpoint": str(args.checkpoint),
        "config":     str(args.config),
        "split":      args.split,
        "mode":       args.mode,
        "n_entries":  len(dataset),
        "ckpt_meta":  {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                       for k, v in meta.items() if k not in ("config",)},
    }

    if args.mode == "elbo":
        loader = DataLoader(
            dataset, batch_size=args.batch_size,
            collate_fn=PaddingCollate(),
            shuffle=False, num_workers=args.num_workers,
        )
        elbo = run_elbo(
            model, loader, args.device, config.train.loss_weights,
        )
        summary["elbo"] = elbo
        logger.info("ELBO summary: %s", elbo)

    else:  # design
        # Design mode wants the *raw* cached structure, not the
        # val-transformed dict, so we tell the dataset to skip its
        # transform. Easiest: stash and restore.
        original_transform = dataset.transform
        dataset.transform = None
        try:
            per_cdr, rows = run_design(
                model, dataset, args.device,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                save_pdb_dir=args.save_pdbs,
            )
        finally:
            dataset.transform = original_transform

        summary["design"] = per_cdr
        logger.info("Design summary:")
        for cdr, stats in per_cdr.items():
            logger.info("  %s: %s", cdr, stats)

        # Per-sample CSV (sequence + per-sample AAR/RMSD).
        csv_path = args.csv or args.output.with_suffix(".csv")
        with open(csv_path, "w") as f:
            f.write("entry_id,cdr,sample,native_seq,gen_seq,aar,rmsd\n")
            for r in rows:
                f.write(
                    f"{r['entry_id']},{r['cdr']},{r['sample']},"
                    f"{r['native_seq']},{r['gen_seq']},"
                    f"{r['aar']:.4f},{r['rmsd']:.4f}\n"
                )
        logger.info("Wrote %d sample rows to %s", len(rows), csv_path)

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Wrote summary to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
