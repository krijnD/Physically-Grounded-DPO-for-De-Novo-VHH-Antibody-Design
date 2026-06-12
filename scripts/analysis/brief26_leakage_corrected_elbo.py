"""Brief 26 -- leakage-corrected ELBO recompute on the 29-entry shared holdout.

Re-evaluates the anchor (seed42_jfix) and expanded (seed42_jfix_expanded) DiffAb
checkpoints on the SAME 29 entries -- the "old test" split that underpinned the
original SQ1 comparison (Brief 06: val ELBO 0.7316 -> 0.6363, -13 %). Brief 23
flagged 3 added training entries clustering with 2 of these 29 holdout entries
at >=85 % CDR identity (7q6c_K matched by 7jkm_K + 7o31_X at 100 %; 7n9v_J
matched by 5o2u_D at 88.8 %). This script produces the leakage-corrected
percentage improvement.

Inputs (on Snellius):
- Anchor checkpoint:   runs/vhh_ft/seed42_jfix/checkpoints/best_ema.pt
- Expanded checkpoint: runs/vhh_ft/seed42_jfix_expanded/checkpoints/best_ema.pt
- Config that builds the eval dataset: configs/diffab_ft/vhh_ft_expanded.yml
  (expanded LMDB contains all 29 shared-holdout entries; the floor LMDB
  also contains them but per Brief 23's re-clustering some are in floor train --
  using the expanded LMDB lets us score both models on the same preprocessed
  structures regardless of split assignment).

Why a custom per-entry loop (not scripts/diffab_ft/evaluate.py --mode elbo):
The trainer's evaluate.py averages losses with ValidationLossTape and reports a
single per-split number; it has no per-entry output. This script reuses the
same model/dataset machinery but iterates with batch_size=1 to emit per-entry
ELBO. To smooth the single-sample (random t, random CDR mask) variance, each
entry is forwarded n_repeats times (default 8) with a fresh seed and the
losses are averaged.

To make the anchor/expanded comparison strictly apples-to-apples we re-seed
torch's RNG to the SAME value before each model's pass, so the i-th forward
pass on entry e draws the same random t and same masking pattern for both
models. Differences in per-entry ELBO are then attributable to the model
weights, not to evaluation noise.

Outputs (under --out-dir, default tmp_brief26/):
- per_entry_elbo.csv            COMMITTED -- 29 entries x 2 models
                                Columns: entry_id, model, elbo_overall,
                                         elbo_rot, elbo_pos, elbo_seq
- leakage_corrected_summary.txt COMMITTED -- headline numbers
"""
from __future__ import annotations

import argparse
import csv
import sys
from copy import deepcopy
from pathlib import Path

import torch

# ---- Project paths (mirror of scripts/diffab_ft/evaluate.py) -------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "diffab"))

from diffab.datasets import get_dataset  # noqa: E402
from diffab.models import get_model  # noqa: E402
from diffab.utils.data import PaddingCollate  # noqa: E402
from diffab.utils.misc import load_config, seed_all  # noqa: E402
from diffab.utils.train import recursive_to, sum_weighted_losses  # noqa: E402

import src.diffab_ft.datasets  # noqa: E402, F401  (registers vhh_andd)


SHARED_HOLDOUT_29 = [
    "7f5h_C", "7n9v_J", "7ndf_C", "7ph3_C", "7ph4_C", "7q6c_K", "7qbf_B",
    "7qia_C", "7r74_B", "7sk7_K", "7vfa_D", "7vke_B", "7vq0_D", "7wd2_C",
    "7xrp_B", "7zlg_K", "8acf_K", "8cy6_D", "8elq_B", "8fcz_C", "8gsi_F",
    "8hbg_E", "8oud_D", "8pyr_D", "8qot_B", "8r61_C", "8tb7_N", "8u4v_K",
    "8wo4_G",
]
LEAKED_HOLDOUT_2 = ["7q6c_K", "7n9v_J"]


def _load_checkpoint_into(model, ckpt_path, device):
    """Best-effort state-dict loader; mirrors scripts/diffab_ft/evaluate.py."""
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(ck, dict) and "model" in ck:
        sd, meta = ck["model"], {k: v for k, v in ck.items() if k != "model"}
    else:
        sd, meta = ck, {}
    result = model.load_state_dict(sd, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(f"  Non-strict load: {len(result.missing_keys)} missing, "
              f"{len(result.unexpected_keys)} unexpected.")
    return meta


@torch.no_grad()
def per_entry_elbo(model, dataset, indices, ids, loss_weights, device,
                   n_repeats, seed):
    """Forward each (entry, repeat) and average per-entry across repeats.

    Re-seeds torch globally before the loop so the sequence of random
    (t, mask) draws is identical across models -- the only thing differing
    between the anchor and expanded passes is the model weights.
    """
    seed_all(seed)
    collate = PaddingCollate()
    model.eval()

    out: dict[str, dict[str, float]] = {}
    for entry_pos, (idx, eid) in enumerate(zip(indices, ids)):
        agg: dict[str, float] = {}
        for _ in range(n_repeats):
            item = dataset[idx]  # re-applies stochastic transform
            batch = recursive_to(collate([item]), device)
            loss_dict = model(batch)
            overall = sum_weighted_losses(loss_dict, loss_weights)
            for k, v in loss_dict.items():
                agg[k] = agg.get(k, 0.0) + float(v.item() if torch.is_tensor(v) else v)
            agg["overall"] = agg.get("overall", 0.0) + float(
                overall.item() if torch.is_tensor(overall) else overall
            )
        out[eid] = {k: v / n_repeats for k, v in agg.items()}
        print(f"  [{entry_pos + 1:2d}/{len(ids)}] {eid}: "
              f"overall={out[eid]['overall']:.4f} "
              f"(rot={out[eid].get('rot', float('nan')):.4f} "
              f"pos={out[eid].get('pos', float('nan')):.4f} "
              f"seq={out[eid].get('seq', float('nan')):.4f})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor-checkpoint", required=True, type=Path)
    ap.add_argument("--expanded-checkpoint", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path,
                    help="YAML config defining model architecture + dataset block. "
                         "Recommend configs/diffab_ft/vhh_ft_expanded.yml "
                         "(expanded LMDB has all 29 shared-holdout entries).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-repeats", type=int, default=8,
                    help="Forward passes per entry; per-entry ELBO is the mean. "
                         "Smooths the single-draw t/mask variance.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="tmp_brief26")
    args = ap.parse_args()

    if not args.anchor_checkpoint.exists():
        sys.exit(f"ERROR: anchor checkpoint not found: {args.anchor_checkpoint}")
    if not args.expanded_checkpoint.exists():
        sys.exit(f"ERROR: expanded checkpoint not found: {args.expanded_checkpoint}")
    if not args.config.exists():
        sys.exit(f"ERROR: config not found: {args.config}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config, _ = load_config(str(args.config))

    # Build the dataset using the val block (so the val transform --
    # mask_multiple_cdrs + merge_chains + patch_around_anchor -- is applied
    # to each fetched entry, matching how Brief 06 measured val ELBO).
    val_block = deepcopy(config.dataset.val)
    print(f"Building dataset from {val_block.manifest_path} "
          f"(split={val_block.split!r}, LMDB at {val_block.processed_dir}).")
    dataset = get_dataset(val_block)
    db_ids = list(dataset.db_ids) if dataset.db_ids else []
    print(f"LMDB has {len(db_ids)} entries; val split as loaded: {len(dataset)}.")

    # Override ids_in_split with the 29 entries (intersected with the LMDB).
    # The base dataset's __getitem__ does `id = self.ids_in_split[i]; get_structure(id);
    # transform(...)` -- so this re-targets the iteration without rebuilding.
    db_id_set = set(db_ids)
    avail_29 = [eid for eid in SHARED_HOLDOUT_29 if eid in db_id_set]
    missing_29 = [eid for eid in SHARED_HOLDOUT_29 if eid not in db_id_set]
    if missing_29:
        print(f"WARN: {len(missing_29)}/29 shared-holdout entries not in LMDB: "
              f"{missing_29}")
    if len(avail_29) < 27:
        # We need at least 27 (29 minus the 2 affected leakage entries) for the
        # filtered number to be meaningful. Fewer than that and the comparison is
        # not the one the brief defines.
        sys.exit(f"ERROR: only {len(avail_29)}/29 entries available in LMDB; "
                 "comparison cannot be made.")
    dataset.ids_in_split = avail_29
    indices = list(range(len(avail_29)))
    print(f"Targeting {len(avail_29)}/29 shared-holdout entries.")
    for eid in LEAKED_HOLDOUT_2:
        in_or_out = "IN" if eid in db_id_set else "MISSING"
        print(f"  Leakage entry {eid}: {in_or_out}")

    # Build the model architecture once; load weights per checkpoint.
    print(f"\nConstructing model on {args.device}.")
    model = get_model(config.model).to(args.device)
    loss_weights = config.train.loss_weights

    # ---- Run both checkpoints, identical seed/sequence ------------------
    all_rows: list[dict] = []
    per_entry_overall: dict[str, dict[str, float]] = {"anchor": {}, "expanded": {}}
    for name, ckpt_path in [("anchor", args.anchor_checkpoint),
                             ("expanded", args.expanded_checkpoint)]:
        print(f"\n=== {name.upper()} ===")
        print(f"Loading {ckpt_path}")
        meta = _load_checkpoint_into(model, ckpt_path, args.device)
        if "iteration" in meta:
            print(f"  iter={meta.get('iteration')}  val_loss={meta.get('val_loss')}")
        scores = per_entry_elbo(model, dataset, indices, avail_29,
                                 loss_weights, args.device,
                                 n_repeats=args.n_repeats, seed=args.seed)
        for eid, s in scores.items():
            row = {"entry_id": eid, "model": name}
            row.update({f"elbo_{k}": s[k] for k in ("overall", "rot", "pos", "seq")
                         if k in s})
            all_rows.append(row)
            per_entry_overall[name][eid] = s["overall"]

    # ---- Write per-entry CSV --------------------------------------------
    csv_path = out_dir / "per_entry_elbo.csv"
    fieldnames = ["entry_id", "model", "elbo_overall", "elbo_rot",
                   "elbo_pos", "elbo_seq"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nWrote {csv_path} ({len(all_rows)} rows)")

    # ---- Summary --------------------------------------------------------
    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    a_full = _mean(list(per_entry_overall["anchor"].values()))
    e_full = _mean(list(per_entry_overall["expanded"].values()))
    pct_full = (a_full - e_full) / a_full * 100 if a_full else float("nan")

    clean_ids = [e for e in avail_29 if e not in LEAKED_HOLDOUT_2]
    a_clean = _mean([per_entry_overall["anchor"][e] for e in clean_ids])
    e_clean = _mean([per_entry_overall["expanded"][e] for e in clean_ids])
    pct_clean = (a_clean - e_clean) / a_clean * 100 if a_clean else float("nan")

    lines = []
    lines.append("=== SQ1 LEAKAGE-CORRECTED ELBO SUMMARY ===")
    lines.append(f"Config:      {args.config}")
    lines.append(f"Anchor:      {args.anchor_checkpoint}")
    lines.append(f"Expanded:    {args.expanded_checkpoint}")
    lines.append(f"n_repeats:   {args.n_repeats}  (per-entry forward passes; "
                  "ELBO is the mean over t/mask noise)")
    lines.append(f"seed:        {args.seed}  (same RNG seed for both models -> "
                  "matched t/mask draws)")
    lines.append("")
    lines.append(f"UNFILTERED (n={len(avail_29)} shared holdout entries):")
    lines.append(f"  anchor   ELBO = {a_full:.4f}")
    lines.append(f"  expanded ELBO = {e_full:.4f}")
    lines.append(f"  Delta%       = {pct_full:+.2f}%   (Brief 06 SQ1 claim: -13.0%)")
    lines.append("")
    lines.append(f"FILTERED (n={len(clean_ids)}, excluding 7q6c_K + 7n9v_J):")
    lines.append(f"  anchor   ELBO = {a_clean:.4f}")
    lines.append(f"  expanded ELBO = {e_clean:.4f}")
    lines.append(f"  Delta%       = {pct_clean:+.2f}%")
    lines.append("")
    lines.append(f"LEAKAGE EFFECT: {pct_full - pct_clean:+.2f} pp")
    lines.append("  (positive = unfiltered inflated by leakage; "
                  "negative = leakage hurt unfiltered)")
    lines.append("")
    lines.append("PER-ENTRY for the 2 affected holdout entries:")
    for eid in LEAKED_HOLDOUT_2:
        a = per_entry_overall["anchor"].get(eid)
        e = per_entry_overall["expanded"].get(eid)
        if a is not None and e is not None:
            lines.append(f"  {eid}: anchor {a:.4f} -> expanded {e:.4f}, "
                          f"Delta = {a - e:+.4f}")
        else:
            lines.append(f"  {eid}: NOT EVALUATED (missing from LMDB)")
    if missing_29:
        lines.append("")
        lines.append("CAVEAT -- shared-holdout entries missing from LMDB "
                      f"({len(missing_29)}):")
        for eid in missing_29:
            lines.append(f"  {eid}")

    txt = "\n".join(lines)
    print()
    print(txt)
    (out_dir / "leakage_corrected_summary.txt").write_text(txt + "\n")
    print(f"\nWrote {out_dir / 'leakage_corrected_summary.txt'}")


if __name__ == "__main__":
    main()
