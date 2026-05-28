#!/usr/bin/env python3
"""Filter the pairs parquet to keep only pairs π_ref already agrees with.

Background
----------
The lwref_distribution diagnostic showed that ~35-40% of training pairs
have negative ref_margin (= L_l_ref - L_w_ref), i.e. π_ref already
prefers the AAPR loser over the GT winner. DPO on those pairs asks the
model to contradict the reference, which destabilises training.

We filter by joining the original pairs parquet with the lwref
diagnostic parquet (one row per pair) and keeping only pairs whose
ref_margin exceeds a threshold. Default threshold = 0.0 (drop only the
contradictory pairs); --strict raises it to median for an even cleaner
pool.

Outputs
-------
A new parquet alongside the original:
    <pair_parquet_dir>/pairs_filtered_marginGT<thr>.parquet

Usage
-----
    python scripts/dpo/filter_pairs_by_ref_margin.py \
        --pairs   data/aapr/.../dpo/pairs.parquet \
        --lwref   data/aapr/.../dpo/lwref_distribution.parquet \
        --threshold 0.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True,
                    help="Original pairs parquet (from select_pareto_pairs.py).")
    ap.add_argument("--lwref", type=Path, required=True,
                    help="lwref_distribution.parquet (from diag_lwref_distribution.py).")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="Keep pairs with ref_margin > threshold. Default 0.0.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path. Default: <pairs dir>/pairs_filtered_marginGT<thr>.parquet")
    args = ap.parse_args()

    if not args.pairs.exists():
        print(f"Pairs parquet not found: {args.pairs}", file=sys.stderr)
        return 2
    if not args.lwref.exists():
        print(f"lwref parquet not found: {args.lwref}", file=sys.stderr)
        return 2

    pairs = pd.read_parquet(args.pairs)
    lwref = pd.read_parquet(args.lwref)
    print(f"Original pairs: {len(pairs)}")
    print(f"lwref rows:     {len(lwref)}")

    # Join on pair_id (lwref carries pair_id, gt_id, split, ref_margin, etc.).
    if "pair_id" not in pairs.columns:
        print("ERROR: pairs parquet has no pair_id column.", file=sys.stderr)
        return 2
    if "pair_id" not in lwref.columns:
        print("ERROR: lwref parquet has no pair_id column.", file=sys.stderr)
        return 2

    merged = pairs.merge(
        lwref[["pair_id", "L_w_ref", "L_l_ref", "ref_margin", "mask_count", "split"]],
        on="pair_id", how="left",
    )
    n_missing = merged["ref_margin"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} pairs have no lwref row (dropped).")
        merged = merged.dropna(subset=["ref_margin"])

    thr = float(args.threshold)
    kept = merged[merged["ref_margin"] > thr].reset_index(drop=True)
    dropped = len(merged) - len(kept)
    print(f"\nThreshold: ref_margin > {thr}")
    print(f"Kept:      {len(kept)}  ({100 * len(kept) / len(merged):.1f}%)")
    print(f"Dropped:   {dropped} ({100 * dropped / len(merged):.1f}%)")

    # Per-split breakdown.
    print("\nPer split:")
    for split in ("train", "val"):
        sub = kept[kept["split"] == split]
        sub_orig = merged[merged["split"] == split]
        n_gts = sub["gt_complex_id"].nunique() if "gt_complex_id" in sub.columns else -1
        print(f"  {split}: kept {len(sub):4d} / {len(sub_orig):4d}  "
              f"({100 * len(sub) / max(1, len(sub_orig)):.1f}%)  "
              f"GTs={n_gts}")

    # Per-GT breakdown — drop GTs that ended up with too few pairs.
    if "gt_complex_id" in kept.columns:
        per_gt = kept.groupby("gt_complex_id").size().describe()
        print(f"\nPer-GT pair count (post-filter):")
        print(f"  min={per_gt['min']:.0f}  median={per_gt['50%']:.0f}  "
              f"max={per_gt['max']:.0f}  GTs={int(per_gt['count'])}")

    out = args.out or (
        args.pairs.parent
        / f"pairs_filtered_marginGT{thr:+.1f}.parquet".replace("+", "p").replace("-", "m")
    )
    # Drop the temp join columns from the output so it remains a drop-in
    # replacement for the original pairs parquet schema-wise.
    cols_to_drop = [c for c in ("L_w_ref", "L_l_ref", "ref_margin",
                                 "mask_count", "split")
                    if c in kept.columns]
    kept_out = kept.drop(columns=cols_to_drop)
    kept_out.to_parquet(out, index=False)
    print(f"\nWrote {len(kept_out)} rows → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
