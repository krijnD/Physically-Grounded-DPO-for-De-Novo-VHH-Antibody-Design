#!/usr/bin/env python3
"""Check 3: is the Pareto-dominance pair selection balanced across axes,
or driven by one axis?

If "winner Pareto-dominates loser" is mostly because of one axis (say,
repulsion blowing up on losers while the other two axes barely differ),
then DPO is effectively learning a 1-D preference, not a 3-axis one —
much weaker signal, and easy to over-fit.

Reads the pairs parquet, identifies the per-axis score columns (winner
+ loser), computes per-pair per-axis margins (loser − winner; positive
means loser is worse on that axis, which is what dominance requires),
and reports:
  - Distribution of each axis's margin
  - How many pairs are dominated by ALL THREE axes (true 3-D Pareto)
  - Correlation between margins (high → 1-D effective signal)

Also looks for a winner/loser swap (any pair where the "winner" is
worse on a majority of axes).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> int:
    pairs_path = PROJECT_ROOT / "data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs.parquet"
    pairs = pd.read_parquet(pairs_path)
    print(f"Loaded {len(pairs)} pairs")
    print(f"\nColumns ({len(pairs.columns)}):")
    for c in pairs.columns:
        dtype = pairs[c].dtype
        sample = pairs[c].iloc[0]
        if isinstance(sample, str) and len(sample) > 60:
            sample = sample[:60] + "..."
        print(f"  {c:<30} {str(dtype):<12} e.g. {sample!r}")

    # Heuristic: find paired (winner_*, loser_*) score columns.
    score_keywords = ("etotal", "energy", "delta_g", "deltag", "dg",
                      "repulsion", "rep", "attraction", "nonrep", "_e_",
                      "score", "phys")
    win_score_cols, los_score_cols = {}, {}
    for c in pairs.columns:
        lc = c.lower()
        if any(k in lc for k in score_keywords):
            if lc.startswith("winner_") or "_winner_" in lc or lc.endswith("_winner"):
                key = lc.replace("winner_", "").replace("_winner", "").strip("_")
                win_score_cols[key] = c
            elif lc.startswith("loser_") or "_loser_" in lc or lc.endswith("_loser"):
                key = lc.replace("loser_", "").replace("_loser", "").strip("_")
                los_score_cols[key] = c

    common_axes = sorted(set(win_score_cols) & set(los_score_cols))
    print(f"\nMatched score axes (winner + loser): {common_axes}")
    if not common_axes:
        print("\nNo paired (winner_*, loser_*) score columns found.")
        print("Per the handoff, the AAPR judge parquet at "
              "data/results/andd_judge_test_full.parquet has winner-side scores;")
        print("the loser-side scores would be elsewhere (per-AAPR-sample judge output).")
        print("If pairs.parquet doesn't carry them, we need to join with the judge")
        print("output to get per-axis margins. Print the head of pairs.parquet so we know:")
        print(pairs.head().to_string())
        return 0

    # Per-axis margin = loser − winner; we want this POSITIVE (loser worse).
    print(f"\n=== Per-axis margin (loser − winner; positive = loser worse) ===\n")
    margins = {}
    for axis in common_axes:
        wcol, lcol = win_score_cols[axis], los_score_cols[axis]
        w = pd.to_numeric(pairs[wcol], errors="coerce")
        l = pd.to_numeric(pairs[lcol], errors="coerce")
        m = (l - w).dropna()
        margins[axis] = m
        print(f"{axis:<25}  n={len(m):4d}  "
              f"min={m.min():+.2f}  q10={m.quantile(.1):+.2f}  "
              f"median={m.median():+.2f}  q90={m.quantile(.9):+.2f}  "
              f"max={m.max():+.2f}  "
              f"frac_positive={(m > 0).mean():.3f}")

    # 3-axis Pareto check.
    if len(common_axes) >= 2:
        m_df = pd.DataFrame(margins)
        all_positive = (m_df > 0).all(axis=1)
        print(f"\n3-axis Pareto check ({len(common_axes)} axes, all margins > 0):")
        print(f"  {all_positive.sum():4d} / {len(m_df)} pairs ({100*all_positive.mean():.1f}%)")

        majority_negative = (m_df < 0).sum(axis=1) > len(common_axes) // 2
        if majority_negative.any():
            print(f"\n!!  {majority_negative.sum()} pairs have a majority of NEGATIVE axes")
            print(f"    (i.e., 'winner' is worse on most axes — possible label swap)")

        print(f"\nPairwise correlation between axis margins:")
        print(m_df.corr().round(3).to_string())
        print("\nReadability:")
        print("  High pairwise correlation (~ 1) → axes move together → effective 1-D signal.")
        print("  Low correlation (~ 0) → independent axes → true 3-D Pareto signal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
