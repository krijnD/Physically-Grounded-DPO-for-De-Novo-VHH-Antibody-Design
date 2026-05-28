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
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_axes_json(s: str) -> dict:
    """Parse one axes_{winner,loser} JSON blob into a flat dict."""
    if not isinstance(s, str):
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def main() -> int:
    pairs_path = PROJECT_ROOT / "data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs.parquet"
    pairs = pd.read_parquet(pairs_path)
    print(f"Loaded {len(pairs)} pairs")

    # Parse the JSON axes columns. Each row → dict of axis → score.
    win_axes = pairs["axes_winner"].apply(parse_axes_json)
    los_axes = pairs["axes_loser"].apply(parse_axes_json)
    win_df = pd.json_normalize(win_axes).add_prefix("w_")
    los_df = pd.json_normalize(los_axes).add_prefix("l_")
    df = pd.concat([pairs[["pair_id", "gt_complex_id", "dominance_margin"]],
                    win_df, los_df], axis=1)

    # Discover axes — any name that appears as both w_<x> and l_<x>.
    w_axes = {c[2:] for c in df.columns if c.startswith("w_")}
    l_axes = {c[2:] for c in df.columns if c.startswith("l_")}
    axes = sorted(w_axes & l_axes)
    print(f"\nAxes parsed from JSON ({len(axes)}): {axes}")

    print(f"\n=== Per-axis margin (loser − winner; positive = loser is worse) ===\n")
    margins = {}
    for ax in axes:
        m = pd.to_numeric(df[f"l_{ax}"], errors="coerce") - pd.to_numeric(df[f"w_{ax}"], errors="coerce")
        m = m.dropna()
        margins[ax] = m
        print(f"{ax:<25}  n={len(m):4d}  "
              f"min={m.min():+8.2f}  q10={m.quantile(.1):+8.2f}  "
              f"median={m.median():+8.2f}  q90={m.quantile(.9):+8.2f}  "
              f"max={m.max():+8.2f}  "
              f"frac_pos={(m > 0).mean():.3f}  "
              f"frac_zero={(m == 0).mean():.3f}")

    m_df = pd.DataFrame(margins)

    # True Pareto: loser is strictly worse on ALL axes.
    all_pos = (m_df > 0).all(axis=1)
    weak_pareto = ((m_df >= 0).all(axis=1) & (m_df > 0).any(axis=1))
    print(f"\nPareto dominance counts ({len(axes)} axes):")
    print(f"  strict (all margins > 0):  {all_pos.sum():4d} / {len(m_df)} ({100*all_pos.mean():.1f}%)")
    print(f"  weak   (all >= 0, any > 0): {weak_pareto.sum():4d} / {len(m_df)} ({100*weak_pareto.mean():.1f}%)")

    any_neg = (m_df < 0).any(axis=1)
    print(f"  has at least one NEGATIVE axis (winner worse on some axis): "
          f"{any_neg.sum():4d} / {len(m_df)} ({100*any_neg.mean():.1f}%)")
    majority_neg = (m_df < 0).sum(axis=1) > len(axes) / 2
    print(f"  majority negative (possible label swap): {majority_neg.sum():4d}")

    print(f"\nPairwise correlation between axis margins:")
    print(m_df.corr().round(3).to_string())
    print("\nReadability:")
    print("  Pairwise correlation ~ 1 between axes  → axes move together → effective 1-D signal.")
    print("  Correlation ~ 0                        → independent → true multi-D Pareto.")

    print(f"\ndominance_margin (existing scalar in parquet):")
    dm = pd.to_numeric(df["dominance_margin"], errors="coerce").dropna()
    print(f"  min={dm.min():.2f}  q10={dm.quantile(.1):.2f}  median={dm.median():.2f}  "
          f"q90={dm.quantile(.9):.2f}  max={dm.max():.2f}")

    # Which axis drives dominance_margin most?
    print(f"\nCorrelation of each axis margin with dominance_margin:")
    for ax in axes:
        rho = m_df[ax].corr(dm)
        print(f"  {ax:<25} ρ={rho:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
