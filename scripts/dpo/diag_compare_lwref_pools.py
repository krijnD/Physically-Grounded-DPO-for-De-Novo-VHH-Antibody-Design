#!/usr/bin/env python3
"""Compare two lwref-per-channel parquets and abort if the winner-side
values are byte-identical — the smoking-gun detector for the manifest-
vs-loader drift that caused the 2026-06-08 incident.

Background
----------
Brief 17 §7 swapped the ``winner_pdb_path`` column from GT-crystal paths
to decoy paths, expecting the per-channel ref-loss diagnostic to read
different winner structures. It didn't: ``PairDataset.__getitem__`` was
pulling winners from the LMDB by ``gt_complex_id`` and ignoring
``winner_pdb_path`` entirely. The bug was silent — the diag script
finished cleanly, the output parquets were written to different paths,
but the per-channel reward summaries matched byte-for-byte across the
floor and decoy pools.

This validator catches that class of bug post-hoc. Given two lwref
parquets that *should* differ on the winner side (e.g., a "floor" run
and a "decoy" run on the same loser pool), it joins on ``pair_id`` and
asserts the ``L_w_ref_{rot,pos,seq}`` columns differ in mean.

Exit codes
----------
0   pools differ as expected (winner substitution had an effect).
1   pools are byte-identical on the winner side (BUG: substitution
    was a no-op; investigate the loader).
2   input error (missing file, schema mismatch).

Usage
-----
::

    python scripts/dpo/diag_compare_lwref_pools.py \\
        --floor  data/aapr/.../dpo/lwref_per_channel_floor.parquet \\
        --decoy  data/aapr/.../dpo/lwref_per_channel_decoy_t10.parquet \\
        --tol    1e-6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CHANNELS = ("rot", "pos", "seq")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--floor", required=True, type=Path,
                    help="Reference (e.g., GT-crystal winners) parquet.")
    ap.add_argument("--decoy", required=True, type=Path,
                    help="Alternative (e.g., decoy winners) parquet that "
                         "MUST differ from --floor on the winner side.")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="Per-pair |L_w_ref_floor - L_w_ref_decoy| tolerance; "
                         "if every pair is within tol on every channel, "
                         "treat as byte-identical and fail. Default 1e-6.")
    ap.add_argument("--min-frac-changed", type=float, default=0.5,
                    help="Minimum fraction of pairs that must show a "
                         "delta > tol on at least one channel. Default 0.5 "
                         "(half the pool should move at decoy_t=10).")
    args = ap.parse_args()

    for p in (args.floor, args.decoy):
        if not p.exists():
            print(f"ERROR: parquet not found: {p}", file=sys.stderr)
            return 2

    floor = pd.read_parquet(args.floor)
    decoy = pd.read_parquet(args.decoy)
    print(f"floor: {len(floor)} rows from {args.floor}")
    print(f"decoy: {len(decoy)} rows from {args.decoy}")

    for col in ("pair_id",) + tuple(f"L_w_ref_{ch}" for ch in CHANNELS):
        for name, df in (("floor", floor), ("decoy", decoy)):
            if col not in df.columns:
                print(f"ERROR: {name} parquet missing column {col!r}; "
                      f"have {list(df.columns)}", file=sys.stderr)
                return 2

    cols = [f"L_w_ref_{ch}" for ch in CHANNELS] + [f"L_l_ref_{ch}" for ch in CHANNELS]
    merged = floor[["pair_id"] + cols].merge(
        decoy[["pair_id"] + cols], on="pair_id", suffixes=("_floor", "_decoy"),
    )
    if not len(merged):
        print(f"ERROR: zero pair_ids joined across the two parquets", file=sys.stderr)
        return 2

    print(f"joined: {len(merged)} pairs")
    print()
    print("=== Winner-side channel deltas (decoy − floor) ===")
    per_channel_changed = {}
    for ch in CHANNELS:
        delta = merged[f"L_w_ref_{ch}_decoy"] - merged[f"L_w_ref_{ch}_floor"]
        n_changed = (delta.abs() > args.tol).sum()
        frac_changed = n_changed / len(merged)
        per_channel_changed[ch] = frac_changed
        print(
            f"  L_w_ref_{ch}: "
            f"mean(decoy)={merged[f'L_w_ref_{ch}_decoy'].mean():+.4f}  "
            f"mean(floor)={merged[f'L_w_ref_{ch}_floor'].mean():+.4f}  "
            f"Δmean={delta.mean():+.4f}  "
            f"|Δ|>tol on {n_changed}/{len(merged)} pairs ({frac_changed * 100:.1f}%)"
        )
    print()
    print("=== Loser-side channel deltas (decoy − floor) ===")
    print("  (expected ≈ 0 — losers are not swapped; tiny rounding only)")
    for ch in CHANNELS:
        delta = merged[f"L_l_ref_{ch}_decoy"] - merged[f"L_l_ref_{ch}_floor"]
        n_changed = (delta.abs() > args.tol).sum()
        print(
            f"  L_l_ref_{ch}: "
            f"Δmean={delta.mean():+.4f}  "
            f"|Δ|>tol on {n_changed}/{len(merged)} pairs"
        )

    print()
    any_changed_pair = (
        (
            (merged[[f"L_w_ref_{ch}_decoy" for ch in CHANNELS]].to_numpy()
             - merged[[f"L_w_ref_{ch}_floor" for ch in CHANNELS]].to_numpy())
            ** 2
        ).sum(axis=1) > args.tol ** 2
    )
    frac_any = float(any_changed_pair.mean())
    print(f"Fraction of pairs with ANY winner-side channel changed: {frac_any * 100:.1f}%")

    max_channel_frac = max(per_channel_changed.values())
    if max_channel_frac < args.min_frac_changed:
        print()
        print("=" * 70)
        print(
            f"FAIL: winner-side substitution had effectively no effect "
            f"(max channel frac-changed = {max_channel_frac * 100:.1f}% < "
            f"{args.min_frac_changed * 100:.1f}% threshold)."
        )
        print(
            "The decoy parquet's winners are matching the floor parquet's "
            "winners byte-for-byte (within tol). Likely cause: the loader "
            "is ignoring winner_pdb_path. See src/dpo/dataset.py "
            "_resolve_winner_source — was the patch reverted?"
        )
        print("=" * 70)
        return 1

    print()
    print(
        f"PASS: pools differ as expected "
        f"(max channel frac-changed = {max_channel_frac * 100:.1f}%)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
