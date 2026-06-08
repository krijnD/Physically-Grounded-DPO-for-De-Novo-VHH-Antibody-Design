#!/usr/bin/env python3
"""Brief 17 §9.1 — tabulate the t_decoy sweep + pick the winning t.

Reads the floor lwref parquet (t=0 baseline) and every available
``lwref_per_channel_decoy_t{N}.parquet`` under the dpo/ directory,
computes per-channel iter-0 implicit reward in BOTH the brief-§9 units
(no-T, matches the +2.28 figure in §3) and the diag-script-§8 units
(T-scaled), prints a side-by-side table across all t values, and
classifies each candidate t against the brief §9 / §9.1 rule:

  PASS    if |reward_rot| ≤ 0.1  AND  |reward_pos| ≤ 0.1   (no-T)
  PARTIAL if |reward_rot| ≤ 0.3  AND  |reward_pos| ≤ 0.3   (no-T)
  FAIL    otherwise

If multiple t-values PASS, picks the one closest to the (0, 0) origin
in (reward_rot, reward_pos) L2 distance (smallest absolute deviation),
per the orchestrator's tiebreak rule (item 6 of the §9.1 supplement).

Exit codes
----------
0   at least one t passed (PASS or PARTIAL). Picked t printed on the
    last stdout line as ``PICKED_T=<int>``.
1   no t qualified for even PARTIAL — orchestrator must decide whether
    to expand the sweep or pivot to informative-null writeup.
2   input error (missing files, schema mismatch).

Usage
-----
::

    python scripts/dpo/tabulate_decoy_sweep.py \\
        --dpo-dir data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo \\
        --floor-name lwref_per_channel_floor.parquet \\
        --decoy-glob 'lwref_per_channel_decoy_t*.parquet'
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd

CHANNELS = ("rot", "pos", "seq")
PASS_TOL = 0.1
PARTIAL_TOL = 0.3
ABORT_THRESHOLD = 1.0


def _per_channel_reward(df: pd.DataFrame, T: int) -> dict[str, dict[str, float]]:
    """Return per-channel reward stats. Reward in no-T units = L_l - L_w."""
    out: dict[str, dict[str, float]] = {}
    for ch in CHANNELS:
        # no-T units: matches the brief §3 / §9 gate tolerance scale.
        r = df[f"L_l_ref_{ch}"] - df[f"L_w_ref_{ch}"]
        out[ch] = {
            "mean":   float(r.mean()),
            "median": float(r.median()),
            "q10":    float(r.quantile(0.1)),
            "q90":    float(r.quantile(0.9)),
            "mean_T": float(r.mean() * T),
        }
    return out


def _classify(rewards: dict[str, dict[str, float]]) -> str:
    rot = abs(rewards["rot"]["mean"])
    pos = abs(rewards["pos"]["mean"])
    if rewards["rot"]["mean"] > ABORT_THRESHOLD or rewards["pos"]["mean"] > ABORT_THRESHOLD:
        return "ABORT"
    if rot <= PASS_TOL and pos <= PASS_TOL:
        return "PASS"
    if rot <= PARTIAL_TOL and pos <= PARTIAL_TOL:
        return "PARTIAL"
    return "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dpo-dir", type=Path,
                    default=Path("data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo"),
                    help="Directory containing the lwref parquets.")
    ap.add_argument("--floor-name", default="lwref_per_channel_floor.parquet",
                    help="Floor (GT-crystal winners) parquet filename.")
    ap.add_argument("--decoy-glob", default="lwref_per_channel_decoy_t*.parquet",
                    help="Glob (relative to --dpo-dir) matching the "
                         "decoy lwref parquets; t is extracted from the "
                         "filename via regex ...decoy_t(\\d+).parquet.")
    ap.add_argument("--T", type=int, default=100,
                    help="Diffusion horizon (informational T-scaling).")
    args = ap.parse_args()

    if not args.dpo_dir.is_dir():
        print(f"ERROR: --dpo-dir not found: {args.dpo_dir}", file=sys.stderr)
        return 2

    floor_path = args.dpo_dir / args.floor_name
    if not floor_path.exists():
        print(f"ERROR: floor parquet not found: {floor_path}", file=sys.stderr)
        return 2

    print(f"Loading floor:  {floor_path}")
    floor_df = pd.read_parquet(floor_path)
    floor_rewards = _per_channel_reward(floor_df, args.T)

    decoy_paths = sorted(args.dpo_dir.glob(args.decoy_glob))
    if not decoy_paths:
        print(f"ERROR: no decoy parquets matched {args.decoy_glob} under "
              f"{args.dpo_dir}", file=sys.stderr)
        return 2

    pat = re.compile(r"decoy_t(\d+)\.parquet$")
    decoys: list[tuple[int, Path, dict[str, dict[str, float]]]] = []
    for p in decoy_paths:
        m = pat.search(p.name)
        if not m:
            print(f"WARN: skipping {p.name} (regex no match)")
            continue
        t = int(m.group(1))
        df = pd.read_parquet(p)
        rewards = _per_channel_reward(df, args.T)
        decoys.append((t, p, rewards))
        print(f"Loading decoy:  t={t:>2}  {p}")
    decoys.sort(key=lambda x: x[0])

    # ── Side-by-side table ────────────────────────────────────────
    print()
    print("=" * 110)
    print("Per-channel iter-0 implicit reward across t_decoy values")
    print("=" * 110)
    print(f"{'t':>4} | {'reward_rot (mean / T·mean)':>30} | "
          f"{'reward_pos (mean / T·mean)':>30} | "
          f"{'reward_seq (mean / T·mean)':>30} | verdict")
    print("-" * 110)

    rows = [(0, "floor", floor_rewards)] + [(t, str(p), r) for t, p, r in decoys]
    verdicts: list[tuple[int, str, dict, float]] = []
    for t, label, rewards in rows:
        verdict = _classify(rewards) if t > 0 else "(reference)"
        r_rot, r_pos = rewards["rot"]["mean"], rewards["pos"]["mean"]
        l2 = math.sqrt(r_rot ** 2 + r_pos ** 2)
        verdicts.append((t, verdict, rewards, l2))
        print(
            f"{t:>4} | "
            f"{rewards['rot']['mean']:>+9.4f} / {rewards['rot']['mean_T']:>+9.2f}    | "
            f"{rewards['pos']['mean']:>+9.4f} / {rewards['pos']['mean_T']:>+9.2f}    | "
            f"{rewards['seq']['mean']:>+9.4f} / {rewards['seq']['mean_T']:>+9.2f}    | "
            f"{verdict}"
        )
    print("=" * 110)
    print(f"PASS zone:    |reward_rot| ≤ {PASS_TOL}   AND  |reward_pos| ≤ {PASS_TOL}    (no-T units)")
    print(f"PARTIAL zone: |reward_rot| ≤ {PARTIAL_TOL}   AND  |reward_pos| ≤ {PARTIAL_TOL}    (no-T units)")
    print(f"ABORT zone:   reward_rot   > +{ABORT_THRESHOLD}   OR   reward_pos > +{ABORT_THRESHOLD}    (no-T units)")

    # ── Pick winning t ─────────────────────────────────────────────
    print()
    candidates_pass = [v for v in verdicts if v[1] == "PASS"]
    candidates_partial = [v for v in verdicts if v[1] == "PARTIAL"]

    if candidates_pass:
        candidates_pass.sort(key=lambda v: v[3])  # L2 to origin
        picked = candidates_pass[0]
        print(f"PICK: t={picked[0]} (PASS; L2 to origin = {picked[3]:.4f}).")
        if len(candidates_pass) > 1:
            others = [f"t={v[0]} (L2={v[3]:.4f})" for v in candidates_pass[1:]]
            print(f"  Other PASS candidates: {', '.join(others)}")
        print(f"PICKED_T={picked[0]}")
        return 0
    if candidates_partial:
        candidates_partial.sort(key=lambda v: v[3])
        picked = candidates_partial[0]
        print(f"PICK: t={picked[0]} (PARTIAL; L2 to origin = {picked[3]:.4f}). "
              f"Proceed but flag in deliverable.")
        if len(candidates_partial) > 1:
            others = [f"t={v[0]} (L2={v[3]:.4f})" for v in candidates_partial[1:]]
            print(f"  Other PARTIAL candidates: {', '.join(others)}")
        print(f"PICKED_T={picked[0]}")
        return 0

    print("PICK: none. No t value satisfies PASS or PARTIAL.")
    print("→ Per orchestrator §9.1 step 4: ping orchestrator with the "
          "full sweep table before proceeding. Options are to expand "
          "the sweep (e.g. t=6, t=11) or pivot to informative-null "
          "writeup.")
    print("PICKED_T=NONE")
    return 1


if __name__ == "__main__":
    sys.exit(main())
