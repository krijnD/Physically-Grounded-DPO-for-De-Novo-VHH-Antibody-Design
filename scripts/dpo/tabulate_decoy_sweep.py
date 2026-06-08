#!/usr/bin/env python3
"""Brief 17 §9.1 / §9.2 — tabulate the t_decoy sweep + pick the winning t.

Reads the floor lwref parquet (t=0 baseline) and every available
``lwref_per_channel_decoy_t{N}.parquet`` under the dpo/ directory,
computes per-channel iter-0 implicit reward in BOTH no-T units (the
brief §3 / §9 scale that matches the +2.28 figure) and T-scaled units
(diag-script-§8 scale), prints a side-by-side table across all t
values, and picks the winning t for proceeding to §10–§12.

Decision rule (Brief 17.2 §9.2, configurable via flags)
--------------------------------------------------------
Among t values in the **candidate window** ``[--candidate-min-t,
--candidate-max-t]`` (default [1, 6] — the Stage-1 sweep range; t≥7
values are mapping-only and not eligible for selection, because at
high t the decoy approaches an AAPR loser by construction), pick the
one minimizing L2 distance ``sqrt(reward_rot² + reward_pos²)`` to the
origin. Declare ``PICKED_T=<X>`` if that t satisfies BOTH:

  * ``|reward_rot| ≤ --rot-tol`` (default 0.3)
  * ``|reward_pos| ≤ --pos-tol`` (default 0.1)

Otherwise print ``PICKED_T=NONE`` and exit 1; orchestrator decides
whether to expand the sweep or pivot to informative-null writeup.

The classification column in the table uses the symmetric PASS/
PARTIAL/FAIL/ABORT zones from §9.1 for human interpretability, but
the picking rule above is what determines ``PICKED_T``.

Exit codes
----------
0   a candidate t passed (PICKED_T set; ready for §10–§12).
1   no candidate t passed — close at §9.2 or expand sweep.
2   input error (missing files, schema mismatch).

Usage
-----
::

    # Default candidate window [1, 6] per Brief 17.2 Stage 1
    python scripts/dpo/tabulate_decoy_sweep.py

    # Tighten candidate window or tolerances
    python scripts/dpo/tabulate_decoy_sweep.py \\
        --candidate-min-t 1 --candidate-max-t 6 \\
        --rot-tol 0.3 --pos-tol 0.1
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd

CHANNELS = ("rot", "pos", "seq")


def _per_channel_reward(df: pd.DataFrame, T: int) -> dict[str, dict[str, float]]:
    """Return per-channel reward stats. Reward in no-T units = L_l - L_w."""
    out: dict[str, dict[str, float]] = {}
    for ch in CHANNELS:
        r = df[f"L_l_ref_{ch}"] - df[f"L_w_ref_{ch}"]
        out[ch] = {
            "mean":   float(r.mean()),
            "median": float(r.median()),
            "q10":    float(r.quantile(0.1)),
            "q90":    float(r.quantile(0.9)),
            "mean_T": float(r.mean() * T),
        }
    return out


def _composite_stats(df: pd.DataFrame) -> tuple[float, float]:
    """Composite ref_margin mean + pct_pos. Uses pre-computed
    ``ref_margin`` column when present (no-T units), otherwise derives
    from L_l_ref_composite - L_w_ref_composite."""
    if "ref_margin" in df.columns:
        m = df["ref_margin"]
    else:
        m = df["L_l_ref_composite"] - df["L_w_ref_composite"]
    return float(m.mean()), float((m > 0).mean() * 100.0)


def _classify_zone(rewards: dict[str, dict[str, float]],
                   rot_tol: float, pos_tol: float,
                   strict_tol: float = 0.1,
                   abort_threshold: float = 1.0) -> str:
    """Symmetric PASS/PARTIAL/FAIL classification for table display.

    Independent of the picking rule (which uses asymmetric rot/pos
    tolerances). Kept symmetric here so the table column reads the
    same way humans interpreted the §9.1 deliverable.
    """
    rot = rewards["rot"]["mean"]
    pos = rewards["pos"]["mean"]
    if rot > abort_threshold or pos > abort_threshold:
        return "ABORT"
    if abs(rot) <= strict_tol and abs(pos) <= strict_tol:
        return "PASS"
    if abs(rot) <= rot_tol and abs(pos) <= pos_tol:
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
    ap.add_argument("--candidate-min-t", type=int, default=1,
                    help="Lowest t eligible for selection. Brief 17.2 "
                         "Stage-1 range starts at 1.")
    ap.add_argument("--candidate-max-t", type=int, default=6,
                    help="Highest t eligible for selection. Brief 17.2 "
                         "caps the candidate window at 6 — t≥7 values "
                         "are mapping-only (decoy → loser asymptote).")
    ap.add_argument("--rot-tol", type=float, default=0.3,
                    help="|reward_rot| upper bound for PARTIAL pass.")
    ap.add_argument("--pos-tol", type=float, default=0.1,
                    help="|reward_pos| upper bound for PARTIAL pass.")
    ap.add_argument("--output-csv", type=Path, default=None,
                    help="Optional CSV path to dump the full sweep table "
                         "(machine-readable; for the §9.2 deliverable).")
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
    floor_comp, floor_pct = _composite_stats(floor_df)

    decoy_paths = sorted(args.dpo_dir.glob(args.decoy_glob))
    if not decoy_paths:
        print(f"ERROR: no decoy parquets matched {args.decoy_glob} under "
              f"{args.dpo_dir}", file=sys.stderr)
        return 2

    pat = re.compile(r"decoy_t(\d+)\.parquet$")
    decoys: list[tuple[int, Path, dict, float, float]] = []
    for p in decoy_paths:
        m = pat.search(p.name)
        if not m:
            print(f"WARN: skipping {p.name} (regex no match)")
            continue
        t = int(m.group(1))
        df = pd.read_parquet(p)
        rewards = _per_channel_reward(df, args.T)
        comp_mean, comp_pct = _composite_stats(df)
        decoys.append((t, p, rewards, comp_mean, comp_pct))
        print(f"Loading decoy:  t={t:>3}  {p}")
    decoys.sort(key=lambda x: x[0])

    # ── Side-by-side table ────────────────────────────────────────
    print()
    print("=" * 132)
    print("Per-channel iter-0 implicit reward across t_decoy values  "
          "(reward = L_l_ref − L_w_ref, no-T units; brief §3 / §9 scale)")
    print("=" * 132)
    print(
        f"{'t':>4} | {'rot':>8} | {'pos':>8} | {'seq':>8} | "
        f"{'composite':>10} | {'pct_pos':>7} | {'L2(rot,pos)':>12} | "
        f"{'window':>9} | verdict"
    )
    print("-" * 132)

    table_rows: list[dict] = []
    rows = (
        [(0, "floor", floor_rewards, floor_comp, floor_pct)]
        + [(t, str(p), r, cm, cp) for t, p, r, cm, cp in decoys]
    )

    candidates: list[tuple[int, dict, float]] = []  # (t, rewards, L2)
    for t, label, rewards, comp_mean, comp_pct in rows:
        r_rot = rewards["rot"]["mean"]
        r_pos = rewards["pos"]["mean"]
        r_seq = rewards["seq"]["mean"]
        l2 = math.sqrt(r_rot ** 2 + r_pos ** 2)
        in_window = (t > 0 and args.candidate_min_t <= t <= args.candidate_max_t)
        window_label = "candidate" if in_window else ("baseline" if t == 0 else "mapping")
        if t == 0:
            verdict = "(reference)"
        else:
            verdict = _classify_zone(rewards, args.rot_tol, args.pos_tol)
            if in_window:
                candidates.append((t, rewards, l2))
        print(
            f"{t:>4} | {r_rot:>+8.4f} | {r_pos:>+8.4f} | {r_seq:>+8.4f} | "
            f"{comp_mean:>+10.4f} | {comp_pct:>6.1f}% | {l2:>12.4f} | "
            f"{window_label:>9} | {verdict}"
        )
        table_rows.append({
            "t":                       t,
            "reward_rot":              r_rot,
            "reward_pos":              r_pos,
            "reward_seq":              r_seq,
            "reward_rot_T":            rewards["rot"]["mean_T"],
            "reward_pos_T":            rewards["pos"]["mean_T"],
            "reward_seq_T":            rewards["seq"]["mean_T"],
            "composite_margin":        comp_mean,
            "pct_pairs_pos_margin":    comp_pct,
            "l2_rot_pos":              l2,
            "window":                  window_label,
            "verdict":                 verdict,
        })
    print("=" * 132)
    print(
        f"Picking rule (Brief 17.2 §9.2): among t in "
        f"[{args.candidate_min_t}, {args.candidate_max_t}], pick smallest L2; "
        f"require |reward_rot| ≤ {args.rot_tol} AND |reward_pos| ≤ {args.pos_tol}."
    )

    # ── Optional CSV dump for the deliverable ────────────────────
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(table_rows).to_csv(args.output_csv, index=False)
        print(f"Wrote machine-readable sweep table → {args.output_csv}")

    # ── Pick winning t ─────────────────────────────────────────────
    print()
    if not candidates:
        print(
            f"PICK: none. No t in [{args.candidate_min_t}, "
            f"{args.candidate_max_t}] is present in --dpo-dir."
        )
        print("→ Submit Stage-1 jobs at the missing t values, then re-run.")
        print("PICKED_T=NONE")
        return 1

    candidates.sort(key=lambda v: v[2])  # L2 ascending
    best_t, best_rewards, best_l2 = candidates[0]
    rot_ok = abs(best_rewards["rot"]["mean"]) <= args.rot_tol
    pos_ok = abs(best_rewards["pos"]["mean"]) <= args.pos_tol
    print(
        f"Best candidate: t={best_t}  "
        f"reward_rot={best_rewards['rot']['mean']:+.4f}  "
        f"reward_pos={best_rewards['pos']['mean']:+.4f}  "
        f"L2={best_l2:.4f}"
    )
    print(
        f"  rot test: |{best_rewards['rot']['mean']:+.4f}| ≤ {args.rot_tol}  → "
        f"{'OK' if rot_ok else 'FAIL'}"
    )
    print(
        f"  pos test: |{best_rewards['pos']['mean']:+.4f}| ≤ {args.pos_tol}  → "
        f"{'OK' if pos_ok else 'FAIL'}"
    )

    if rot_ok and pos_ok:
        print(f"PICK: t={best_t} satisfies both thresholds. Proceed to §10–§12.")
        if len(candidates) > 1:
            others = ", ".join(f"t={t} (L2={l2:.4f})" for t, _, l2 in candidates[1:])
            print(f"  Other in-window candidates by L2: {others}")
        print(f"PICKED_T={best_t}")
        return 0

    print(
        "PICK: none. Best in-window candidate fails the asymmetric "
        "(rot, pos) tolerance test."
    )
    print(
        "→ Brief 17.2 §9.2 decision: close deliverable at §9.2; full sweep "
        "table + bathtub figure is the deliverable. Orchestrator confirms."
    )
    print("PICKED_T=NONE")
    return 1


if __name__ == "__main__":
    sys.exit(main())
