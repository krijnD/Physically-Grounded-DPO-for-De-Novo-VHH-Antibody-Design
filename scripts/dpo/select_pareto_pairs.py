#!/usr/bin/env python3
"""Pareto pair selection for DPO training (post-AAPR scoring).

Implements the strict-Pareto-dominance pair-selection algorithm specified
in the April 2026 thesis proposal §4.4. For each scored AAPR candidate
(loser y_l), this script picks the parent GT (winner y_w = the natural
VHH the candidate was sampled from) and emits a valid (y_w, y_l) DPO
training pair iff:

  1. y_l fails at least one judge (biology / biophysics / physics) — the
     "Physical-Reject" step of AAPR. Candidates that pass all judges
     are NOT hard negatives and are skipped.
  2. y_w STRICTLY DOMINATES y_l on the 3-axis evaluation vector
     (PSH_outside_zone, cdr_energy_per_res, e_rep):
        y_w ≼ y_l on all 3 axes AND y_w < y_l on at least one.

Ambiguous pairs (y_l better on any axis) are discarded — they'd teach DPO
the wrong lesson per proposal §4.4 Scenario B ("don't bind so tightly").

Pareto axes
-----------
- PSH_outside_zone = max(0, psh - 126.83) + max(0, 79.59 - psh).
  0 inside Gordon's green zone, positive outside, lower is better.
  Treats over-hydrophobic and under-hydrophobic equally bad.

- cdr_energy_per_res: lower is better (more negative = better binder).
  No transform. Maps to the proposal's ΔG_bind axis.

- e_rep: lower is better (less steric repulsion). No transform.

Inputs
------
- AAPR-scored parquet: output of `scripts/test_sabdab_judges.py` on the
  AAPR candidate manifest. Must include `gt_complex_id`, `sample_idx`,
  the 3 verdicts, and the 3 axes.
- GT reference parquet: `data/results/andd_calibration_full.parquet` —
  per-GT scores against which losers are tested for dominance.

Output
------
One parquet, one row per valid pair, with columns:
  pair_id, winner_candidate_id, winner_pdb_path,
  loser_candidate_id, loser_pdb_path,
  gt_complex_id, loser_sample_idx,
  axes_winner (json), axes_loser (json),
  dominance_margin (sum of positive (loser - winner) across axes).

Usage
-----
    python scripts/dpo/select_pareto_pairs.py \\
        --aapr-parquet data/results/aapr_<run_id>_scored.parquet \\
        --gt-parquet   data/results/andd_calibration_full.parquet \\
        --output       data/results/dpo_pairs_<run_id>.parquet

See docs/dpo_pair_selection_context.md for the full Phase B handoff.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
log = logging.getLogger("select_pareto_pairs")

# ── Pareto axes — locked per the conversation that produced this script ──
# Source: April 2026 proposal §4.4 + decisions captured in
# docs/dpo_pair_selection_context.md §3.
PSH_GREEN_LOW = 79.59
PSH_GREEN_HIGH = 126.83
AXES = ["psh_outside_zone", "cdr_energy_per_res", "e_rep"]


def _normalize_complex_id(raw: str) -> str:
    """Strip the chain suffix so AAPR's {pdb}_{chain} joins GT's {pdb}.

    AAPR's `gt_complex_id` comes from DiffAb's dataset entry_id format
    (`{pdb}_{vhh_chain}`, e.g. `7f5h_C`). The GT calibration parquet's
    `candidate_id` is the bare PDB ID (`7f5h`) — there's one curated VHH
    per PDB in the ANDD set, so the chain suffix is informational. Strip
    everything after the first underscore on both sides so the join is
    symmetric and robust to future format drift.
    """
    return raw.split("_", 1)[0] if raw else raw


def psh_outside_zone(psh: float) -> float:
    """Distance outside Gordon's PSH green zone. 0 inside, positive outside."""
    if pd.isna(psh):
        return float("nan")
    return max(0.0, psh - PSH_GREEN_HIGH) + max(0.0, PSH_GREEN_LOW - psh)


def is_loser_eligible(row: pd.Series) -> bool:
    """Loser pre-filter: candidate must fail ≥1 judge (decision #2)."""
    return (
        row.get("biology_verdict") != "pass"
        or row.get("biophysics_verdict") != "pass"
        or row.get("physics_verdict") != "pass"
    )


def has_valid_axes(row: pd.Series) -> bool:
    """All three axes must be populated. NaN is disqualifying — we cannot
    compare Pareto with missing components."""
    for a in ("psh_score", "cdr_energy_per_res", "e_rep"):
        v = row.get(a)
        if v is None or pd.isna(v):
            return False
    return True


def strictly_dominates(winner: dict, loser: dict) -> tuple[bool, float]:
    """Strict Pareto dominance test on the three transformed axes.

    Returns (dominates, margin). `margin` is the sum of positive
    (loser - winner) across axes — i.e. how much "worse" the loser is on
    aggregate. Returned only when dominance holds; 0.0 otherwise.

    Strict rule (per proposal §4.4):
      ∀ axis: winner[axis] ≤ loser[axis]
      ∃ axis: winner[axis] < loser[axis]
    """
    no_worse = True
    strictly_better_anywhere = False
    margin = 0.0
    for a in AXES:
        diff = loser[a] - winner[a]   # positive = winner better
        if diff < 0:
            no_worse = False
            break
        if diff > 0:
            strictly_better_anywhere = True
            margin += diff
    if no_worse and strictly_better_anywhere:
        return True, margin
    return False, 0.0


def build_winner_pool(gt_df: pd.DataFrame) -> dict[str, dict]:
    """Index GT rows by candidate_id (matches AAPR's gt_complex_id) and
    project onto the 3 Pareto axes.

    Returns a dict {gt_id: {axes..., pdb_path, candidate_id}} for O(1)
    lookup during the loser scan.
    """
    pool: dict[str, dict] = {}
    n_dropped = 0
    for _, row in gt_df.iterrows():
        if not has_valid_axes(row):
            n_dropped += 1
            continue
        gid = _normalize_complex_id(str(row["candidate_id"]))
        pool[gid] = {
            "candidate_id":      gid,
            "pdb_path":          row.get("complex_pdb_path") or row.get("pdb_filepath"),
            "psh_outside_zone":  psh_outside_zone(row["psh_score"]),
            "cdr_energy_per_res": float(row["cdr_energy_per_res"]),
            "e_rep":             float(row["e_rep"]),
            # Raw axes too — useful for the output's audit columns.
            "psh_score":         float(row["psh_score"]),
        }
    log.info(
        "GT pool: %d valid winners (%d dropped for missing axes)",
        len(pool), n_dropped,
    )
    return pool


def select_pairs(
    aapr_df: pd.DataFrame,
    winner_pool: dict[str, dict],
) -> pd.DataFrame:
    """Iterate scored AAPR candidates, emit valid (winner, loser) pairs."""
    rows: list[dict] = []
    stats = {
        "total": len(aapr_df),
        "skipped_no_gt":     0,
        "skipped_pass_all":  0,
        "skipped_nan_axes":  0,
        "skipped_no_dominance": 0,
        "pairs_emitted":     0,
    }
    for _, row in aapr_df.iterrows():
        gid = row.get("gt_complex_id")
        if gid is None or pd.isna(gid):
            stats["skipped_no_gt"] += 1
            continue
        gid = _normalize_complex_id(str(gid))
        winner = winner_pool.get(gid)
        if winner is None:
            stats["skipped_no_gt"] += 1
            continue
        if not is_loser_eligible(row):
            stats["skipped_pass_all"] += 1
            continue
        if not has_valid_axes(row):
            stats["skipped_nan_axes"] += 1
            continue

        loser = {
            "psh_outside_zone":   psh_outside_zone(row["psh_score"]),
            "cdr_energy_per_res": float(row["cdr_energy_per_res"]),
            "e_rep":              float(row["e_rep"]),
            "psh_score":          float(row["psh_score"]),
        }
        dominates, margin = strictly_dominates(winner, loser)
        if not dominates:
            stats["skipped_no_dominance"] += 1
            continue

        cand_id = str(row["candidate_id"])
        rows.append({
            "pair_id":             f"{gid}__vs__{cand_id}",
            "winner_candidate_id": winner["candidate_id"],
            "winner_pdb_path":     winner["pdb_path"],
            "loser_candidate_id":  cand_id,
            "loser_pdb_path":      row.get("complex_pdb_path"),
            "gt_complex_id":       gid,
            "loser_sample_idx":    (int(row["sample_idx"])
                                    if not pd.isna(row.get("sample_idx"))
                                    else None),
            "axes_winner":         json.dumps({a: winner[a] for a in AXES + ["psh_score"]}),
            "axes_loser":          json.dumps({a: loser[a] for a in AXES + ["psh_score"]}),
            "dominance_margin":    margin,
        })
        stats["pairs_emitted"] += 1

    log.info("=" * 60)
    log.info("Selection stats:")
    for k, v in stats.items():
        log.info("  %-25s  %d", k, v)
    log.info("=" * 60)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--aapr-parquet", required=True, type=Path,
        help="Scored AAPR candidate parquet (from test_sabdab_judges.py on the "
             "AAPR manifest). Must include gt_complex_id, sample_idx, the 3 "
             "verdicts, and the 3 axes.",
    )
    parser.add_argument(
        "--gt-parquet", required=True, type=Path,
        help="GT reference parquet (default: data/results/andd_calibration_full.parquet). "
             "Provides the winner pool; rows are indexed by candidate_id matching "
             "AAPR's gt_complex_id.",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output parquet path for the DPO pairs.",
    )
    args = parser.parse_args()

    if not args.aapr_parquet.exists():
        log.error("AAPR parquet not found: %s", args.aapr_parquet)
        return 2
    if not args.gt_parquet.exists():
        log.error("GT parquet not found: %s", args.gt_parquet)
        return 2

    log.info("Reading AAPR-scored parquet: %s", args.aapr_parquet)
    aapr_df = pd.read_parquet(args.aapr_parquet)
    log.info("  %d candidate rows", len(aapr_df))

    log.info("Reading GT reference parquet: %s", args.gt_parquet)
    gt_df = pd.read_parquet(args.gt_parquet)
    log.info("  %d GT rows", len(gt_df))

    winner_pool = build_winner_pool(gt_df)
    if not winner_pool:
        log.error("Empty winner pool — every GT row failed has_valid_axes(). "
                  "Cannot select pairs.")
        return 1

    pairs_df = select_pairs(aapr_df, winner_pool)
    if pairs_df.empty:
        log.warning(
            "No valid pairs emitted. See selection stats above. Consider the "
            "§6 escalation options in docs/dpo_pair_selection_context.md "
            "(relax pre-filter, increase K, cross-GT pairing)."
        )
    else:
        log.info("Per-GT pair count distribution:")
        log.info("\n%s", pairs_df.groupby("gt_complex_id").size().describe())
        log.info("Dominance margin distribution:")
        log.info("\n%s", pairs_df["dominance_margin"].describe())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_parquet(args.output, index=False)
    log.info("Wrote %d pairs → %s", len(pairs_df), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
