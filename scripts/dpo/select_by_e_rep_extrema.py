#!/usr/bin/env python3
"""E2 sample-vs-sample DPO pair selection by E_Rep extrema.

Brief 22 / E2-prep. Builds the "matched-manifold sample-vs-sample rescue"
pair pool for E2: both winner and loser are π_ref samples (NOT the GT
crystal), removing the real-vs-synthetic membership gap that confounds
E1-B. The winner is the LOWER-E_Rep sample (more physics-favourable);
the loser is the HIGHER-E_Rep sample.

Krijn's directive (post-spec review):
- Scope = floor's 188 GTs (HARD) — keeps E2 directly comparable to the
  E1-B contrast and the floor π_θ training set.
- Pairing = MULTIPLE pairs per GT, E_Rep-gap-thresholded, capped per GT,
  preferring the largest gaps. Target n_pairs ≈ 500–800 (avoids
  reviewer-4's "too few pairs" ambiguity on null).
- D0 pre-check: report within-GT E_Rep spread BEFORE pairing, drop GTs
  with degenerate spread (range < 1.0 REU) where winner ≈ loser.

Trainer compatibility
---------------------
The output parquet's schema mirrors the existing floor pairs.parquet
(see :func:`src.dpo.dataset.PairDataset._resolve_winner_source` at
``src/dpo/dataset.py:418-454``). The critical column is
``winner_provenance = "sample_min_erep"`` — any non-empty value routes
the winner through the disk-parse path, identical to how the loser is
loaded. This lets the existing trainer + dataloader handle E2 without
code changes.

Inputs
------
- ``data/aapr/ftseed42_jfix_trainval_K8_20260525/scored.parquet``
  K=8 samples per GT × 210 GTs after AAPR judging; includes ``e_rep``,
  ``psh_score``, ``cdr_energy_per_res``, ``complex_pdb_path``, etc.
- ``data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_filtered_marginGTp0.0.parquet``
  The 188 unique GTs of the floor (bare-PDB ``gt_complex_id``).

Outputs
-------
- ``data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_sample_minErep_brief22.parquet``
  E2 pair pool, schema mirrors the floor pairs.parquet + a
  ``winner_provenance="sample_min_erep"`` sentinel + ``e_rep_gap`` audit.
- ``data/analysis_outputs/e2_d0_within_gt_spread.csv``
  Per-GT D0 within-GT E_Rep spread table (one row per restricted-floor GT).

Usage
-----
::

    python scripts/dpo/select_by_e_rep_extrema.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the floor's chain-stripping helper so the join is symmetric.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_pareto_pairs import _normalize_complex_id, psh_outside_zone  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
log = logging.getLogger("select_by_e_rep_extrema")

# ── Selection constants ───────────────────────────────────────────────
# THRESHOLD_E_REP_REU: minimum (loser.e_rep − winner.e_rep) to emit a
#   pair. Conservative — sits between the q25 (~89 REU) and the floor
#   for "meaningful" physics gap. Calibrated on the observed per-GT
#   spread (median range = 139 REU); ~all GTs contribute under this.
# CAP_PER_GT: max pairs per GT to prevent the largest-spread GTs from
#   dominating the pool. With CAP=4 + THRESHOLD=50.0 we land at ~678
#   pairs over the 186 surviving GTs.
THRESHOLD_E_REP_REU = 50.0
CAP_PER_GT = 4

# Same axes as the floor pairs.parquet for trainer-side compatibility.
AXES_JSON_KEYS = ("psh_outside_zone", "cdr_energy_per_res", "e_rep", "psh_score")

# Drop a GT entirely if its within-GT E_Rep range is below this (REU).
# At range < 1.0 the "min-vs-max" pair is noise; winner ≈ loser would
# poison DPO with non-information.
NARROW_RANGE_DROP_REU = 1.0

# ── Paths (campaign-relative; running from repo root) ─────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
SCORED_PARQUET = REPO_ROOT / "data/aapr/ftseed42_jfix_trainval_K8_20260525/scored.parquet"
FLOOR_PAIRS_PARQUET = (
    REPO_ROOT
    / "data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_filtered_marginGTp0.0.parquet"
)
OUTPUT_PAIRS_PARQUET = (
    REPO_ROOT
    / "data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_sample_minErep_brief22.parquet"
)
OUTPUT_D0_CSV = REPO_ROOT / "data/analysis_outputs/e2_d0_within_gt_spread.csv"


def _axes_json(row: pd.Series) -> str:
    """Build the floor-compatible axes JSON for one scored row.

    Mirrors the structure used by ``select_pareto_pairs.py`` so the
    trainer's existing JSON parsing path (if any) keeps working
    unchanged.
    """
    return json.dumps(
        {
            "psh_outside_zone": psh_outside_zone(float(row["psh_score"])),
            "cdr_energy_per_res": float(row["cdr_energy_per_res"]),
            "e_rep": float(row["e_rep"]),
            "psh_score": float(row["psh_score"]),
        }
    )


def main() -> int:
    # ── 1. Load + normalize ────────────────────────────────────────────
    log.info("Loading scored parquet: %s", SCORED_PARQUET)
    scored = pd.read_parquet(SCORED_PARQUET)
    n_raw = len(scored)
    log.info("  raw scored rows: %d", n_raw)

    scored = scored.dropna(subset=["e_rep"]).copy()
    n_post_nan = len(scored)
    log.info("  post-NaN-e_rep drop: %d (dropped %d)", n_post_nan, n_raw - n_post_nan)

    scored["gt_norm"] = scored["gt_complex_id"].astype(str).map(_normalize_complex_id)

    log.info("Loading floor pairs parquet: %s", FLOOR_PAIRS_PARQUET)
    floor = pd.read_parquet(FLOOR_PAIRS_PARQUET)
    floor_gts = set(floor["gt_complex_id"].astype(str).unique())
    log.info("  floor GTs (bare PDB): %d unique", len(floor_gts))

    scored_in = scored[scored["gt_norm"].isin(floor_gts)].copy()
    n_post_restrict = len(scored_in)
    log.info(
        "  post floor-GT restrict: %d rows across %d GTs",
        n_post_restrict,
        scored_in["gt_norm"].nunique(),
    )

    # Sanity guard: the orchestrator already verified floor GTs ⊆ scored GTs.
    missing = floor_gts - set(scored_in["gt_norm"].unique())
    if missing:
        log.warning(
            "  %d floor GTs are missing from scored after NaN drop: %s",
            len(missing),
            sorted(missing),
        )

    # ── 2. D0 pre-check ─────────────────────────────────────────────────
    agg = (
        scored_in.groupby("gt_norm")["e_rep"]
        .agg(["min", "max", "median", "count"])
        .reset_index()
        .rename(
            columns={
                "min": "e_rep_min",
                "max": "e_rep_max",
                "median": "e_rep_median",
                "count": "n_samples",
            }
        )
    )
    agg["e_rep_range"] = agg["e_rep_max"] - agg["e_rep_min"]
    agg["kept"] = agg["e_rep_range"] >= NARROW_RANGE_DROP_REU

    log.info("=" * 60)
    log.info("D0 within-GT E_Rep spread (restricted to floor's 188 GTs):")
    log.info("  GTs present:               %d", len(agg))
    log.info("  GTs with range < %.1f REU: %d", NARROW_RANGE_DROP_REU, (~agg["kept"]).sum())
    log.info("  e_rep_range describe:")
    desc = agg["e_rep_range"].describe()
    for k, v in desc.items():
        log.info("    %-8s  %.3f", k, v)

    narrow = agg.loc[~agg["kept"], ["gt_norm", "e_rep_range", "n_samples"]]
    if not narrow.empty:
        log.info("  Narrow-range GTs dropped:")
        for _, r in narrow.iterrows():
            log.info(
                "    %s  range=%.4f  n_samples=%d",
                r["gt_norm"],
                r["e_rep_range"],
                int(r["n_samples"]),
            )

    OUTPUT_D0_CSV.parent.mkdir(parents=True, exist_ok=True)
    agg_out = agg[
        [
            "gt_norm",
            "n_samples",
            "e_rep_min",
            "e_rep_max",
            "e_rep_median",
            "e_rep_range",
            "kept",
        ]
    ].sort_values("gt_norm")
    agg_out.to_csv(OUTPUT_D0_CSV, index=False)
    log.info("  Wrote D0 CSV → %s", OUTPUT_D0_CSV)

    kept_gts = set(agg.loc[agg["kept"], "gt_norm"].tolist())
    log.info("  GTs eligible for pairing:  %d", len(kept_gts))

    # ── 3. Pair selection — multi-pair, gap-thresholded, per-GT capped ─
    log.info("=" * 60)
    log.info(
        "Pairing: THRESHOLD_E_REP_REU=%.2f, CAP_PER_GT=%d",
        THRESHOLD_E_REP_REU,
        CAP_PER_GT,
    )

    rows: list[dict] = []
    per_gt_counts: list[int] = []
    zero_gts: list[str] = []

    for gt in sorted(kept_gts):
        g = (
            scored_in[scored_in["gt_norm"] == gt]
            .sort_values("e_rep")
            .reset_index(drop=True)
        )

        # All (winner_idx, loser_idx) with gap > THRESHOLD, sorted desc by gap.
        candidate_pairs: list[tuple[int, int, float]] = []
        for i in range(len(g)):
            e_w = float(g.loc[i, "e_rep"])
            for j in range(i + 1, len(g)):
                gap = float(g.loc[j, "e_rep"]) - e_w
                if gap > THRESHOLD_E_REP_REU:
                    candidate_pairs.append((i, j, gap))
        # Largest-gap-first: maximises within-GT signal per emitted pair.
        candidate_pairs.sort(key=lambda t: -t[2])
        chosen = candidate_pairs[:CAP_PER_GT]

        if not chosen:
            zero_gts.append(gt)
            per_gt_counts.append(0)
            continue

        per_gt_counts.append(len(chosen))
        for win_i, los_j, gap in chosen:
            w = g.loc[win_i]
            l = g.loc[los_j]
            win_sidx = int(w["sample_idx"])
            los_sidx = int(l["sample_idx"])
            pair_id = f"{gt}__sampair__{win_sidx:04d}__vs__{los_sidx:04d}"
            rows.append(
                {
                    "pair_id": pair_id,
                    "winner_candidate_id": str(w["candidate_id"]),
                    "winner_pdb_path": w["complex_pdb_path"],
                    "loser_candidate_id": str(l["candidate_id"]),
                    "loser_pdb_path": l["complex_pdb_path"],
                    "gt_complex_id": gt,
                    "loser_sample_idx": los_sidx,
                    "axes_winner": _axes_json(w),
                    "axes_loser": _axes_json(l),
                    # E2 is single-axis (e_rep); margin == gap. Informational only.
                    "dominance_margin": float(gap),
                    # Load-bearing sentinel — see src/dpo/dataset.py:418-454.
                    # Any non-empty value routes winner through disk-parse,
                    # mirroring the loser path. DO NOT drop or rename.
                    "winner_provenance": "sample_min_erep",
                    # Audit column. Equal to dominance_margin here, but kept
                    # under its own name so downstream filters / plots can
                    # discriminate E2 from Pareto-margin pools.
                    "e_rep_gap": float(gap),
                }
            )

    pairs_df = pd.DataFrame(rows)
    n_pairs = len(pairs_df)
    counts_arr = np.array(per_gt_counts, dtype=int) if per_gt_counts else np.array([0])

    log.info("=" * 60)
    log.info("Selection summary:")
    log.info("  n_pairs total:                       %d", n_pairs)
    log.info("  GTs eligible (range ≥ %.1f REU):     %d", NARROW_RANGE_DROP_REU, len(kept_gts))
    log.info("  GTs contributing ≥1 pair:            %d", int((counts_arr > 0).sum()))
    log.info("  GTs contributing 0 pairs:            %d", len(zero_gts))
    log.info(
        "  Per-GT pair counts: min=%d  median=%.1f  max=%d  mean=%.2f",
        int(counts_arr.min()),
        float(np.median(counts_arr)),
        int(counts_arr.max()),
        float(counts_arr.mean()),
    )
    if zero_gts:
        log.info("  Zero-pair GTs (all same-GT gaps ≤ THRESHOLD):")
        # Print in chunks of 10 per line for readability.
        for k in range(0, len(zero_gts), 10):
            log.info("    %s", ", ".join(zero_gts[k : k + 10]))

    if not (500 <= n_pairs <= 800):
        log.warning(
            "n_pairs=%d is OUTSIDE the target window [500, 800]. "
            "Consider adjusting THRESHOLD_E_REP_REU or CAP_PER_GT.",
            n_pairs,
        )

    # ── 4. Write parquet + sanity preview ──────────────────────────────
    OUTPUT_PAIRS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_parquet(OUTPUT_PAIRS_PARQUET, index=False)
    log.info("Wrote %d pairs → %s", n_pairs, OUTPUT_PAIRS_PARQUET)

    if not pairs_df.empty:
        log.info("First 3 output rows (subset):")
        preview = pairs_df[
            [
                "pair_id",
                "gt_complex_id",
                "winner_candidate_id",
                "loser_candidate_id",
                "e_rep_gap",
                "winner_provenance",
            ]
        ].head(3)
        log.info("\n%s", preview.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
