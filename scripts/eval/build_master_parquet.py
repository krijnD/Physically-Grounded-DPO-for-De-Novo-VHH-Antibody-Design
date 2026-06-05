#!/usr/bin/env python3
"""Assemble the Brief 11 master parquet + summary table.

Joins the judges scored parquet with the ΔG parquet on candidate_id,
decodes candidate_id into variant / test_set / entry_id / cdr / sample
columns, computes a 3-axis TNP Green/Amber/Red flag against the locked
p80 bands in src/common/config.py, and emits both:

  - the master parquet (one row per design candidate, all metrics
    + per-axis band-membership booleans + composite gar_flag);
  - a markdown summary table per (variant × test_set) suitable for
    pasting into the Brief 11 §4 deliverable. The GT calibration row
    (n=465 ANDD natural VHHs from data/results/andd_calibration_full
    .parquet) is prepended as the reference comparator.

Gar flag definition (campaign-wide convention):
    Green = all 3 thresholded TNP axes inside their locked bands
            (PSH, PPC, compactness — the only axes the biophysics judge
            actually thresholds). CDR3 length is metadata-only.
    Amber = 2 of 3 axes inside band.
    Red   = ≤ 1 of 3 axes inside band.

Brief 11 §3 mentions "all four TNP axes" in passing but the production
biophysics judge (src/biophysics_judge/judge.py) only gates on 3 —
length is informational. The 3-axis definition matches what the judge
itself decides and keeps the figure caption defensible.

Usage::

    python scripts/eval/build_master_parquet.py \\
        --judged          data/eval/design_samples_judged_all.parquet \\
        --dg              data/eval/design_samples_dG_all.parquet \\
        --gt-calibration  data/results/andd_calibration_full.parquet \\
        --output-parquet  data/eval/design_samples_master.parquet \\
        --output-summary  docs/figures/phase_b/summary_table.md
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("build_master_parquet")

# Locked thresholds — mirror src/common/config.py exactly. If those move,
# update here.
PSH_LOW, PSH_HIGH = 79.59, 126.83
PPC_MAX = 0.39
COMP_LOW, COMP_HIGH = 0.81, 1.57


def _gar_flag(n_pass: int) -> str:
    """3-axis TNP Green/Amber/Red."""
    if n_pass >= 3:
        return "Green"
    if n_pass == 2:
        return "Amber"
    return "Red"


def _decode_candidate_id(s: str) -> pd.Series:
    """Split <variant>__<test_set>__<entry_id>__<cdr>__s<NNNN> → 5 fields."""
    parts = s.split("__")
    if len(parts) != 5:
        return pd.Series({
            "variant": None, "test_set": None,
            "entry_id": None, "cdr": None, "sample": None,
        })
    return pd.Series({
        "variant":   parts[0],
        "test_set":  parts[1],
        "entry_id":  parts[2],
        "cdr":       parts[3],
        "sample":    int(parts[4][1:]) if parts[4].startswith("s") else None,
    })


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--judged", required=True, type=Path)
    ap.add_argument("--dg", required=True, type=Path)
    ap.add_argument("--gt-calibration", required=True, type=Path)
    ap.add_argument("--output-parquet", required=True, type=Path)
    ap.add_argument("--output-summary", required=True, type=Path)
    args = ap.parse_args()

    for p in (args.judged, args.dg, args.gt_calibration):
        if not p.exists():
            logger.error("Input missing: %s", p)
            return 2

    judged = pd.read_parquet(args.judged)
    dG = pd.read_parquet(args.dg)
    gt = pd.read_parquet(args.gt_calibration)
    logger.info("judged %d rows | dG %d rows | gt %d rows", len(judged), len(dG), len(gt))

    decoded = judged["candidate_id"].apply(_decode_candidate_id)
    master = pd.concat([judged, decoded], axis=1)
    master = master.merge(
        dG[["candidate_id", "dG_separated", "dG_cross", "dSASA"]],
        on="candidate_id",
        how="left",
    )

    # Per-axis band membership (3 thresholded TNP axes)
    master["psh_in_band"] = master["psh_score"].between(PSH_LOW, PSH_HIGH, inclusive="both")
    master["ppc_in_band"] = master["ppc_score"] <= PPC_MAX
    master["comp_in_band"] = master["compactness"].between(COMP_LOW, COMP_HIGH, inclusive="both")
    master["n_tnp_pass"] = (
        master[["psh_in_band", "ppc_in_band", "comp_in_band"]].sum(axis=1)
    )
    master["gar_flag"] = master["n_tnp_pass"].apply(_gar_flag)

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(args.output_parquet, index=False)
    logger.info(
        "Wrote master parquet: %s  (%d rows, %d cols)",
        args.output_parquet, len(master), len(master.columns),
    )

    # ── GT calibration reference row ──
    gt_valid = gt[gt["is_valid"]] if "is_valid" in gt.columns else gt
    # Compute GAR on GT for an apples-to-apples scorecard row (NOT all-Green
    # by definition — that's the calibration set's distribution under the
    # same locked thresholds).
    gt_check = gt_valid.copy()
    gt_check["psh_in_band"] = gt_check["psh_score"].between(PSH_LOW, PSH_HIGH, inclusive="both")
    gt_check["ppc_in_band"] = gt_check["ppc_score"] <= PPC_MAX
    gt_check["comp_in_band"] = gt_check["compactness"].between(COMP_LOW, COMP_HIGH, inclusive="both")
    gt_check["n_tnp_pass"] = gt_check[["psh_in_band", "ppc_in_band", "comp_in_band"]].sum(axis=1)
    gt_check["gar_flag"] = gt_check["n_tnp_pass"].apply(_gar_flag)
    gt_gar = gt_check["gar_flag"].value_counts(normalize=True)

    gt_summary = {
        "variant": "GT_calibration", "test_set": "—",
        "n": len(gt_valid),
        "median_psh":   float(gt_valid["psh_score"].median()),
        "median_ppc":   float(gt_valid["ppc_score"].median()),
        "median_comp":  float(gt_valid["compactness"].median()),
        "median_erep":  float(gt_valid["e_rep"].median()),
        "median_cdr_e": float(gt_valid["cdr_energy_per_res"].median()),
        "median_dG":    None,
        "pct_green":    100.0 * float(gt_gar.get("Green", 0.0)),
        "pct_amber":    100.0 * float(gt_gar.get("Amber", 0.0)),
        "pct_red":      100.0 * float(gt_gar.get("Red", 0.0)),
        "biophys_pass": 100.0 * float((gt_valid["biophysics_verdict"] == "pass").mean()),
        "physics_pass": 100.0 * float((gt_valid["physics_verdict"] == "pass").mean()),
    }

    # ── Per-(variant, test_set) aggregation ──
    summary = (
        master.groupby(["variant", "test_set"])
        .agg(
            n=("candidate_id", "count"),
            median_psh=("psh_score", "median"),
            median_ppc=("ppc_score", "median"),
            median_comp=("compactness", "median"),
            median_erep=("e_rep", "median"),
            median_cdr_e=("cdr_energy_per_res", "median"),
            median_dG=("dG_separated", "median"),
            pct_green=("gar_flag", lambda s: 100.0 * s.eq("Green").mean()),
            pct_amber=("gar_flag", lambda s: 100.0 * s.eq("Amber").mean()),
            pct_red=("gar_flag", lambda s: 100.0 * s.eq("Red").mean()),
            biophys_pass=("biophysics_verdict", lambda s: 100.0 * s.eq("pass").mean()),
            physics_pass=("physics_verdict", lambda s: 100.0 * s.eq("pass").mean()),
        )
        .reset_index()
    )

    full = pd.concat([pd.DataFrame([gt_summary]), summary], ignore_index=True)

    # ── Markdown table ──
    lines: list[str] = []
    lines.append(
        "| Variant | Test | n | median PSH | median PPC | median compactness | "
        "median e_rep | median CDR_E/res | median ΔG | % Green | % Amber | % Red | "
        "Biophys pass | Physics pass |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for _, row in full.iterrows():
        dG_str = (
            f"{row['median_dG']:.2f}"
            if row["median_dG"] is not None and not pd.isna(row["median_dG"])
            else "—"
        )
        lines.append(
            f"| {row['variant']} | {row['test_set']} | {int(row['n'])} | "
            f"{row['median_psh']:.2f} | {row['median_ppc']:.3f} | {row['median_comp']:.3f} | "
            f"{row['median_erep']:.2f} | {row['median_cdr_e']:.3f} | {dG_str} | "
            f"{row['pct_green']:.1f}% | {row['pct_amber']:.1f}% | {row['pct_red']:.1f}% | "
            f"{row['biophys_pass']:.1f}% | {row['physics_pass']:.1f}% |"
        )
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text("\n".join(lines) + "\n")
    logger.info("Wrote summary table: %s", args.output_summary)
    print()
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
