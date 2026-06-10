"""E0 — per-channel reference-margin decomposition on the 928 real training pairs.

Loads:
  - data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/lwref_per_channel_floor.parquet
    (1492 rows; per-channel L_w_ref / L_l_ref at fixed t=50)
  - data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_filtered_marginGTp0.0.parquet
    (928 rows; the pair_ids the floor pi_theta trained on)

Inner-joins on `pair_id` (expects 928 rows), computes per-channel margins
m_c = L_l_ref_c - L_w_ref_c for c in {rot, pos, seq}, summarises, compares
against the bathtub t=0 row, and emits:
  - data/analysis_outputs/e0_per_channel_ref_margin_928.csv
  - docs/figures/phase2/per_channel_ref_margin_real_pairs.png
  - docs/figures/phase2/per_channel_ref_margin_real_pairs.pdf

Runnable as:
    python scripts/thesis/e0_per_channel_ref_margin_real_pairs.py
from the repo root.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

LWREF_PARQUET = (
    REPO_ROOT
    / "data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/lwref_per_channel_floor.parquet"
)
PAIRS_PARQUET = (
    REPO_ROOT
    / "data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_filtered_marginGTp0.0.parquet"
)

OUT_CSV = REPO_ROOT / "data/analysis_outputs/e0_per_channel_ref_margin_928.csv"
OUT_PNG = REPO_ROOT / "docs/figures/phase2/per_channel_ref_margin_real_pairs.png"
OUT_PDF = REPO_ROOT / "docs/figures/phase2/per_channel_ref_margin_real_pairs.pdf"

# Bathtub t=0 reference values (Brief 17 §8 / Table A.12 convention; no beta).
BATHTUB_T0 = {"rot": 2.606, "pos": 0.125, "seq": -0.226}

CHANNELS = ("rot", "pos", "seq")


def main() -> None:
    print(f"[E0] loading lwref per-channel parquet:\n    {LWREF_PARQUET}")
    lwref = pd.read_parquet(LWREF_PARQUET)
    print(f"      shape={lwref.shape}, splits={sorted(lwref['split'].unique())}")

    print(f"[E0] loading filtered pairs parquet:\n    {PAIRS_PARQUET}")
    pairs = pd.read_parquet(PAIRS_PARQUET)
    print(f"      shape={pairs.shape}, n_unique pair_id={pairs['pair_id'].nunique()}")

    # Inner-join on pair_id; keep only the columns we need from `pairs` to avoid clutter.
    keep_from_pairs = ["pair_id"]
    if "gt_complex_id" in pairs.columns:
        keep_from_pairs.append("gt_complex_id")
    merged = lwref.merge(
        pairs[keep_from_pairs], on="pair_id", how="inner", validate="one_to_one"
    )
    print(f"[E0] inner-join on pair_id -> {merged.shape[0]} rows")
    assert merged.shape[0] == 928, (
        f"Expected 928 rows after inner-join, got {merged.shape[0]}"
    )

    # Per-channel margins: m_c = L_l_ref_c - L_w_ref_c (winner-minus-loser convention
    # flipped: positive means "loser is harder for reference", i.e. higher margin in
    # favour of the winner — matches Brief 17 §8 / Table A.12).
    for c in CHANNELS:
        merged[f"m_{c}"] = merged[f"L_l_ref_{c}"] - merged[f"L_w_ref_{c}"]
    merged["m_composite"] = merged[[f"m_{c}" for c in CHANNELS]].sum(axis=1)

    # ---- Summary statistics ----
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    rows = []
    for c in CHANNELS:
        s = merged[f"m_{c}"]
        q = s.quantile(quantiles)
        rows.append(
            {
                "channel": c,
                "n": int(s.shape[0]),
                "mean": float(s.mean()),
                "median": float(q.loc[0.50]),
                "q10": float(q.loc[0.10]),
                "q25": float(q.loc[0.25]),
                "q75": float(q.loc[0.75]),
                "q90": float(q.loc[0.90]),
                "std": float(s.std(ddof=1)),
            }
        )
    stats = pd.DataFrame(rows).set_index("channel")

    print("\n[E0] per-channel summary on 928 real training pairs (single t=50):")
    with pd.option_context(
        "display.float_format", lambda v: f"{v:+.4f}", "display.width", 160
    ):
        print(stats.to_string())

    # ---- Comparison vs bathtub t=0 ----
    cmp_rows = []
    for c in CHANNELS:
        median_obs = float(stats.loc[c, "median"])
        bathtub_val = BATHTUB_T0[c]
        cmp_rows.append(
            {
                "channel": c,
                "median_928pairs": median_obs,
                "bathtub_t0": bathtub_val,
                "delta_vs_bathtub_t0": median_obs - bathtub_val,
            }
        )
    cmp = pd.DataFrame(cmp_rows).set_index("channel")
    print(
        "\n[E0] median(m_c) on 928 pairs minus bathtub t=0 (rot +2.606 / pos +0.125 / seq -0.226):"
    )
    with pd.option_context(
        "display.float_format", lambda v: f"{v:+.4f}", "display.width", 160
    ):
        print(cmp.to_string())

    # ---- CSV output (one row per pair) ----
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_cols = ["pair_id", "gt_id", "m_rot", "m_pos", "m_seq", "m_composite"]
    merged[out_cols].to_csv(OUT_CSV, index=False)
    print(f"\n[E0] wrote per-pair CSV: {OUT_CSV}")

    # ---- Figure: 3-panel horizontal histogram ----
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, c in zip(axes, CHANNELS):
        vals = merged[f"m_{c}"].to_numpy()
        median_val = float(np.median(vals))
        bathtub_val = BATHTUB_T0[c]
        ax.hist(vals, bins=60, color="#4C72B0", alpha=0.78, edgecolor="white")
        ax.axvline(
            median_val,
            color="#C44E52",
            linewidth=2.0,
            label=f"median = {median_val:+.3f}",
        )
        ax.axvline(
            bathtub_val,
            color="#55A868",
            linewidth=2.0,
            linestyle="--",
            label=f"bathtub t=0 = {bathtub_val:+.3f}",
        )
        ax.set_title(f"m_{c}")
        ax.set_xlabel(f"m_{c} = L_l_ref_{c} - L_w_ref_{c}")
        ax.set_ylabel("count" if c == "rot" else "")
        ax.legend(loc="best", fontsize=8)
        ax.text(
            0.02,
            0.98,
            f"median {median_val:+.3f}\nmean {vals.mean():+.3f}\nstd  {vals.std(ddof=1):.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85),
        )
    fig.suptitle(
        "E0 — per-channel reference margin on the 928 floor pairs (single t=50)"
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"[E0] wrote figure PNG: {OUT_PNG}")
    print(f"[E0] wrote figure PDF: {OUT_PDF}")


if __name__ == "__main__":
    main()
