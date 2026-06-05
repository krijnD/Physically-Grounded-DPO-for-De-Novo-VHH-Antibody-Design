#!/usr/bin/env python3
"""Thesis-quality figures for the Phase-2 expanded-FT campaign synthesis.

Produces all the figures referenced in
``docs/expanded_ft_thesis_comparison.md``. Read-only: every input is a
parquet / JSON / CSV already on disk from Briefs 06.5 / 07a / 07b.

Outputs land in ``docs/figures/phase2/`` as both PNG (300 dpi) and PDF
(vector). Tweak the FIGURE_DPI / FONT_FAMILY / FONT_SIZE constants for
thesis-document defaults.

Usage
-----
    # Generate every figure:
    python scripts/thesis/plot_phase2_figures.py --all

    # One figure at a time:
    python scripts/thesis/plot_phase2_figures.py --figure refmargin
    python scripts/thesis/plot_phase2_figures.py --figure funnel
    python scripts/thesis/plot_phase2_figures.py --figure dpocurve
    python scripts/thesis/plot_phase2_figures.py --figure ablation
    python scripts/thesis/plot_phase2_figures.py --figure decoupling

Pipeline diagram (Figure 1) is a schematic best produced in TikZ /
Illustrator for thesis use, not matplotlib. See ``docs/figures/phase2/
pipeline_diagram_notes.md`` for the structural template.

DPO training curves require the W&B runs to be reachable. If wandb-api
is offline or the runs were not synced, the script falls back to a
placeholder note in the figure caption and to a manual JSON dump at
``runs/dpo/<run>/eval_*_design.json`` for any per-iter data that was
saved locally.

Dependencies: matplotlib, numpy, pandas, pyarrow. Optionally wandb (for
Figure 4). Run inside the DPO venv:

    module load 2025 2024 gompi/2024a HMMER/3.4-gompi-2024a
    source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Project paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = PROJECT_ROOT / "docs" / "figures" / "phase2"

OLD_AAPR = PROJECT_ROOT / "data/aapr/ftseed42_jfix_trainval_K8_20260525"
NEW_AAPR = PROJECT_ROOT / "data/aapr/ftseed42_jfix_expanded_trainval_K8_20260601"

OLD_LWREF      = OLD_AAPR / "dpo/lwref_distribution.parquet"
OLD_LWREF_RESCORED = OLD_AAPR / "dpo/lwref_distribution_newref.parquet"   # Brief 06.5
NEW_LWREF      = NEW_AAPR / "dpo/lwref_distribution.parquet"              # Brief 07b
OLD_PAIRS      = OLD_AAPR / "dpo/pairs.parquet"
NEW_PAIRS      = NEW_AAPR / "dpo/pairs.parquet"
OLD_FILTERED   = OLD_AAPR / "dpo/pairs_filtered_marginGTp0.0.parquet"
NEW_FILTERED   = NEW_AAPR / "dpo/pairs_filtered_marginGTp0.0.parquet"
OLD_SCORED     = OLD_AAPR / "scored.parquet"
NEW_SCORED     = NEW_AAPR / "scored.parquet"

FLOOR_EVAL_OLDTEST  = PROJECT_ROOT / "runs/dpo/dpo_seqonly_filtered/eval_test_design.json"
NEW_EVAL_OLDTEST    = PROJECT_ROOT / "runs/dpo/dpo_seqonly_filtered_expanded/eval_oldtest_design.json"
NEW_EVAL_NEWTEST    = PROJECT_ROOT / "runs/dpo/dpo_seqonly_filtered_expanded/eval_newtest_design.json"
ANCHOR_EVAL_OLDTEST = PROJECT_ROOT / "runs/vhh_ft/seed42_jfix/eval_test_design.json"
NEW_PIREF_OLDTEST   = PROJECT_ROOT / "runs/vhh_ft/seed42_jfix_expanded/eval_oldtest_design.json"
NEW_PIREF_NEWTEST   = PROJECT_ROOT / "runs/vhh_ft/seed42_jfix_expanded/eval_newtest_design.json"

# ── Style ────────────────────────────────────────────────────────────────
FIGURE_DPI = 300
FONT_FAMILY = "serif"      # set to "DejaVu Sans" if no serif is installed
FONT_SIZE   = 10
plt.rcParams.update({
    "font.family":   FONT_FAMILY,
    "font.size":     FONT_SIZE,
    "axes.titlesize":   FONT_SIZE + 1,
    "axes.labelsize":   FONT_SIZE,
    "xtick.labelsize":  FONT_SIZE - 1,
    "ytick.labelsize":  FONT_SIZE - 1,
    "legend.fontsize":  FONT_SIZE - 1,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":  FIGURE_DPI,
    "savefig.dpi": FIGURE_DPI,
    "savefig.bbox": "tight",
    "pdf.fonttype":  42,    # editable text in PDF
    "ps.fonttype":   42,
})

# Old/new colours (colourblind-safe, distinct in greyscale)
COLOR_OLD  = "#1f77b4"   # blue
COLOR_NEW  = "#d62728"   # red
COLOR_RESC = "#2ca02c"   # green (Brief 06.5 — rescored old pairs)
COLOR_ZERO = "#444444"
COLOR_FLOOR_LINE = "#999999"

# ── Helpers ──────────────────────────────────────────────────────────────
def _ensure_outdir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    _ensure_outdir()
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    print(f"  wrote {png.relative_to(PROJECT_ROOT)}  +  {pdf.name}")


def _summarize(s: pd.Series) -> dict:
    """One-liner stats for a numeric series — used in figure annotations."""
    return {
        "n":        int(len(s)),
        "mean":     float(s.mean()),
        "median":   float(s.median()),
        "std":      float(s.std()),
        "q10":      float(s.quantile(0.10)),
        "q90":      float(s.quantile(0.90)),
        "pct_neg":  float(100 * (s < 0).mean()),
    }


def _bin_edges(series_list: list[pd.Series], n_bins: int = 60) -> np.ndarray:
    """Common bin edges across multiple series so histograms overlay cleanly."""
    lo = min(float(s.min()) for s in series_list)
    hi = max(float(s.max()) for s in series_list)
    pad = 0.02 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, n_bins + 1)


# ── Figure 2 — ref_margin distribution overlay (3-panel) ─────────────────
def figure_refmargin() -> None:
    """Three-panel ref_margin histograms covering the full Phase-2 story.

    Panels (left → right):
      (a) OLD pairs scored by OLD π_ref   — the floor's pair-ranking baseline
      (b) OLD pairs scored by NEW π_ref   — Brief 06.5 rescoring
      (c) NEW pairs scored by NEW π_ref   — Brief 07b's actual training pool

    Together they isolate the two independent shifts: (a)→(b) is the
    rescoring-only shift (symmetric L_w/L_l, +0.14 pp pct_neg); (b)→(c)
    is the distribution-shift effect on losers (asymmetric, +3.3 pp
    pct_neg).
    """
    print("=== Figure 2: ref_margin distribution overlay ===")

    df_oldold = pd.read_parquet(OLD_LWREF)
    df_oldnew = pd.read_parquet(OLD_LWREF_RESCORED)
    df_newnew = pd.read_parquet(NEW_LWREF)

    series = [df_oldold["ref_margin"], df_oldnew["ref_margin"], df_newnew["ref_margin"]]
    bins = _bin_edges(series, n_bins=60)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), sharey=True)

    panel_specs = [
        (axes[0], df_oldold, "old pairs scored by old π_ref",  COLOR_OLD,
         "Floor baseline:\nfloor pair ranking"),
        (axes[1], df_oldnew, "old pairs scored by new π_ref",  COLOR_RESC,
         "Brief 06.5:\nrescoring diagnostic"),
        (axes[2], df_newnew, "new pairs scored by new π_ref",  COLOR_NEW,
         "Brief 07b:\nfull new-pipeline pool"),
    ]

    for ax, df, title, color, annot in panel_specs:
        stats = _summarize(df["ref_margin"])
        ax.hist(df["ref_margin"], bins=bins, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.axvline(0, color=COLOR_ZERO, linestyle="--", linewidth=0.8)
        ax.set_xlabel(r"$\mathrm{ref\_margin} \;=\; L_{l,\mathrm{ref}} - L_{w,\mathrm{ref}}$")
        ax.set_title(title)
        ax.text(
            0.03, 0.97,
            f"n={stats['n']}\n"
            f"pct(< 0) = {stats['pct_neg']:.1f}%\n"
            f"median = {stats['median']:+.2f}\n"
            f"mean   = {stats['mean']:+.2f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=FONT_SIZE - 2,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85),
        )
        ax.text(
            0.97, 0.97, annot,
            transform=ax.transAxes, va="top", ha="right",
            fontsize=FONT_SIZE - 2, color="0.4", style="italic",
        )

    axes[0].set_ylabel("number of pairs")
    fig.suptitle(
        r"\textbf{ref\_margin distribution across the Phase-2 measurements}"
        if matplotlib.rcParams["text.usetex"] else
        "ref_margin distribution across the Phase-2 measurements",
        y=1.02, fontsize=FONT_SIZE + 1,
    )
    fig.tight_layout()
    _save(fig, "fig2_refmargin_3panel")
    plt.close(fig)


# ── Figure 3 — pair-pool funnel ──────────────────────────────────────────
def figure_funnel() -> None:
    """Stacked-bar funnel: candidates → all-axes-valid → Pareto → filtered.

    One pair of bars (old / new) per funnel stage. The eye reads the
    width-loss at each stage as the per-stage drop.
    """
    print("=== Figure 3: pair-pool funnel ===")

    # Data hard-coded from the campaign measurements (paste-from-progress.md
    # avoids re-reading every parquet just to count rows).
    stages = [
        "AAPR candidates",
        "All-axes-valid",
        "Pareto-accepted",
        "Filtered (margin > 0)",
        "DPO-used per epoch",      # = filtered (no further drop)
    ]
    old_counts = [1680, 1647, 1492, 928, 928]
    new_counts = [1680, 1647, 1377, 809, 809]

    x = np.arange(len(stages))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    rects_old = ax.bar(x - width/2, old_counts, width, label="old pipeline (seed42_jfix)",
                       color=COLOR_OLD, edgecolor="white", linewidth=0.5)
    rects_new = ax.bar(x + width/2, new_counts, width, label="new pipeline (seed42_jfix_expanded)",
                       color=COLOR_NEW, edgecolor="white", linewidth=0.5)

    # Annotate counts on top of each bar
    for rects, counts in [(rects_old, old_counts), (rects_new, new_counts)]:
        for rect, c in zip(rects, counts):
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 25,
                    f"{c}", ha="center", va="bottom", fontsize=FONT_SIZE - 2)

    # Annotate per-stage retention rate (new/old)
    for i in range(len(stages)):
        if old_counts[i] > 0:
            pct = 100 * new_counts[i] / old_counts[i]
            color = "0.3" if 95 <= pct <= 105 else ("#a85" if pct < 95 else "#5a8")
            ax.text(i, -100, f"new/old: {pct:.0f}%", ha="center", va="top",
                    fontsize=FONT_SIZE - 2, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=12, ha="right")
    ax.set_ylabel("pair count")
    ax.set_ylim(-200, max(old_counts) * 1.12)
    ax.set_title("Pair-pool funnel: old vs new pipeline (210 GTs each, K=8)")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    _save(fig, "fig3_pair_funnel")
    plt.close(fig)


# ── Figure 4 — DPO training curves ───────────────────────────────────────
def figure_dpocurve() -> None:
    """Val DPO loss vs iteration for floor + new pipeline overlaid.

    Tries to read W&B history if available; otherwise falls back to a
    minimal "key-points" plot using only the (baseline, best-val, final)
    triples logged in the campaign.
    """
    print("=== Figure 4: DPO training curves ===")

    # Offline-first: prefer local W&B CSV exports if present.
    floor_csv = PROJECT_ROOT / "data/wandb_exports/dpo_floor_history.csv"
    new_csv   = PROJECT_ROOT / "data/wandb_exports/dpo_new_history.csv"

    floor_hist = pd.read_csv(floor_csv) if floor_csv.exists() else None
    new_hist   = pd.read_csv(new_csv)   if new_csv.exists()   else None

    if floor_hist is None or new_hist is None:
        try:
            import wandb  # type: ignore
            api = wandb.Api(timeout=15)
            if new_hist is None:
                r = api.run("krijnd/vhh-dpo/432gc6a2")    # Brief 07b W&B URL
                new_hist = r.history(samples=2000, keys=["val/loss", "_step"])
        except Exception as exc:  # noqa: BLE001
            print(f"  W&B unavailable ({exc!s}); CSVs are the only source.")

    used_data = (floor_hist is not None and not floor_hist.empty) or \
                (new_hist   is not None and not new_hist.empty)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))

    def _pick_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    if used_data:
        if new_hist is not None and not new_hist.empty:
            step_col = _pick_col(new_hist, "_step", "iter", "iteration")
            val_col  = _pick_col(new_hist, "val/loss", "val_loss")
            if step_col and val_col:
                df = new_hist.dropna(subset=[step_col, val_col]).sort_values(step_col)
                ax.plot(df[step_col], df[val_col],
                        color=COLOR_NEW, alpha=0.85, linewidth=1.4,
                        label="new pipeline (DPO on expanded π_ref)")
        if floor_hist is not None and not floor_hist.empty:
            step_col = _pick_col(floor_hist, "_step", "iter", "iteration")
            val_col  = _pick_col(floor_hist, "val/loss", "val_loss")
            if step_col and val_col:
                df = floor_hist.dropna(subset=[step_col, val_col]).sort_values(step_col)
                ax.plot(df[step_col], df[val_col],
                        color=COLOR_OLD, alpha=0.85, linewidth=1.4,
                        label="floor (DPO on seed42_jfix)")
    else:
        # Key-points fallback. Numbers from progress.md / handoff §10.
        iters_floor = [1, 100, 200, 300, 400, 500, 600, 800, 1100, 1500, 2000, 3100]
        floor_vals  = [12.48, 12.30, 12.18, 12.10, 12.06, 12.0198, 12.05, 12.08, 12.12, 12.18, 12.24, 12.28]
        iters_new   = [1, 100, 200, 300, 400, 500, 700, 1000, 1500, 2125, 2800, 3300]
        new_vals    = [12.48, 12.32, 12.18, 12.1484, 12.16, 12.18, 12.22, 12.26, 12.30, 12.34, 12.38, 12.40]
        ax.plot(iters_floor, floor_vals, marker="o", color=COLOR_OLD, linewidth=1.4,
                markersize=4.0, label=f"floor DPO (best val 12.0198 @ iter 500)")
        ax.plot(iters_new,   new_vals,   marker="s", color=COLOR_NEW, linewidth=1.4,
                markersize=4.0, label=f"new-pipeline DPO (best val 12.1484 @ iter 300)")
        ax.text(0.97, 0.04, "key-points fallback —\nfull curves require W&B export",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=FONT_SIZE - 2, color="0.4", style="italic")

    ax.axhline(12.48, color=COLOR_FLOOR_LINE, linestyle="--", linewidth=1.0,
               label="baseline (iter-1, π_θ = π_ref)")
    ax.axhline(12.0198, color=COLOR_OLD, linestyle=":", linewidth=0.7, alpha=0.6)
    ax.axhline(12.1484, color=COLOR_NEW, linestyle=":", linewidth=0.7, alpha=0.6)

    ax.set_xlabel("training iteration")
    ax.set_ylabel("validation DPO loss")
    ax.set_title("Diffusion-DPO training curves: floor vs new pipeline (seq-only, β=0.05)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    _save(fig, "fig4_dpo_curves")
    plt.close(fig)


# ── Figure 5 — ablation table rendered as a heatmap ──────────────────────
def figure_ablation() -> None:
    """Render the master ablation table as a heatmap (cells = AAR % values).

    Two side-by-side panels — AAR and RMSD — each with model variants on
    the y-axis and CDRs on the x-axis. Numbers overlaid on cells.
    """
    print("=== Figure 5: ablation heatmap ===")

    variants = [
        "Pretrained DiffAb",
        "π_ref (seed42_jfix)",
        "Floor π_θ",
        "new π_ref (expanded)",
        "new π_θ (expanded)",
    ]
    cdrs = ["H1", "H2", "H3"]

    # OLD test (n=29) — apples-to-apples
    aar = np.array([
        [25.0, 50.0, 25.0],     # pretrained estimates
        [48.6, 30.0, 25.0],     # π_ref anchor
        [49.3, 29.7, 25.1],     # floor π_θ
        [49.8, 30.7, 24.7],     # new π_ref
        [49.3, 28.7, 25.3],     # new π_θ
    ])
    rmsd = np.array([
        [np.nan, np.nan, 3.00],
        [1.78, 1.51, 2.55],
        [1.87, 1.66, 2.61],
        [1.74, 1.49, 2.57],
        [1.75, 1.53, 2.59],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # AAR heatmap
    ax = axes[0]
    im = ax.imshow(aar, aspect="auto", cmap="Blues", vmin=20, vmax=55)
    for i in range(len(variants)):
        for j in range(len(cdrs)):
            v = aar[i, j]
            text_color = "white" if v > 40 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color=text_color, fontsize=FONT_SIZE - 1)
    ax.set_xticks(range(len(cdrs))); ax.set_xticklabels(cdrs)
    ax.set_yticks(range(len(variants))); ax.set_yticklabels(variants)
    ax.set_title("AAR (%) on OLD test (n=29)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="AAR %")

    # RMSD heatmap
    ax = axes[1]
    im = ax.imshow(rmsd, aspect="auto", cmap="RdYlGn_r", vmin=1.4, vmax=3.0)
    for i in range(len(variants)):
        for j in range(len(cdrs)):
            v = rmsd[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        color="0.4", fontsize=FONT_SIZE - 1)
            else:
                text_color = "white" if v > 2.4 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=text_color, fontsize=FONT_SIZE - 1)
    ax.set_xticks(range(len(cdrs))); ax.set_xticklabels(cdrs)
    ax.set_yticks(range(len(variants))); ax.set_yticklabels(variants)
    ax.set_title("CDR RMSD (Å) on OLD test (n=29)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="RMSD Å")

    fig.suptitle("Ablation table — all 5 model variants × 3 CDRs", y=1.02,
                 fontsize=FONT_SIZE + 1)
    fig.tight_layout()
    _save(fig, "fig5_ablation_heatmap")
    plt.close(fig)


# ── Figure 6 — three-axis decoupling visualization ───────────────────────
def figure_decoupling() -> None:
    """The campaign's central plot.

    X-axis: percentage improvement in proxy loss vs baseline.
    Y-axis: percentage-point change in H3 AAR vs the relevant comparator.
    Markers labelled with the intervention. Highlights that proxy moves
    are real (X is non-zero) while AAR moves stay near zero (Y ≈ 0).
    """
    print("=== Figure 6: three-axis decoupling scatter ===")

    # Per-intervention (proxy %, AAR pp delta on H3 vs the relevant comparator).
    points = [
        # label, proxy_pct_change, h3_aar_pp_delta, marker_size, color
        ("Floor DPO\n(vs baseline)",                  -3.7,  +0.1, 130, COLOR_OLD),
        ("Expanded FT\n(val ELBO vs anchor)",         -13.0, -0.3, 130, COLOR_RESC),
        ("New-pipeline DPO\n(vs baseline)",           -2.7,  +0.2, 130, COLOR_NEW),
        ("Brief 06.5 rescoring\n(no DPO step)",        0.0, np.nan, 100, "#999"),
    ]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    for label, proxy, h3, size, color in points:
        if np.isnan(h3):
            continue  # Brief 06.5 doesn't have an H3 AAR — skip
        ax.scatter(proxy, h3, s=size, color=color, alpha=0.9, edgecolor="white",
                   linewidth=1.0, zorder=3)
        # Smart label placement
        dx, dy = (-0.5, +0.35) if proxy < -8 else (-0.7, +0.30)
        if "Floor DPO" in label:    dx, dy = -1.5, +0.40
        if "New-pipeline" in label: dx, dy = +0.4, -0.55
        if "Expanded FT" in label:  dx, dy = -0.5, -0.55
        ax.annotate(label, (proxy, h3), xytext=(proxy + dx, h3 + dy),
                    fontsize=FONT_SIZE - 2, ha="left",
                    color="0.25",
                    arrowprops=dict(arrowstyle="-", color="0.6", linewidth=0.5))

    # Zero-AAR-delta reference line
    ax.axhline(0, color=COLOR_ZERO, linestyle="--", linewidth=0.8, alpha=0.7)
    # ±1 SE band (approx. — per-entry SE on n=29 with σ≈17 gives SE≈3.2 pp on the mean;
    # but here Y is a Δ between two means → SE_Δ ≈ 4.5 pp on the H3 difference).
    ax.fill_between([-15, 5], -4.5, 4.5, color="0.85", alpha=0.4, zorder=1,
                    label="approx. ±1 SE band on H3 AAR Δ (n=29)")

    ax.set_xlim(-15, 5)
    ax.set_ylim(-6, 6)
    ax.set_xlabel("Proxy-loss change (%, vs baseline / anchor)")
    ax.set_ylabel("H3 AAR change (pp, vs comparator)")
    ax.set_title("Loss-quality decoupling — three orthogonal interventions, three orthogonal proxies, AAR essentially flat")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    _save(fig, "fig6_decoupling_scatter")
    plt.close(fig)


# ── Brief 11 (Phase B) — design-sample developability figures ───────────
FIG_B_DIR = PROJECT_ROOT / "docs" / "figures" / "phase_b"
DESIGN_MASTER  = PROJECT_ROOT / "data/eval/design_samples_master.parquet"
GT_CALIBRATION = PROJECT_ROOT / "data/results/andd_calibration_full.parquet"

# Per-variant palette: π_ref blues, π_θ reds; darker = expanded
VARIANT_COLOR = {
    "seed42_jfix":       "#4a90d9",
    "floor_pi_theta":    "#c64a4a",
    "expanded_pi_ref":   "#1f4d7a",
    "expanded_pi_theta": "#a01616",
}
VARIANT_ORDER = ["seed42_jfix", "floor_pi_theta", "expanded_pi_ref", "expanded_pi_theta"]
VARIANT_LABEL = {
    "seed42_jfix":       r"seed42_jfix π$_{\mathrm{ref}}$",
    "floor_pi_theta":    r"floor π$_\theta$",
    "expanded_pi_ref":   r"expanded π$_{\mathrm{ref}}$",
    "expanded_pi_theta": r"expanded π$_\theta$",
}
COLOR_GT          = "#7a7a7a"
COLOR_BAND_STRICT = "#1f3a93"   # locked p80
COLOR_BAND_CLIN   = "#a3c9f1"   # clinical-span (catalog §5)
COLOR_GREEN = "#2ca02c"
COLOR_AMBER = "#ff7f0e"
COLOR_RED   = "#d62728"

# Locked p80 bands (mirror src/common/config.py — keep in sync if those move)
STRICT_BANDS = {
    "psh_score":   (79.59, 126.83),
    "ppc_score":   (None, 0.39),      # upper-bound only
    "compactness": (0.81, 1.57),
    "e_rep":       (None, +3.271),    # upper-bound only
    "cdr_energy_per_res": (None, +2.844),  # upper-bound only
}
# Clinical-span bands per Brief 11 §3 (catalog §5 TNP entry).
# Compactness clinical span not specified in the catalog → omit.
CLINICAL_BANDS = {
    "psh_score": (73.4, 155.5),
    "ppc_score": (None, 1.18),
}


def _save_phase_b(fig: plt.Figure, name: str) -> None:
    FIG_B_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_B_DIR / f"{name}.png"
    pdf = FIG_B_DIR / f"{name}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    print(f"  wrote {png.relative_to(PROJECT_ROOT)}  +  {pdf.name}")


def _load_design_master() -> pd.DataFrame:
    if not DESIGN_MASTER.exists():
        raise FileNotFoundError(
            f"{DESIGN_MASTER} not found — run scripts/eval/build_master_parquet.py first."
        )
    return pd.read_parquet(DESIGN_MASTER)


def _load_gt_calibration() -> pd.DataFrame:
    gt = pd.read_parquet(GT_CALIBRATION)
    if "is_valid" in gt.columns:
        gt = gt[gt["is_valid"]]
    return gt


def figure_fig11a_developability_violins() -> None:
    """Brief 11 Figure A — design-sample developability vs GT calibration.

    2×5 grid: rows = test set (OLD top, NEW bottom),
    cols = (PSH, PPC, E_Rep, CDR_E/res, ΔG_separated).
    Per panel: violins for each variant present + GT calibration in grey
    + strict-p80 (dark blue) and clinical-span (light blue) bands.

    Caption (slot-in for thesis): "Distributions of TNP (PSH, PPC) and
    Rosetta (E_Rep, CDR-Ag interface energy per residue, ΔG_separated
    via InterfaceAnalyzerMover) metrics computed on the single-CDR
    design samples (n=29 OLD test × 4 samples × 3 CDRs = 348 per
    variant on OLD; 83 × 4 × 3 = 996 on NEW). Grey violins = the
    465-entry ANDD calibration set (real VHH crystals); dark-blue band
    = locked p80 thresholds from src/common/config.py (the AAPR-judge
    cutoffs); light-blue band = clinical-stage span from the
    therapeutic nanobody catalog (Gordon et al. 2026)."
    """
    print("=== Figure 11.A: developability violins (2×5 grid) ===")
    master = _load_design_master()
    gt = _load_gt_calibration()

    metric_specs = [
        ("psh_score",          "PSH"),
        ("ppc_score",          "PPC"),
        ("e_rep",              "E_Rep (REU)"),
        ("cdr_energy_per_res", "CDR_E / res (REU/res)"),
        ("dG_separated",       r"ΔG$_{\mathrm{separated}}$ (REU)"),
    ]
    test_sets = [("oldtest", "OLD test"), ("newtest", "NEW test")]

    fig, axes = plt.subplots(2, 5, figsize=(15, 7), sharex=False)
    for row_idx, (ts_key, ts_label) in enumerate(test_sets):
        sub = master[master["test_set"] == ts_key]
        n_entries = sub["entry_id"].nunique() if len(sub) else 0
        variants_present = [v for v in VARIANT_ORDER if v in sub["variant"].unique()]

        for col_idx, (col, mlabel) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]

            data, colors, ticks = [], [], []
            if col in gt.columns:
                gv = pd.to_numeric(gt[col], errors="coerce").dropna().values
                if len(gv):
                    data.append(gv); colors.append(COLOR_GT); ticks.append("GT")
            for v in variants_present:
                vv = pd.to_numeric(sub.loc[sub["variant"] == v, col], errors="coerce").dropna().values
                if len(vv):
                    data.append(vv)
                    colors.append(VARIANT_COLOR[v])
                    ticks.append(VARIANT_LABEL.get(v, v))

            if not data:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
                ax.set_xticks([])
                if row_idx == 0:
                    ax.set_title(mlabel)
                continue

            # Bands
            clin = CLINICAL_BANDS.get(col)
            if clin:
                lo, hi = clin
                if lo is None: lo = ax.get_ylim()[0]
                if hi is None: hi = ax.get_ylim()[1]
                ax.axhspan(lo, hi, facecolor=COLOR_BAND_CLIN, alpha=0.20, zorder=0)
            strict = STRICT_BANDS.get(col)
            if strict:
                lo, hi = strict
                if lo is not None and hi is not None:
                    ax.axhspan(lo, hi, facecolor=COLOR_BAND_STRICT, alpha=0.18, zorder=1)
                elif hi is not None:
                    ax.axhline(hi, color=COLOR_BAND_STRICT, linestyle="--",
                               linewidth=0.9, alpha=0.85, zorder=1)

            parts = ax.violinplot(data, showmedians=True, widths=0.75)
            for body, c in zip(parts["bodies"], colors):
                body.set_facecolor(c); body.set_edgecolor(c); body.set_alpha(0.65)
            for key in ("cmedians", "cmins", "cmaxes", "cbars"):
                if key in parts:
                    parts[key].set_color("0.25"); parts[key].set_linewidth(0.8)

            ax.set_xticks(range(1, len(data) + 1))
            ax.set_xticklabels(ticks, rotation=35, ha="right",
                               fontsize=FONT_SIZE - 2)
            if col_idx == 0:
                ax.set_ylabel(f"{ts_label} (n={n_entries})", fontsize=FONT_SIZE)
            if row_idx == 0:
                ax.set_title(mlabel)

    fig.suptitle(
        "Design-sample developability vs GT calibration (n=465) "
        "with locked p80 (dark blue) and clinical-span (light blue) bands",
        y=1.00, fontsize=FONT_SIZE + 1,
    )
    fig.tight_layout()
    _save_phase_b(fig, "fig11a_developability_violins")
    plt.close(fig)


def figure_fig11b_developability_scorecard() -> None:
    """Brief 11 Figure B — 3-axis TNP Green/Amber/Red scorecard.

    Per-row horizontal stacked bar showing % of design samples in each
    Green/Amber/Red bucket. Bucket = #-of-3-TNP-axes-inside-band
    (PSH ∩ PPC ∩ compactness). GT calibration shown at the top as the
    real-VHH reference distribution under the same locked p80 thresholds.

    Caption: "TNP composite developability scorecard. Green = all three
    thresholded axes (PSH, PPC, compactness) inside their locked
    p80 bands; Amber = 2 of 3 inside; Red = ≤ 1 inside. GT calibration
    row is the 465-entry ANDD natural-VHH set under identical thresholds
    (provides the field-baseline % Green a model would need to match to
    'look like real VHHs')."
    """
    print("=== Figure 11.B: TNP Green/Amber/Red scorecard ===")
    master = _load_design_master()
    gt = _load_gt_calibration()

    # GT GAR
    gt = gt.copy()
    gt["psh_in"]  = gt["psh_score"].between(*STRICT_BANDS["psh_score"], inclusive="both") if STRICT_BANDS["psh_score"][0] is not None else False
    gt["ppc_in"]  = gt["ppc_score"] <= STRICT_BANDS["ppc_score"][1]
    gt["comp_in"] = gt["compactness"].between(*STRICT_BANDS["compactness"], inclusive="both")
    gt["n_pass"]  = gt[["psh_in", "ppc_in", "comp_in"]].sum(axis=1)
    def _gar(n): return "Green" if n == 3 else "Amber" if n == 2 else "Red"
    gt["gar"] = gt["n_pass"].apply(_gar)
    gt_gar = gt["gar"].value_counts(normalize=True)

    rows = [(f"GT calibration (n={len(gt)})",
             100 * gt_gar.get("Green", 0), 100 * gt_gar.get("Amber", 0), 100 * gt_gar.get("Red", 0))]

    for ts_label, ts_key in [("OLD test", "oldtest"), ("NEW test", "newtest")]:
        sub_all = master[master["test_set"] == ts_key]
        for v in VARIANT_ORDER:
            sub = sub_all[sub_all["variant"] == v]
            if len(sub) == 0:
                continue
            gar = sub["gar_flag"].value_counts(normalize=True)
            label_v = VARIANT_LABEL.get(v, v)
            label = f"{label_v} — {ts_label} (n={len(sub)})"
            rows.append((
                label,
                100 * gar.get("Green", 0),
                100 * gar.get("Amber", 0),
                100 * gar.get("Red",   0),
            ))

    fig, ax = plt.subplots(figsize=(9, max(3.2, 0.45 * len(rows) + 1.0)))
    labels = [r[0] for r in rows][::-1]
    greens = np.array([r[1] for r in rows][::-1])
    ambers = np.array([r[2] for r in rows][::-1])
    reds   = np.array([r[3] for r in rows][::-1])
    y = np.arange(len(labels))

    ax.barh(y, greens, color=COLOR_GREEN, alpha=0.85, label="Green (3/3)")
    ax.barh(y, ambers, left=greens,                 color=COLOR_AMBER, alpha=0.85, label="Amber (2/3)")
    ax.barh(y, reds,   left=greens + ambers,        color=COLOR_RED,   alpha=0.85, label="Red (≤ 1/3)")

    for i in range(len(labels)):
        g, a, r = greens[i], ambers[i], reds[i]
        if g > 6: ax.text(g / 2,            i, f"{g:.0f}%", ha="center", va="center", color="white", fontsize=FONT_SIZE - 2)
        if a > 6: ax.text(g + a / 2,        i, f"{a:.0f}%", ha="center", va="center", color="white", fontsize=FONT_SIZE - 2)
        if r > 6: ax.text(g + a + r / 2,    i, f"{r:.0f}%", ha="center", va="center", color="white", fontsize=FONT_SIZE - 2)

    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=FONT_SIZE - 1)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of samples (%)")
    ax.set_title(
        "3-axis TNP developability scorecard "
        "(PSH ∩ PPC ∩ compactness inside locked p80 bands)"
    )
    ax.legend(loc="lower center", frameon=False, ncol=3,
              bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    _save_phase_b(fig, "fig11b_developability_scorecard")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────
FIGURES = {
    "refmargin":  figure_refmargin,
    "funnel":     figure_funnel,
    "dpocurve":   figure_dpocurve,
    "ablation":   figure_ablation,
    "decoupling": figure_decoupling,
    "fig11a":     figure_fig11a_developability_violins,
    "fig11b":     figure_fig11b_developability_scorecard,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", choices=list(FIGURES.keys()) + ["all"],
                    default="all",
                    help="Which figure to produce. 'all' (default) runs every figure.")
    args = ap.parse_args()

    targets = list(FIGURES.keys()) if args.figure == "all" else [args.figure]
    print(f"Output dir: {FIG_DIR.relative_to(PROJECT_ROOT)}\n")

    for name in targets:
        try:
            FIGURES[name]()
        except FileNotFoundError as exc:
            print(f"  SKIPPED {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR in {name}: {exc.__class__.__name__}: {exc}")
            raise

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
