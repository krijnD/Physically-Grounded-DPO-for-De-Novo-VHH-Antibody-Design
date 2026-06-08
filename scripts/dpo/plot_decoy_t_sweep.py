#!/usr/bin/env python3
"""Brief 17.2 §9.2 — bathtub plot of per-channel reward vs decoy depth.

Loads the floor lwref parquet (t=0 reference) and every available
``lwref_per_channel_decoy_t{N}.parquet`` under the dpo/ directory,
computes per-channel iter-0 implicit reward (no-T units), and renders
a single-panel curve showing how rot / pos / seq margins evolve as
the decoy depth ``t`` runs from 0 (= GT) to ``T`` (= AAPR loser).

The orchestrator's hypothesis (Brief 17.2): the rot curve is bathtub-
shaped — positive at t=0 (crystal shortcut), dips below zero around
t=7–10 (overshoot regime), and returns to ≈0 at t=T=100 by symmetry
(decoy → π_ref sample → loser-distribution). Confirming or refuting
this geometry is the §9.2 deliverable headline.

Output: docs/figures/phase2/decoy_t_sweep.png + .pdf (300 dpi PNG,
vector PDF). Style matches scripts/thesis/plot_phase2_figures.py
(serif, 10 pt, FIGURE_DPI=300) so the figure drops cleanly into the
thesis document.

CLI
---
::

    python scripts/dpo/plot_decoy_t_sweep.py
    python scripts/dpo/plot_decoy_t_sweep.py --picked-t 4
    python scripts/dpo/plot_decoy_t_sweep.py \\
        --dpo-dir data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo \\
        --output-dir docs/figures/phase2 \\
        --picked-t 4
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CHANNELS = ("rot", "pos", "seq")
CHANNEL_COLOR = {"rot": "#C0392B", "pos": "#2E86AB", "seq": "#6C757D"}
CHANNEL_LABEL = {
    "rot": "rotation (rot)",
    "pos": "position (pos)",
    "seq": "sequence (seq)",
}

FIGURE_DPI = 300
FONT_FAMILY = "serif"
FONT_SIZE = 10


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family":      FONT_FAMILY,
        "font.size":        FONT_SIZE,
        "axes.titlesize":   FONT_SIZE + 1,
        "axes.labelsize":   FONT_SIZE,
        "xtick.labelsize":  FONT_SIZE - 1,
        "ytick.labelsize":  FONT_SIZE - 1,
        "legend.fontsize":  FONT_SIZE - 1,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.05,
    })


def _per_channel_reward(df: pd.DataFrame) -> dict[str, float]:
    """Mean implicit reward (no-T units) per channel."""
    return {
        ch: float((df[f"L_l_ref_{ch}"] - df[f"L_w_ref_{ch}"]).mean())
        for ch in CHANNELS
    }


def _load_sweep(dpo_dir: Path, floor_name: str,
                decoy_glob: str) -> pd.DataFrame:
    floor_path = dpo_dir / floor_name
    if not floor_path.exists():
        raise FileNotFoundError(f"floor parquet not found: {floor_path}")
    rows: list[dict] = []
    rows.append({"t": 0, **_per_channel_reward(pd.read_parquet(floor_path))})

    pat = re.compile(r"decoy_t(\d+)\.parquet$")
    for p in sorted(dpo_dir.glob(decoy_glob)):
        m = pat.search(p.name)
        if not m:
            continue
        t = int(m.group(1))
        df = pd.read_parquet(p)
        rows.append({"t": t, **_per_channel_reward(df)})
    out = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    return out


def _plot(df: pd.DataFrame, *, picked_t: int | None,
          output_path_png: Path, output_path_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    # Shaded PASS / PARTIAL zones for rot+pos. Pos is the tight bound.
    ax.axhspan(-0.1, 0.1, color="#2E86AB", alpha=0.08,
               label="PASS zone (|r| ≤ 0.1)")
    ax.axhspan(-0.3, 0.3, color="#2E86AB", alpha=0.04,
               label="PARTIAL zone (|r| ≤ 0.3)")
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)

    for ch in CHANNELS:
        ax.plot(
            df["t"], df[ch],
            color=CHANNEL_COLOR[ch], linewidth=1.6, marker="o",
            markersize=4.5, markerfacecolor=CHANNEL_COLOR[ch],
            markeredgecolor="white", markeredgewidth=0.6,
            label=CHANNEL_LABEL[ch],
        )

    if picked_t is not None:
        ax.axvline(picked_t, color="#000000", linewidth=1.0,
                   linestyle=":", alpha=0.7)
        ax.annotate(
            f"picked\nt = {picked_t}",
            xy=(picked_t, ax.get_ylim()[1]),
            xytext=(8, -4),
            textcoords="offset points",
            ha="left", va="top",
            fontsize=FONT_SIZE - 1,
        )

    # Symlog x with linthresh=10: linear 0..10, log beyond — clean
    # rendering of both the dense Stage-1 region and the Stage-2 tail.
    ax.set_xscale("symlog", linthresh=10, linscale=1.0)
    ax.set_xlim(-0.5, 110)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100])
    ax.set_xticklabels(
        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "20", "50", "100"]
    )
    ax.set_xlabel(r"Decoy depth $t_{\mathrm{decoy}}$ "
                  r"(forward-noise step, $T=100$ diffusion horizon)")
    ax.set_ylabel(r"Mean implicit reward  $\overline{L_{l,\mathrm{ref}} - L_{w,\mathrm{ref}}}$  (no-$T$ units)")
    ax.set_title("Per-channel iter-0 implicit reward vs decoy depth")
    ax.grid(True, which="major", linewidth=0.4, alpha=0.4)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.2, linestyle=":")
    ax.legend(loc="upper right", framealpha=0.92)

    fig.tight_layout()
    output_path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path_png, dpi=FIGURE_DPI)
    fig.savefig(output_path_pdf)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dpo-dir", type=Path,
                    default=Path("data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo"))
    ap.add_argument("--floor-name", default="lwref_per_channel_floor.parquet")
    ap.add_argument("--decoy-glob", default="lwref_per_channel_decoy_t*.parquet")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("docs/figures/phase2"))
    ap.add_argument("--stem", default="decoy_t_sweep",
                    help="Filename stem; outputs <stem>.png and <stem>.pdf.")
    ap.add_argument("--picked-t", type=int, default=None,
                    help="Highlight this t value with a vertical "
                         "annotation. Omit if no t passed.")
    args = ap.parse_args()

    _setup_style()
    try:
        df = _load_sweep(args.dpo_dir, args.floor_name, args.decoy_glob)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if df.empty or len(df) < 2:
        print(f"ERROR: insufficient data points to plot (n={len(df)})",
              file=sys.stderr)
        return 2

    print(f"Loaded {len(df)} t-values: {df['t'].tolist()}")
    out_png = args.output_dir / f"{args.stem}.png"
    out_pdf = args.output_dir / f"{args.stem}.pdf"
    _plot(df, picked_t=args.picked_t,
          output_path_png=out_png, output_path_pdf=out_pdf)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
