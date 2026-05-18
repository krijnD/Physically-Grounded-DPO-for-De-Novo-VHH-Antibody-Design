"""Percentile-on-training-set calibration analysis for the three judges.

Replicates AbDPO (Zhou et al., NeurIPS 2024) Appendix E.1 methodology on the
curated ANDD GT set: for each Physics / Biophysics scalar, compute the empirical
50–95th percentiles over natural VHH–antigen complexes, plus a bootstrap 95% CI
on the 80th-percentile headline (AbDPO's chosen success-rate cutoff).

Run twice — once on the `pack_cdrs` refinement arm parquet, once on the
`full` (full-complex repack + FastRelax) arm. The pack-vs-full comparison
informs §4.3 of docs/threshold_calibration_context.md.

Usage
-----
    python scripts/calibration/percentile_analysis.py \\
        --pack-parquet data/results/andd_calibration_pack.parquet \\
        --full-parquet data/results/andd_calibration_full.parquet \\
        --out-dir      docs/calibration/

Outputs
-------
- percentiles_pack.csv, percentiles_full.csv  — §4.1 per-scalar tables.
- sap_per_position.csv                        — §4.2 SAP-per-Kabat-position.
- figures/ecdf_<scalar>.png                   — §4.3 visual pack-vs-full.
- pack_vs_full_summary.md                     — §4.3 narrative + per-scalar reco.
"""

from __future__ import annotations

import argparse
import ast
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make `from src.common.config import Config` work when running from project root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import Config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("percentile_analysis")


# --- Configuration ----------------------------------------------------------

PERCENTILES = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]   # AbDPO Table 4 grid
HEADLINE_P = 80                                          # AbDPO's chosen cutoff
N_BOOTSTRAP = 1000
SEED = 42
MIN_ROWS_PER_ARM = 200                                   # quality bar (post-dedup; pre-dedup ANDD has ~458)

PHYSICS_SCALARS = [
    "e_rep",
    "cdr_energy_per_res",
    "cdr_e_total_sidechain",
    "cdr_ag_e_nonrep_sidechain",
    "cdr_ag_e_rep_sidechain",
]

BIOPHYSICS_SCALARS = [
    "psh_score",
    "ppc_score",
    "compactness",
]

SCALAR_UNITS = {
    "e_rep":                     "REU/residue",
    "cdr_energy_per_res":        "REU/residue",
    "cdr_e_total_sidechain":     "REU/residue (side-chain only)",
    "cdr_ag_e_nonrep_sidechain": "REU/residue (CDR↔Ag attractive)",
    "cdr_ag_e_rep_sidechain":    "REU/residue (CDR↔Ag repulsive)",
    "psh_score":                 "dimensionless (PSH)",
    "ppc_score":                 "dimensionless (PPC)",
    "compactness":               "dimensionless (expected radius / length)",
}

# Direction in which the scalar "improves" under physically faithful refinement.
# Used to flag scalars that move the wrong way under `full` vs `pack`
# (cf. 7B2P-class FastRelax pathologies, §8.3 of the context doc).
SCALAR_DIRECTION = {
    "e_rep":                     "lower_is_better",
    "cdr_energy_per_res":        "lower_is_better",
    "cdr_e_total_sidechain":     "lower_is_better",
    "cdr_ag_e_nonrep_sidechain": "lower_is_better",
    "cdr_ag_e_rep_sidechain":    "lower_is_better",
    "psh_score":                 "in_range",  # green zone [PSH_GREEN_LOW, PSH_GREEN_HIGH]
    "ppc_score":                 "lower_is_better",  # PPC_MAX cap
    "compactness":               "in_range",  # [COMPACTNESS_LOW, COMPACTNESS_HIGH]
}

# Current literature/sentinel values shown alongside empirical values for
# direct visual comparison in the report header.
CURRENT_THRESHOLDS = {
    "e_rep":              ("E_REP_REJECT", Config.E_REP_REJECT, "REU"),
    "cdr_energy_per_res": ("CDR_ENERGY_PER_RES_REJECT", Config.CDR_ENERGY_PER_RES_REJECT, "REU/residue"),
    "psh_score":          ("PSH green zone", (Config.PSH_GREEN_LOW, Config.PSH_GREEN_HIGH), "dimensionless"),
    "ppc_score":          ("PPC_MAX", Config.PPC_MAX, "dimensionless"),
    "compactness":        ("Compactness range", (Config.COMPACTNESS_LOW, Config.COMPACTNESS_HIGH), "dimensionless"),
}


# --- Data loading -----------------------------------------------------------

@dataclass
class ArmData:
    name: str
    df: pd.DataFrame
    mean_cdr_len: float
    n_pre_dedup: int = 0  # row count before sequence deduplication

    @property
    def n_rows(self) -> int:
        return len(self.df)


def load_arm(name: str, path: Path, dedupe_by: str = "raw_sequence") -> ArmData:
    """Read one arm's merged parquet, drop physics-error rows, deduplicate by sequence.

    Physics `error` rows are dropped because PyRosetta crashed before any
    scalar was populated; they carry no signal for the percentile.
    Per-scalar NaN drops happen later inside `percentile_table`.

    The ANDD curated set contains ~52% sequence redundancy — one nanobody
    (CPAPFTRDCFDVTSTTYAY CDR3) appears in 133 PDB structures. Computing
    percentiles on the raw 458 rows over-weights heavily-studied molecules
    and inflates the apparent saturation at high-end values. We deduplicate
    by `raw_sequence` so each unique VHH contributes once to the
    natural-population reference; structural variance across multiple PDB
    snapshots of the same molecule is data noise, not signal about the
    natural-VHH distribution. Pass ``dedupe_by="none"`` to disable.
    """
    df = pd.read_parquet(path)
    n_before = len(df)
    df = df[df["physics_verdict"] != "error"].copy()
    n_after_err = len(df)
    log.info(
        "loaded %s arm: %d rows (dropped %d error rows)",
        name, n_after_err, n_before - n_after_err,
    )

    if dedupe_by != "none":
        if dedupe_by not in df.columns:
            raise RuntimeError(
                f"{name} arm: --dedupe-by={dedupe_by!r} but column missing from parquet"
            )
        seq_counts = df[dedupe_by].value_counts()
        n_unique = seq_counts.size
        top5_dups = seq_counts[seq_counts >= 2].head(5)
        df = df.drop_duplicates(subset=[dedupe_by]).copy()
        log.info(
            "[%s] deduplicated by %s: %d → %d unique sequences "
            "(%d duplicates removed, top counts: %s)",
            name, dedupe_by, n_after_err, len(df), n_after_err - len(df),
            ", ".join(f"{c}×" for c in top5_dups.tolist()) or "none",
        )

    if len(df) < MIN_ROWS_PER_ARM:
        log.warning(
            "%s arm has only %d rows after error-drop + dedup (min=%d) — "
            "percentile CIs will be wide. Proceeding anyway.",
            name, len(df), MIN_ROWS_PER_ARM,
        )

    mean_cdr_len = float(df["cdr_length"].mean())
    log.info("[%s] mean CDR length = %.2f", name, mean_cdr_len)
    return ArmData(name=name, df=df, mean_cdr_len=mean_cdr_len, n_pre_dedup=n_after_err)


# --- Percentile computation -------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    percentile: int,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap 95% CI on the chosen percentile."""
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot[i] = np.quantile(sample, percentile / 100)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return lo, hi


def percentile_table(arm: ArmData, scalars: list[str]) -> pd.DataFrame:
    """One row per scalar with deciles + p80 bootstrap CI + ×mean_cdr_len scaling.

    The `×mean_cdr_len` column rescales per-residue percentiles back to
    AbDPO's CDR-summed scale (Table 4) so the thesis Appendix can be compared
    directly against the original paper. See §6 of context doc.
    """
    rows = []
    for col in scalars:
        if col not in arm.df.columns:
            log.warning("[%s] %s not in parquet — skipping", arm.name, col)
            continue
        vals = arm.df[col].dropna().to_numpy()
        if len(vals) == 0:
            log.warning("[%s] %s has zero valid values — skipping", arm.name, col)
            continue

        row: dict[str, object] = {
            "scalar": col,
            "unit":   SCALAR_UNITS.get(col, ""),
            "n":      len(vals),
            "mean":   float(np.mean(vals)),
            "std":    float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
        }
        for p in PERCENTILES:
            row[f"p{p}"] = float(np.quantile(vals, p / 100))

        # Per-residue scalars scaled back to AbDPO's CDR-summed convention
        # (so reviewers can sanity-check against AbDPO Table 4 directly).
        is_per_residue = "REU/residue" in SCALAR_UNITS.get(col, "")
        if is_per_residue:
            for p in PERCENTILES:
                row[f"p{p}_cdrsum"] = row[f"p{p}"] * arm.mean_cdr_len  # type: ignore[operator]

        # Bootstrap CI on the headline percentile.
        ci_lo, ci_hi = bootstrap_ci(vals, HEADLINE_P)
        row[f"p{HEADLINE_P}_ci_lo"] = ci_lo
        row[f"p{HEADLINE_P}_ci_hi"] = ci_hi

        rows.append(row)

    return pd.DataFrame(rows)


_NP_FLOAT_RE = re.compile(r"np\.float64\(([^)]+)\)")


def _parse_sap_dict(raw: object) -> dict[str, float]:
    """Parse the `sap_scores` column entry into {flag_name: score}.

    The Biology Judge writes the dict via `repr()` of a Python dict whose
    values are numpy scalars, yielding entries like
        {'W47_BULKY_INDOLE_RISK': np.float64(-0.04), ...}
    which `ast.literal_eval` rejects. We strip the `np.float64(…)` wrapper
    textually before parsing.
    """
    if not isinstance(raw, str):
        return dict(raw) if raw is not None else {}  # already a dict
    cleaned = _NP_FLOAT_RE.sub(r"\1", raw)
    return ast.literal_eval(cleaned)


def sap_per_flag_table(pack: ArmData, full: ArmData) -> pd.DataFrame:
    """SAP distribution per biology-flag name, per arm.

    The Biology Judge keys `sap_scores` by flag name (e.g.
    `V37_CAVITY_RISK`, `L45_GATEKEEPER_RISK`, …, `CDR3_HYDROPHOBIC_OVERRIDE_RISK`)
    rather than bare Kabat positions. Flag names encode (residue,
    Kabat position, risk-type) and are the natural unit for any
    position-specific SAP threshold the user might lock in §5.6.

    Flags are auto-discovered from the data; one (arm × flag) row.
    """
    # Discover the universe of flag names present in either arm.
    flag_names: set[str] = set()
    for arm in (pack, full):
        for raw in arm.df["sap_scores"].dropna():
            flag_names.update(_parse_sap_dict(raw).keys())

    rows = []
    for arm in (pack, full):
        # Index per-row dicts once per arm.
        per_row_dicts = [_parse_sap_dict(s) for s in arm.df["sap_scores"].dropna()]
        for flag in sorted(flag_names):
            scores = [d[flag] for d in per_row_dicts if flag in d]
            if not scores:
                continue
            vals = np.asarray(scores, dtype=float)
            row: dict[str, object] = {
                "arm":  arm.name,
                "flag": flag,
                "n":    len(vals),
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
            }
            for p in PERCENTILES:
                row[f"p{p}"] = float(np.quantile(vals, p / 100))
            ci_lo, ci_hi = bootstrap_ci(vals, HEADLINE_P)
            row[f"p{HEADLINE_P}_ci_lo"] = ci_lo
            row[f"p{HEADLINE_P}_ci_hi"] = ci_hi
            rows.append(row)
    return pd.DataFrame(rows)


# --- ECDF figures -----------------------------------------------------------

def ecdf_plot(
    pack_vals: np.ndarray,
    full_vals: np.ndarray,
    scalar: str,
    pack_p80_ci: tuple[float, float],
    full_p80_ci: tuple[float, float],
    out_path: Path,
) -> None:
    """ECDFs of pack and full overlaid, with bootstrap-CI bands at p80.

    X-axis clipped to the combined [p1, p99] window so the bulk distribution
    stays legible even when outliers stretch the natural range.
    """
    combined = np.concatenate([pack_vals, full_vals])
    x_lo = float(np.quantile(combined, 0.01))
    x_hi = float(np.quantile(combined, 0.99))

    fig, ax = plt.subplots(figsize=(7, 4.2))

    for vals, label, color in [
        (pack_vals, f"pack (n={len(pack_vals)})", "tab:blue"),
        (full_vals, f"full (n={len(full_vals)})", "tab:orange"),
    ]:
        xs = np.sort(vals)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post", label=label, color=color, lw=1.6)

    # P80 bootstrap CI bands (translucent).
    ax.axvspan(*pack_p80_ci, color="tab:blue", alpha=0.15, label=f"pack p80 CI [{pack_p80_ci[0]:.3f}, {pack_p80_ci[1]:.3f}]")
    ax.axvspan(*full_p80_ci, color="tab:orange", alpha=0.15, label=f"full p80 CI [{full_p80_ci[0]:.3f}, {full_p80_ci[1]:.3f}]")

    # Horizontal line at 0.80 to make the p80 readout obvious.
    ax.axhline(0.80, color="grey", lw=0.7, ls="--", alpha=0.6)

    ax.set_xlabel(f"{scalar}  [{SCALAR_UNITS.get(scalar, '')}]")
    ax.set_ylabel("ECDF")
    ax.set_title(f"{scalar} — ANDD GT (N≈465 VHH+antigen)")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- Pack-vs-full summary report --------------------------------------------

def _ci_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Return True if two intervals overlap."""
    return not (a[1] < b[0] or b[1] < a[0])


def pack_vs_full_summary(pack: ArmData, full: ArmData, scalars: list[str], dedupe_by: str) -> str:
    """Markdown report comparing the two refinement regimes."""
    dedup_note = (
        f"deduplicated by `{dedupe_by}` (was {pack.n_pre_dedup}/{full.n_pre_dedup} pre-dedup)"
        if dedupe_by != "none"
        else "no deduplication"
    )
    lines = [
        "# Pack vs Full — Calibration Arm Comparison",
        "",
        f"- Pack arm: N = {pack.n_rows} (after error-row drop, {dedup_note}), mean CDR length = {pack.mean_cdr_len:.2f}",
        f"- Full arm: N = {full.n_rows} (after error-row drop, {dedup_note}), mean CDR length = {full.mean_cdr_len:.2f}",
        f"- Bootstrap: {N_BOOTSTRAP} iterations, seed={SEED}, 95% CI on p{HEADLINE_P}",
        "",
        "## Per-scalar comparison",
        "",
        "| Scalar | n_pack | n_full | pack p80 [CI] | full p80 [CI] | CI overlap? | median shift (full − pack) | flag |",
        "|---|---:|---:|---|---|:-:|---:|:--|",
    ]

    for col in scalars:
        pack_vals = pack.df[col].dropna().to_numpy()
        full_vals = full.df[col].dropna().to_numpy()
        if len(pack_vals) == 0 or len(full_vals) == 0:
            continue

        pack_p80 = float(np.quantile(pack_vals, HEADLINE_P / 100))
        full_p80 = float(np.quantile(full_vals, HEADLINE_P / 100))
        pack_ci = bootstrap_ci(pack_vals, HEADLINE_P)
        full_ci = bootstrap_ci(full_vals, HEADLINE_P)
        overlap = _ci_overlap(pack_ci, full_ci)

        median_shift = float(np.median(full_vals) - np.median(pack_vals))

        # Anomaly flag: lower-is-better scalars whose median moved upward
        # under `full` indicate FastRelax-driven inflation (7B2P-class).
        flag = ""
        if SCALAR_DIRECTION.get(col) == "lower_is_better" and median_shift > 0:
            flag = "⚠ full > pack on lower-is-better"

        lines.append(
            f"| `{col}` | {len(pack_vals)} | {len(full_vals)} | "
            f"{pack_p80:+.3f} [{pack_ci[0]:+.3f}, {pack_ci[1]:+.3f}] | "
            f"{full_p80:+.3f} [{full_ci[0]:+.3f}, {full_ci[1]:+.3f}] | "
            f"{'yes' if overlap else 'no'} | "
            f"{median_shift:+.3f} | {flag} |"
        )

    lines += [
        "",
        "## §4.3 recommendation",
        "",
        "Apply this decision table per scalar:",
        "",
        "- **CIs overlap** → `pack` is acceptable (faster, no GT/AAPR asymmetry).",
        "- **CIs disjoint, full shifts favourably (lower)** → `full` for GT, `pack` for AAPR (matches AbDPO).",
        "- **CIs disjoint, full shifts unfavourably (⚠ flag)** → add an "
        "`fa_rep_post − fa_rep_pre > 0` exclusion filter before recomputing.",
        "",
        "## Current config thresholds (for visual comparison)",
        "",
    ]
    for col, (name, val, unit) in CURRENT_THRESHOLDS.items():
        lines.append(f"- `{col}` ({name}): {val} [{unit}]")

    return "\n".join(lines) + "\n"


# --- Console summary --------------------------------------------------------

def log_headline(arm: ArmData, scalars: list[str]) -> None:
    """One log line per scalar in a copy-pasteable headline format."""
    for col in scalars:
        vals = arm.df[col].dropna().to_numpy()
        if len(vals) == 0:
            continue
        p80 = float(np.quantile(vals, HEADLINE_P / 100))
        ci = bootstrap_ci(vals, HEADLINE_P)
        log.info(
            "[%s] %-34s p%d = %+.3f [%+.3f, %+.3f]   n=%d",
            arm.name, col, HEADLINE_P, p80, ci[0], ci[1], len(vals),
        )


# --- Main -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack-parquet", type=Path, required=True)
    parser.add_argument("--full-parquet", type=Path, required=True)
    parser.add_argument("--out-dir",      type=Path, required=True)
    parser.add_argument(
        "--dedupe-by", type=str, default="raw_sequence",
        choices=["raw_sequence", "cdr3_sequence", "none"],
        help="Deduplicate by this column before percentile computation. "
             "ANDD has ~52%% sequence redundancy; default 'raw_sequence' makes each "
             "unique VHH contribute once. Pass 'none' to reproduce pre-dedup results.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    figs_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Phase A: load arms.
    pack = load_arm("pack", args.pack_parquet, dedupe_by=args.dedupe_by)
    full = load_arm("full", args.full_parquet, dedupe_by=args.dedupe_by)

    all_scalars = PHYSICS_SCALARS + BIOPHYSICS_SCALARS

    # §4.1 — per-arm percentile tables.
    log.info("computing per-scalar percentile tables (§4.1)…")
    pack_table = percentile_table(pack, all_scalars)
    full_table = percentile_table(full, all_scalars)
    pack_table.to_csv(out_dir / "percentiles_pack.csv", index=False, float_format="%.4f")
    full_table.to_csv(out_dir / "percentiles_full.csv", index=False, float_format="%.4f")
    log.info("wrote %s", out_dir / "percentiles_pack.csv")
    log.info("wrote %s", out_dir / "percentiles_full.csv")

    # §4.2 — SAP per biology-flag name.
    log.info("computing SAP per biology flag (§4.2)…")
    sap_table = sap_per_flag_table(pack, full)
    if not sap_table.empty:
        sap_table.to_csv(out_dir / "sap_per_flag.csv", index=False, float_format="%.4f")
        log.info("wrote %s (%d rows)", out_dir / "sap_per_flag.csv", len(sap_table))
    else:
        log.warning("no SAP flags found in either arm — sap_per_flag.csv not written")

    # §4.3 — ECDF figures + summary.
    log.info("rendering pack-vs-full ECDFs (§4.3)…")
    for col in all_scalars:
        pack_vals = pack.df[col].dropna().to_numpy()
        full_vals = full.df[col].dropna().to_numpy()
        if len(pack_vals) == 0 or len(full_vals) == 0:
            continue
        pack_ci = bootstrap_ci(pack_vals, HEADLINE_P)
        full_ci = bootstrap_ci(full_vals, HEADLINE_P)
        ecdf_plot(pack_vals, full_vals, col, pack_ci, full_ci, figs_dir / f"ecdf_{col}.png")
    log.info("wrote ECDF figures to %s/", figs_dir)

    summary = pack_vs_full_summary(pack, full, all_scalars, dedupe_by=args.dedupe_by)
    (out_dir / "pack_vs_full_summary.md").write_text(summary)
    log.info("wrote %s", out_dir / "pack_vs_full_summary.md")

    # Console headlines for at-a-glance review.
    log.info("=" * 60)
    log.info("HEADLINE p%d (with bootstrap 95%% CI):", HEADLINE_P)
    log_headline(pack, all_scalars)
    log_headline(full, all_scalars)
    log.info("=" * 60)
    log.info("done.")


if __name__ == "__main__":
    main()
