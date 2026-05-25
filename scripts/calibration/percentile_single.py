#!/usr/bin/env python3
"""Single-mode percentile calibration for the Physics + Biophysics judges.

Adapted from ``percentile_analysis.py`` (pack-vs-full comparator) for the
2026-05-22-onwards regime where calibration runs only ``refinement_mode=none``.
Same percentile grid (50-95, headline p80), same bootstrap CI, same dedup
defaults — just one parquet in, one CSV out.

Usage
-----
    python scripts/calibration/percentile_single.py \\
        --parquet data/results/andd_judge_test_full.parquet \\
        --out-dir docs/calibration/

Outputs
-------
- ``percentiles_none.csv``  — per-scalar percentile table (p50…p95 + p80 CI).
- Console summary listing the new p80 thresholds suitable for pasting into
  ``src/common/config.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import Config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("percentile_single")

PERCENTILES = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
HEADLINE_P = 80
N_BOOTSTRAP = 1000
SEED = 42

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


@dataclass
class ArmData:
    name: str
    df: pd.DataFrame
    n_pre_dedup: int = 0

    @property
    def n_rows(self) -> int:
        return len(self.df)


def load_arm(name: str, path: Path, dedupe_by: str = "raw_sequence") -> ArmData:
    """Load parquet, drop physics-error rows, dedupe by sequence."""
    df = pd.read_parquet(path)
    n_total = len(df)
    log.info("[%s] loaded %d rows from %s", name, n_total, path)

    # Drop rows with NaN on the primary physics scalars (PyRosetta crashes).
    physics_present = [c for c in ("e_rep", "cdr_energy_per_res") if c in df.columns]
    if physics_present:
        before = len(df)
        df = df.dropna(subset=physics_present, how="all")
        if before - len(df) > 0:
            log.info("[%s] dropped %d rows with no physics scalars (PyRosetta error)",
                     name, before - len(df))

    n_pre_dedup = len(df)

    if dedupe_by != "none":
        if dedupe_by not in df.columns:
            log.warning("[%s] dedupe column %r missing — skipping dedup", name, dedupe_by)
        else:
            before = len(df)
            df = df.drop_duplicates(subset=[dedupe_by])
            log.info("[%s] deduped by %s: %d → %d rows", name, dedupe_by, before, len(df))

    return ArmData(name=name, df=df, n_pre_dedup=n_pre_dedup)


def bootstrap_ci(vals: np.ndarray, p: int) -> tuple[float, float]:
    """95% bootstrap CI on the p-th percentile."""
    rng = np.random.default_rng(SEED)
    n = len(vals)
    if n < 10:
        return (float("nan"), float("nan"))
    samples = np.array([
        np.quantile(rng.choice(vals, n, replace=True), p / 100)
        for _ in range(N_BOOTSTRAP)
    ])
    return (float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)))


def percentile_table(arm: ArmData, scalars: list[str]) -> pd.DataFrame:
    """Per-scalar table: percentile grid + p80 bootstrap CI."""
    rows: list[dict] = []
    for col in scalars:
        if col not in arm.df.columns:
            log.warning("[%s] scalar %r missing — skipping", arm.name, col)
            continue
        vals = arm.df[col].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            log.warning("[%s] scalar %r is all-NaN — skipping", arm.name, col)
            continue

        row = {"scalar": col, "n": len(vals)}
        for p in PERCENTILES:
            row[f"p{p}"] = float(np.quantile(vals, p / 100))

        ci_lo, ci_hi = bootstrap_ci(vals, HEADLINE_P)
        row[f"p{HEADLINE_P}_ci_lo"] = ci_lo
        row[f"p{HEADLINE_P}_ci_hi"] = ci_hi
        rows.append(row)

    return pd.DataFrame(rows)


def print_headline(table: pd.DataFrame) -> None:
    """Print the threshold-ready summary (p80 + CI per scalar)."""
    print()
    print("=" * 78)
    print(f"  p{HEADLINE_P} thresholds (paste into src/common/config.py)")
    print("=" * 78)
    for _, r in table.iterrows():
        ci = (r[f"p{HEADLINE_P}_ci_lo"], r[f"p{HEADLINE_P}_ci_hi"])
        print(f"  {r['scalar']:<32}  p80 = {r['p80']:>+8.3f}   "
              f"CI [{ci[0]:>+7.3f}, {ci[1]:>+7.3f}]   (n={int(r['n'])})")
    print("=" * 78)

    # Specifically call out the two values that go into Config
    if "e_rep" in table["scalar"].values:
        e = table.loc[table["scalar"] == "e_rep", "p80"].iloc[0]
        print(f"\n  E_REP_REJECT             = {e:+.3f}   # REU. p80 (none-mode, J-anchor-fixed)")
    if "cdr_energy_per_res" in table["scalar"].values:
        c = table.loc[table["scalar"] == "cdr_energy_per_res", "p80"].iloc[0]
        print(f"  CDR_ENERGY_PER_RES_REJECT = {c:+.3f}   # REU/res. p80 (none-mode, J-anchor-fixed)")
    print()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, required=True,
                   help="Merged judges parquet (output of merge_chunks.py).")
    p.add_argument("--out-dir", type=Path, default=Path("docs/calibration"),
                   help="Directory for the percentiles_none.csv output.")
    p.add_argument("--dedupe-by", default="raw_sequence",
                   choices=["raw_sequence", "cdr3_sequence", "none"],
                   help="Column to deduplicate on (default: raw_sequence).")
    args = p.parse_args()

    arm = load_arm("none", args.parquet, dedupe_by=args.dedupe_by)
    log.info("[none] n_rows after dedup = %d (pre-dedup %d)", arm.n_rows, arm.n_pre_dedup)

    scalars = PHYSICS_SCALARS + BIOPHYSICS_SCALARS
    table = percentile_table(arm, scalars)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "percentiles_none.csv"
    table.to_csv(csv_path, index=False, float_format="%.4f")
    log.info("Wrote %s", csv_path)

    print_headline(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
