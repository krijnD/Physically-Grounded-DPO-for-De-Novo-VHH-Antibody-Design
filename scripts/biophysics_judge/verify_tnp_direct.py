#!/usr/bin/env python3
"""Verify ``tnp_direct.score_pdb`` against prior TNP-CLI metrics.

Reads any pipeline output parquet that has ``complex_pdb_path``,
``nanobody_chain_id``, ``raw_sequence`` and the old PSH/PPC/PNC/
Compactness columns, calls ``score_pdb`` on each row's DiffAb structure,
and writes a side-by-side parquet plus a printed summary.

Use it to answer two questions in one pass:
  1. Does the new code run without erroring on real candidates?
  2. How much do metrics shift when the geometry switches from NB2
     re-fold to the DiffAb structure?

Usage:
    python scripts/biophysics_judge/verify_tnp_direct.py \\
        --input  data/results/<some_canary>.parquet \\
        --output data/results/tnp_direct_vs_cli.parquet \\
        --limit  20

Optional ``--limit`` keeps the dry run fast; drop it for the full set.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.biophysics_judge.tnp_direct import score_pdb
from src.common.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("verify_tnp_direct")


_OLD_COLS = ["psh_score", "ppc_score", "pnc_score", "compactness", "cdr_length", "cdr3_length"]
_NEW_COLS = [f"{c}_new" for c in _OLD_COLS]


def _in_band(value: float | None, lo: float, hi: float | None = None) -> bool:
    if value is None or pd.isna(value):
        return False
    if hi is None:
        return value <= lo
    return lo <= value <= hi


def _passes_thresholds(psh, ppc, compactness) -> bool:
    return (
        _in_band(psh, Config.PSH_GREEN_LOW, Config.PSH_GREEN_HIGH)
        and _in_band(ppc, Config.PPC_MAX)
        and _in_band(compactness, Config.COMPACTNESS_LOW, Config.COMPACTNESS_HIGH)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True,
                   help="Source parquet with complex_pdb_path + old TNP metrics.")
    p.add_argument("--output", required=True,
                   help="Destination parquet (side-by-side old vs new).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N rows (sanity / smoke test).")
    p.add_argument("--monomer-dir", default=None,
                   help="Where to write extracted monomer PDBs "
                        "(default: <output>/monomers/).")
    args = p.parse_args()

    src_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    monomer_dir = Path(args.monomer_dir) if args.monomer_dir else out_path.parent / "monomers"

    df = pd.read_parquet(src_path)
    logger.info("Loaded %d rows from %s", len(df), src_path)

    required = {"candidate_id", "raw_sequence", "complex_pdb_path", "nanobody_chain_id"}
    missing = required - set(df.columns)
    if missing:
        logger.error("Input parquet is missing required columns: %s", missing)
        sys.exit(2)

    # Restrict to rows that actually had a structure on disk last time.
    df = df[df["complex_pdb_path"].notna()].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).copy()
    logger.info("Scoring %d candidates with score_pdb …", len(df))

    new_rows: list[dict] = []
    n_ok = n_fail = 0
    t0 = time.time()

    for idx, row in df.iterrows():
        candidate_id = str(row["candidate_id"])
        complex_pdb = row["complex_pdb_path"]
        chain = row["nanobody_chain_id"] or "H"
        seq = row["raw_sequence"]

        record: dict = {"candidate_id": candidate_id}
        try:
            result = score_pdb(
                complex_pdb_path=complex_pdb,
                nanobody_chain_id=chain,
                candidate_id=candidate_id,
                sequence=seq,
                output_dir=monomer_dir,
            )
        except Exception as e:
            n_fail += 1
            logger.warning("[%d/%d] %s — score_pdb FAILED: %s",
                           idx + 1, len(df), candidate_id, e)
            for col in _NEW_COLS:
                record[col] = None
            record["new_error"] = str(e)
            new_rows.append(record)
            continue

        n_ok += 1
        record.update({
            "psh_score_new": result.psh,
            "ppc_score_new": result.ppc,
            "pnc_score_new": result.pnc,
            "compactness_new": result.compactness,
            "cdr_length_new": result.cdr_length,
            "cdr3_length_new": result.cdr3_length,
            "new_error": None,
        })
        new_rows.append(record)

        if (idx + 1) % 10 == 0 or idx + 1 == len(df):
            elapsed = time.time() - t0
            logger.info("  [%d/%d] %.1fs elapsed (%.1fs/cand). ok=%d fail=%d",
                        idx + 1, len(df), elapsed, elapsed / (idx + 1), n_ok, n_fail)

    new_df = pd.DataFrame(new_rows)
    # Side-by-side join on candidate_id; keep all _OLD_COLS for diffing.
    keep = ["candidate_id", *_OLD_COLS, "complex_pdb_path", "nanobody_chain_id"]
    keep = [c for c in keep if c in df.columns]
    merged = df[keep].merge(new_df, on="candidate_id", how="left")

    # Per-metric delta columns
    for old in _OLD_COLS:
        new = f"{old}_new"
        if old in merged.columns and new in merged.columns:
            merged[f"{old}_delta"] = merged[new] - merged[old]

    merged.to_parquet(out_path, index=False)
    logger.info("Wrote %s", out_path)

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("SUMMARY  (n=%d, ok=%d, fail=%d)", len(df), n_ok, n_fail)
    logger.info("-" * 60)

    for old in _OLD_COLS:
        delta_col = f"{old}_delta"
        if delta_col not in merged.columns:
            continue
        s = merged[delta_col].dropna()
        if len(s) == 0:
            continue
        logger.info(
            "%-15s  Δ mean %+8.3f  median %+8.3f  std %7.3f  min %+8.3f  max %+8.3f",
            old, s.mean(), s.median(), s.std(), s.min(), s.max(),
        )

    # Pass-rate at Gordon et al. 2026 thresholds (PSH band, PPC max, Compactness band)
    def _passrate(psh_col, ppc_col, comp_col):
        passes = merged.apply(
            lambda r: _passes_thresholds(r.get(psh_col), r.get(ppc_col), r.get(comp_col)),
            axis=1,
        )
        return passes.sum(), len(passes)

    old_pass, old_n = _passrate("psh_score", "ppc_score", "compactness")
    new_pass, new_n = _passrate("psh_score_new", "ppc_score_new", "compactness_new")
    logger.info("-" * 60)
    logger.info("Pass at Gordon thresholds (PSH band + PPC<%.2f + Compactness band):",
                Config.PPC_MAX)
    logger.info("  old (NB2 fold):       %3d / %3d  (%.1f%%)",
                old_pass, old_n, 100 * old_pass / max(old_n, 1))
    logger.info("  new (DiffAb struct):  %3d / %3d  (%.1f%%)",
                new_pass, new_n, 100 * new_pass / max(new_n, 1))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
