#!/usr/bin/env python3
"""Merge ANDD-curated and SAbDab-curated CSVs into a unified pool.

PDB-code dedup at the manifest-build boundary (Brief 05 §4.4). For
overlapping PDBs (a PDB appears in both sources), apply this tiebreak:

  1. Better resolution (lower numeric; NaN → +inf, loses)
  2. More non-null metadata columns (among antigen_type, antigen_name,
     date, resolution, method)
  3. Newer deposition date (NaT loses)
  4. Prefer ANDD (richer original schema)

Output columns (the manifest-relevant subset; §4.5 consumes this):
  PDB_ID, H_Chain Auth Asym ID, Ag_Auth Asym ID, Ab/Nano H_Chain AA,
  antigen_type, antigen_name, date, resolution, method, source

`source` tags origin and which side won an overlap, so the §4.5 step
can audit decisions.

CDR-cluster (sequence-identity) dedup is NOT done here — that's §4.6.
This step only collapses exact PDB-code duplicates.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("merge_curated")

# SAbDab-side column names are fixed (set by sabdab_to_andd_csv.py).
SAB_COLS = {
    "resolution":   "resolution",
    "date":         "date",
    "antigen_type": "antigen_type",
    "antigen_name": "antigen_name",
    "method":       "method",
}


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column matching any candidate (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def detect_andd_cols(df: pd.DataFrame) -> dict[str, str | None]:
    """Heuristically find ANDD's metadata columns. Logs detections."""
    detected = {
        "resolution":   find_col(df, ["Resolution", "resolution", "Resol"]),
        "date":         find_col(df, ["Date", "date", "Deposition Date",
                                       "Deposit Date", "Release Date"]),
        "antigen_type": find_col(df, ["antigen_type", "Antigen_Type",
                                       "Ag_Type", "Ag Type"]),
        "antigen_name": find_col(df, ["antigen_name", "Antigen_Name",
                                       "Ag_Name", "Ag Name", "Antigen"]),
        "method":       find_col(df, ["method", "Method", "Exptl_Method",
                                       "Experimental Method"]),
    }
    logger.info("ANDD column detection:")
    for k, v in detected.items():
        logger.info("  %-13s -> %s", k, v if v else "(none)")
    return detected


def _safe_float(v) -> float:
    try:
        f = float(v)
        # Guard NaN: NaN comparisons all return False; force to +inf so the
        # NaN-resolution row LOSES the resolution tiebreak instead of
        # tying with a real number.
        return f if f == f else float("inf")
    except (TypeError, ValueError):
        return float("inf")


def _safe_date(v) -> pd.Timestamp:
    ts = pd.to_datetime(v, errors="coerce")
    return ts if pd.notna(ts) else pd.Timestamp("1900-01-01")


def _nn_count(row, cols: list[str | None]) -> int:
    return sum(1 for c in cols if c and pd.notna(row.get(c)))


def pick_winner(
    andd_row: pd.Series,
    sab_row: pd.Series,
    andd_cols: dict[str, str | None],
) -> str:
    """Apply the §4.4 tiebreak. Returns 'ANDD' or 'SAbDab'."""
    # 1) Resolution: lower wins
    ra = _safe_float(andd_row.get(andd_cols["resolution"]) if andd_cols["resolution"] else None)
    rs = _safe_float(sab_row.get(SAB_COLS["resolution"]))
    if ra < rs:
        return "ANDD"
    if rs < ra:
        return "SAbDab"

    # 2) More non-null metadata (over the 5-column metadata set)
    na = _nn_count(andd_row, list(andd_cols.values()))
    ns = _nn_count(sab_row, list(SAB_COLS.values()))
    if na > ns:
        return "ANDD"
    if ns > na:
        return "SAbDab"

    # 3) Newer date wins
    da = _safe_date(andd_row.get(andd_cols["date"]) if andd_cols["date"] else None)
    ds = _safe_date(sab_row.get(SAB_COLS["date"]))
    if da > ds:
        return "ANDD"
    if ds > da:
        return "SAbDab"

    # 4) Final tiebreak
    return "ANDD"


def project_row(
    row: pd.Series,
    source_label: str,
    cols: dict[str, str | None],
) -> dict:
    """Map a source-specific row to the unified output schema."""
    def g(key: str):
        c = cols.get(key)
        return row.get(c) if c else None

    return {
        "PDB_ID":               row.get("PDB_ID"),
        "H_Chain Auth Asym ID": row.get("H_Chain Auth Asym ID"),
        "Ag_Auth Asym ID":      row.get("Ag_Auth Asym ID"),
        "Ab/Nano H_Chain AA":   row.get("Ab/Nano H_Chain AA"),
        "antigen_type":         g("antigen_type"),
        "antigen_name":         g("antigen_name"),
        "date":                 g("date"),
        "resolution":           g("resolution"),
        "method":               g("method"),
        "source":               source_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--andd-csv", required=True, type=Path)
    parser.add_argument("--sabdab-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for p in (args.andd_csv, args.sabdab_csv):
        if not p.exists():
            sys.exit(f"Input not found: {p}")
    if args.output_csv.exists() and not args.overwrite:
        sys.exit(f"Output exists: {args.output_csv} (use --overwrite)")

    andd = pd.read_csv(args.andd_csv)
    sab  = pd.read_csv(args.sabdab_csv)
    logger.info("ANDD:   %d rows x %d cols", len(andd), len(andd.columns))
    logger.info("SAbDab: %d rows x %d cols", len(sab), len(sab.columns))

    andd_cols = detect_andd_cols(andd)

    # Lowercase the join key
    andd["_pdb_lower"] = andd["PDB_ID"].astype(str).str.lower()
    sab["_pdb_lower"]  = sab["PDB_ID"].astype(str).str.lower()

    # Each curate output is already PDB-deduped (curate dedupes its input
    # before the per-row loop), but be paranoid: drop any residual dups.
    andd = andd.drop_duplicates(subset="_pdb_lower", keep="first")
    sab  = sab.drop_duplicates(subset="_pdb_lower", keep="first")

    andd_set = set(andd["_pdb_lower"])
    sab_set  = set(sab["_pdb_lower"])
    overlap   = andd_set & sab_set
    andd_only = andd_set - sab_set
    sab_only  = sab_set - andd_set

    logger.info("ANDD-only:   %d", len(andd_only))
    logger.info("SAbDab-only: %d", len(sab_only))
    logger.info("Overlap:     %d", len(overlap))
    logger.info("Combined unique: %d", len(andd_set | sab_set))

    andd_idx = andd.set_index("_pdb_lower")
    sab_idx  = sab.set_index("_pdb_lower")

    rows: list[dict] = []
    for p in sorted(andd_only):
        rows.append(project_row(andd_idx.loc[p], "ANDD", andd_cols))
    for p in sorted(sab_only):
        rows.append(project_row(sab_idx.loc[p], "SAbDab", SAB_COLS))

    n_andd_wins = n_sab_wins = 0
    for p in sorted(overlap):
        winner = pick_winner(andd_idx.loc[p], sab_idx.loc[p], andd_cols)
        if winner == "ANDD":
            n_andd_wins += 1
            rows.append(project_row(andd_idx.loc[p], "ANDD+SAbDab[A-won]",
                                     andd_cols))
        else:
            n_sab_wins += 1
            rows.append(project_row(sab_idx.loc[p], "ANDD+SAbDab[S-won]",
                                     SAB_COLS))

    logger.info("Overlap tiebreaks: ANDD-won=%d  SAbDab-won=%d",
                n_andd_wins, n_sab_wins)

    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    logger.info("=" * 60)
    logger.info("Wrote %d combined rows to %s", len(out), args.output_csv)
    logger.info("Source breakdown:")
    logger.info("%s", out["source"].value_counts().to_string())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
