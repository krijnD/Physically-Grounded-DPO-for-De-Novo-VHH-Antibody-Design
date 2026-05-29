#!/usr/bin/env python3
"""Translate the SAbDab nano summary TSV into a curate_andd.py-compatible CSV.

The output CSV carries the column subset curate_andd.py reads
(``PDB_ID``, ``H_Chain Auth Asym ID``, ``Ag_Auth Asym ID``,
``Ab/Nano H_Chain AA``, ``Ab_or_Nano``, ``Predicted_or_Not``) PLUS the
SAbDab metadata we'll need at Brief 05 §4.5 to build the unified manifest
(antigen_type, antigen_name, date, resolution, method).

Filtering rules (Brief 05 §4.3):
  - Drop rows whose PDB is not in ``--pdb-dir``.
  - Drop rows with an empty antigen_chain (apo entries; curate would
    reject them as ``no_antigen`` anyway, filter early to save runtime).
  - For PDBs with multiple antigen-bearing rows in the summary, keep
    the row with the longest ``antigen_chain`` (most antigens →
    captures the full binding interface).
  - Drop rows whose H-chain sequence can't be extracted from the PDB.

Sequence source: ATOM-record (via the same ``extract_chain_sequence``
helper curate uses). Choosing ATOM over SEQRES means curate's exact-seq
disambiguation branch fires on the same string curate would extract
internally — collapses to a no-op for unambiguous picks and lets the
patched identity-rescue branch handle the residual ambiguities.

Usage:
    python "data scripts/sabdab_to_andd_csv.py" \\
        --summary-tsv /projects/0/hpmlprjs/interns/krijn/sabdab_nano_dataset_IgLM/sabdab_nano_summary.tsv \\
        --pdb-dir     /projects/0/hpmlprjs/interns/krijn/sabdab_nano_dataset_IgLM/filtered_vhh_pdbs \\
        --output-csv  data/raw/curate_full/SAbDab_for_curate.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.sabdab_loader import extract_chain_sequence  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("sabdab_to_andd")


def _antigen_chain_count(s) -> int:
    """Number of distinct antigen-chain tokens in a pipe-delimited string."""
    if pd.isna(s):
        return 0
    return len([t for t in str(s).split("|") if t.strip()])


def _norm_antigen_chain(s) -> str:
    """SAbDab 'A | E' → ANDD 'A,E' (curate's _normalize_csv_letter splits on
    comma; pipes would be treated as a single token)."""
    if pd.isna(s):
        return ""
    return ",".join(t.strip() for t in str(s).split("|") if t.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--summary-tsv", required=True, type=Path,
                        help="SAbDab nano summary TSV.")
    parser.add_argument("--pdb-dir", required=True, type=Path,
                        help="Directory of downloaded SAbDab PDB files.")
    parser.add_argument("--output-csv", required=True, type=Path,
                        help="Path for the curate-compatible CSV.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting an existing output CSV.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    if not args.summary_tsv.exists():
        sys.exit(f"Summary TSV not found: {args.summary_tsv}")
    if not args.pdb_dir.exists() or not args.pdb_dir.is_dir():
        sys.exit(f"PDB dir not found: {args.pdb_dir}")
    if args.output_csv.exists() and not args.overwrite:
        sys.exit(f"Output exists: {args.output_csv} (use --overwrite)")

    # Load
    df = pd.read_csv(args.summary_tsv, sep="\t")
    n_total = len(df)
    logger.info("Loaded %d rows from %s", n_total, args.summary_tsv)
    logger.info("Columns: %s", list(df.columns))

    required = {"pdb", "Hchain", "antigen_chain"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Summary TSV is missing required columns: {sorted(missing)}")

    # Build the case-insensitive PDB lookup
    pdb_lookup: dict[str, Path] = {p.stem.lower(): p for p in args.pdb_dir.glob("*.pdb")}
    logger.info("Discovered %d PDB files in %s", len(pdb_lookup), args.pdb_dir)

    df["_pdb_lower"] = df["pdb"].astype(str).str.lower()
    n_before_on_disk = len(df)
    df = df[df["_pdb_lower"].isin(pdb_lookup)].copy()
    logger.info("Restricted to %d/%d rows with PDB on disk",
                len(df), n_before_on_disk)

    # Drop rows with empty antigen_chain (apo entries — curate would reject
    # them as no_antigen anyway; pre-filter saves wallclock).
    n_before_ag = len(df)
    df = df.dropna(subset=["antigen_chain"]).copy()
    df = df[df["antigen_chain"].astype(str).str.strip() != ""].copy()
    logger.info("After dropping no-antigen rows: %d/%d",
                len(df), n_before_ag)

    # Per-PDB dedup: keep the row with the most antigen chains. Tie-break
    # is stable on input order.
    df["_n_antigens"] = df["antigen_chain"].apply(_antigen_chain_count)
    n_before_dedup = len(df)
    df = (
        df.sort_values(["_pdb_lower", "_n_antigens"],
                       ascending=[True, False], kind="stable")
          .drop_duplicates(subset="_pdb_lower", keep="first")
    )
    logger.info("After per-PDB dedup (kept longest antigen_chain): %d/%d",
                len(df), n_before_dedup)

    # Pull the H-chain ATOM-record sequence for each row.
    out_rows: list[dict] = []
    skipped_no_chain = 0
    skipped_no_seq = 0
    for i, (_, row) in enumerate(df.iterrows(), 1):
        pdb_id = str(row["_pdb_lower"])
        h_chain = str(row["Hchain"]).strip()
        if not h_chain or h_chain.lower() == "nan":
            skipped_no_chain += 1
            continue
        pdb_path = pdb_lookup[pdb_id]
        try:
            seq = extract_chain_sequence(str(pdb_path), h_chain)
        except Exception as e:
            logger.debug("extract_chain_sequence failed for %s_%s: %s",
                         pdb_id, h_chain, e)
            seq = None
        if not seq:
            skipped_no_seq += 1
            continue

        out_rows.append({
            # curate-required columns
            "PDB_ID":               pdb_id,
            "H_Chain Auth Asym ID": h_chain,
            "Ag_Auth Asym ID":      _norm_antigen_chain(row["antigen_chain"]),
            "Ab/Nano H_Chain AA":   seq,
            "Ab_or_Nano":           "Nanobody/VHH",
            "Predicted_or_Not":     "real",
            # SAbDab metadata, carried through for the §4.5 manifest build
            "antigen_type":         row.get("antigen_type"),
            "antigen_name":         row.get("antigen_name"),
            "date":                 row.get("date"),
            "resolution":           row.get("resolution"),
            "method":               row.get("method"),
        })

        if i % 100 == 0:
            logger.info("  ...processed %d/%d", i, len(df))

    if skipped_no_chain:
        logger.warning("Skipped %d rows: empty/NaN Hchain.", skipped_no_chain)
    if skipped_no_seq:
        logger.warning("Skipped %d rows: H-chain ATOM sequence empty / "
                       "chain not present.", skipped_no_seq)

    out_df = pd.DataFrame(out_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    logger.info("=" * 60)
    logger.info("Wrote %d rows to %s", len(out_df), args.output_csv)
    logger.info("Summary funnel:")
    logger.info("  raw summary rows:           %d", n_total)
    logger.info("  after PDB-on-disk filter:   %d", n_before_ag)
    logger.info("  after no-antigen filter:    %d", n_before_dedup)
    logger.info("  after per-PDB dedup:        %d", len(df))
    logger.info("  emitted (Hchain seq found): %d", len(out_df))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
