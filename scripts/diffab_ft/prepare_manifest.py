#!/usr/bin/env python3
"""Build a DiffAb-compatible SAbDab summary TSV from the curated ANDD CSV.

DiffAb's dataset loader (third_party/diffab/diffab/datasets/sabdab.py)
reads a tab-separated summary file with a fixed schema. This script
translates the ANDD curation output (PDB_ID, H_Chain Auth Asym ID,
Ag_Auth Asym ID, ...) into that schema so DiffAb's existing loader can
ingest our 465 VHH+antigen complexes with no fork surgery.

Per row, we:
  1. Skip rows whose ``curation_status`` is not ``"ok"`` (failed entries
     are intentionally not used for fine-tuning).
  2. Open the PDB file to read header metadata (deposition date,
     resolution, experimental method) — these are authoritative and may
     not be present in the ANDD CSV.
  3. Emit one TSV row in DiffAb's schema:

       pdb  Hchain  Lchain  antigen_chain  antigen_type  antigen_name
       date resolution  method  scfv

     Lchain is always empty (VHH = single-domain). antigen_chain is
     pipe-delimited as DiffAb expects ("A | B" not "A,B"). antigen_type
     is hard-coded to "protein" for protein antigens (or
     "protein | protein | ..." for multi-chain antigens), which passes
     DiffAb's ALLOWED_AG_TYPES filter.

Cryo-EM and other entries without a single resolution number write
``resolution = "NOT"`` — DiffAb's default loader filters those out, but
our subclass (added in a later step) will accept them.

The output TSV is intentionally a drop-in for DiffAb's
``sabdab_summary_all.tsv`` so we can later swap to the real SAbDab
summary if needed for ablations.

Usage:
    python scripts/diffab_ft/prepare_manifest.py \\
        --curated-csv /path/to/ANDD_VHH_curated_diffab.csv \\
        --pdb-dir     /path/to/VHH_structures_post_diffab \\
        --output-tsv  data/datasets/diffab_manifest.tsv
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Project root is two levels up from this script (scripts/diffab_ft/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.pdb_utils import load_structure  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("prepare_manifest")


# DiffAb's expected schema, in column order.
MANIFEST_COLUMNS = [
    "pdb",
    "Hchain",
    "Lchain",
    "antigen_chain",
    "antigen_type",
    "antigen_name",
    "date",
    "resolution",
    "method",
    "scfv",
]


def _read_pdb_header(pdb_path: Path) -> dict:
    """Extract deposition date, resolution, and method from a PDB file.

    Returns a dict with keys 'date' (str MM/DD/YY or None), 'resolution'
    (float or None), 'method' (str uppercase or None). Missing or
    unparseable fields are returned as None — caller decides defaults.
    """
    structure = load_structure(str(pdb_path), structure_id=pdb_path.stem)
    header = structure.header or {}

    raw_date = header.get("deposition_date")
    date_str = None
    if raw_date:
        try:
            date_str = datetime.strptime(raw_date, "%Y-%m-%d").strftime("%m/%d/%y")
        except ValueError:
            logger.warning(
                "Unparseable deposition_date %r in %s; leaving blank.",
                raw_date, pdb_path.name,
            )

    resolution = header.get("resolution")
    if resolution is not None:
        try:
            resolution = float(resolution)
        except (TypeError, ValueError):
            resolution = None

    method = header.get("structure_method")
    if method:
        method = method.upper().strip()

    return {"date": date_str, "resolution": resolution, "method": method}


def _split_chain_list(value: object) -> list[str]:
    """Parse a comma-separated chain ID string ("A,B,C") into a clean list."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        return [c.strip() for c in value.split(",") if c.strip()]
    return [str(value).strip()] if str(value).strip() else []


def build_manifest_row(curated_row: dict, pdb_dir: Path) -> dict | None:
    """Translate one curated CSV row into a DiffAb manifest TSV row.

    Returns None if the PDB file is missing or malformed; the caller
    accumulates a count of skips for the summary.
    """
    pdb_id = str(curated_row["PDB_ID"]).strip().lower()
    h_chain = str(curated_row["H_Chain Auth Asym ID"]).strip()
    ag_chains = _split_chain_list(curated_row.get("Ag_Auth Asym ID"))

    if not h_chain:
        logger.warning("Row %s: H_Chain Auth Asym ID is empty, skipping.", pdb_id)
        return None
    if not ag_chains:
        logger.warning("Row %s: Ag_Auth Asym ID is empty, skipping.", pdb_id)
        return None

    pdb_path = pdb_dir / f"{pdb_id}.pdb"
    if not pdb_path.exists():
        # Try uppercase variant (some PDB mirrors use upper-case filenames).
        alt = pdb_dir / f"{pdb_id.upper()}.pdb"
        if alt.exists():
            pdb_path = alt
        else:
            logger.warning("Row %s: PDB file not found at %s, skipping.",
                           pdb_id, pdb_path)
            return None

    try:
        header = _read_pdb_header(pdb_path)
    except Exception as exc:  # noqa: BLE001 — defensive boundary
        logger.warning("Row %s: failed to parse PDB header (%s: %s), skipping.",
                       pdb_id, exc.__class__.__name__, exc)
        return None

    # DiffAb pipe-delimits antigen chains; antigen_type is one "protein"
    # token per chain so it passes ALLOWED_AG_TYPES.
    antigen_chain_field = " | ".join(ag_chains)
    antigen_type_field = " | ".join(["protein"] * len(ag_chains))

    return {
        "pdb": pdb_id,
        "Hchain": h_chain,
        "Lchain": "",                        # VHH = no light chain
        "antigen_chain": antigen_chain_field,
        "antigen_type": antigen_type_field,
        "antigen_name": "",                  # not used (we override splitting)
        "date": header["date"] or "",
        "resolution": (
            f"{header['resolution']:.2f}"
            if header["resolution"] is not None else "NOT"
        ),
        "method": header["method"] or "",
        "scfv": "False",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--curated-csv", required=True, type=Path,
                        help="Curated ANDD CSV (output of curate_andd.py).")
    parser.add_argument("--pdb-dir", required=True, type=Path,
                        help="Directory containing PDB files (one per PDB_ID).")
    parser.add_argument("--output-tsv", required=True, type=Path,
                        help="Path for the DiffAb-compatible manifest TSV.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting an existing output TSV.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N curated rows (debug).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    if not args.curated_csv.exists():
        logger.error("Curated CSV not found: %s", args.curated_csv)
        sys.exit(1)
    if not args.pdb_dir.exists() or not args.pdb_dir.is_dir():
        logger.error("PDB dir not found or not a directory: %s", args.pdb_dir)
        sys.exit(1)
    if args.output_tsv.exists() and not args.overwrite:
        logger.error(
            "Output TSV already exists: %s (use --overwrite to replace).",
            args.output_tsv,
        )
        sys.exit(1)
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.curated_csv)
    logger.info("Loaded %d rows from %s", len(df), args.curated_csv)

    if "curation_status" in df.columns:
        before = len(df)
        df = df[df["curation_status"] == "ok"].reset_index(drop=True)
        logger.info("Kept %d rows with curation_status == 'ok' (dropped %d).",
                    len(df), before - len(df))
    else:
        logger.warning(
            "No 'curation_status' column found — assuming all rows are curated."
        )

    if args.limit is not None:
        df = df.head(args.limit)
        logger.info("Limiting to first %d rows for debugging.", len(df))

    rows: list[dict] = []
    skipped_missing_pdb = 0
    skipped_bad_header = 0
    skipped_empty_field = 0
    for i, row in df.iterrows():
        pdb_id = str(row.get("PDB_ID", "?")).strip()
        manifest_row = build_manifest_row(row.to_dict(), args.pdb_dir)
        if manifest_row is None:
            # The exact reason was already logged inside build_manifest_row;
            # we tally a coarse breakdown here for the final summary.
            pdb_path = args.pdb_dir / f"{pdb_id.lower()}.pdb"
            alt_path = args.pdb_dir / f"{pdb_id.upper()}.pdb"
            if not (pdb_path.exists() or alt_path.exists()):
                skipped_missing_pdb += 1
            elif not str(row.get("H_Chain Auth Asym ID") or "").strip() or \
                 not _split_chain_list(row.get("Ag_Auth Asym ID")):
                skipped_empty_field += 1
            else:
                skipped_bad_header += 1
            continue
        rows.append(manifest_row)

        if (i + 1) % 50 == 0:
            logger.info("Processed %d / %d rows.", i + 1, len(df))

    out_df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    out_df.to_csv(args.output_tsv, sep="\t", index=False)

    logger.info("=" * 60)
    logger.info("Manifest written: %s", args.output_tsv)
    logger.info("  Curated input rows:     %d", len(df))
    logger.info("  Manifest entries:       %d", len(out_df))
    logger.info("  Skipped (missing PDB):  %d", skipped_missing_pdb)
    logger.info("  Skipped (empty fields): %d", skipped_empty_field)
    logger.info("  Skipped (header parse): %d", skipped_bad_header)
    if len(out_df):
        n_no_res = (out_df["resolution"] == "NOT").sum()
        n_no_date = (out_df["date"] == "").sum()
        n_no_method = (out_df["method"] == "").sum()
        logger.info("  Entries without resolution: %d", n_no_res)
        logger.info("  Entries without date:       %d", n_no_date)
        logger.info("  Entries without method:     %d", n_no_method)
        logger.info("  Method breakdown:")
        for method, count in out_df["method"].value_counts().items():
            logger.info("    %-30s %d", method or "(empty)", count)


if __name__ == "__main__":
    main()
