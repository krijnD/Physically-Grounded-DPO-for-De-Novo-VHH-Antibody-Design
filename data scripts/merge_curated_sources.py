#!/usr/bin/env python3
"""Merge ANDD-curated and SAbDab-curated CSVs into a unified pool.

Two-stage pipeline (Brief 05 §4.4):

  1. PDB-code dedup. For each PDB in the union of the two curated sets,
     pick one row. Overlap tiebreak (after metadata is unified, see below)
     collapses to "prefer ANDD" — the original brief's resolution/date
     comparisons all tie because both sides end up with identical RCSB
     metadata.

  2. Metadata layer. ANDD's CSV does not carry RCSB resolution/date/method
     (only Update_Date and Experimental_Method, neither directly usable
     in the manifest). To produce a manifest-ready output we attach
     `antigen_type, antigen_name, date, resolution, method` from a single
     canonical source PER PDB:
       (a) SAbDab nano summary if the PDB is listed there (1186 PDBs);
       (b) PDB-file HEADER/REMARK records as fallback (BioPython).

Output columns:
  PDB_ID, H_Chain Auth Asym ID, Ag_Auth Asym ID, Ab/Nano H_Chain AA,
  antigen_type, antigen_name, date, resolution, method,
  source ("ANDD" / "SAbDab" / "ANDD+SAbDab"),
  metadata_source ("sabdab_summary" / "pdb_header" / "missing")

CDR-cluster sequence-identity dedup is intentionally deferred to §4.6.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("merge_curated")


# ── SAbDab summary metadata lookup ───────────────────────────────────────
def build_summary_lookup(summary_path: Path) -> dict[str, dict]:
    """Per-PDB metadata from the SAbDab nano summary.

    Multi-row PDBs are collapsed by picking the row with the longest
    antigen_chain (matches sabdab_to_andd_csv.py policy).
    """
    df = pd.read_csv(summary_path, sep="\t")

    def _n_ag(s) -> int:
        if pd.isna(s):
            return 0
        return len([t for t in str(s).split("|") if t.strip()])

    df["_pdb_lower"] = df["pdb"].astype(str).str.lower()
    df["_n_ag"] = df["antigen_chain"].apply(_n_ag)
    df = (
        df.sort_values(["_pdb_lower", "_n_ag"],
                        ascending=[True, False], kind="stable")
          .drop_duplicates(subset="_pdb_lower", keep="first")
    )

    wanted = ["antigen_type", "antigen_name", "date", "resolution", "method"]
    have = [c for c in wanted if c in df.columns]
    missing = set(wanted) - set(have)
    if missing:
        logger.warning("SAbDab summary missing columns: %s", sorted(missing))

    lookup = df.set_index("_pdb_lower")[have].to_dict("index")
    logger.info("SAbDab summary lookup built: %d PDB codes (cols: %s)",
                len(lookup), have)
    return lookup


# ── PDB-header fallback ──────────────────────────────────────────────────
_PARSER = PDBParser(QUIET=True)


def parse_pdb_header(pdb_path: Path) -> dict:
    """Pull RCSB-equivalent metadata from a PDB file's HEADER/REMARK
    records. Returns the same five-key dict shape as the summary lookup;
    antigen_type cannot be derived and is left None (defaulted later)."""
    try:
        struct = _PARSER.get_structure("x", str(pdb_path))
        h = struct.header
        return {
            "antigen_type": None,
            "antigen_name": None,
            "date":         h.get("deposition_date"),
            "resolution":   h.get("resolution"),
            "method":       (h.get("structure_method") or "").upper() or None,
        }
    except Exception as e:
        logger.debug("PDB header parse failed for %s: %s", pdb_path, e)
        return {k: None for k in ("antigen_type", "antigen_name",
                                   "date", "resolution", "method")}


def find_pdb(pdb_id_lower: str, pdb_dirs: list[Path]) -> Path | None:
    """Locate a PDB file by id (case-insensitive) across multiple dirs."""
    candidates = [f"{pdb_id_lower}.pdb", f"{pdb_id_lower.upper()}.pdb"]
    for d in pdb_dirs:
        for name in candidates:
            p = d / name
            if p.exists():
                return p
    return None


def get_metadata(pdb_lower: str,
                 summary_lookup: dict[str, dict],
                 pdb_dirs: list[Path]) -> tuple[dict, str]:
    """Return (metadata_dict, source_tag)."""
    if pdb_lower in summary_lookup:
        return dict(summary_lookup[pdb_lower]), "sabdab_summary"
    p = find_pdb(pdb_lower, pdb_dirs)
    if p is None:
        return ({k: None for k in ("antigen_type", "antigen_name",
                                    "date", "resolution", "method")},
                "missing")
    return parse_pdb_header(p), "pdb_header"


# ── Per-row projection ───────────────────────────────────────────────────
def project_row(src_row: pd.Series,
                source_label: str,
                metadata: dict,
                metadata_source: str) -> dict:
    return {
        "PDB_ID":               src_row.get("PDB_ID"),
        "H_Chain Auth Asym ID": src_row.get("H_Chain Auth Asym ID"),
        "Ag_Auth Asym ID":      src_row.get("Ag_Auth Asym ID"),
        "Ab/Nano H_Chain AA":   src_row.get("Ab/Nano H_Chain AA"),
        "antigen_type":         metadata.get("antigen_type"),
        "antigen_name":         metadata.get("antigen_name"),
        "date":                 metadata.get("date"),
        "resolution":           metadata.get("resolution"),
        "method":               metadata.get("method"),
        "source":               source_label,
        "metadata_source":      metadata_source,
    }


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--andd-csv", required=True, type=Path)
    parser.add_argument("--sabdab-csv", required=True, type=Path)
    parser.add_argument("--sabdab-summary-tsv", required=True, type=Path,
                        help="SAbDab nano summary TSV (canonical metadata source).")
    parser.add_argument("--pdb-dirs", required=True, type=Path, nargs="+",
                        help="One or more PDB directories to search for the "
                             "header-fallback path (typically the ANDD and "
                             "SAbDab download dirs).")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for p in (args.andd_csv, args.sabdab_csv, args.sabdab_summary_tsv):
        if not p.exists():
            sys.exit(f"Input not found: {p}")
    for d in args.pdb_dirs:
        if not d.exists() or not d.is_dir():
            sys.exit(f"PDB dir not found: {d}")
    if args.output_csv.exists() and not args.overwrite:
        sys.exit(f"Output exists: {args.output_csv} (use --overwrite)")

    andd = pd.read_csv(args.andd_csv)
    sab  = pd.read_csv(args.sabdab_csv)
    logger.info("ANDD:   %d rows x %d cols", len(andd), len(andd.columns))
    logger.info("SAbDab: %d rows x %d cols", len(sab), len(sab.columns))

    summary_lookup = build_summary_lookup(args.sabdab_summary_tsv)

    andd["_pdb_lower"] = andd["PDB_ID"].astype(str).str.lower()
    sab["_pdb_lower"]  = sab["PDB_ID"].astype(str).str.lower()
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
    meta_source_counts: dict[str, int] = {"sabdab_summary": 0,
                                          "pdb_header": 0,
                                          "missing": 0}

    n = 0
    total = len(andd_set | sab_set)

    def _emit(pdb: str, src_row: pd.Series, source_label: str):
        nonlocal n
        n += 1
        meta, msrc = get_metadata(pdb, summary_lookup, args.pdb_dirs)
        meta_source_counts[msrc] += 1
        rows.append(project_row(src_row, source_label, meta, msrc))
        if n % 200 == 0:
            logger.info("  ...metadata attached for %d/%d", n, total)

    for pdb in sorted(andd_only):
        _emit(pdb, andd_idx.loc[pdb], "ANDD")
    for pdb in sorted(sab_only):
        _emit(pdb, sab_idx.loc[pdb], "SAbDab")
    for pdb in sorted(overlap):
        # Both rows end up with identical metadata via summary_lookup,
        # so the brief's tiebreak (resolution / nn-metadata / date) all
        # tie and we fall through to "prefer ANDD" (§4.4 final rule).
        # We still tag source as "ANDD+SAbDab" so the audit is honest.
        _emit(pdb, andd_idx.loc[pdb], "ANDD+SAbDab")

    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    logger.info("=" * 60)
    logger.info("Wrote %d combined rows to %s", len(out), args.output_csv)
    logger.info("Source breakdown:")
    logger.info("%s", out["source"].value_counts().to_string())
    logger.info("Metadata-source breakdown:")
    for k, v in meta_source_counts.items():
        logger.info("  %-15s %d", k, v)
    # Quick sanity numbers
    res = pd.to_numeric(out["resolution"], errors="coerce")
    logger.info("Resolution non-null: %d / %d", res.notna().sum(), len(out))
    logger.info("Date non-null:       %d / %d",
                out["date"].notna().sum(), len(out))
    logger.info("antigen_type non-null: %d / %d",
                out["antigen_type"].notna().sum(), len(out))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
