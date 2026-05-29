#!/usr/bin/env python3
"""Translate combined_curated.csv into the canonical DiffAb manifest TSV.

Manifest schema (matches data/datasets/diffab_manifest.tsv):
    pdb, Hchain, Lchain, antigen_chain, antigen_type, antigen_name,
    date, resolution, method, scfv

Transformations:
  - pdb         lowercase
  - Hchain      from curate's verified ``H_Chain Auth Asym ID``
  - Lchain      empty (VHH-only, matches current manifest)
  - antigen_chain  curate's comma-delim ("A,E") → manifest pipe-delim
                   ("A | E"); single chains unchanged
  - antigen_type   default NaN → "protein" (the §4.5 fallback for
                   ANDD-only PDBs without SAbDab summary coverage —
                   curate already verified ≥5Å contact so the chain
                   IS a binding partner; protein is the modal class)
  - date           normalize to MM/DD/YY (BioPython/SAbDab give YYYY-MM-DD)
  - scfv           "False" (VHH-only, hardcoded)

Filters (Brief 05 §4.5):
  F1. Drop rows with empty antigen_chain.
  F2. Drop rows with resolution > 3.0Å (or missing). EXEMPTION: entries
      already in --current-manifest-tsv pass F2 unconditionally — the
      resolution bar is a quality gate for newly added expansion data,
      not a retroactive purge. Empirical justification: the current
      manifest contains entries with resolution > 3.0Å (e.g. 7b2m=3.39Å
      in train, 12/30 of current test PDBs > 3.0Å), so a uniform F2
      would silently regress the existing pool.
  F3. Drop rows whose antigen_type is not a pipe-separated combination
      of {protein, peptide}. Any other type (DNA / RNA / Hapten / etc.)
      is excluded. Applies uniformly — completeness, not quality.

Test-set preservation audit:
  Reads data/datasets/clustering/cluster_splits.json :: splits.test and
  asserts every current-test PDB survives all filters. With the F2
  exemption above, this passes by construction unless F1/F3 drops a
  current-test PDB — which would indicate a data corruption upstream.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("build_manifest")

ALLOWED_ANTIGEN_ATOMS = {"protein", "peptide"}


def _is_allowed_antigen_type(t) -> bool:
    if pd.isna(t):
        return True  # NaN defaulted to "protein" below
    tokens = [x.strip().lower() for x in str(t).split("|") if x.strip()]
    if not tokens:
        return True
    return all(x in ALLOWED_ANTIGEN_ATOMS for x in tokens)


def _comma_to_pipe(s) -> str:
    """ "A,B" -> "A | B"; "" stays "". Single chains unchanged."""
    if pd.isna(s):
        return ""
    parts = [t.strip() for t in str(s).split(",") if t.strip()]
    return " | ".join(parts)


def _norm_date(v) -> str:
    """Best-effort parse → MM/DD/YY (matches existing manifest format).

    BioPython's deposition_date is typically 'YYYY-MM-DD'; SAbDab summary
    is the same. ANDD's CSV doesn't carry a usable date (Update_Date is
    the ANDD-DB timestamp). Anything that fails to parse → empty string.
    """
    if pd.isna(v) or str(v).strip() in ("", "nan", "NaT"):
        return ""
    ts = pd.to_datetime(v, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.strftime("%m/%d/%y")


def _norm_resolution(v) -> float | None:
    """Parse to float; ≤ 0 → None (NMR / sentinel)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f <= 0:
        return None
    return f


def _norm_method(v) -> str:
    """Upper-case match the existing manifest convention."""
    if pd.isna(v):
        return ""
    return str(v).strip().upper()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--combined-csv", required=True, type=Path,
                        help="Output of merge_curated_sources.py.")
    parser.add_argument("--current-splits-json", required=True, type=Path,
                        help="data/datasets/clustering/cluster_splits.json "
                             "(used for test-set preservation audit).")
    parser.add_argument("--current-manifest-tsv", required=True, type=Path,
                        help="Current 465-entry manifest. PDBs already in "
                             "this manifest are exempt from F2 (resolution "
                             "cap) — see module docstring.")
    parser.add_argument("--output-tsv", required=True, type=Path,
                        help="Path for the expanded manifest TSV.")
    parser.add_argument("--resolution-cap", type=float, default=3.0,
                        help="Drop rows with resolution > this (default 3.0).")
    parser.add_argument("--default-antigen-type", default="protein",
                        help="Value used when antigen_type is NaN "
                             "(default 'protein').")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.combined_csv.exists():
        sys.exit(f"Input CSV not found: {args.combined_csv}")
    if not args.current_splits_json.exists():
        sys.exit(f"Current splits JSON not found: {args.current_splits_json}")
    if not args.current_manifest_tsv.exists():
        sys.exit(f"Current manifest TSV not found: {args.current_manifest_tsv}")
    if args.output_tsv.exists() and not args.overwrite:
        sys.exit(f"Output exists: {args.output_tsv} (use --overwrite)")

    df = pd.read_csv(args.combined_csv)
    logger.info("Loaded %d combined-curated rows", len(df))

    # Current manifest PDBs are exempt from F2 (resolution cap).
    cur_manifest = pd.read_csv(args.current_manifest_tsv, sep="\t")
    cur_manifest_pdbs = set(cur_manifest["pdb"].astype(str).str.lower())
    logger.info("Current manifest PDBs (F2-exempt): %d", len(cur_manifest_pdbs))

    # Load current test PDB IDs for preservation audit
    with open(args.current_splits_json) as f:
        splits = json.load(f)
    cur_test_entries = splits["splits"]["test"]  # e.g. ['7f5g_B', ...]
    cur_test_pdbs = {e.rsplit("_", 1)[0].lower() for e in cur_test_entries}
    logger.info("Current test entries: %d  (unique PDBs: %d)",
                len(cur_test_entries), len(cur_test_pdbs))

    # ── Stage 1: project to manifest schema ──────────────────────────
    out = pd.DataFrame({
        "pdb":           df["PDB_ID"].astype(str).str.lower(),
        "Hchain":        df["H_Chain Auth Asym ID"].astype(str).str.strip(),
        "Lchain":        "",
        "antigen_chain": df["Ag_Auth Asym ID"].apply(_comma_to_pipe),
        "antigen_type":  df["antigen_type"].where(
                            df["antigen_type"].notna(),
                            args.default_antigen_type),
        "antigen_name":  df["antigen_name"].fillna("").astype(str),
        "date":          df["date"].apply(_norm_date),
        "resolution":    df["resolution"].apply(_norm_resolution),
        "method":        df["method"].apply(_norm_method),
        "scfv":          "False",
    })

    # Resolution buckets (pre-filter, for reporting)
    res = pd.to_numeric(out["resolution"], errors="coerce")
    n_le_25  = int((res <= 2.5).sum())
    n_25_30  = int(((res > 2.5) & (res <= 3.0)).sum())
    n_gt_30  = int((res > 3.0).sum())
    n_no_res = int(res.isna().sum())
    logger.info("Resolution buckets (pre-filter):")
    logger.info("  <= 2.5 Å:        %d", n_le_25)
    logger.info("  2.5 - 3.0 Å:     %d", n_25_30)
    logger.info("  > 3.0 Å:         %d", n_gt_30)
    logger.info("  missing/<=0:     %d", n_no_res)

    # ── Stage 2: filters (track which test PDBs hit each) ────────────
    def _audit_drop(mask: pd.Series, name: str) -> None:
        dropped_pdbs = set(out.loc[mask, "pdb"])
        n_dropped = mask.sum()
        n_test_hit = len(dropped_pdbs & cur_test_pdbs)
        logger.info("Filter %s: drops %d rows  (%d are current-test PDBs)",
                    name, int(n_dropped), n_test_hit)
        if n_test_hit > 0:
            sample = sorted(dropped_pdbs & cur_test_pdbs)[:10]
            logger.warning("  ↳ current-test PDBs dropped: %s%s",
                           ", ".join(sample),
                           " ..." if n_test_hit > 10 else "")

    # F1: empty antigen_chain
    f1 = out["antigen_chain"].astype(str).str.strip() == ""
    _audit_drop(f1, "F1[empty antigen_chain]")

    # F2: resolution > cap (or missing). Exempt current-manifest PDBs —
    # the resolution bar gates NEW expansion data quality, not the
    # already-validated existing pool.
    res_now = pd.to_numeric(out["resolution"], errors="coerce")
    is_existing = out["pdb"].isin(cur_manifest_pdbs)
    f2_quality = (res_now.isna()) | (res_now > args.resolution_cap)
    f2 = f2_quality & ~is_existing
    n_f2_exempt = int((f2_quality & is_existing).sum())
    logger.info("F2 exemption: %d existing-manifest PDBs passed through "
                "despite resolution > %.1f Å (or missing).",
                n_f2_exempt, args.resolution_cap)
    _audit_drop(f2, f"F2[resolution > {args.resolution_cap}Å or missing, "
                    "new entries only]")

    # F3: antigen_type not allowed
    f3 = ~out["antigen_type"].apply(_is_allowed_antigen_type)
    _audit_drop(f3, "F3[antigen_type not protein/peptide combo]")

    keep = ~(f1 | f2 | f3)
    kept = out.loc[keep].reset_index(drop=True).copy()
    logger.info("After all filters: %d / %d rows", len(kept), len(out))

    # ── Test-set preservation hard check ─────────────────────────────
    kept_pdbs = set(kept["pdb"])
    missing_test_pdbs = cur_test_pdbs - kept_pdbs
    if missing_test_pdbs:
        logger.error(
            "Test-set preservation FAILED — %d current-test PDBs missing "
            "from filtered manifest: %s",
            len(missing_test_pdbs),
            ", ".join(sorted(missing_test_pdbs)[:20]),
        )
        sys.exit(2)
    logger.info("Test-set preservation OK: all %d current-test PDBs present.",
                len(cur_test_pdbs))

    # ── Stage 3: write ────────────────────────────────────────────────
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(args.output_tsv, sep="\t", index=False)
    logger.info("Wrote %d rows to %s", len(kept), args.output_tsv)

    # ── Final stats ──────────────────────────────────────────────────
    res_kept = pd.to_numeric(kept["resolution"], errors="coerce")
    logger.info("=" * 60)
    logger.info("Final manifest stats")
    logger.info("  rows:                 %d", len(kept))
    logger.info("  antigen-bearing rows: %d (= rows, F1 dropped empties)",
                int((kept["antigen_chain"].str.strip() != "").sum()))
    logger.info("  multi-chain antigens: %d", int(kept["antigen_chain"].str.contains(r"\|").sum()))
    logger.info("  resolution mean/median: %.2f / %.2f Å",
                res_kept.mean(), res_kept.median())
    logger.info("  resolution range:     %.2f - %.2f Å",
                res_kept.min(), res_kept.max())
    dates = pd.to_datetime(kept["date"], format="%m/%d/%y", errors="coerce")
    dates_v = dates.dropna()
    if len(dates_v) > 0:
        logger.info("  date range:           %s -> %s (n=%d valid)",
                    dates_v.min().date(), dates_v.max().date(), len(dates_v))
    logger.info("  method breakdown:")
    for m, n in kept["method"].value_counts().items():
        logger.info("    %-25s %d", m, n)
    logger.info("  antigen_type breakdown:")
    for t, n in kept["antigen_type"].value_counts().head(10).items():
        logger.info("    %-25s %d", str(t), n)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
