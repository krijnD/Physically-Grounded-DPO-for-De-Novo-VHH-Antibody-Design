#!/usr/bin/env python3
"""ANDD dataset curation script.

Verifies the VHH chain and antigen chain(s) of each ANDD entry against the
actual PDB geometry (not the CSV metadata, which can misidentify the
biological target vs. the VHH's real binding partner — see Nb35 / 7F1O).
Writes a curated CSV that is a drop-in replacement for the input: same
schema, same columns, but with ``H_Chain Auth Asym ID`` and
``Ag_Auth Asym ID`` overwritten by structure-verified values. Entries that
cannot be curated are written to a separate rejected CSV for auditing.

Non-destructive by design: never modifies the input CSV or the PDB
directory.  The output CSV path must differ from the input, and existing
output files are not clobbered unless ``--overwrite-output`` is passed.

Algorithm (per row):
  1. Enumerate chains in the PDB.
  2. For each chain, try ANARCI (abnumber); keep chains that parse as
     VH-type and whose length falls in [100, 160] — a permissive VHH band
     that covers His-tags / linkers (Nb35 in 7F1O is 160 residues).
  3. Pick the VHH: if one candidate, use it; if many, prefer exact
     sequence match against the CSV's ``Ab/Nano H_Chain AA`` column,
     else pick the shortest.
  4. For each non-VHH chain, count VHH residues with any heavy atom
     within ``--contact-cutoff`` (default 5 Å) of any of its heavy atoms.
     Chains with ≥ ``--min-contact-residues`` (default 5) are antigens.
  5. If no VHH → reject as ``no_vhh``.  If no antigen → reject as
     ``no_antigen``.  Otherwise write curated row.

Usage:
    python "data scripts/curate_andd.py" \\
        --input-csv    /path/to/ANDD_VHH_with_structure.csv \\
        --pdb-dir      /path/to/VHH_structures_post_iglm \\
        --output-csv   /path/to/ANDD_VHH_curated.csv \\
        --rejected-csv /path/to/ANDD_VHH_rejected.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from abnumber import Chain as AbnumberChain
from abnumber.exceptions import ChainParseError
from Bio.PDB.NeighborSearch import NeighborSearch

# Ensure project root is on sys.path so we can reuse src/ utilities.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.pdb_utils import load_structure  # noqa: E402
from src.common.sabdab_loader import extract_chain_sequence  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("curate_andd")

# ── VHH classification thresholds ─────────────────────────────────────────
# Lower bound: typical VH domain is ~115 residues; we allow 100 for
# truncated constructs. Upper bound: Nb35 in 7F1O is 160 (His-tag + linker),
# which is the longest "VHH" we realistically expect.
VHH_MIN_LEN = 100
VHH_MAX_LEN = 160


# ── Helpers ───────────────────────────────────────────────────────────────
def _list_chain_ids(structure) -> list[str]:
    """Chain letters in the first model, preserving PDB order."""
    return [chain.id for chain in structure[0].get_chains()]


def _identify_vhh_candidates(pdb_path: str, chain_ids: list[str]) -> list[dict]:
    """Return chains that ANARCI classifies as VH-type within the VHH length band.

    Each dict has keys ``chain_id`` and ``sequence``.  Chains that fail to
    parse as antibodies (most non-Ig proteins) are silently skipped.
    """
    candidates: list[dict] = []
    for chain_id in chain_ids:
        seq = extract_chain_sequence(pdb_path, chain_id)
        if seq is None:
            continue
        if not (VHH_MIN_LEN <= len(seq) <= VHH_MAX_LEN):
            continue
        try:
            chain = AbnumberChain(seq, scheme="kabat", assign_germline=False)
        except ChainParseError:
            continue
        except Exception as e:
            logger.debug(
                "ANARCI raised non-parse error on chain %s of %s: %s",
                chain_id, pdb_path, e,
            )
            continue
        if chain.chain_type != "H":
            continue
        candidates.append({"chain_id": chain_id, "sequence": seq})
    return candidates


def _pick_vhh(candidates: list[dict], csv_seq: str | None) -> tuple[dict, bool]:
    """Pick the best VHH candidate. Returns (chosen, is_ambiguous).

    - 1 candidate → it, unambiguous.
    - >1 and CSV seq matches one exactly → that one, unambiguous.
    - >1 and no exact match → shortest (heuristic), flagged ambiguous.
    """
    if len(candidates) == 1:
        return candidates[0], False
    if csv_seq:
        for c in candidates:
            if c["sequence"] == csv_seq:
                return c, False
    # Heuristic fallback
    return min(candidates, key=lambda c: len(c["sequence"])), True


def _count_contacts_per_chain(
    structure, vhh_chain_id: str, cutoff: float
) -> dict[str, int]:
    """For each non-VHH chain, count VHH residues with any heavy atom within
    ``cutoff`` Å of any heavy atom of that chain.

    Returns a dict {chain_id: residue_contact_count}.  Hetero residues
    (waters, ligands) are skipped on both sides.  Same K-D tree pattern as
    src/masking/paratope_detector.py.
    """
    model = structure[0]
    try:
        vhh_chain = model[vhh_chain_id]
    except KeyError:
        return {}

    vhh_residues = [
        r for r in vhh_chain.get_residues() if not r.id[0].strip()
    ]

    results: dict[str, int] = {}
    for chain in model.get_chains():
        if chain.id == vhh_chain_id:
            continue
        heavy_atoms = [
            a for a in chain.get_atoms()
            if a.element != "H" and not a.parent.id[0].strip()
        ]
        if not heavy_atoms:
            continue

        ns = NeighborSearch(heavy_atoms)
        n_contacts = 0
        for residue in vhh_residues:
            for atom in residue.get_atoms():
                if atom.element == "H":
                    continue
                if ns.search(atom.coord, cutoff):
                    n_contacts += 1
                    break  # residue counted; move to next residue
        if n_contacts > 0:
            results[chain.id] = n_contacts
    return results


def _build_pdb_lookup(pdb_dir: Path) -> dict[str, Path]:
    """Case-insensitive map from PDB id → .pdb file path."""
    return {p.stem.lower(): p for p in pdb_dir.glob("*.pdb")}


def _normalize_csv_seq(raw) -> str | None:
    """Return a stripped sequence string, or None for missing values."""
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    if s in ("", "nan", "\\"):
        return None
    return s


# ── Core per-row routine ──────────────────────────────────────────────────
def _curate_row(
    row: pd.Series,
    pdb_lookup: dict[str, Path],
    contact_cutoff: float,
    min_contact_residues: int,
) -> tuple[dict, bool]:
    """Process one CSV row. Returns (curated_row_dict, is_ok).

    ``is_ok=True`` means the row should go to the curated CSV; ``False``
    means the rejected CSV.  Original row values are preserved on rejection
    (we never overwrite H_Chain / Ag_Auth when we couldn't verify them).
    """
    pdb_id = str(row["PDB_ID"]).strip()
    out: dict = row.to_dict()
    # Traceability fields — recorded for every row, ok or rejected.
    out["curation_vhh_chain_original"] = row.get("H_Chain Auth Asym ID")
    out["curation_antigen_chains_original"] = row.get("Ag_Auth Asym ID")
    out["curation_vhh_contacts_per_chain"] = None
    out["curation_status"] = None
    out["curation_notes"] = ""

    pdb_path = pdb_lookup.get(pdb_id.lower())
    if pdb_path is None:
        out["curation_status"] = "load_failed"
        out["curation_notes"] = f"No PDB file found for {pdb_id} in input directory."
        return out, False

    try:
        structure = load_structure(str(pdb_path), pdb_id)
    except Exception as e:
        out["curation_status"] = "load_failed"
        out["curation_notes"] = f"Biopython failed to parse PDB: {e}"
        return out, False

    chain_ids = _list_chain_ids(structure)
    candidates = _identify_vhh_candidates(str(pdb_path), chain_ids)

    if not candidates:
        out["curation_status"] = "no_vhh"
        out["curation_notes"] = (
            f"No VH-type chain of length [{VHH_MIN_LEN},{VHH_MAX_LEN}] "
            f"found among chains {chain_ids}."
        )
        return out, False

    csv_seq = _normalize_csv_seq(row.get("Ab/Nano H_Chain AA"))
    vhh, is_ambiguous = _pick_vhh(candidates, csv_seq)

    contacts = _count_contacts_per_chain(
        structure, vhh["chain_id"], contact_cutoff
    )
    out["curation_vhh_contacts_per_chain"] = json.dumps(contacts)

    antigen_chains = sorted(
        ch for ch, n in contacts.items() if n >= min_contact_residues
    )
    if not antigen_chains:
        out["curation_status"] = "no_antigen"
        out["curation_notes"] = (
            f"VHH chain {vhh['chain_id']}: no non-VHH chain has "
            f">= {min_contact_residues} residues within {contact_cutoff} Å "
            f"(contacts per chain: {contacts})."
        )
        return out, False

    # ── Curated row: overwrite chain + antigen columns, handle seq mismatch ──
    out["H_Chain Auth Asym ID"] = vhh["chain_id"]
    out["Ag_Auth Asym ID"] = ",".join(antigen_chains)

    notes: list[str] = []
    status = "ok"

    if csv_seq is None:
        out["Ab/Nano H_Chain AA"] = vhh["sequence"]
        notes.append(
            f"CSV 'Ab/Nano H_Chain AA' was empty; filled from PDB chain "
            f"{vhh['chain_id']}."
        )
    elif csv_seq != vhh["sequence"]:
        out["Ab/Nano H_Chain AA"] = vhh["sequence"]
        status = "ok_sequence_mismatch"
        notes.append(
            f"CSV sequence (len {len(csv_seq)}) differs from PDB chain "
            f"{vhh['chain_id']} (len {len(vhh['sequence'])}); PDB sequence used."
        )

    if is_ambiguous:
        status = "ambiguous_vhh"
        cand_ids = [c["chain_id"] for c in candidates]
        notes.append(
            f"Multiple VH candidates {cand_ids}; picked {vhh['chain_id']} "
            f"(shortest, no CSV match). Review recommended."
        )

    out["curation_status"] = status
    out["curation_notes"] = " | ".join(notes)
    return out, True


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-csv", required=True, type=Path,
                        help="ANDD CSV with columns PDB_ID, H_Chain Auth Asym ID, "
                             "Ag_Auth Asym ID, Ab/Nano H_Chain AA.")
    parser.add_argument("--pdb-dir", required=True, type=Path,
                        help="Directory containing the PDB files.")
    parser.add_argument("--output-csv", required=True, type=Path,
                        help="Path for the curated CSV (must differ from input).")
    parser.add_argument("--rejected-csv", required=True, type=Path,
                        help="Path for the rejected CSV (must differ from input).")
    parser.add_argument("--contact-cutoff", type=float, default=5.0,
                        help="Heavy-atom distance cutoff in Å (default: 5.0).")
    parser.add_argument("--min-contact-residues", type=int, default=5,
                        help="Min VHH residues in contact to count a chain "
                             "as antigen (default: 5).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N rows (for testing).")
    parser.add_argument("--overwrite-output", action="store_true",
                        help="Allow overwriting existing output/rejected CSVs.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    # ── Path validation ──
    if not args.input_csv.exists():
        logger.error("Input CSV not found: %s", args.input_csv)
        sys.exit(1)
    if not args.pdb_dir.exists() or not args.pdb_dir.is_dir():
        logger.error("PDB dir not found or not a directory: %s", args.pdb_dir)
        sys.exit(1)

    # Non-destructive guards.
    resolved_input = args.input_csv.resolve()
    for label, path in (("output", args.output_csv), ("rejected", args.rejected_csv)):
        if path.resolve() == resolved_input:
            logger.error(
                "--%s-csv (%s) is the same file as --input-csv — refusing "
                "to overwrite input.", label, path,
            )
            sys.exit(2)
        if path.exists() and not args.overwrite_output:
            logger.error(
                "--%s-csv already exists at %s. Pass --overwrite-output "
                "to replace it.", label, path,
            )
            sys.exit(2)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.rejected_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── Load input ──
    logger.info("Reading %s", args.input_csv)
    df = pd.read_csv(args.input_csv)
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset="PDB_ID", keep="first")
    logger.info(
        "Loaded %d rows (%d unique PDB_IDs after dedup)", n_before_dedup, len(df)
    )

    pdb_lookup = _build_pdb_lookup(args.pdb_dir)
    logger.info("Discovered %d PDB files in %s", len(pdb_lookup), args.pdb_dir)

    # Keep only rows whose PDB is actually present in the input directory.
    # The ANDD CSV covers 1300 PDBs; a typical --pdb-dir is a subset-specific
    # slice (e.g. VHH_structures_post_iglm has 264). Applying --limit before
    # this filter would sample the first N alphabetical entries and almost
    # always miss the subset, producing 20/20 "load_failed".
    n_before_filter = len(df)
    df = df[df["PDB_ID"].astype(str).str.lower().isin(pdb_lookup)]
    logger.info(
        "Restricted to %d/%d rows with a PDB in %s",
        len(df), n_before_filter, args.pdb_dir,
    )

    if args.limit:
        df = df.head(args.limit)
        logger.info("Limited to first %d rows for testing", args.limit)

    # ── Curate ──
    curated: list[dict] = []
    rejected: list[dict] = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows(), 1):
        pdb_id = str(row["PDB_ID"]).strip()
        try:
            out_row, ok = _curate_row(
                row, pdb_lookup, args.contact_cutoff, args.min_contact_residues,
            )
        except Exception as e:
            logger.exception("[%d/%d] %s: curation crashed", i, total, pdb_id)
            out_row = row.to_dict()
            out_row.setdefault("curation_vhh_chain_original",
                               row.get("H_Chain Auth Asym ID"))
            out_row.setdefault("curation_antigen_chains_original",
                               row.get("Ag_Auth Asym ID"))
            out_row["curation_vhh_contacts_per_chain"] = None
            out_row["curation_status"] = "load_failed"
            out_row["curation_notes"] = f"Unhandled exception: {e}"
            ok = False

        if ok:
            curated.append(out_row)
            logger.info(
                "[%d/%d] %s: %s (VHH=%s, Ag=%s)",
                i, total, pdb_id, out_row["curation_status"],
                out_row["H_Chain Auth Asym ID"], out_row["Ag_Auth Asym ID"],
            )
        else:
            rejected.append(out_row)
            logger.warning(
                "[%d/%d] %s: REJECTED (%s) — %s",
                i, total, pdb_id,
                out_row.get("curation_status"),
                out_row.get("curation_notes"),
            )

    # ── Write outputs ──
    pd.DataFrame(curated).to_csv(args.output_csv, index=False)
    pd.DataFrame(rejected).to_csv(args.rejected_csv, index=False)

    # ── Summary ──
    status_counts: dict[str, int] = {}
    for row in curated + rejected:
        s = row.get("curation_status") or "unknown"
        status_counts[s] = status_counts.get(s, 0) + 1

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("  Total processed:  %d", total)
    logger.info("  Curated:          %d → %s", len(curated), args.output_csv)
    logger.info("  Rejected:         %d → %s", len(rejected), args.rejected_csv)
    for s, n in sorted(status_counts.items()):
        logger.info("    %-25s %d", s, n)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
