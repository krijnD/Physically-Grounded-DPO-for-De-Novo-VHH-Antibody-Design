#!/usr/bin/env python3
"""ANDD dataset curation script.

Verifies the VHH chain and antigen chain(s) of each ANDD entry against the
actual PDB geometry (not the CSV metadata, which can misidentify the
biological target vs. the VHH's real binding partner — see Nb35 / 7F1O).
Writes a curated CSV that is a drop-in replacement for the input: same
schema, same columns, but with ``H_Chain Auth Asym ID`` and
``Ag_Auth Asym ID`` overwritten by structure-verified values. Entries that
cannot be curated are written to a separate rejected CSV for auditing.

The sequence column ``Ab/Nano H_Chain AA`` is **not** modified. CSV stores
the SEQRES-equivalent full construct sequence, whereas a PDB ATOM-derived
sequence only includes residues with resolved coordinates; those routinely
differ by unresolved termini / loop gaps, so comparing them is not
informative about data quality.

Non-destructive by design: never modifies the input CSV or the PDB
directory.  The output CSV path must differ from the input, and existing
output files are not clobbered unless ``--overwrite-output`` is passed.

Algorithm (per row):
  1. Enumerate chains in the PDB.
  2. For each chain, try ANARCI (abnumber); keep chains that parse as
     VH-type and whose length falls in [100, 160] — a permissive VHH band
     that covers His-tags / linkers (Nb35 in 7F1O is 160 residues).
  3. Pick the VHH. If only one candidate, use it. If many, try in order:
     (a) any letter listed in CSV's ``H_Chain Auth Asym ID`` (which may
         be comma-separated for homodimers, e.g. ``"B, D, F, H"``) that
         matches a candidate chain — most direct annotation hint, robust
         to SEQRES-vs-ATOM divergence;
     (b) exact match against the CSV's ``Ab/Nano H_Chain AA`` sequence;
     (c) otherwise flag as ``ambiguous_vhh`` and reject — we prefer a
         smaller but certain dataset to silent heuristic overrides of
         CSV annotations.
  4. For each chain NOT in the VH-type set (i.e. excluding every chain
     that passed step 2, not just the picked VHH), count VHH residues
     with any heavy atom within ``--contact-cutoff`` (default 5 Å) of any
     of its heavy atoms. Chains with ≥ ``--min-contact-residues``
     (default 5) are antigens. Excluding all VH-type chains prevents
     multi-VHH assemblies / VHH–Fab / anti-idiotypic pairs from being
     scored as antigens (observed in ~20% of ANDD VHH PDBs).
  5. Rejection statuses: ``load_failed`` (missing file or parse error),
     ``no_vhh`` (no candidate passed ANARCI filters), ``ambiguous_vhh``
     (multiple candidates, no reliable hint), ``no_antigen`` (no non-VHH
     chain met the contact threshold). Otherwise write curated row.

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
import re
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

# ── CDR3 annotation sanity thresholds ─────────────────────────────────────
# abnumber silently truncates CDR3 on malformed / truncated constructs.
# Observed failure: 7F1G has CDR3="AGR" (len 3) because its ATOM record
# ends "…TSAGRRGPGTQVTVSS" with no WG[x]GT J-motif — the real C-terminal
# framework is missing. We reject such VHH candidates rather than emit
# bad CDR3 annotations downstream.
MIN_CDR3_LEN = 5
MAX_CDR3_LEN = 30
# Canonical J-region motif immediately following CDR3; its absence means
# the construct is truncated and CDR3 boundaries are unreliable.
#
# Position 1 is relaxed from strict ``W`` to ``[WRK]``: camelid / engineered
# VHHs are observed with W→R or W→K substitutions at the otherwise highly
# conserved FR4 start residue (W103). A diagnostic pass over the ANDD
# post-DiffAb slice found 166/178 no_vhh rejections were caused by this
# exact substitution — the chains are otherwise canonical VH with correct
# CDR3 length and full FR4 geometry (e.g. ``...YAYRGQGTQVTVSS``).
_J_MOTIF_RE = re.compile(r"[WRK]G[A-Z]GT")


# ── Helpers ───────────────────────────────────────────────────────────────
def _list_chain_ids(structure) -> list[str]:
    """Chain letters in the first model, preserving PDB order."""
    return [chain.id for chain in structure[0].get_chains()]


def _identity(a: str, b: str) -> float:
    """Length-normalized identity over the min-aligned prefix.

    Lifted verbatim from data scripts/diagnose_rejections.py — used by
    _pick_vhh's identity-rescue fallback to disambiguate VH candidates
    that share the same domain modulo unresolved termini.
    """
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / n


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

        # CDR3 annotation sanity: length must be plausible, and the
        # construct must contain a canonical J-motif. See module-level
        # note on 7F1G ("AGR") for the motivating failure mode.
        cdr3 = chain.cdr3_seq
        if not (MIN_CDR3_LEN <= len(cdr3) <= MAX_CDR3_LEN):
            logger.debug(
                "Chain %s of %s: CDR3 len %d outside [%d,%d] — "
                "skipping candidate (abnumber annotation unreliable).",
                chain_id, pdb_path, len(cdr3), MIN_CDR3_LEN, MAX_CDR3_LEN,
            )
            continue
        if not _J_MOTIF_RE.search(seq):
            logger.debug(
                "Chain %s of %s: no WG[x]GT J-motif — construct appears "
                "truncated, skipping candidate.",
                chain_id, pdb_path,
            )
            continue

        candidates.append({"chain_id": chain_id, "sequence": seq})
    return candidates


def _pick_vhh(
    candidates: list[dict],
    csv_seq: str | None,
    csv_letters: set[str] | None,
    *,
    identity_threshold: float = 0.95,
) -> tuple[dict, bool]:
    """Pick the best VHH candidate. Returns (chosen, is_ambiguous).

    Priority:
    - 1 candidate → it, unambiguous.
    - >1 and any CSV-listed letter matches a candidate chain → pick the
      first such candidate, unambiguous. ANDD lists every equivalent
      chain for homodimers (``"B, D, F, H"`` etc.); any match is correct
      because the chains carry the same sequence, and the picked
      candidate's chain_id is what flows downstream.
    - >1, no letter match, CSV seq matches one candidate exactly → that
      one, unambiguous. Rarely fires because CSV sequence is SEQRES-style
      (full biological construct) while candidate sequences are
      ATOM-derived (resolved residues only).
    - >1 and exactly one candidate matches CSV seq at >= identity_threshold
      (prefix-aligned identity) → that one, unambiguous. Same VH domain
      modulo unresolved termini. Brief 03 §4.4 confirmed: rescues 7/32
      previously-ambiguous entries on the post_diffab subset, zero
      false-pick risk.
    - >1 and no hint resolves the tie → shortest (heuristic), flagged
      ambiguous. Caller should treat ambiguous picks as rejections, not
      silent overrides of CSV data.
    """
    if len(candidates) == 1:
        return candidates[0], False
    if csv_letters:
        for c in candidates:
            if c["chain_id"] in csv_letters:
                return c, False
    if csv_seq:
        for c in candidates:
            if c["sequence"] == csv_seq:
                return c, False
        # Identity-rescue fallback: same VH domain modulo unresolved
        # termini. Only commit if exactly one candidate clears the bar.
        scored = [(c, _identity(csv_seq, c["sequence"])) for c in candidates]
        above = [(c, s) for c, s in scored if s >= identity_threshold]
        if len(above) == 1:
            return above[0][0], False
    # Heuristic fallback — caller should reject.
    return min(candidates, key=lambda c: len(c["sequence"])), True


def _count_contacts_per_chain(
    structure,
    vhh_chain_id: str,
    exclude_chain_ids: set[str],
    cutoff: float,
) -> dict[str, int]:
    """For each chain not in ``exclude_chain_ids``, count VHH residues with
    any heavy atom within ``cutoff`` Å of any heavy atom of that chain.

    ``exclude_chain_ids`` must contain at least the picked VHH plus every
    other VH-type chain in the structure. Empirically ~20% of ANDD VHH
    PDBs have additional VH-type chains (multi-VHH assemblies, VHH–Fab
    complexes, anti-idiotypic pairs); treating those as antigen candidates
    pollutes the curated set with antibody-antibody contacts that DiffAb
    should not see as ground-truth epitopes.

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
        if chain.id in exclude_chain_ids:
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


def _normalize_csv_letter(raw) -> set[str] | None:
    """Return a set of chain-letter strings, or None for missing values.

    ANDD's ``H_Chain Auth Asym ID`` column can hold a single letter
    (``"H"``) or — for homodimers / multi-copy VHH assemblies — a
    comma-separated list of every equivalent chain (``"B, D, F, H"``,
    ``"A, B, C, D, E, F, G, H"``). Empty values, ``nan``, and the
    backslash sentinel ``\\`` return None. The whitespace after each
    comma varies by row, so we tokenize on comma and strip.
    """
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    if s in ("", "nan", "\\"):
        return None
    letters = {tok.strip() for tok in s.split(",") if tok.strip()}
    return letters or None


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
            f"No VH-type chain passed filters (length [{VHH_MIN_LEN},"
            f"{VHH_MAX_LEN}], CDR3 length [{MIN_CDR3_LEN},{MAX_CDR3_LEN}], "
            f"WG[x]GT J-motif present) among chains {chain_ids}."
        )
        return out, False

    csv_seq = _normalize_csv_seq(row.get("Ab/Nano H_Chain AA"))
    csv_letters = _normalize_csv_letter(row.get("H_Chain Auth Asym ID"))
    vhh, is_ambiguous = _pick_vhh(candidates, csv_seq, csv_letters)

    # Reject on genuine ambiguity: multiple VH-type candidates with no
    # reliable hint (CSV letters missing or none match a candidate, and
    # CSV sequence did not match any candidate exactly). For a DPO
    # training set we prefer fewer but certain VHH / antigen assignments
    # over heuristic guesses — see README "Step 3" for rationale.
    if is_ambiguous:
        cand_ids = [c["chain_id"] for c in candidates]
        out["curation_status"] = "ambiguous_vhh"
        out["curation_notes"] = (
            f"Multiple VH candidates {cand_ids}; no CSV H_Chain letter "
            f"(from {sorted(csv_letters) if csv_letters else []}) matches "
            f"a candidate and CSV sequence did not match any candidate "
            f"exactly. Rejected to avoid uncertain VHH / antigen assignment."
        )
        return out, False

    # Exclude ALL VH-type chains from antigen scoring — not just the picked
    # VHH. ~20% of ANDD PDBs have additional VH chains (multi-VHH
    # assemblies, VHH–Fab, anti-idiotypic) that would otherwise pass the
    # contact threshold and be mis-annotated as antigens.
    all_vhh_chain_ids = {c["chain_id"] for c in candidates}
    contacts = _count_contacts_per_chain(
        structure, vhh["chain_id"], all_vhh_chain_ids, contact_cutoff
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

    # ── Curated row: overwrite only chain + antigen columns ──
    # The CSV's Ab/Nano H_Chain AA comes from SEQRES / the full biological
    # construct; PDB extract_chain_sequence only includes residues with
    # resolved ATOM coordinates, so they routinely differ by unresolved
    # termini or loop gaps. That divergence is crystallography, not a data
    # issue, so we leave the CSV sequence untouched.
    out["H_Chain Auth Asym ID"] = vhh["chain_id"]
    out["Ag_Auth Asym ID"] = ",".join(antigen_chains)
    out["curation_status"] = "ok"
    out["curation_notes"] = ""
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

    # ANDD rows per PDB are redundant (a single PDB can appear 2–80× from
    # different sources). ``Predicted_or_Not`` takes values ``real``,
    # ``\`` (unlabelled sentinel), or ``predicted``. A naive ``keep="first"``
    # dedup can retain a ``predicted`` row for a PDB that also has a ``real``
    # row, silently putting a model-designed sequence into the curated set.
    # Sort by preference first (real > unlabelled > predicted) so the
    # retained row after dedup is the least-contaminated annotation.
    _label_priority = {"real": 0, "\\": 1, "predicted": 2}
    if "Predicted_or_Not" in df.columns:
        df = (
            df.assign(_label_p=df["Predicted_or_Not"].map(
                lambda v: _label_priority.get(str(v).strip(), 1)
            ))
            .sort_values(["PDB_ID", "_label_p"], kind="stable")
            .drop(columns="_label_p")
        )
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
