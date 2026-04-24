#!/usr/bin/env python3
"""Diagnostic for ANDD_VHH_rejected_diffab.csv.

Reports how many rejected rows could be recovered by loosening a *single*
filter, without reintroducing silent guesses. Two reports:

  1. ambiguous_vhh rescue potential
     For each row where multiple VH candidates exist and neither the CSV
     H_Chain letter nor the CSV sequence matched exactly, check whether a
     weaker-but-still-strong match would disambiguate:
       * substring containment (CSV seq ⊂ exactly one candidate, or vice
         versa)
       * high-identity match (exactly one candidate with ≥ threshold %
         identity over the aligned length)

  2. no_vhh single-filter-fail classification
     For every chain of every PDB that produced a no_vhh rejection,
     record which step of _identify_vhh_candidates dropped it:
         length_oob, abnumber_parse_error, not_H_type,
         cdr3_len_oob, no_J_motif, ok
     Then for each row, report its "best-progress" chain (the chain that
     made it furthest). If the best-progress reason is a single filter,
     that row would be recoverable by relaxing only that filter.

Usage:
    python "data scripts/diagnose_rejections.py" \\
        --rejected-csv /path/to/ANDD_VHH_rejected_diffab.csv \\
        --pdb-dir      /path/to/VHH_structures_post_diffab \\
        [--identity-threshold 0.95]
"""

import argparse
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from abnumber import Chain as AbnumberChain
from abnumber.exceptions import ChainParseError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.pdb_utils import load_structure  # noqa: E402
from src.common.sabdab_loader import extract_chain_sequence  # noqa: E402

# Mirror constants from curate_andd.py so this diagnostic stays faithful to
# the production filter chain. Keep in sync manually if those change.
VHH_MIN_LEN = 100
VHH_MAX_LEN = 160
MIN_CDR3_LEN = 5
MAX_CDR3_LEN = 30
_J_MOTIF_RE = re.compile(r"WG[A-Z]GT")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("diagnose_rejections")


# ── ambiguous_vhh rescue ──────────────────────────────────────────────────
def _identify_vh_candidates(pdb_path: str, chain_ids: list[str]) -> list[dict]:
    """Same pipeline as curate_andd._identify_vhh_candidates."""
    out = []
    for cid in chain_ids:
        seq = extract_chain_sequence(pdb_path, cid)
        if seq is None:
            continue
        if not (VHH_MIN_LEN <= len(seq) <= VHH_MAX_LEN):
            continue
        try:
            ch = AbnumberChain(seq, scheme="kabat", assign_germline=False)
        except ChainParseError:
            continue
        except Exception:
            continue
        if ch.chain_type != "H":
            continue
        cdr3 = ch.cdr3_seq
        if not (MIN_CDR3_LEN <= len(cdr3) <= MAX_CDR3_LEN):
            continue
        if not _J_MOTIF_RE.search(seq):
            continue
        out.append({"chain_id": cid, "sequence": seq})
    return out


def _identity(a: str, b: str) -> float:
    """Simple length-normalized identity over min-aligned prefix.

    For detecting "same VH domain with unresolved termini" — not a
    general alignment. Good enough to flag unambiguous candidates.
    """
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / n


def _diagnose_ambiguous(row, pdb_lookup, identity_threshold: float) -> dict:
    pdb_id = str(row["PDB_ID"]).strip()
    result = {
        "pdb_id": pdb_id,
        "csv_letter": row.get("curation_vhh_chain_original"),
        "n_candidates": 0,
        "substring_unique": False,
        "identity_unique": False,
        "rescue_possible": False,
        "best_match_chain": None,
        "best_identity": None,
    }
    pdb_path = pdb_lookup.get(pdb_id.lower())
    if pdb_path is None:
        return result
    try:
        structure = load_structure(str(pdb_path), pdb_id)
    except Exception:
        return result
    chain_ids = [c.id for c in structure[0].get_chains()]
    candidates = _identify_vh_candidates(str(pdb_path), chain_ids)
    result["n_candidates"] = len(candidates)
    csv_seq = row.get("Ab/Nano H_Chain AA")
    if not isinstance(csv_seq, str) or not csv_seq.strip():
        return result
    csv_seq = csv_seq.strip()

    # substring containment test
    substring_hits = [
        c for c in candidates
        if csv_seq in c["sequence"] or c["sequence"] in csv_seq
    ]
    result["substring_unique"] = len(substring_hits) == 1

    # identity test (simple prefix-aligned identity)
    scored = [(c, _identity(csv_seq, c["sequence"])) for c in candidates]
    above = [(c, s) for c, s in scored if s >= identity_threshold]
    result["identity_unique"] = len(above) == 1
    if scored:
        best = max(scored, key=lambda x: x[1])
        result["best_match_chain"] = best[0]["chain_id"]
        result["best_identity"] = round(best[1], 3)

    result["rescue_possible"] = (
        result["substring_unique"] or result["identity_unique"]
    )
    return result


# ── no_vhh single-filter diagnosis ────────────────────────────────────────
# Ordinal stage values — "how far did this chain progress through the
# filter chain". A row's best-progress chain is the one with the highest
# stage value. If a row's best chain stopped at a single filter, relaxing
# only that filter would recover the row.
STAGE_EXTRACT_FAIL = 0
STAGE_LENGTH_OOB = 1
STAGE_PARSE_ERROR = 2
STAGE_NOT_H = 3
STAGE_CDR3_OOB = 4
STAGE_NO_J_MOTIF = 5
STAGE_OK = 6

_STAGE_NAMES = {
    STAGE_EXTRACT_FAIL: "extract_fail",
    STAGE_LENGTH_OOB: "length_oob",
    STAGE_PARSE_ERROR: "abnumber_parse_error",
    STAGE_NOT_H: "not_H_type",
    STAGE_CDR3_OOB: "cdr3_len_oob",
    STAGE_NO_J_MOTIF: "no_J_motif",
    STAGE_OK: "ok",
}


def _stage_of_chain(pdb_path: str, chain_id: str) -> tuple[int, dict]:
    """Return (stage, details) describing how far this chain progressed."""
    details = {"chain_id": chain_id}
    seq = extract_chain_sequence(pdb_path, chain_id)
    if seq is None:
        return STAGE_EXTRACT_FAIL, details
    details["length"] = len(seq)
    if not (VHH_MIN_LEN <= len(seq) <= VHH_MAX_LEN):
        return STAGE_LENGTH_OOB, details
    try:
        ch = AbnumberChain(seq, scheme="kabat", assign_germline=False)
    except ChainParseError:
        return STAGE_PARSE_ERROR, details
    except Exception:
        return STAGE_PARSE_ERROR, details
    details["chain_type"] = ch.chain_type
    if ch.chain_type != "H":
        return STAGE_NOT_H, details
    cdr3 = ch.cdr3_seq
    details["cdr3_len"] = len(cdr3)
    if not (MIN_CDR3_LEN <= len(cdr3) <= MAX_CDR3_LEN):
        return STAGE_CDR3_OOB, details
    if not _J_MOTIF_RE.search(seq):
        return STAGE_NO_J_MOTIF, details
    return STAGE_OK, details


def _diagnose_no_vhh(row, pdb_lookup) -> dict:
    pdb_id = str(row["PDB_ID"]).strip()
    result = {"pdb_id": pdb_id, "best_stage": None, "best_stage_name": None}
    pdb_path = pdb_lookup.get(pdb_id.lower())
    if pdb_path is None:
        return result
    try:
        structure = load_structure(str(pdb_path), pdb_id)
    except Exception:
        return result
    chain_ids = [c.id for c in structure[0].get_chains()]
    best = STAGE_EXTRACT_FAIL
    best_details: dict = {}
    for cid in chain_ids:
        stage, details = _stage_of_chain(str(pdb_path), cid)
        if stage > best:
            best = stage
            best_details = details
    result["best_stage"] = best
    result["best_stage_name"] = _STAGE_NAMES[best]
    result["best_details"] = best_details
    return result


# ── main ──────────────────────────────────────────────────────────────────
def _build_pdb_lookup(pdb_dir: Path) -> dict:
    return {p.stem.lower(): p for p in pdb_dir.glob("*.pdb")}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rejected-csv", required=True, type=Path)
    parser.add_argument("--pdb-dir", required=True, type=Path)
    parser.add_argument("--identity-threshold", type=float, default=0.95)
    parser.add_argument("--out-ambiguous", type=Path, default=None,
                        help="Optional CSV path for per-row ambiguous diagnosis.")
    parser.add_argument("--out-no-vhh", type=Path, default=None,
                        help="Optional CSV path for per-row no_vhh diagnosis.")
    args = parser.parse_args()

    df = pd.read_csv(args.rejected_csv)
    pdb_lookup = _build_pdb_lookup(args.pdb_dir)
    print(f"Loaded {len(df)} rejected rows; {len(pdb_lookup)} PDB files available.\n")

    # ── ambiguous_vhh ──
    amb = df[df["curation_status"] == "ambiguous_vhh"]
    print(f"━━━ ambiguous_vhh: {len(amb)} rows ━━━")
    amb_results = [
        _diagnose_ambiguous(row, pdb_lookup, args.identity_threshold)
        for _, row in amb.iterrows()
    ]
    n_substring = sum(1 for r in amb_results if r["substring_unique"])
    n_identity = sum(1 for r in amb_results if r["identity_unique"])
    n_either = sum(1 for r in amb_results if r["rescue_possible"])
    n_neither = sum(1 for r in amb_results if not r["rescue_possible"])
    print(f"  Unique substring match            : {n_substring} / {len(amb)}")
    print(f"  Unique identity ≥ {args.identity_threshold:.2f} match   : {n_identity} / {len(amb)}")
    print(f"  Recoverable by either rule        : {n_either} / {len(amb)}")
    print(f"  Genuinely ambiguous (no signal)   : {n_neither} / {len(amb)}")
    if args.out_ambiguous:
        pd.DataFrame(amb_results).to_csv(args.out_ambiguous, index=False)
        print(f"  → wrote per-row details to {args.out_ambiguous}")

    # ── no_vhh ──
    novhh = df[df["curation_status"] == "no_vhh"]
    print(f"\n━━━ no_vhh: {len(novhh)} rows ━━━")
    novhh_results = [_diagnose_no_vhh(row, pdb_lookup) for _, row in novhh.iterrows()]
    stage_counter = Counter(r["best_stage_name"] for r in novhh_results)
    print("  Best-progress stage distribution (closer to 'ok' = easier to recover):")
    for stage in ["extract_fail", "length_oob", "abnumber_parse_error",
                  "not_H_type", "cdr3_len_oob", "no_J_motif", "ok"]:
        n = stage_counter.get(stage, 0)
        print(f"    {stage:25s} {n:4d}")

    # How many would be recovered if we relaxed exactly one filter?
    recover_cdr3 = stage_counter.get("cdr3_len_oob", 0)
    recover_jmotif = stage_counter.get("no_J_motif", 0)
    print("\n  Rows recoverable by relaxing *one* filter:")
    print(f"    drop CDR3 length bounds           : +{recover_cdr3}")
    print(f"    drop J-motif requirement          : +{recover_jmotif}")
    print(f"    drop both                         : +{recover_cdr3 + recover_jmotif}")

    if args.out_no_vhh:
        # Flatten best_details into the output row for inspection.
        flat = []
        for r in novhh_results:
            d = {k: v for k, v in r.items() if k != "best_details"}
            d.update({f"bd_{k}": v for k, v in r.get("best_details", {}).items()})
            flat.append(d)
        pd.DataFrame(flat).to_csv(args.out_no_vhh, index=False)
        print(f"  → wrote per-row details to {args.out_no_vhh}")


if __name__ == "__main__":
    main()
