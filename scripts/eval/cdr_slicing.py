"""
Brief 15 Track 1 — ANARCI-based CDR slicing helper for design PDBs.

Why this exists
---------------
The Brief 12 + 13 dispatchers hard-coded CDR_WINDOWS by Chothia author-numbered
resseq:
    {"H1": (26, 32), "H2": (52, 56), "H3": (95, 102)}

That window is correct for the GT PDBs (Chothia author-numbered) but wrong for
the design PDBs. There are two design-PDB sources in the campaign:

  1. data/eval/judged_chunks/all_variants/vhh_monomers/<v>__<t>__<e>__<c>__s<n>.pdb
     — IMGT-renumbered judged-chunk PDBs, VHH chain renamed to "H".
     At these PDBs, resseq 95-102 lands on the conserved FR3 framework
     `KPEDTAVY` motif (NOT the H3 CDR). Read by per_position_modal_picks.py.

  2. runs/<variant>/eval_<testset>_pdbs/<entry>/<cdr>/sample_NNNN.pdb
     — DiffAb raw outputs, heterogeneous numbering. Mostly sequential 1→N,
     some starting at higher resseq, some carry insertion codes from the
     input PDB. Resseq 95-102 grabs actual H3 CDR for most entries here
     (but the slice may be truncated or shifted depending on the entry's
     numbering). Read by run_caar_epif1_array.py + run_scrmsd_array.py via
     CLI args (--design-pdb-roots / --pdb-roots).

This helper provides a single CDR-slicing function that works on BOTH PDB
sources by running ANARCII on the VHH heavy-chain sequence and mapping the
Chothia-numbered CDR positions (H1: 26-32, H2: 52-56, H3: 95-102) back to
PDB residues via chain-order indexing. No more hard-coded resseq windows on
the design side.

GT side (Brief 12/13) still uses the original CDR_WINDOWS resseq lookup —
that's correct for Chothia author-numbered GT PDBs and unchanged.

API
---
    from cdr_slicing import slice_cdrs
    cdrs = slice_cdrs("path/to/design.pdb", vhh_chain_hint=None)
    # cdrs == {"H1": [Residue, ...], "H2": [Residue, ...], "H3": [Residue, ...]}
    # Returns None if ANARCII fails. Falls back to a Cys-anchor recipe.

The result residue lists are in chain order; AA1 lookup is the caller's job.

Implementation notes
--------------------
- Uses legacy ANARCI (`pip install anarci`, package import `anarci`) via
  `run_anarci(seq_list, scheme="chothia", allow={"H"})`. Returns a 4-tuple
  `(input_seqs, numbering, alignment_details, hit_tables)` where
  `numbering[0][0]` is the first VHH domain as `(numbering_tuples, query_start,
  query_end)` with numbering_tuples = list of `((pos, icode), aa)`.

  Anarcii v2.x's `Anarcii.to_scheme()` raises if called before `number()`
  and its mutate-after-number behavior is finicky; the legacy ANARCI path
  is more reliable and accepts `scheme=` directly.

- `query_start` / `query_end` give the seq indices of the V-domain envelope.
  Residues outside the envelope (e.g., constant-region overhang) are excluded
  from CDR assignment.
- Numbering entries with `aa == "-"` are gap positions in the canonical
  scheme; they don't consume a seq position and are skipped.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

from Bio.PDB import PDBParser
from Bio.PDB.Residue import Residue

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
       "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
       "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
       "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}

# Chothia CDR position windows (canonical; matches GT-side CDR_WINDOWS used
# by run_caar_epif1_array.py + run_scrmsd_array.py)
CHOTHIA_CDR_BOUNDS = {"H1": (26, 32), "H2": (52, 56), "H3": (95, 102)}


def _extract_heavy_chain(structure, vhh_chain_hint: Optional[str] = None):
    """Return (chain_obj, residue_list, seq_str) for the VHH heavy chain.

    Resolution order:
      1. If vhh_chain_hint is given AND that chain has ≥ 100 standard
         residues, use it.
      2. Otherwise, first chain with 100-160 standard residues.
    Returns (None, [], "") on failure.
    """
    # Build per-chain residue caches
    chain_residues_map = {}
    for chain in structure.get_chains():
        residues = [r for r in chain.get_residues()
                    if r.id[0] == " " and r.get_resname() in AA3]
        chain_residues_map[chain.id] = (chain, residues)

    chain, residues = None, []
    if vhh_chain_hint is not None and vhh_chain_hint in chain_residues_map:
        c, r = chain_residues_map[vhh_chain_hint]
        if 100 <= len(r) <= 160:
            chain, residues = c, r
    if chain is None:
        for cid, (c, r) in chain_residues_map.items():
            if 100 <= len(r) <= 160:
                chain, residues = c, r
                break
    if chain is None:
        return None, [], ""
    seq = "".join(AA3[r.get_resname()] for r in residues)
    return chain, residues, seq


def slice_cdrs_anarci(pdb_path: str,
                      vhh_chain_hint: Optional[str] = None,
                      verbose: bool = False) -> Optional[dict]:
    """Slice CDR-H1/H2/H3 residues using legacy ANARCI Chothia numbering.

    Returns {"H1": [Residue, ...], "H2": [...], "H3": [...]} (lists in chain
    order) on success, None on any failure (parse error, no heavy chain,
    ANARCI error, etc.).
    """
    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("x", pdb_path)
    except Exception as e:
        if verbose:
            print(f"  parse_fail: {e}", file=sys.stderr)
        return None
    chain, residues, seq = _extract_heavy_chain(struct, vhh_chain_hint)
    if chain is None or len(seq) < 100:
        if verbose:
            print("  no_heavy_chain (or seq < 100 aa)", file=sys.stderr)
        return None

    try:
        from anarci import run_anarci
        result = run_anarci([("query", seq)], scheme="chothia",
                            allow={"H"}, ncpu=1)
    except Exception as e:
        if verbose:
            print(f"  anarci_run_fail: {e}", file=sys.stderr)
        return None
    if not result or len(result) < 2:
        return None
    numbering_list = result[1]
    if not numbering_list or not numbering_list[0]:
        if verbose:
            print(f"  anarci_no_domain (chain may not be VHH)", file=sys.stderr)
        return None

    # Take the first (and usually only) heavy domain. ANARCI returns it as
    # (numbering_tuples, query_start, query_end) where query_start/end are
    # 0-indexed seq positions of the V-domain envelope.
    domain = numbering_list[0][0]
    if not isinstance(domain, tuple) or len(domain) != 3:
        if verbose:
            print(f"  anarci_unexpected_domain_shape: {type(domain)}",
                  file=sys.stderr)
        return None
    numbering, query_start, query_end = domain

    # Build map: seq_index → chothia_pos. seq_index = position in the input
    # `seq` string (which is also chain-order index in `residues`). ANARCI
    # numbering starts at query_start of the V-domain envelope.
    # Gap entries (aa == "-") in the canonical scheme don't consume a seq
    # position; skip them.
    seq_to_chothia = {}
    seq_idx = query_start
    for (pos_num, icode), aa in numbering:
        if aa == "-":
            continue
        if seq_idx > query_end:
            break
        if seq_idx < len(seq) and seq[seq_idx] == aa:
            seq_to_chothia[seq_idx] = pos_num
        seq_idx += 1

    cdrs = {"H1": [], "H2": [], "H3": []}
    for i, res in enumerate(residues):
        cpos = seq_to_chothia.get(i)
        if cpos is None:
            continue
        for cdr, (lo, hi) in CHOTHIA_CDR_BOUNDS.items():
            if lo <= cpos <= hi:
                cdrs[cdr].append(res)
                break
    return cdrs


def slice_cdrs_cys_anchor(pdb_path: str,
                          vhh_chain_hint: Optional[str] = None,
                          verbose: bool = False) -> Optional[dict]:
    """Fallback: CDR slicing via conserved cysteines + canonical motifs.

    H1: between the first conserved Cys (~chain-order pos 22) and the FR2
        Trp motif (`W[FY][RQ]Q` typical for VHH FR2).
    H2: 14 residues after FR2 Trp + 6 residues long (rough heuristic).
    H3: between the LAST Cys (end of FR3) and the FR4 `WG[A-Z]G` motif.

    Less precise than ANARCII but useful when ANARCII fails. Tested only
    on VHH; mAbs not supported.
    """
    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("x", pdb_path)
    except Exception as e:
        if verbose:
            print(f"  parse_fail: {e}", file=sys.stderr)
        return None
    chain, residues, seq = _extract_heavy_chain(struct, vhh_chain_hint)
    if chain is None or len(seq) < 100:
        return None

    cys_idxs = [i for i, aa in enumerate(seq) if aa == "C"]
    if len(cys_idxs) < 2:
        if verbose:
            print(f"  cys_count<2 in seq", file=sys.stderr)
        return None
    cys2 = cys_idxs[-1]  # last Cys = end of FR3 in VHH

    # FR4 motif search
    m_fr4 = re.search(r"WG[A-Z]G", seq[cys2:])
    if not m_fr4:
        if verbose:
            print(f"  no_FR4_motif after Cys2 at idx {cys2}", file=sys.stderr)
        return None
    h3_start = cys2 + 1
    h3_end = cys2 + m_fr4.start()
    cdrs = {"H1": [], "H2": [], "H3": residues[h3_start:h3_end]}

    # H1: heuristic — 3 residues after Cys1 to FR2 Trp.
    cys1 = cys_idxs[0]
    h1_start = cys1 + 3
    fr2 = re.search(r"W[FY][RQ]Q", seq[h1_start:h1_start + 25])
    if fr2:
        h1_end = h1_start + fr2.start()
    else:
        h1_end = h1_start + 7
    cdrs["H1"] = residues[h1_start:h1_end]

    # H2: 14 residues after FR2 Trp end + ~6 residues long
    fr2_search = re.search(r"W[FY][RQ]Q", seq)
    if fr2_search:
        h2_start = fr2_search.end() + 14
        h2_end = h2_start + 6
        if h2_end <= len(residues):
            cdrs["H2"] = residues[h2_start:h2_end]

    return cdrs


def slice_cdrs(pdb_path: str,
               vhh_chain_hint: Optional[str] = None,
               verbose: bool = False) -> Optional[dict]:
    """Slice CDR-H1/H2/H3 from a VHH design PDB.

    Tries ANARCII (Chothia scheme) first; falls back to Cys-anchor recipe
    on failure. Returns {"H1": [...], "H2": [...], "H3": [...]} or None.
    """
    cdrs = slice_cdrs_anarci(pdb_path, vhh_chain_hint, verbose)
    if cdrs is None or not (cdrs["H1"] and cdrs["H2"] and cdrs["H3"]):
        if verbose:
            print("  ANARCII failed or returned empty CDR → Cys-anchor fallback",
                  file=sys.stderr)
        cdrs = slice_cdrs_cys_anchor(pdb_path, vhh_chain_hint, verbose)
    return cdrs


def cdrs_as_aa1(cdrs: dict) -> dict:
    """Convenience: convert {cdr: [Residue, ...]} → {cdr: 'AA1AA1...'}."""
    if cdrs is None:
        return None
    return {
        cdr: "".join(AA3.get(r.get_resname(), "X") for r in cdrs[cdr])
        for cdr in ("H1", "H2", "H3")
    }


def slice_cdrs_with_chothia(pdb_path: str,
                            vhh_chain_hint: Optional[str] = None,
                            verbose: bool = False) -> Optional[dict]:
    """Like slice_cdrs but returns {cdr: [(chothia_pos, Residue), ...]}.

    Each tuple gives the Chothia position number that ANARCI assigned to that
    residue. This lets callers (e.g., the CAAR dispatcher) align residues
    across GT and design PDBs by Chothia position number — the ChimeraBench
    convention — rather than by raw PDB resseq.

    Returns None if ANARCI fails (no Cys-anchor fallback, since the
    Cys-anchor recipe doesn't assign Chothia positions).
    """
    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("x", pdb_path)
    except Exception as e:
        if verbose:
            print(f"  parse_fail: {e}", file=sys.stderr)
        return None
    chain, residues, seq = _extract_heavy_chain(struct, vhh_chain_hint)
    if chain is None or len(seq) < 100:
        return None

    try:
        from anarci import run_anarci
        result = run_anarci([("query", seq)], scheme="chothia",
                            allow={"H"}, ncpu=1)
    except Exception as e:
        if verbose:
            print(f"  anarci_run_fail: {e}", file=sys.stderr)
        return None
    if not result or len(result) < 2:
        return None
    numbering_list = result[1]
    if not numbering_list or not numbering_list[0]:
        return None
    domain = numbering_list[0][0]
    if not isinstance(domain, tuple) or len(domain) != 3:
        return None
    numbering, query_start, query_end = domain

    seq_to_chothia = {}
    seq_idx = query_start
    for (pos_num, icode), aa in numbering:
        if aa == "-":
            continue
        if seq_idx > query_end:
            break
        if seq_idx < len(seq) and seq[seq_idx] == aa:
            seq_to_chothia[seq_idx] = pos_num
        seq_idx += 1

    cdrs = {"H1": [], "H2": [], "H3": []}
    for i, res in enumerate(residues):
        cpos = seq_to_chothia.get(i)
        if cpos is None:
            continue
        for cdr, (lo, hi) in CHOTHIA_CDR_BOUNDS.items():
            if lo <= cpos <= hi:
                cdrs[cdr].append((cpos, res))
                break
    return cdrs


def cdrs_chothia_dict(cdrs_with_chothia: dict) -> dict:
    """Convenience: {cdr: [(pos, Residue), ...]} → {cdr: {pos: aa1}}."""
    if cdrs_with_chothia is None:
        return None
    return {
        cdr: {pos: AA3.get(r.get_resname(), "X")
              for pos, r in cdrs_with_chothia[cdr]}
        for cdr in ("H1", "H2", "H3")
    }
