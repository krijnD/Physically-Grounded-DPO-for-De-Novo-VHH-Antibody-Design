"""PDB utilities for biophysics scoring on DiffAb-generated structures.

Two helpers compose into the preprocessing that ``tnp_direct.score_pdb``
runs before calling ``theraprofnano``:

  * ``extract_vhh_monomer`` — single chain, ATOM only, no hydrogens,
    chain relabeled to "H".
  * ``renumber_to_imgt`` — rewrite ATOM residue numbers / insertion
    codes to the IMGT scheme via ``abnumber`` (drops residues outside
    the FV region).

Three preprocessing details matter for downstream TNP compatibility:

1. ``CreateAnnotation`` in TNP's Hydrophobicity_and_Charge_Assigner
   hardcodes the chain label to "H" (``nb_structure = {'H': PDBchain(...,'H')}``),
   so the extracted monomer must use chain id "H".
2. TNP's CLI normally strips hydrogens after NanoBodyBuilder2 folding via
   its ``pdb_remove_hydrogens.py`` script. The metric functions therefore
   assume a hydrogen-free structure with no HETATM, and a single altloc
   per atom — we strip those here.
3. ``parse_nb`` (compactness) and ``is_CDR`` (charge/hydrophobicity)
   both look up residues by their PDB residue number under the IMGT
   scheme. DiffAb-generated PDBs carry the source crystal's original
   numbering, which doesn't line up with IMGT — so we renumber.
"""

import logging
import warnings
from pathlib import Path

from Bio import BiopythonWarning
from Bio.PDB import PDBParser

logger = logging.getLogger(__name__)

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def extract_vhh_monomer(
    complex_pdb_path: str | Path,
    source_chain_id: str,
    output_path: str | Path,
    target_chain_id: str = "H",
) -> Path:
    """Write a single-chain VHH PDB extracted from a complex PDB.

    Implemented as a column-precise text rewrite — Biopython's
    PDBIO+Select doesn't play well with in-place chain renames, so we
    avoid the tree entirely. Reads the source PDB line-by-line, keeps
    only ATOM records whose chain id matches ``source_chain_id``,
    drops hydrogens / HETATM / non-A altlocs, and rewrites the chain id
    column to ``target_chain_id`` (defaults to "H" because TNP's
    ``CreateAnnotation`` hardcodes "H").

    Args:
        complex_pdb_path: Source PDB (typically a DiffAb-generated
            VHH+antigen complex, or a crystal complex).
        source_chain_id: Chain id of the VHH in the source PDB.
        output_path: Destination PDB path. Parent dir is created.
        target_chain_id: Chain id to assign in the output PDB.

    Returns:
        ``output_path`` as a Path.

    Raises:
        ValueError: if no ATOM records for ``source_chain_id`` are found.
    """
    complex_pdb_path = Path(complex_pdb_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(target_chain_id) != 1:
        raise ValueError(f"target_chain_id must be a single char, got {target_chain_id!r}")

    n_written = 0
    with open(complex_pdb_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            if not line.startswith("ATOM"):
                # Drop everything that isn't ATOM: HEADER/TITLE we don't
                # need, HETATM (waters/ligands) we never want, TER lines
                # that reference dropped residues, MODEL/ENDMDL boundaries
                # that confuse a single-chain output.
                continue
            if len(line) < 22:
                continue
            chain_col = line[21]
            if chain_col != source_chain_id:
                continue
            # Drop hydrogens. PDB element column is cols 77-78 (0-idx 76:78).
            # If absent or whitespace, infer from atom-name col 13-14
            # (0-idx 12:14): hydrogens start with " H" or "1H"/"2H".
            element = line[76:78].strip() if len(line) >= 78 else ""
            atom_name = line[12:16].strip()
            if element == "H" or (not element and atom_name.startswith("H")):
                continue
            # Drop non-blank, non-A altlocs (col 17, 0-idx 16).
            altloc = line[16] if len(line) > 16 else " "
            if altloc not in (" ", "A"):
                continue
            # Rewrite chain id column to target.
            fout.write(line[:21] + target_chain_id + line[22:])
            n_written += 1
        fout.write("END\n")

    if n_written == 0:
        raise ValueError(
            f"No ATOM records for chain {source_chain_id!r} in {complex_pdb_path}. "
            f"Check that the chain id is correct."
        )
    return output_path


def renumber_to_imgt(monomer_pdb: str | Path, output_pdb: str | Path) -> Path:
    """Rewrite ``monomer_pdb`` with IMGT residue numbers via ``abnumber``.

    Residues outside the antibody FV region (signal peptide, C-terminal
    tags) are dropped because TNP's metric functions iterate over all
    chain residues and would otherwise misclassify them as framework.

    The output keeps only ATOM records for residues that mapped to an
    IMGT position; all other lines (HEADER, TER, END, etc.) are passed
    through unchanged so the file remains a valid PDB.
    """
    from abnumber import Chain as AbChain
    from abnumber.exceptions import ChainParseError

    monomer_pdb = Path(monomer_pdb)
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Extract ordered AA sequence from the monomer's ATOM records ──
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiopythonWarning)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("monomer", str(monomer_pdb))

    chain = next(iter(next(structure.get_models()).get_chains()))
    ordered_residues = [
        res for res in chain
        if res.id[0] == " " and res.get_resname() in _THREE_TO_ONE
    ]
    if not ordered_residues:
        raise ValueError(f"No standard residues found in {monomer_pdb}")

    seq = "".join(_THREE_TO_ONE[res.get_resname()] for res in ordered_residues)

    # ── 2. Get IMGT numbering for the sequence ──
    try:
        ab = AbChain(seq, scheme="imgt", assign_germline=False)
    except ChainParseError as e:
        raise ValueError(f"abnumber failed to parse {monomer_pdb}: {e}") from e

    fv = ab.seq
    fv_start = seq.find(fv)
    if fv_start < 0:
        raise ValueError(
            f"abnumber FV region ({fv[:30]}…) not found in input sequence "
            f"({seq[:30]}…) for {monomer_pdb}"
        )

    positions = list(ab.positions.keys())
    if len(positions) != len(fv):
        raise ValueError(
            f"abnumber returned {len(positions)} positions for an "
            f"FV of length {len(fv)} in {monomer_pdb}"
        )

    # ── 3. Build (orig_resseq, orig_icode) → (imgt_num, imgt_icode) map ──
    remap: dict[tuple[int, str], tuple[int, str]] = {}
    for i, pos in enumerate(positions):
        res = ordered_residues[fv_start + i]
        orig_resseq = res.id[1]
        orig_icode = res.id[2] if res.id[2] != " " else " "
        new_resseq = pos.number
        new_icode = pos.letter if pos.letter else " "
        remap[(orig_resseq, orig_icode)] = (new_resseq, new_icode)

    # ── 4. Rewrite the PDB file, column-precise on ATOM records ──
    # PDB ATOM/HETATM record layout (1-indexed): cols 23-26 = resseq (i4),
    # col 27 = icode (a1). In Python 0-indexed slicing that's [22:26] and
    # [26:27].
    with open(monomer_pdb) as fin, open(output_pdb, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    resseq = int(line[22:26])
                except ValueError:
                    continue
                icode = line[26] if len(line) > 26 else " "
                new = remap.get((resseq, icode))
                if new is None:
                    continue  # outside FV — drop
                new_resseq, new_icode = new
                fout.write(
                    line[:22]
                    + f"{new_resseq:>4d}"
                    + new_icode
                    + line[27:]
                )
            elif line.startswith("TER"):
                # Drop TER lines — they reference residues that may have
                # been dropped or renumbered, and TNP doesn't need them.
                continue
            else:
                fout.write(line)

    return output_pdb


def pack_sidechains(input_pdb: str | Path, output_pdb: str | Path) -> Path:
    """Place / repack side chains via PyRosetta's fixed-backbone packer.

    DiffAb writes backbone atoms only (N, CA, C, O, CB) for the residues
    it regenerated (CDRs in the multi-CDR π_ref scope). TNP's surface
    metrics — especially PSH — read solvent-accessible side-chain
    surface, so a backbone-only CDR inflates PSH by ~+50 REU vs. a
    structure with full atoms (validated on the 2026-05-21 seed42_dedup
    canary).

    This wrapper:
      1. Loads the input PDB into a Pose; PyRosetta autoplaces missing
         heavy atoms in idealized positions.
      2. Runs ``PackRotamersMover`` with ``RestrictToRepacking`` on
         every residue (single-chain monomer — no antigen to preserve).
         Framework residues with already-good crystal rotamers usually
         stay put; CDR residues get sampled into the Dunbrack 2010
         library and pick a low-energy rotamer.
      3. Dumps the packed pose, preserving residue numbering /
         insertion codes (PyRosetta's PDBInfo carries them through).

    Reuses ``physics_judge.rosetta_scorer._ensure_init`` so PyRosetta
    initializes exactly once even when both judges run in the same
    process.

    Args:
        input_pdb: Path to the IMGT-numbered VHH monomer with possibly
            incomplete CDR side chains.
        output_pdb: Destination PDB.

    Returns:
        ``output_pdb`` as a Path.
    """
    from src.physics_judge.rosetta_scorer import _ensure_init, repack_complex

    input_pdb = Path(input_pdb)
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    _ensure_init()
    import pyrosetta

    pose = pyrosetta.pose_from_pdb(str(input_pdb))
    repack_complex(pose)  # logs pre→post score delta
    pose.dump_pdb(str(output_pdb))
    return output_pdb
