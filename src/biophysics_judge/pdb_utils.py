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
from Bio.PDB import PDBIO, PDBParser, Select

logger = logging.getLogger(__name__)

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


class _VHHMonomerSelect(Select):
    """Biopython selector: keep one chain, ATOM records only,
    no hydrogens, single altloc."""

    def __init__(self, source_chain_id: str):
        self._source_chain_id = source_chain_id

    def accept_chain(self, chain) -> bool:
        return chain.id == self._source_chain_id

    def accept_residue(self, residue) -> bool:
        # hetflag is " " for ATOM records, "H_*" for HETATM, "W" for water
        return residue.id[0] == " "

    def accept_atom(self, atom) -> bool:
        if atom.element == "H":
            return False
        altloc = atom.get_altloc()
        return altloc in ("", "A")


def extract_vhh_monomer(
    complex_pdb_path: str | Path,
    source_chain_id: str,
    output_path: str | Path,
    target_chain_id: str = "H",
) -> Path:
    """Write a single-chain VHH PDB extracted from a complex PDB.

    Args:
        complex_pdb_path: Source PDB (typically a DiffAb-generated
            VHH+antigen complex, or a crystal complex).
        source_chain_id: Chain id of the VHH in the source PDB.
        output_path: Destination PDB path. Parent dir is created.
        target_chain_id: Chain id to assign in the output PDB. Defaults
            to "H" because TNP's ``CreateAnnotation`` hardcodes "H".

    Returns:
        ``output_path`` as a Path.
    """
    complex_pdb_path = Path(complex_pdb_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiopythonWarning)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("complex", str(complex_pdb_path))

    model = structure[0]
    available = [c.id for c in model.get_chains()]
    if source_chain_id not in available:
        raise ValueError(
            f"Chain {source_chain_id!r} not found in {complex_pdb_path}. "
            f"Available chains: {available}"
        )

    if source_chain_id != target_chain_id:
        # Rename the source chain to the target id. If a chain already
        # exists at the target id (e.g. an antigen named "H"), rename
        # it to a free letter so we don't collide before applying Select.
        existing_target = next(
            (c for c in model.get_chains() if c.id == target_chain_id), None,
        )
        if existing_target is not None and existing_target.id != source_chain_id:
            free = next(
                (
                    letter for letter in "ZYXWVUTSRQPONMLKJI"
                    if letter not in available
                ),
                None,
            )
            if free is None:
                raise ValueError(
                    f"Cannot relabel chain {target_chain_id!r} — all fallback "
                    f"letters in use ({available})."
                )
            existing_target.id = free
        model[source_chain_id].id = target_chain_id

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path), _VHHMonomerSelect(target_chain_id))
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

    chain = next(iter(structure[0].get_chains()))
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
