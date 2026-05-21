"""PDB utilities for biophysics scoring on DiffAb-generated structures.

Provides ``extract_vhh_monomer`` that converts a multi-chain complex PDB
(VHH + antigen) into a single-chain VHH-only PDB suitable for
``theraprofnano``'s metric functions.

Two preprocessing details matter for downstream TNP compatibility:

1. ``CreateAnnotation`` in TNP's Hydrophobicity_and_Charge_Assigner
   hardcodes the chain label to "H" (``nb_structure = {'H': PDBchain(...,'H')}``),
   so the extracted monomer must use chain id "H" regardless of the chain
   id the source PDB used.
2. TNP's CLI normally strips hydrogens after NanoBodyBuilder2 folding via
   its ``pdb_remove_hydrogens.py`` script. The metric functions therefore
   assume a hydrogen-free structure with no HETATM, and a single altloc
   per atom — we strip those here.
"""

import logging
import warnings
from pathlib import Path

from Bio import BiopythonWarning
from Bio.PDB import PDBIO, PDBParser, Select

logger = logging.getLogger(__name__)


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
