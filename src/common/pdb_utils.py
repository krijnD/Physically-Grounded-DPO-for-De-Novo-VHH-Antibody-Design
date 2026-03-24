"""Utilities for loading and working with PDB structural data."""

import warnings

from Bio import BiopythonWarning
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure


def load_structure(pdb_path: str, structure_id: str = "candidate") -> Structure:
    """Parse a PDB file into a Biopython SMCRA Structure object.

    Args:
        pdb_path: Path to the .pdb file.
        structure_id: Label for the structure (used internally by Biopython).

    Returns:
        Biopython Structure object for downstream spatial analysis.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiopythonWarning)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(structure_id, pdb_path)
    return structure
