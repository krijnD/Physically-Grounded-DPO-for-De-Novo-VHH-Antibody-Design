"""Structure-based paratope detection using Biopython NeighborSearch.

Identifies VHH nanobody residues at the antigen interface by measuring
the physical distance between heavy atoms.  A residue is classified as
paratope if ANY of its heavy atoms falls within a distance cutoff of
ANY antigen heavy atom.

Follows the same NeighborSearch K-D tree pattern used in
``biology_judge/sap_calculator.py``.

References:
- Structural paratope definition (5.0 Angstrom cutoff):
  Leem et al. (2022), contrastive Ab-Ag specificity prediction
- VHH framework paratope contributions:
  Mitchell & Colwell (2018), VHH vs VH binding site comparison
"""

import logging

from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Structure import Structure

from src.common.config import Config

logger = logging.getLogger(__name__)


def _parse_antigen_chain_ids(antigen_chain_ids: str) -> set[str]:
    """Normalize antigen chain ID string to a set of single letters.

    Handles the formats used throughout the pipeline:
    - ``"A"`` → ``{"A"}``
    - ``"A|C|B"`` → ``{"A", "C", "B"}``
    - ``"ACB"`` → ``{"A", "C", "B"}``

    Mirrors the cleanup logic in ``pipeline.py`` (line 151).
    """
    cleaned = antigen_chain_ids.replace(" ", "").replace("|", "")
    return set(cleaned)


def detect_paratope_residues(
    structure: Structure,
    nanobody_chain_id: str,
    antigen_chain_ids: str,
    distance_cutoff: float = Config.PARATOPE_DISTANCE_CUTOFF,
) -> set[int]:
    """Identify nanobody residues at the paratope interface.

    Uses a K-D tree (Biopython ``NeighborSearch``) to find all VHH
    residues that have any heavy atom within ``distance_cutoff``
    Angstroms of any antigen heavy atom.

    Args:
        structure: Biopython Structure (parsed complex PDB).
        nanobody_chain_id: Chain letter of the nanobody (e.g. ``"H"``).
        antigen_chain_ids: Chain letter(s) of the antigen.  Supports
            ``"A"``, ``"A|C|B"``, or ``"ACB"`` formats.
        distance_cutoff: Distance threshold in Angstroms (default 5.0).

    Returns:
        Set of PDB residue sequence numbers (``resseq`` integers) on
        the nanobody chain that are within the cutoff of the antigen.
        Empty set if chains are not found.
    """
    model = structure[0]
    ag_chains = _parse_antigen_chain_ids(antigen_chain_ids)

    # Collect all heavy atoms from antigen chain(s).
    antigen_atoms = []
    for chain_letter in ag_chains:
        try:
            chain = model[chain_letter]
        except KeyError:
            logger.warning(
                "Antigen chain %s not found in structure", chain_letter
            )
            continue
        for atom in chain.get_atoms():
            if atom.element != "H":
                antigen_atoms.append(atom)

    if not antigen_atoms:
        logger.warning(
            "No antigen heavy atoms found for chains %s", antigen_chain_ids
        )
        return set()

    # Build K-D tree on antigen atoms.
    ns = NeighborSearch(antigen_atoms)

    # For each nanobody heavy atom, check proximity to antigen.
    try:
        nb_chain = model[nanobody_chain_id]
    except KeyError:
        logger.warning(
            "Nanobody chain %s not found in structure", nanobody_chain_id
        )
        return set()

    paratope_resseqs: set[int] = set()

    for residue in nb_chain.get_residues():
        # Skip hetero atoms (water, ligands).
        hetflag = residue.id[0]
        if hetflag.strip():
            continue

        for atom in residue.get_atoms():
            if atom.element == "H":
                continue
            neighbors = ns.search(atom.coord, distance_cutoff)
            if neighbors:
                paratope_resseqs.add(residue.id[1])
                break  # One contact is enough to classify as paratope.

    logger.info(
        "Paratope detection: %d residues within %.1f A of antigen (chains %s)",
        len(paratope_resseqs),
        distance_cutoff,
        antigen_chain_ids,
    )

    return paratope_resseqs
