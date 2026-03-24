"""Localized Spatial Aggregation Propensity (SAP) calculator.

Computes a SAP-proxy score within a spherical radius around a target
residue by combining:
  1. Shrake-Rupley SASA (solvent-accessible surface area)
  2. K-D tree NeighborSearch for spatial boundary
  3. Hydrophobicity scale weighting

A high positive score indicates exposed hydrophobic surface (aggregation
risk). A negative score indicates the region is shielded by polar/charged
residues (conformational rescue by CDR loops).
"""

import logging

from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Selection import unfold_entities
from Bio.PDB.Structure import Structure

from src.common.config import Config

logger = logging.getLogger(__name__)


def calculate_localized_sap(
    structure: Structure,
    target_res_id: int,
    radius: float = Config.SAP_RADIUS,
    chain_id: str = "A",
) -> float:
    """Compute a localized SAP score around a specific framework residue.

    The score quantifies whether the target residue's hydrophobic
    environment is shielded (negative/low score) or exposed (high score).

    Args:
        structure: Biopython Structure object (already parsed from PDB).
        target_res_id: Kabat residue number to evaluate (e.g. 45 for L45).
        radius: Spherical search radius in Angstroms around the target CA.
        chain_id: Chain identifier in the PDB (default "A" for single-domain).

    Returns:
        Localized SAP score. Values above Config.SAP_SAFETY_THRESHOLD
        indicate unshielded hydrophobic exposure.
    """
    # 1. Compute SASA for all residues using Shrake-Rupley rolling ball
    sr = ShrakeRupley(probe_radius=1.40, n_points=100)
    sr.compute(structure, level="R")

    # 2. Locate the target residue
    model = structure[0]
    try:
        chain = model[chain_id]
    except KeyError:
        # Fall back to first chain if chain_id not found
        chain = next(model.get_chains())

    try:
        target_res = chain[(" ", target_res_id, " ")]
    except KeyError:
        logger.warning(
            "Residue %d not found in chain %s — returning failsafe score.",
            target_res_id, chain_id,
        )
        return 999.0

    # 3. Build K-D tree and find neighbors within radius
    atom_list = unfold_entities(structure, "A")
    ns = NeighborSearch(atom_list)

    try:
        target_ca = target_res["CA"]
    except KeyError:
        logger.warning("No CA atom for residue %d — returning failsafe.", target_res_id)
        return 999.0

    neighbor_atoms = ns.search(target_ca.coord, radius)
    neighbor_residues = {atom.get_parent() for atom in neighbor_atoms}

    # 4. Compute the localized SAP score
    # Hydrophobic residues with high SASA inflate the score (aggregation risk).
    # Polar residues (negative scale values) reduce the score (shielding).
    localized_sap = 0.0
    for res in neighbor_residues:
        res_name = res.get_resname()
        hydrophobicity = Config.HYDROPHOBICITY_SCALE.get(res_name)
        if hydrophobicity is None:
            continue
        sasa = getattr(res, "sasa", 0.0)
        localized_sap += hydrophobicity * sasa

    logger.debug(
        "SAP for residue %d: %.2f (radius=%.1f, neighbors=%d)",
        target_res_id, localized_sap, radius, len(neighbor_residues),
    )

    return localized_sap
