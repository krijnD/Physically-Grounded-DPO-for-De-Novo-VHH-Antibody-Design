"""Localized normalized Spatial Aggregation Propensity (SAP) calculator.

Implements a Chennamsetty-style normalized SAP localized to a target
residue's neighborhood:

    SAP_local(target) = (1 / N) · Σⱼ (SASA_j / SASA_max_j) × hydropathy_BM_j

where j ranges over residues with any atom within ``SAP_RADIUS`` of the
target residue's CA atom, ``SASA_j`` is the Shrake-Rupley solvent-
accessible surface area of residue j, ``SASA_max_j`` is the theoretical
max from Tien et al. (2013), and ``hydropathy_BM_j`` is the
Black & Mould (1991) normalized hydrophobicity.

Each per-neighbor contribution is bounded in [-1, +1] (fraction-exposed
∈ [0, 1] times hydropathy ∈ [-1, +1]), so the average is also bounded
in [-1, +1]. This makes the metric scale-invariant to neighborhood size
and lets us cite a literature-derived threshold rather than fitting to
the project's own data.

Sign convention:
  - Positive value → exposed-hydrophobic neighborhood (aggregation risk)
  - Negative value → polar-shielded neighborhood (compensated by CDR loops
    folding over the FR2 patch — the conformational rescue from sdAb B,
    Uto et al. 2025)
  - Magnitude near 0 → neutral / mixed neighborhood

References:
  - Chennamsetty, Voynov, Kayser, Helk, Trout (2009) PNAS — original SAP
  - Black & Mould (1991) Proteins — normalized hydrophobicity scale
  - Tien, Meyer, Sydykova, Spielman, Wilke (2013) PLOS ONE — max SASA values
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
    """Compute a normalized localized SAP score around a framework residue.

    Args:
        structure: Biopython Structure object (already parsed from PDB).
        target_res_id: Kabat residue number to evaluate (e.g. 45 for L45).
        radius: Spherical search radius in Angstroms around the target CA.
        chain_id: Chain identifier in the PDB (default "A" for single-domain).

    Returns:
        Mean SASA-weighted Black & Mould hydropathy of residues within
        ``radius`` of the target CA. Bounded in [-1, +1]. Values above
        ``Config.SAP_SAFETY_THRESHOLD`` indicate unshielded hydrophobic
        exposure. Returns the failsafe ``999.0`` when the target residue
        cannot be located (so the judge fails-safe rather than passing).
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

    # 4. Sum normalized contributions and average over neighbors.
    # Per-neighbor contribution = fraction_exposed × hydropathy_BM ∈ [-1, +1].
    total = 0.0
    n_counted = 0
    for res in neighbor_residues:
        res_name = res.get_resname()
        hydro = Config.BLACK_MOULD_HYDROPHOBICITY.get(res_name)
        max_sasa = Config.MAX_RESIDUE_SASA.get(res_name)
        if hydro is None or max_sasa is None:
            # Skip non-canonical residues / hetero-atoms (HOH, etc.)
            continue
        sasa = getattr(res, "sasa", 0.0)
        # Clamp fraction at 1.0 — Shrake-Rupley with n_points=100 can
        # occasionally exceed Tien's theoretical max for highly-exposed
        # surface residues due to sampling noise.
        fraction_exposed = min(sasa / max_sasa, 1.0)
        total += fraction_exposed * hydro
        n_counted += 1

    if n_counted == 0:
        logger.warning(
            "No canonical-residue neighbors of residue %d — returning 0.0.",
            target_res_id,
        )
        return 0.0

    sap = total / n_counted
    logger.debug(
        "SAP for residue %d: %.3f (radius=%.1f, neighbors=%d)",
        target_res_id, sap, radius, n_counted,
    )
    return sap
