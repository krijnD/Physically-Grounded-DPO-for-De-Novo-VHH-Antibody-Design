"""PyRosetta wrapper for Physics Judge energy computations.

Encapsulates all PyRosetta operations behind a clean interface so that
no other module needs to import PyRosetta directly.  Provides:

  - CDR loop refinement via constrained FastRelax on CDR H1/H2/H3
  - E_Rep extraction (fa_rep at the interface) for steric clash detection
  - delta_G_bind via InterfaceAnalyzerMover for binding affinity

Uses lazy initialization: PyRosetta is initialized exactly once on first
call to any scoring function.

References:
  - Zhou et al. (NeurIPS 2024), AbDPO — energy decomposition methodology
  - Alford et al. (2017), ref2015 score function
"""

import logging
from dataclasses import dataclass
from typing import Optional

from src.common.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy PyRosetta initialization
# ---------------------------------------------------------------------------
_INITIALIZED = False


def _ensure_init(flags: str = Config.PYROSETTA_FLAGS) -> None:
    """Initialize PyRosetta exactly once per process."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    import pyrosetta  # noqa: delayed import

    pyrosetta.init(flags)
    _INITIALIZED = True
    logger.info("PyRosetta initialized with flags: %s", flags)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------
@dataclass
class PhysicsScores:
    """Container for Physics Judge energy metrics."""

    e_rep: float
    delta_g: Optional[float] = None  # None when fast-failed on e_rep


# ---------------------------------------------------------------------------
# Pose loading
# ---------------------------------------------------------------------------
def load_complex_pose(complex_pdb_path: str):
    """Load a PDB file into a PyRosetta Pose.

    Args:
        complex_pdb_path: Path to a PDB containing the nanobody–antigen complex.

    Returns:
        A ``pyrosetta.Pose`` object.
    """
    _ensure_init()
    import pyrosetta

    pose = pyrosetta.pose_from_pdb(complex_pdb_path)
    logger.debug(
        "Loaded complex from %s — %d residues, %d chains",
        complex_pdb_path,
        pose.total_residue(),
        pose.num_chains(),
    )
    return pose


# ---------------------------------------------------------------------------
# CDR loop refinement (CCD)
# ---------------------------------------------------------------------------
def _find_pose_loop_residues(
    pose,
    nanobody_chain_id: str,
    cdr_ranges: list[tuple[int, int]] = Config.VHH_CDR_RANGES,
) -> list[tuple[int, int]]:
    """Map Kabat CDR ranges to Pose residue indices.

    Iterates the nanobody chain in the Pose and maps PDB residue numbers
    to Pose-internal (1-based) residue indices for each CDR range.

    Returns:
        List of (start_pose_idx, end_pose_idx) tuples.  Empty if the
        chain cannot be found or no CDR residues are identified.
    """
    import pyrosetta  # noqa: delayed import

    pdb_info = pose.pdb_info()
    if pdb_info is None:
        logger.warning("Pose has no PDB info — cannot identify CDR loops.")
        return []

    loops: list[tuple[int, int]] = []
    for cdr_start, cdr_end in cdr_ranges:
        start_idx = None
        end_idx = None
        for res_num in range(cdr_start, cdr_end + 1):
            pose_idx = pdb_info.pdb2pose(nanobody_chain_id, res_num)
            if pose_idx == 0:
                # residue not found in pose (numbering mismatch)
                continue
            if start_idx is None:
                start_idx = pose_idx
            end_idx = pose_idx

        if start_idx is not None and end_idx is not None:
            loops.append((start_idx, end_idx))
        else:
            logger.debug(
                "CDR range %d–%d not found on chain %s",
                cdr_start,
                cdr_end,
                nanobody_chain_id,
            )

    return loops


def refine_cdr_loops(
    pose,
    nanobody_chain_id: str,
    cdr_ranges: list[tuple[int, int]] = Config.VHH_CDR_RANGES,
    outer_cycles: int = Config.CCD_OUTER_CYCLES,
    max_inner_cycles: int = Config.CCD_MAX_INNER_CYCLES,
) -> None:
    """Apply constrained FastRelax to CDR loops in place.

    Resolves steric clashes from crystal packing or structure prediction
    artifacts that would otherwise inflate fa_rep scores. Uses FastRelax
    with a MoveMap restricted to CDR backbone + side-chain DOFs plus
    side-chain repacking of interface neighbors.

    Falls back to CCD loop refinement if FastRelax is unavailable (should
    not happen in practice with any recent PyRosetta build).

    Args:
        pose: PyRosetta Pose (modified in place).
        nanobody_chain_id: Chain letter of the nanobody (e.g. ``"H"``).
        cdr_ranges: Kabat-numbered CDR boundaries.
        outer_cycles: Not used by FastRelax (kept for API compat).
        max_inner_cycles: Not used by FastRelax (kept for API compat).
    """
    import pyrosetta
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.protocols.relax import FastRelax

    loop_ranges = _find_pose_loop_residues(pose, nanobody_chain_id, cdr_ranges)
    if not loop_ranges:
        logger.warning("No CDR loops identified — skipping refinement.")
        return

    # Build a MoveMap: allow backbone + chi moves ONLY on CDR residues,
    # plus chi (side-chain) repacking on a ±2 residue shell around each CDR.
    mm = MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)

    cdr_residues: set[int] = set()
    for start, end in loop_ranges:
        for i in range(start, end + 1):
            mm.set_bb(i, True)
            mm.set_chi(i, True)
            cdr_residues.add(i)

    # Allow side-chain repacking of neighbors (±2 shell) to relieve clashes
    for start, end in loop_ranges:
        for i in range(max(1, start - 2), min(pose.total_residue(), end + 2) + 1):
            if i not in cdr_residues:
                mm.set_chi(i, True)

    sfxn = pyrosetta.create_score_function("ref2015")
    relax = FastRelax()
    relax.set_scorefxn(sfxn)
    relax.set_movemap(mm)
    # Use a single round for speed — enough to resolve major clashes
    relax.max_iter(200)

    relax.apply(pose)

    logger.debug(
        "FastRelax refinement applied to %d CDR loop(s) (%d residues) on chain %s",
        len(loop_ranges),
        len(cdr_residues),
        nanobody_chain_id,
    )


# ---------------------------------------------------------------------------
# Energy computation
# ---------------------------------------------------------------------------
def compute_e_rep(pose, interface: str = Config.ROSETTA_INTERFACE) -> float:
    """Compute the total fa_rep (steric repulsion) energy at the interface.

    Uses the ref2015 score function.  Sums per-residue fa_rep contributions
    across all residues at the nanobody–antigen interface.

    Args:
        pose: Scored or unscored PyRosetta Pose (will be scored in place).
        interface: Interface definition string (e.g. ``"H_A"``).

    Returns:
        Total fa_rep energy in REU at the interface.
    """
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreType
    from pyrosetta.rosetta.core.select.residue_selector import (
        InterGroupInterfaceByVectorSelector,
        ChainSelector,
    )

    # Score the pose with ref2015
    sfxn = pyrosetta.create_score_function("ref2015")
    sfxn(pose)

    # Parse interface string "H_A" → nanobody chains "H", antigen chains "A"
    parts = interface.split("_")
    nb_chains = parts[0]
    ag_chains = parts[1] if len(parts) > 1 else ""

    nb_selector = ChainSelector(nb_chains)
    ag_selector = ChainSelector(ag_chains)

    interface_selector = InterGroupInterfaceByVectorSelector(
        nb_selector, ag_selector
    )
    interface_mask = interface_selector.apply(pose)

    # Sum fa_rep over interface residues
    energies = pose.energies()
    fa_rep_type = ScoreType.fa_rep
    total_fa_rep = 0.0
    for i in range(1, pose.total_residue() + 1):
        if interface_mask[i]:
            total_fa_rep += energies.residue_total_energies(i)[fa_rep_type]

    logger.debug("E_Rep at interface %s: %.3f REU", interface, total_fa_rep)
    return total_fa_rep


def compute_delta_g(pose, interface: str = Config.ROSETTA_INTERFACE) -> float:
    """Compute binding free energy (delta_G) via InterfaceAnalyzerMover.

    Separates the complex into unbound chains, repacks the exposed
    interface residues (``pack_separated=True``), and computes:

        delta_G = E_complex - (E_antibody + E_antigen)

    Args:
        pose: PyRosetta Pose of the complex (will be modified by the mover).
        interface: Interface definition string (e.g. ``"H_A"``).

    Returns:
        Binding free energy (dG_separated) in REU.  More negative = tighter.
    """
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    iam = InterfaceAnalyzerMover()
    iam.set_interface(interface)
    iam.set_pack_separated(True)
    iam.set_scorefunction(
        __import__("pyrosetta").create_score_function("ref2015")
    )
    iam.apply(pose)

    dg = iam.get_interface_dG()
    logger.debug("delta_G at interface %s: %.3f REU", interface, dg)
    return dg


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def score_complex(
    complex_pdb_path: str,
    nanobody_chain_id: str = "H",
    interface: str = Config.ROSETTA_INTERFACE,
    cdr_ranges: list[tuple[int, int]] = Config.VHH_CDR_RANGES,
    ccd_outer_cycles: int = Config.CCD_OUTER_CYCLES,
    ccd_max_inner_cycles: int = Config.CCD_MAX_INNER_CYCLES,
    e_rep_fast_fail: float = Config.E_REP_REJECT,
) -> PhysicsScores:
    """Score a nanobody–antigen complex for steric clashes and binding affinity.

    Pipeline:
      1. Load PDB → Pose
      2. FastRelax CDR loops (resolve steric clashes)
      3. Compute E_Rep (fast)
      4. If E_Rep ≤ threshold → compute delta_G (expensive)
         If E_Rep > threshold → skip delta_G (fast-fail)

    Args:
        complex_pdb_path: Path to the complex PDB file.
        nanobody_chain_id: Chain letter of the nanobody.
        interface: PyRosetta interface string (e.g. ``"H_A"``).
        cdr_ranges: Kabat-numbered CDR boundaries for CCD refinement.
        ccd_outer_cycles: CCD outer cycles (AbDPO default: 1).
        ccd_max_inner_cycles: CCD inner cycles (AbDPO default: 10).
        e_rep_fast_fail: E_Rep threshold for skipping delta_G computation.

    Returns:
        :class:`PhysicsScores` with ``e_rep`` always populated and
        ``delta_g`` populated only if E_Rep passes the fast-fail gate.
    """
    _ensure_init()

    pose = load_complex_pose(complex_pdb_path)

    # CCD refinement — skip gracefully on failure
    try:
        refine_cdr_loops(
            pose,
            nanobody_chain_id,
            cdr_ranges=cdr_ranges,
            outer_cycles=ccd_outer_cycles,
            max_inner_cycles=ccd_max_inner_cycles,
        )
    except Exception:
        logger.warning(
            "CCD refinement failed for %s — scoring without refinement.",
            complex_pdb_path,
            exc_info=True,
        )

    # Step 1: E_Rep (cheap — fast-fail gate)
    e_rep = compute_e_rep(pose, interface)

    if e_rep > e_rep_fast_fail:
        logger.info(
            "E_Rep %.3f > %.1f REU — fast-failing, skipping delta_G.",
            e_rep,
            e_rep_fast_fail,
        )
        return PhysicsScores(e_rep=e_rep, delta_g=None)

    # Step 2: delta_G (expensive — only if E_Rep passes)
    delta_g = compute_delta_g(pose, interface)

    return PhysicsScores(e_rep=e_rep, delta_g=delta_g)
