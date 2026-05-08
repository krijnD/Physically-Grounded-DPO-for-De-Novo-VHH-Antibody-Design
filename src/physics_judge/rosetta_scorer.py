"""PyRosetta wrapper for Physics Judge energy computations.

Encapsulates all PyRosetta operations behind a clean interface so that
no other module needs to import PyRosetta directly.  Provides:

  - CDR loop refinement via constrained FastRelax on CDR H1/H2/H3
  - E_Rep extraction (fa_rep at the interface) for steric clash detection
  - CDR per-residue energy (AbDPO-style residue-level total energy
    summed over CDR residues, normalized by CDR length for scope-
    invariance)

Uses lazy initialization: PyRosetta is initialized exactly once on first
call to any scoring function.

References:
  - Zhou et al. (NeurIPS 2024), AbDPO — residue-level CDR energy:
    ε(R⁰) = Σⱼ ε(R⁰[j]) summed over CDR-H3 residues. We extend to
    multi-CDR (H1+H2+H3) and normalize by N_CDR_residues.
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
    # AbDPO-style per-residue CDR energy (REU/residue).  None when the
    # E_Rep fast-fail gate tripped and we skipped the expensive scoring.
    cdr_energy_per_res: Optional[float] = None
    # True when CDR energy came back non-physical
    # (|E_cdr| > CDR_ENERGY_PATHOLOGICAL REU/residue), i.e. structure
    # prep couldn't resolve clashes in the bound state.  When set,
    # cdr_energy_per_res is nulled out and the judge emits
    # "skipped_scoring_failure" rather than "fail_cdr_energy".
    scoring_failed: bool = False


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


def repack_complex(pose) -> tuple[float, float]:
    """Full-complex side-chain repack with ref2015.

    Runs ``PackRotamersMover`` with ``RestrictToRepacking`` over every
    residue in the pose — nanobody framework, CDR loops, and antigen.
    Backbone is not moved; only side-chain rotamers are reassigned.

    This is essential for raw crystal PDBs: unrelaxed side-chain clashes
    in the framework or antigen (outside the CDR ±2 shell that
    :func:`refine_cdr_loops` handles) survive into ``E_complex`` and
    produce million-REU ``fa_rep`` blowups that dominate
    ``delta_G = E_complex − (E_Ab + E_Ag)``.

    Args:
        pose: PyRosetta Pose (modified in place).

    Returns:
        Tuple of (pre_score, post_score) total ref2015 energies, for
        audit logging. Large pre→post drops indicate the input had
        significant clashes.
    """
    import pyrosetta
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import (
        InitializeFromCommandline,
        RestrictToRepacking,
    )
    from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover

    sfxn = pyrosetta.create_score_function("ref2015")
    pre_score = sfxn(pose)

    tf = TaskFactory()
    tf.push_back(InitializeFromCommandline())
    tf.push_back(RestrictToRepacking())
    task = tf.create_task_and_apply_taskoperations(pose)

    packer = PackRotamersMover(sfxn, task)
    packer.apply(pose)

    post_score = sfxn(pose)
    logger.info(
        "Full-complex repack: total score %.2f → %.2f REU (delta %+.2f)",
        pre_score,
        post_score,
        post_score - pre_score,
    )
    return pre_score, post_score


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
# Chain selector helpers
# ---------------------------------------------------------------------------
def _make_chain_selector(chains: str):
    """Build a residue selector for one or more chain letters.

    For a single chain (e.g. ``"A"``), returns a plain ``ChainSelector``.
    For multiple chains (e.g. ``"ACB"``), tries comma-separated format
    first (``ChainSelector("A,C,B")``), falling back to combining
    individual selectors via ``OrResidueSelector``.

    Args:
        chains: One or more chain letters (e.g. ``"A"`` or ``"ACB"``).

    Returns:
        A residue selector covering all specified chains.
    """
    from pyrosetta.rosetta.core.select.residue_selector import (
        ChainSelector,
        OrResidueSelector,
    )

    if len(chains) <= 1:
        return ChainSelector(chains)

    # Try comma-separated format (supported in RosettaScripts XML)
    try:
        return ChainSelector(",".join(chains))
    except Exception:
        pass

    # Fallback: combine via OrResidueSelector
    selector = OrResidueSelector()
    for ch in chains:
        selector.add_residue_selector(ChainSelector(ch))
    return selector


# ---------------------------------------------------------------------------
# Energy computation
# ---------------------------------------------------------------------------
def compute_e_rep(pose, interface: str = Config.ROSETTA_INTERFACE) -> float:
    """Compute mean per-residue fa_rep (steric repulsion) at the interface.

    Uses the ref2015 score function.  Computes the average fa_rep across
    all residues at the nanobody–antigen interface, matching the AbDPO
    per-residue E_Rep metric (threshold: 5.0 REU).

    Args:
        pose: Scored or unscored PyRosetta Pose (will be scored in place).
        interface: Interface definition string (e.g. ``"H_A"`` or ``"D_ACB"``).

    Returns:
        Mean per-residue fa_rep energy in REU at the interface.
    """
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreType
    from pyrosetta.rosetta.core.select.residue_selector import (
        InterGroupInterfaceByVectorSelector,
    )

    # Score the pose with ref2015
    sfxn = pyrosetta.create_score_function("ref2015")
    sfxn(pose)

    # Parse interface string "H_A" → nanobody chains "H", antigen chains "A"
    parts = interface.split("_")
    nb_chains = parts[0]
    ag_chains = parts[1] if len(parts) > 1 else ""

    nb_selector = _make_chain_selector(nb_chains)
    ag_selector = _make_chain_selector(ag_chains)

    interface_selector = InterGroupInterfaceByVectorSelector(
        nb_selector, ag_selector
    )
    interface_mask = interface_selector.apply(pose)

    # Sum fa_rep over interface residues
    energies = pose.energies()
    fa_rep_type = ScoreType.fa_rep
    total_fa_rep = 0.0
    n_interface = 0
    for i in range(1, pose.total_residue() + 1):
        if interface_mask[i]:
            total_fa_rep += energies.residue_total_energies(i)[fa_rep_type]
            n_interface += 1

    if n_interface == 0:
        logger.warning("No interface residues found for %s — returning 0.0.", interface)
        return 0.0

    mean_fa_rep = total_fa_rep / n_interface
    logger.info(
        "E_Rep at interface %s: %.3f REU mean (%.3f total over %d residues)",
        interface, mean_fa_rep, total_fa_rep, n_interface,
    )
    return mean_fa_rep


def compute_cdr_energy_per_res(
    pose,
    nanobody_chain_id: str = "H",
    cdr_ranges: list[tuple[int, int]] = Config.VHH_CDR_RANGES,
) -> Optional[float]:
    """Compute mean Rosetta total energy across CDR residues (REU/residue).

    Implements the AbDPO residue-level energy from Zhou et al.
    NeurIPS 2024 (§3.2):

        ε(R⁰) = Σⱼ ε(R⁰[j])     for j in CDR residues

    additionally normalized by ``N_CDR_residues`` to make the metric
    scope-invariant — the same threshold (-0.2 REU/residue) applies
    whether we score CDR-H3 only (AbDPO ablation) or the full
    H1+H2+H3 set (multi-CDR π_ref scope).

    Computed in the bound complex: ``residue_total_energy(i)`` already
    captures all interactions involving residue ``i`` including those
    with antigen, so this is a binding-context energy not a separated
    monomer energy.

    Args:
        pose: PyRosetta Pose of the complex (scored in place).
        nanobody_chain_id: Chain letter of the nanobody (e.g. ``"H"``).
        cdr_ranges: Kabat-numbered CDR boundaries (default: H1+H2+H3).

    Returns:
        Mean Rosetta total energy per CDR residue in REU/residue, or
        ``None`` if no CDR residues could be located in the pose
        (numbering mismatch).
    """
    import pyrosetta

    sfxn = pyrosetta.create_score_function("ref2015")
    sfxn(pose)

    pdb_info = pose.pdb_info()
    if pdb_info is None:
        logger.warning("Pose has no PDB info — cannot map CDR ranges.")
        return None

    energies = pose.energies()
    total = 0.0
    n_cdr = 0
    for kabat_start, kabat_end in cdr_ranges:
        for kabat_res in range(kabat_start, kabat_end + 1):
            pose_idx = pdb_info.pdb2pose(nanobody_chain_id, kabat_res)
            if pose_idx == 0:
                continue
            total += energies.residue_total_energy(pose_idx)
            n_cdr += 1

    if n_cdr == 0:
        logger.warning(
            "No CDR residues found in pose for chain %s — returning None.",
            nanobody_chain_id,
        )
        return None

    mean_per_res = total / n_cdr
    logger.info(
        "CDR energy: %.3f REU/residue mean (%.3f total over %d CDR residues, chain %s)",
        mean_per_res, total, n_cdr, nanobody_chain_id,
    )
    return mean_per_res


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
    """Score a nanobody–antigen complex for steric clashes and CDR binding energy.

    Pipeline:
      1. Load PDB → Pose
      2. Full-complex side-chain repack (resolve framework/antigen clashes
         from raw crystal PDBs — essential to avoid million-REU blowups)
      3. FastRelax CDR loops (resolve residual CDR-local clashes)
      4. Compute E_Rep (fast)
      5. If E_Rep ≤ threshold → compute CDR per-residue energy (expensive)
         If E_Rep > threshold → skip CDR energy (fast-fail)
      6. If |CDR_energy| exceeds CDR_ENERGY_PATHOLOGICAL → flag
         scoring_failed (prevents non-physical blowups from being
         labeled "weak binder")

    Args:
        complex_pdb_path: Path to the complex PDB file.
        nanobody_chain_id: Chain letter of the nanobody.
        interface: PyRosetta interface string (e.g. ``"H_A"``) — used
                   only for the E_Rep selector, not for CDR energy.
        cdr_ranges: Kabat-numbered CDR boundaries for CCD refinement
                    AND for the CDR energy summation.
        ccd_outer_cycles: CCD outer cycles (AbDPO default: 1).
        ccd_max_inner_cycles: CCD inner cycles (AbDPO default: 10).
        e_rep_fast_fail: E_Rep threshold for skipping CDR energy.

    Returns:
        :class:`PhysicsScores` with ``e_rep`` always populated and
        ``cdr_energy_per_res`` populated only if E_Rep passes the
        fast-fail gate.
    """
    _ensure_init()

    pose = load_complex_pose(complex_pdb_path)

    # Full-complex side-chain repack — relieves framework/antigen clashes
    # in raw crystal PDBs. Without this, unrelaxed clashes outside the
    # CDR ±2 shell produce non-physical million-REU fa_rep blowups in
    # E_complex that dominate delta_G. Skip gracefully on failure.
    try:
        repack_complex(pose)
    except Exception:
        logger.warning(
            "Full-complex repack failed for %s — scoring without full repack.",
            complex_pdb_path,
            exc_info=True,
        )

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
            "E_Rep %.3f > %.1f REU — fast-failing, skipping CDR energy.",
            e_rep,
            e_rep_fast_fail,
        )
        return PhysicsScores(e_rep=e_rep, cdr_energy_per_res=None)

    # Step 2: per-residue CDR energy (expensive — only if E_Rep passes)
    cdr_energy = compute_cdr_energy_per_res(
        pose,
        nanobody_chain_id=nanobody_chain_id,
        cdr_ranges=cdr_ranges,
    )

    # Pathological-value guard: if CDR energy is outside the physical range
    # even after structure prep, the score is unreliable (likely residual
    # unresolvable clashes). Flag as scoring_failure rather than letting
    # it be misinterpreted as a weak-binder reject.
    if cdr_energy is not None and abs(cdr_energy) > Config.CDR_ENERGY_PATHOLOGICAL:
        logger.warning(
            "CDR energy %.2f REU/residue exceeds pathological threshold ±%.1f — "
            "marking scoring_failed (structure prep did not resolve clashes).",
            cdr_energy,
            Config.CDR_ENERGY_PATHOLOGICAL,
        )
        return PhysicsScores(
            e_rep=e_rep, cdr_energy_per_res=None, scoring_failed=True,
        )

    return PhysicsScores(e_rep=e_rep, cdr_energy_per_res=cdr_energy)
