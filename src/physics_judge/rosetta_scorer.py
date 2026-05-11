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
    # AbDPO Appendix B sub-residue side-chain decomposition
    # (mean REU/residue across CDR residues; ref2015).  Reserved as
    # forward-looking signal for the DPO loss / PCGrad ablation — not
    # used by the AAPR rejection gates.  All three are None when the
    # E_Rep fast-fail tripped or when the antigen could not be mapped.
    #
    #   cdr_e_total_sidechain      — Eq. 10 sub-residue form
    #                                ES_total(A_j^sc), mean over CDR
    #   cdr_ag_e_nonrep_sidechain  — Eq. 11 mean over CDR
    #                                (CDR side-chain ↔ Ag attractive)
    #   cdr_ag_e_rep_sidechain     — Eq. 12 mean over CDR
    #                                (CDR side-chain ↔ Ag repulsive)
    cdr_e_total_sidechain: Optional[float] = None
    cdr_ag_e_nonrep_sidechain: Optional[float] = None
    cdr_ag_e_rep_sidechain: Optional[float] = None


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


def pack_cdr_shell(
    pose,
    nanobody_chain_id: str,
    cdr_ranges: list[tuple[int, int]] = Config.VHH_CDR_RANGES,
) -> tuple[float, float]:
    """Side-chain repack restricted to CDR residues + ±2 framework shell.

    Cheaper than :func:`repack_complex` (~10–30s vs ~30–90s) and more
    faithful for the per-residue CDR energy metric:

      - **Backbone is not moved**, so when scoring DiffAb-generated
        structures we evaluate what the model produced rather than a
        Rosetta-refined version of it (no FastRelax-induced confounding).
      - **Antigen side chains are not repacked**, preserving experimental
        rotamers for crystal complexes and the original-antigen anchor
        in AAPR pairs.
      - **CDR + ±2 framework shell** rotamers are reassigned to the
        Rosetta library, which is required for ``residue_total_energy``
        to be in interpretable units.

    Sufficient for the per-residue CDR energy metric, which is local
    (residue-residue interactions within ~6 Å pairwise cutoff) and so
    is unaffected by distant un-packed framework or antigen side chains.

    Falls back to :func:`repack_complex` if the CDR ranges cannot be
    located in the pose (numbering mismatch).

    Args:
        pose: PyRosetta Pose (modified in place).
        nanobody_chain_id: Chain letter of the nanobody (e.g. ``"H"``).
        cdr_ranges: Kabat-numbered CDR boundaries.

    Returns:
        Tuple of (pre_score, post_score) total ref2015 energies.
    """
    import pyrosetta
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import (
        InitializeFromCommandline,
        OperateOnResidueSubset,
        PreventRepackingRLT,
        RestrictToRepacking,
    )
    from pyrosetta.rosetta.core.select.residue_selector import (
        NotResidueSelector,
        ResidueIndexSelector,
    )
    from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover

    sfxn = pyrosetta.create_score_function("ref2015")
    pre_score = sfxn(pose)

    # Map Kabat CDR ranges to Pose residue indices using the existing helper
    loop_ranges = _find_pose_loop_residues(pose, nanobody_chain_id, cdr_ranges)
    if not loop_ranges:
        logger.warning(
            "No CDR loops identified for chain %s — falling back to "
            "full-complex repack.",
            nanobody_chain_id,
        )
        return repack_complex(pose)

    # CDR + ±2 framework shell, in Pose-internal residue indices.
    repack_indices: set[int] = set()
    for start, end in loop_ranges:
        lo = max(1, start - 2)
        hi = min(pose.total_residue(), end + 2)
        for i in range(lo, hi + 1):
            repack_indices.add(i)

    indices_str = ",".join(str(i) for i in sorted(repack_indices))
    repack_selector = ResidueIndexSelector(indices_str)
    no_repack_selector = NotResidueSelector(repack_selector)

    # TaskFactory: restrict to repacking everywhere (no design), then
    # explicitly prevent repacking outside the CDR-shell selector.
    tf = TaskFactory()
    tf.push_back(InitializeFromCommandline())
    tf.push_back(RestrictToRepacking())
    tf.push_back(OperateOnResidueSubset(PreventRepackingRLT(), no_repack_selector))

    task = tf.create_task_and_apply_taskoperations(pose)
    packer = PackRotamersMover(sfxn, task)
    packer.apply(pose)

    post_score = sfxn(pose)
    logger.info(
        "CDR-shell repack (%d residues): total score %.2f → %.2f REU "
        "(delta %+.2f)",
        len(repack_indices),
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
# AbDPO Appendix B sub-residue side-chain decomposition
# ---------------------------------------------------------------------------
def _resolve_cdr_pose_indices(
    pose,
    nanobody_chain_id: str,
    cdr_ranges: list[tuple[int, int]],
) -> list[int]:
    """Flat list of Pose-internal indices for every CDR residue.

    Returns empty list if no CDR residue can be located (numbering mismatch).
    """
    pdb_info = pose.pdb_info()
    if pdb_info is None:
        return []

    out: list[int] = []
    for kabat_start, kabat_end in cdr_ranges:
        for kabat_res in range(kabat_start, kabat_end + 1):
            pose_idx = pdb_info.pdb2pose(nanobody_chain_id, kabat_res)
            if pose_idx != 0:
                out.append(pose_idx)
    return out


def _resolve_antigen_pose_indices(
    pose,
    nanobody_chain_id: str,
) -> list[int]:
    """Pose-internal indices of every residue NOT on the nanobody chain.

    The Physics Judge does not need to know which chain letters are
    antigen — anything that is not the nanobody is treated as antigen
    for the CDR–Ag pair-energy sum (matches AbDPO Appendix B which sums
    over ``i ∈ [g+1, g+n]`` covering all antigen residues).
    """
    pdb_info = pose.pdb_info()
    if pdb_info is None:
        return []

    out: list[int] = []
    for i in range(1, pose.total_residue() + 1):
        if pdb_info.chain(i) != nanobody_chain_id:
            out.append(i)
    return out


def _break_cdr_disulfides(pose, cdr_pose_idxs: list[int]) -> bool:
    """Break disulfide bonds in ``pose`` where one (or both) ends fall in CDRs.

    Required because ``replace_pose_residue_copying_existing_coordinates``
    leaves the conformation's chemical-bond record intact when swapping a
    disulfide CYS to GLY — the next ``ref2015`` call then aborts on
    ``DisulfideAtomIndices.cc`` with "a disulfide is shown to be
    atomically connected, but the residue type is not a disulfide".
    Camelid VHHs commonly carry a second intra-CDR or CDR–framework
    disulfide (e.g. C96 inside the H3 range), so this is a real case on
    ANDD inputs, not an edge case.

    Strategy:
      1. Enumerate disulfide-state CYS pairs within the CDR set
         (matched by ``SG``–``SG`` distance < 2.5 Å).
      2. Call ``core.conformation.break_disulfide`` — this both removes
         the chemical bond AND strips the ``DISULFIDE`` variant from
         both residues.
      3. If the C++ entry point is unavailable in this PyRosetta build,
         fall back to manually removing the ``DISULFIDE`` variant; this
         is sufficient because the subsequent Gly substitution destroys
         the SG atom that the chemical-bond record references.

    Mutates ``pose`` in place. Returns ``True`` on success (including no
    disulfides to break), ``False`` if a bond touching the CDR cannot be
    broken — caller treats that as a candidate-level failure.
    """
    import pyrosetta
    from pyrosetta.rosetta.core.chemical import VariantType

    # Identify disulfide pairs touching the CDR
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for i in cdr_pose_idxs:
        res_i = pose.residue(i)
        if res_i.name3() != "CYS" or not res_i.has_variant_type(
            VariantType.DISULFIDE
        ):
            continue
        try:
            sg_i = res_i.xyz("SG")
        except Exception:
            continue
        for j in range(1, pose.total_residue() + 1):
            if j == i:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            res_j = pose.residue(j)
            if res_j.name3() != "CYS" or not res_j.has_variant_type(
                VariantType.DISULFIDE
            ):
                continue
            try:
                if (sg_i - res_j.xyz("SG")).norm() < 2.5:
                    pairs.append(key)
                    seen.add(key)
                    break
            except Exception:
                continue

    if not pairs:
        return True

    break_fn = getattr(
        pyrosetta.rosetta.core.conformation, "break_disulfide", None,
    )

    for lo, hi in pairs:
        broken = False
        if break_fn is not None:
            try:
                break_fn(pose.conformation(), lo, hi)
                broken = True
            except Exception:
                logger.debug(
                    "break_disulfide(conf, %d, %d) raised — "
                    "falling back to variant strip.",
                    lo, hi,
                )
        if not broken:
            try:
                from pyrosetta.rosetta.core.pose import (
                    remove_variant_type_from_pose_residue,
                )

                remove_variant_type_from_pose_residue(
                    pose, VariantType.DISULFIDE, lo,
                )
                remove_variant_type_from_pose_residue(
                    pose, VariantType.DISULFIDE, hi,
                )
                broken = True
            except Exception:
                logger.warning(
                    "Could not break disulfide %d-%d — "
                    "sub-residue scoring will fail for this candidate.",
                    lo, hi,
                    exc_info=True,
                )
                return False

        logger.info(
            "Broke CDR-touching disulfide %d-%d before Gly substitution.",
            lo, hi,
        )

    return True


def _make_gly_replacement_pose(pose, cdr_pose_idxs: list[int]):
    """Return a copy of ``pose`` with every CDR residue replaced by Glycine.

    Glycine has no non-backbone heavy atoms, so substituting each CDR
    residue with Gly (preserving backbone coordinates) cleanly isolates
    the side-chain contribution: subtracting the Gly-substituted pose
    energy from the original yields the per-residue and per-residue-pair
    side-chain contributions used in AbDPO Appendix B Eqs. 10–12.

    Implementation: uses ``replace_pose_residue_copying_existing_coordinates``
    which preserves the existing N, CA, C, O, H, HA atoms by identity-
    matching atom names, so the backbone geometry is bit-identical to
    the input. Before substitution, any CDR-touching disulfide bonds
    are broken via :func:`_break_cdr_disulfides` so that the destination
    Gly residue does not retain stale ``SG`` chemical-bond records.

    Returns:
        A new Pose with Gly substitutions applied at each CDR position.
        Returns None if any substitution fails (e.g. terminal cap issue,
        unbreakable disulfide).
    """
    import pyrosetta
    from pyrosetta.rosetta.core.chemical import ChemicalManager
    from pyrosetta.rosetta.core.pose import (
        replace_pose_residue_copying_existing_coordinates,
    )

    rts = ChemicalManager.get_instance().residue_type_set("fa_standard")
    try:
        gly_type = rts.name_map("GLY")
    except Exception:
        logger.warning("Could not resolve GLY residue type from fa_standard set.")
        return None

    pose_bb = pose.clone()

    # Strip CDR-touching disulfides before any Gly substitution — otherwise
    # the SG-SG chemical-bond record survives the residue swap and ref2015
    # asserts on the resulting Gly-with-disulfide-connectivity state.
    if not _break_cdr_disulfides(pose_bb, cdr_pose_idxs):
        return None

    for pose_idx in cdr_pose_idxs:
        try:
            replace_pose_residue_copying_existing_coordinates(
                pose_bb, pose_idx, gly_type,
            )
        except Exception:
            logger.warning(
                "Failed to substitute Gly at pose residue %d — cannot "
                "compute sub-residue energies.",
                pose_idx,
                exc_info=True,
            )
            return None

    return pose_bb


def compute_cdr_sidechain_energies(
    pose,
    nanobody_chain_id: str = "H",
    cdr_ranges: list[tuple[int, int]] = Config.VHH_CDR_RANGES,
) -> Optional[tuple[float, float, float]]:
    """Compute AbDPO Appendix B sub-residue side-chain energies.

    Returns three scalars (REU/residue, averaged over CDR residues for
    scope-invariance — same normalization as
    :func:`compute_cdr_energy_per_res`):

      1. ``cdr_e_total_sidechain``       — Eq. 10 sub-residue form,
                                            mean ES_total(A_j^sc)
      2. ``cdr_ag_e_nonrep_sidechain``   — Eq. 11 mean over CDR,
                                            sum of attractive (fa_atr,
                                            fa_sol, fa_elec, hbond,
                                            lk_ball) for CDR side-chain
                                            ↔ antigen pairs
      3. ``cdr_ag_e_rep_sidechain``      — Eq. 12 mean over CDR,
                                            fa_rep for CDR side-chain
                                            ↔ antigen pairs

    **Implementation: two-pose differencing.** Score the original pose
    with ref2015; build a Gly-substituted copy (each CDR residue replaced
    by Glycine, backbone preserved); score that copy with ref2015. Since
    Glycine has no side-chain heavy atoms, the difference at any
    energy-graph location is exactly the CDR side-chain contribution:

        E_sc(j)        = E_full(j)     − E_bb(j)
        E_sc(j, ag)    = E_full(j, ag) − E_bb(j, ag)

    where ``E_bb`` is evaluated on the Gly-substituted pose. The
    energy-graph subtraction yields per-residue-pair contributions that
    can be decomposed by ScoreType.

    **Relationship to AbDPO's exact form.**

    AbDPO Eq. 11 sums attractive energies over atom-pair types
    ``(A_j^sc, A_i^sc)`` and ``(A_j^sc, A_i^bb)``; Eq. 12 sums repulsive
    energies over all four atom-pair types with 2× weighting on those
    involving the CDR backbone (``A_j^bb``). Implementing exact atom-pair
    masking requires ``EnergyMethod``-level access (Etable.atom_pair_energy,
    HBondSet directionality) — substantially more PyRosetta plumbing.

    The Gly-substitution variant implemented here computes the
    *side-chain contribution of the CDR residue* exactly (since Gly has
    no side-chain atoms), but does *not* distinguish between antigen
    side-chain and antigen backbone atoms on the partner side — the
    antigen side is fully unchanged across both poses, so its bb/sc
    contributions are aggregated. This is consistent with how AbDPO
    actually computes Eq. 11 (the (sc, sc) + (sc, bb) sum covers all
    antigen-atom types) and is a deliberate simplification of Eq. 12's
    2× backbone weighting (the weighting was specific to AbDPO's
    energy-weighted loss and does not generalize to binary DPO).

    Args:
        pose: PyRosetta Pose of the complex (assumed already refined
              consistent with ``score_complex``).  Will be cloned;
              input pose is not modified.
        nanobody_chain_id: Chain letter of the nanobody (e.g. ``"H"``).
        cdr_ranges: Kabat-numbered CDR boundaries.

    Returns:
        ``(cdr_e_total_sidechain, cdr_ag_e_nonrep_sidechain,
        cdr_ag_e_rep_sidechain)`` as a tuple of mean REU/residue values,
        or ``None`` if CDR residues / antigen residues cannot be located
        or Gly substitution fails on any CDR position.
    """
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreType

    sfxn = pyrosetta.create_score_function("ref2015")

    cdr_idxs = _resolve_cdr_pose_indices(pose, nanobody_chain_id, cdr_ranges)
    if not cdr_idxs:
        logger.warning(
            "No CDR residues found for chain %s — cannot compute "
            "sub-residue energies.",
            nanobody_chain_id,
        )
        return None

    ag_idxs = _resolve_antigen_pose_indices(pose, nanobody_chain_id)
    if not ag_idxs:
        logger.warning(
            "No antigen residues found (every residue is on chain %s) — "
            "cannot compute CDR–Ag sub-residue energies.",
            nanobody_chain_id,
        )
        return None

    # Score the original (full) pose
    sfxn(pose)
    e_full = pose.energies()

    # Build and score the Gly-substituted (backbone-only) pose
    pose_bb = _make_gly_replacement_pose(pose, cdr_idxs)
    if pose_bb is None:
        return None
    sfxn(pose_bb)
    e_bb = pose_bb.energies()

    # ── (1) CDR side-chain total energy: per-residue diff ──
    e_total_sc_sum = 0.0
    for j in cdr_idxs:
        e_total_sc_sum += (
            e_full.residue_total_energy(j) - e_bb.residue_total_energy(j)
        )

    # ── (2,3) CDR–Ag side-chain interaction energies: pair-energy diff ──
    # Decompose by ScoreType. NONREP_TYPES are the AbDPO Eq. 11 attractive
    # set (hbond, fa_atr, fa_sol, fa_elec, lk_ball); fa_rep is Eq. 12.
    nonrep_types = (
        ScoreType.fa_atr,
        ScoreType.fa_sol,
        ScoreType.fa_elec,
        ScoreType.lk_ball_wtd,
        ScoreType.hbond_sc,
        ScoreType.hbond_bb_sc,
    )

    graph_full = e_full.energy_graph()
    graph_bb = e_bb.energy_graph()

    e_nonrep_sc_sum = 0.0
    e_rep_sc_sum = 0.0
    for j in cdr_idxs:
        for i in ag_idxs:
            edge_full = graph_full.find_energy_edge(j, i)
            edge_bb = graph_bb.find_energy_edge(j, i)
            for st in nonrep_types:
                v_full = edge_full[st] if edge_full is not None else 0.0
                v_bb = edge_bb[st] if edge_bb is not None else 0.0
                e_nonrep_sc_sum += (v_full - v_bb)
            v_full_rep = (
                edge_full[ScoreType.fa_rep] if edge_full is not None else 0.0
            )
            v_bb_rep = edge_bb[ScoreType.fa_rep] if edge_bb is not None else 0.0
            e_rep_sc_sum += (v_full_rep - v_bb_rep)

    n_cdr = len(cdr_idxs)
    e_total_sc_mean = e_total_sc_sum / n_cdr
    e_nonrep_sc_mean = e_nonrep_sc_sum / n_cdr
    e_rep_sc_mean = e_rep_sc_sum / n_cdr

    logger.info(
        "Sub-residue (AbDPO App. B): E_total_sc=%.3f, E_nonRep_sc=%.3f, "
        "E_Rep_sc=%.3f REU/residue (mean over %d CDR residues, %d antigen residues)",
        e_total_sc_mean, e_nonrep_sc_mean, e_rep_sc_mean, n_cdr, len(ag_idxs),
    )

    return (e_total_sc_mean, e_nonrep_sc_mean, e_rep_sc_mean)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def score_complex(
    complex_pdb_path: str,
    nanobody_chain_id: str = "H",
    interface: str = Config.ROSETTA_INTERFACE,
    cdr_ranges: list[tuple[int, int]] = Config.VHH_CDR_RANGES,
    refinement_mode: str = "pack_cdrs",
    ccd_outer_cycles: int = Config.CCD_OUTER_CYCLES,
    ccd_max_inner_cycles: int = Config.CCD_MAX_INNER_CYCLES,
    e_rep_fast_fail: float = Config.E_REP_REJECT,
) -> PhysicsScores:
    """Score a nanobody–antigen complex for steric clashes and CDR binding energy.

    Pipeline:
      1. Load PDB → Pose
      2. Refinement (per ``refinement_mode``):
           - ``"pack_cdrs"`` (default): CDR + ±2 shell side-chain repack
             only (~10–30s). Recommended for both calibration on GT
             crystals and AAPR on model outputs — backbone is preserved
             so the metric is faithful to the input structure.
           - ``"full"``: Full-complex side-chain repack + FastRelax on
             CDR loops (~5–6 min). Only useful for paranoid runs on
             ill-prepared crystal PDBs; FastRelax distorts the CDR
             backbone away from the input, which is undesirable for
             AAPR evaluation.
      3. Compute E_Rep (fast)
      4. If E_Rep ≤ threshold → compute CDR per-residue energy
         If E_Rep > threshold → skip CDR energy (fast-fail)
      5. If |CDR_energy| exceeds CDR_ENERGY_PATHOLOGICAL → flag
         scoring_failed (prevents non-physical blowups from being
         labeled "weak binder")

    Args:
        complex_pdb_path: Path to the complex PDB file.
        nanobody_chain_id: Chain letter of the nanobody.
        interface: PyRosetta interface string (e.g. ``"H_A"``) — used
                   only for the E_Rep selector, not for CDR energy.
        cdr_ranges: Kabat-numbered CDR boundaries for refinement AND for
                    the CDR energy summation.
        refinement_mode: ``"pack_cdrs"`` (default) or ``"full"``.
                         See pipeline step 2 above.
        ccd_outer_cycles: CCD outer cycles (full mode only; AbDPO default: 1).
        ccd_max_inner_cycles: CCD inner cycles (full mode only; AbDPO default: 10).
        e_rep_fast_fail: E_Rep threshold for skipping CDR energy.

    Returns:
        :class:`PhysicsScores` with ``e_rep`` always populated and
        ``cdr_energy_per_res`` populated only if E_Rep passes the
        fast-fail gate.

    Raises:
        ValueError: if ``refinement_mode`` is not one of the supported
            values.
    """
    if refinement_mode not in ("pack_cdrs", "full"):
        raise ValueError(
            f"Unknown refinement_mode={refinement_mode!r}; "
            "must be one of: 'pack_cdrs', 'full'."
        )
    _ensure_init()

    pose = load_complex_pose(complex_pdb_path)

    # Refinement — dispatch on requested mode. All branches catch
    # exceptions to avoid losing the candidate to a Rosetta hiccup;
    # the downstream scoring still runs on whatever pose state we have.
    if refinement_mode == "pack_cdrs":
        try:
            pack_cdr_shell(pose, nanobody_chain_id, cdr_ranges=cdr_ranges)
        except Exception:
            logger.warning(
                "CDR-shell repack failed for %s — scoring without refinement.",
                complex_pdb_path,
                exc_info=True,
            )
    else:  # refinement_mode == "full"
        # Full-complex side-chain repack — relieves framework/antigen
        # clashes in raw crystal PDBs.  Skip gracefully on failure.
        try:
            repack_complex(pose)
        except Exception:
            logger.warning(
                "Full-complex repack failed for %s — scoring without full repack.",
                complex_pdb_path,
                exc_info=True,
            )

        # CDR FastRelax — skip gracefully on failure
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
                "CDR FastRelax failed for %s — scoring without refinement.",
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

    # Step 3: sub-residue side-chain decomposition (AbDPO Appendix B).
    # Reuses the same refined Pose — no additional refinement cost.
    # Signal-only (does not gate the verdict); a failure to compute
    # leaves the three fields as None on the result, mirroring the
    # CDR-energy fast-fail behaviour.
    sidechain_values: Optional[tuple[float, float, float]] = None
    try:
        sidechain_values = compute_cdr_sidechain_energies(
            pose,
            nanobody_chain_id=nanobody_chain_id,
            cdr_ranges=cdr_ranges,
        )
    except Exception:
        logger.warning(
            "Sub-residue side-chain energy computation failed for %s — "
            "leaving sidechain fields as None on this candidate.",
            complex_pdb_path,
            exc_info=True,
        )

    if sidechain_values is None:
        return PhysicsScores(e_rep=e_rep, cdr_energy_per_res=cdr_energy)

    e_total_sc, e_nonrep_sc, e_rep_sc = sidechain_values
    return PhysicsScores(
        e_rep=e_rep,
        cdr_energy_per_res=cdr_energy,
        cdr_e_total_sidechain=e_total_sc,
        cdr_ag_e_nonrep_sidechain=e_nonrep_sc,
        cdr_ag_e_rep_sidechain=e_rep_sc,
    )
