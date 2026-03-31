"""Low-level PyRosetta wrapper for Physics Judge energy computations.

Provides lazy PyRosetta initialization, optional CCD loop refinement
to fix prediction artifacts, and two energy extraction functions:

  - compute_e_rep: sums the fa_rep (Lennard-Jones repulsive) term across
    all residues to detect steric clashes (physical hallucinations).
  - compute_delta_g: runs InterfaceAnalyzerMover with apo-state repacking
    to calculate binding free energy for multi-chain complexes.

The ref2015 score function is used exclusively, as established by
AbDPO (Zhou et al., NeurIPS 2024).
"""

import logging

from src.common.config import Config

logger = logging.getLogger(__name__)

_PYROSETTA_INITIALIZED = False


def init_pyrosetta(flags: str = Config.PYROSETTA_FLAGS) -> None:
    """Initialize PyRosetta exactly once per process.

    Args:
        flags: Command-line flags passed to Rosetta.  Defaults to
               silent mode with non-standard residue tolerance.

    Raises:
        ImportError: If PyRosetta is not installed.
    """
    global _PYROSETTA_INITIALIZED
    if _PYROSETTA_INITIALIZED:
        return

    try:
        import pyrosetta
    except ImportError as exc:
        raise ImportError(
            "PyRosetta is required for the Physics Judge. "
            "Install via conda: conda install -c https://conda.rosettacommons.org pyrosetta"
        ) from exc

    pyrosetta.init(extra_options=flags)
    _PYROSETTA_INITIALIZED = True
    logger.info("PyRosetta initialized with flags: %s", flags)


def load_and_refine(
    pdb_path: str,
    cdr_loop_regions: list[tuple[int, int]] | None = None,
    outer_cycles: int = Config.CCD_OUTER_CYCLES,
    max_inner_cycles: int = Config.CCD_MAX_INNER_CYCLES,
):
    """Load a PDB into a PyRosetta Pose and optionally refine CDR loops.

    CCD (Cyclic Coordinate Descent) refinement resolves minor geometric
    imperfections from ML structure predictors (ESMFold, NanoBodyBuilder2)
    that would otherwise inflate fa_rep scores and produce false-positive
    hallucination flags.  Parameters follow AbDPO: outer_cycles=1,
    max_inner_cycles=10.

    Args:
        pdb_path: Path to the PDB file.
        cdr_loop_regions: List of (start, end) Kabat residue pairs
            defining CDR loops to refine.  If None or empty, refinement
            is skipped.
        outer_cycles: Number of CCD outer cycles.
        max_inner_cycles: Maximum CCD inner cycles per outer cycle.

    Returns:
        A scored PyRosetta Pose object.

    Raises:
        RuntimeError: If PyRosetta cannot parse the PDB.
    """
    init_pyrosetta()
    import pyrosetta

    pose = pyrosetta.pose_from_pdb(pdb_path)

    if cdr_loop_regions:
        try:
            from pyrosetta.rosetta.protocols.loops import Loop, Loops
            from pyrosetta.rosetta.protocols.loops.loop_mover.refine import (
                LoopMover_Refine_CCD,
            )

            loops = Loops()
            for start, end in cdr_loop_regions:
                cut = (start + end) // 2
                loops.add_loop(Loop(start, end, cut))

            mover = LoopMover_Refine_CCD(loops)
            mover.outer_cycles(outer_cycles)
            mover.max_inner_cycles(max_inner_cycles)
            mover.apply(pose)
            logger.debug(
                "CCD refinement applied to %d loop region(s).",
                len(cdr_loop_regions),
            )
        except Exception:
            logger.warning(
                "CCD loop refinement failed for %s — proceeding with unrefined pose.",
                pdb_path,
                exc_info=True,
            )

    return pose


def compute_e_rep(pose) -> float:
    """Sum the fa_rep (Lennard-Jones repulsive) energy across all residues.

    The pose MUST be scored before per-residue energies can be accessed;
    this function handles scoring internally.

    Args:
        pose: A PyRosetta Pose object.

    Returns:
        Total repulsion energy in REU (Rosetta Energy Units).
    """
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreType

    sfxn = pyrosetta.get_fa_scorefxn()
    sfxn(pose)  # Score the pose — mandatory before accessing energies

    total_fa_rep = sum(
        pose.energies().residue_total_energies(i)[ScoreType.fa_rep]
        for i in range(1, pose.total_residue() + 1)
    )
    return total_fa_rep


def compute_delta_g(pose, interface: str = Config.ROSETTA_INTERFACE) -> float:
    """Calculate binding free energy via InterfaceAnalyzerMover.

    Performs the full thermodynamic simulation:
      1. Score the bound complex
      2. Separate chains to infinite distance
      3. Repack apo-state interface residues (pack_separated=True)
      4. Score separated chains independently
      5. Return dG_separated = E_complex - (E_chain_A + E_chain_B)

    Args:
        pose: A PyRosetta Pose with >= 2 chains (antibody + antigen).
        interface: Chain interface string (e.g. "H_A" for VHH chain H
                   vs antigen chain A).

    Returns:
        Binding free energy (dG_separated) in REU.
    """
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    iam = InterfaceAnalyzerMover(interface)
    iam.set_pack_separated(True)
    iam.apply(pose)

    return iam.get_interface_dG()


def get_chain_count(pose) -> int:
    """Return the number of chains in a Pose.

    Used to gate delta-G computation (requires >= 2 chains).
    """
    return pose.num_chains()
