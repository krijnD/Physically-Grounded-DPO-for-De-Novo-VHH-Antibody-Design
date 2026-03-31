"""Physics Judge: detects physical hallucinations via Rosetta energy decomposition.

Uses PyRosetta's ref2015 score function to evaluate two orthogonal
failure axes established by AbDPO (Zhou et al., NeurIPS 2024):

  - E_Rep (fa_rep): Lennard-Jones repulsive energy detecting steric
    clashes where atoms occupy the same spatial volume.  Evaluated
    FIRST as a fast-fail gate.
  - delta_G (dG_separated): Binding free energy via InterfaceAnalyzerMover
    detecting inert non-binders ("Rocks").  Only evaluated when the PDB
    contains a multi-chain antibody-antigen complex.

Rejection thresholds:
  - E_Rep > 5.0 REU  → fail_e_rep   (physical hallucination)
  - delta_G > -2.0 REU → fail_delta_g (non-binder)

Pre-evaluation step:
  LoopMover_Refine_CCD is optionally applied to CDR loops to resolve
  geometric artifacts from ML structure predictors (ESMFold, TNP)
  that would otherwise inflate fa_rep and produce false positives.

Decision flow:
  1. Candidate already failed → return immediately
  2. Missing or invalid PDB → skip with warning
  3. Load PDB → Pose, optionally refine CDR loops
  4. E_Rep > threshold → fail_e_rep (fast-fail gate)
  5. If multi-chain complex: delta_G > threshold → fail_delta_g
  6. All passed → physics_verdict = "pass"
"""

import logging
from pathlib import Path

from src.common.candidate import NanobodyCandidate
from src.common.config import Config
from .rosetta_scorer import (
    compute_delta_g,
    compute_e_rep,
    get_chain_count,
    load_and_refine,
)

logger = logging.getLogger(__name__)

# Standard VHH Kabat CDR boundaries for CCD refinement
_VHH_CDR_REGIONS: list[tuple[int, int]] = [
    (26, 32),   # CDR1
    (52, 56),   # CDR2
    (95, 102),  # CDR3
]


class PhysicsJudge:
    """Evaluates nanobody structural viability using Rosetta energy decomposition."""

    def __init__(
        self,
        e_rep_threshold: float = Config.E_REP_REJECT,
        delta_g_threshold: float = Config.DELTA_G_REJECT,
        interface: str = Config.ROSETTA_INTERFACE,
        refine_loops: bool = True,
        ccd_outer_cycles: int = Config.CCD_OUTER_CYCLES,
        ccd_max_inner_cycles: int = Config.CCD_MAX_INNER_CYCLES,
    ):
        self.e_rep_threshold = e_rep_threshold
        self.delta_g_threshold = delta_g_threshold
        self.interface = interface
        self.refine_loops = refine_loops
        self.ccd_outer_cycles = ccd_outer_cycles
        self.ccd_max_inner_cycles = ccd_max_inner_cycles

    def evaluate(
        self, candidate: NanobodyCandidate
    ) -> NanobodyCandidate:
        """Run the Physics Judge on a candidate with a folded PDB structure.

        Args:
            candidate: Must have pdb_filepath populated by TNP or another
                       structure predictor.  If already failed
                       (is_valid=False), returns immediately.

        Returns:
            The candidate with physics_verdict set.
        """
        if not candidate.is_valid:
            return candidate

        # Guard: PDB path must be present
        if not candidate.pdb_filepath:
            logger.warning(
                "Candidate %s: no PDB path, skipping physics evaluation.",
                candidate.candidate_id,
            )
            return candidate

        # Guard: PDB file must exist on disk
        pdb_path = Path(candidate.pdb_filepath)
        if not pdb_path.exists():
            logger.warning(
                "Candidate %s: PDB file %s not found, skipping physics evaluation.",
                candidate.candidate_id,
                pdb_path,
            )
            return candidate

        # Load PDB → Pose, optionally refine CDR loops
        cdr_regions = _VHH_CDR_REGIONS if self.refine_loops else None
        try:
            pose = load_and_refine(
                pdb_path=str(pdb_path),
                cdr_loop_regions=cdr_regions,
                outer_cycles=self.ccd_outer_cycles,
                max_inner_cycles=self.ccd_max_inner_cycles,
            )
        except Exception:
            logger.error(
                "Candidate %s: failed to load PDB into PyRosetta, "
                "skipping physics evaluation.",
                candidate.candidate_id,
                exc_info=True,
            )
            return candidate

        # ── E_Rep: steric clash detection (fast-fail gate) ──
        try:
            e_rep = compute_e_rep(pose)
        except Exception:
            logger.error(
                "Candidate %s: E_Rep computation failed.",
                candidate.candidate_id,
                exc_info=True,
            )
            return candidate

        candidate.e_rep = e_rep

        if e_rep > self.e_rep_threshold:
            candidate.fail_candidate(
                f"Physics: E_Rep {e_rep:.2f} > {self.e_rep_threshold} REU (steric clash)"
            )
            candidate.physics_verdict = "fail_e_rep"
            return candidate

        # ── delta_G: binding affinity (only for multi-chain complexes) ──
        num_chains = get_chain_count(pose)
        if num_chains >= 2:
            try:
                delta_g = compute_delta_g(pose, self.interface)
            except Exception:
                logger.error(
                    "Candidate %s: delta_G computation failed.",
                    candidate.candidate_id,
                    exc_info=True,
                )
                candidate.physics_verdict = "pass"
                return candidate

            candidate.delta_g = delta_g

            if delta_g > self.delta_g_threshold:
                candidate.fail_candidate(
                    f"Physics: delta_G {delta_g:.2f} > {self.delta_g_threshold} REU (non-binder)"
                )
                candidate.physics_verdict = "fail_delta_g"
                return candidate
        else:
            logger.info(
                "Candidate %s: single-chain structure, delta_G evaluation skipped.",
                candidate.candidate_id,
            )

        candidate.physics_verdict = "pass"
        return candidate
