"""Physics Judge: evaluates thermodynamic viability via Rosetta energy decomposition.

Detects two classes of failure (Zhou et al., NeurIPS 2024 — AbDPO):

  1. **Steric clashes ("Physical Hallucinations")**
     E_Rep > 5.0 REU — atoms overlap in the predicted structure.
     Fast-fail gate: checked first because it is cheap to compute.

  2. **Non-binders ("Rocks")**
     delta_G_bind > -2.0 REU — thermodynamically inert interface.
     Only evaluated if E_Rep passes (expensive InterfaceAnalyzer).

Requires a **complex** PDB (nanobody + antigen) and chain identifiers.
This is unlike the Biology/Biophysics judges which operate on the
nanobody monomer alone.

Decision flow:
  1. Candidate already failed upstream → return immediately
  2. No complex PDB available → skip with warning
  3. E_Rep > 5.0 REU → fail_e_rep
  4. delta_G > -2.0 REU → fail_delta_g
  5. Both pass → physics_verdict = "pass"
"""

import logging
from pathlib import Path

from src.common.candidate import NanobodyCandidate
from src.common.config import Config

logger = logging.getLogger(__name__)


class PhysicsJudge:
    """Evaluates nanobody–antigen complex viability using PyRosetta energies."""

    def __init__(
        self,
        e_rep_reject: float = Config.E_REP_REJECT,
        delta_g_reject: float = Config.DELTA_G_REJECT,
    ):
        self.e_rep_reject = e_rep_reject
        self.delta_g_reject = delta_g_reject

    def evaluate(
        self,
        candidate: NanobodyCandidate,
        complex_pdb_path: str,
        nanobody_chain_id: str = "H",
        interface: str = Config.ROSETTA_INTERFACE,
    ) -> NanobodyCandidate:
        """Run the Physics Judge on a nanobody–antigen complex.

        Args:
            candidate: Must have ``is_valid=True`` to be evaluated.
                       Results are written to ``candidate.e_rep``,
                       ``candidate.delta_g``, and ``candidate.physics_verdict``.
            complex_pdb_path: Path to the complex PDB (nanobody + antigen).
            nanobody_chain_id: Chain letter of the nanobody in the PDB.
            interface: PyRosetta interface string (e.g. ``"H_A"``).

        Returns:
            The candidate with physics verdict set.
        """
        if not candidate.is_valid:
            return candidate

        # Guard: complex PDB must exist
        if not complex_pdb_path or not Path(complex_pdb_path).exists():
            logger.warning(
                "Candidate %s: complex PDB not available (%s), "
                "skipping physics evaluation.",
                candidate.candidate_id,
                complex_pdb_path,
            )
            return candidate

        # Score the complex — wrapped in try/except because PyRosetta can
        # throw C++ exceptions on malformed structures.
        try:
            from src.physics_judge.rosetta_scorer import score_complex

            scores = score_complex(
                complex_pdb_path=complex_pdb_path,
                nanobody_chain_id=nanobody_chain_id,
                interface=interface,
            )
        except Exception:
            logger.error(
                "Candidate %s: PyRosetta scoring failed for %s",
                candidate.candidate_id,
                complex_pdb_path,
                exc_info=True,
            )
            candidate.physics_verdict = "error"
            return candidate

        # Populate metrics on the candidate
        candidate.e_rep = scores.e_rep
        candidate.delta_g = scores.delta_g

        # ── E_Rep: steric clash gate ──
        if candidate.e_rep > self.e_rep_reject:
            candidate.fail_candidate(
                f"Physics: E_Rep {candidate.e_rep:.3f} > "
                f"{self.e_rep_reject} REU (steric clash)"
            )
            candidate.physics_verdict = "fail_e_rep"
            return candidate

        # ── delta_G: binding affinity check ──
        if candidate.delta_g is not None and candidate.delta_g > self.delta_g_reject:
            candidate.fail_candidate(
                f"Physics: delta_G {candidate.delta_g:.3f} > "
                f"{self.delta_g_reject} REU (non-binder)"
            )
            candidate.physics_verdict = "fail_delta_g"
            return candidate

        candidate.physics_verdict = "pass"
        return candidate
