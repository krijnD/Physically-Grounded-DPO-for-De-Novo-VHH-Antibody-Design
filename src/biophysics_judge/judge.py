"""Biophysics Judge: evaluates clinical developability via TNP surface metrics.

Applies strict thresholds calibrated against 36 clinical-stage nanobody
therapeutics (Gordon et al., Therapeutic Nanobody Profiler).

Three metrics are thresholded for rejection:
  - PSH (Patches of Surface Hydrophobicity): bounded interval [79.59, 126.83]
  - PPC (Positive Patch Charge): upper bound < 0.39
  - Compactness (CDR3 loop geometry): bounded interval [0.81, 1.57]

Three additional metrics are stored but not used for rejection:
  - PNC (Patches of Negative Charge)
  - Total CDR Length
  - CDR3 Length

Independence contract: this judge always emits a ``biophysics_verdict``
regardless of ``candidate.is_valid``. ``is_valid`` is treated as a
downstream aggregate label, not a gate. If TNP metrics are missing
(e.g. folding failed, or Phase 1 flagged an unparseable sequence) the
verdict is ``"skipped_no_tnp"`` rather than silent None — so the
parquet is self-describing for hard-negative mining.

Decision flow:
  1. Missing TNP metrics → biophysics_verdict = "skipped_no_tnp"
  2. PSH outside green zone → fail_psh
  3. PPC above threshold → fail_ppc
  4. Compactness outside range → fail_compactness
  5. All passed → biophysics_verdict = "pass"
"""

import logging

from src.common.candidate import NanobodyCandidate
from src.common.config import Config

logger = logging.getLogger(__name__)


class BiophysicsJudge:
    """Evaluates nanobody developability using TNP surface metrics."""

    def __init__(
        self,
        psh_low: float = Config.PSH_GREEN_LOW,
        psh_high: float = Config.PSH_GREEN_HIGH,
        ppc_max: float = Config.PPC_MAX,
        compactness_low: float = Config.COMPACTNESS_LOW,
        compactness_high: float = Config.COMPACTNESS_HIGH,
    ):
        self.psh_low = psh_low
        self.psh_high = psh_high
        self.ppc_max = ppc_max
        self.compactness_low = compactness_low
        self.compactness_high = compactness_high

    def evaluate(
        self, candidate: NanobodyCandidate
    ) -> NanobodyCandidate:
        """Run the Biophysics Judge on a candidate with TNP metrics populated.

        Runs independently of any prior judge's verdict. ``is_valid`` is
        only used as an aggregate label downstream — it does not gate
        this judge.

        Args:
            candidate: Candidate with psh_score, ppc_score, and compactness
                       populated by the TNP runner.  If any of these are
                       missing, the judge emits ``"skipped_no_tnp"`` so the
                       output is self-describing.

        Returns:
            The candidate with biophysics_verdict set.
        """
        # Guard: TNP metrics must be present
        if any(
            v is None
            for v in (candidate.psh_score, candidate.ppc_score, candidate.compactness)
        ):
            logger.warning(
                "Candidate %s: missing TNP metrics — biophysics_verdict = skipped_no_tnp.",
                candidate.candidate_id,
            )
            candidate.biophysics_verdict = "skipped_no_tnp"
            return candidate

        # ── PSH: bounded interval (strict green zone) ──
        if candidate.psh_score < self.psh_low or candidate.psh_score > self.psh_high:
            candidate.fail_candidate(
                f"Biophysics: PSH {candidate.psh_score:.2f} outside "
                f"[{self.psh_low}, {self.psh_high}]"
            )
            candidate.biophysics_verdict = "fail_psh"
            return candidate

        # ── PPC: upper bound only ──
        if candidate.ppc_score > self.ppc_max:
            candidate.fail_candidate(
                f"Biophysics: PPC {candidate.ppc_score:.3f} > {self.ppc_max}"
            )
            candidate.biophysics_verdict = "fail_ppc"
            return candidate

        # ── Compactness: bounded interval ──
        if (
            candidate.compactness < self.compactness_low
            or candidate.compactness > self.compactness_high
        ):
            candidate.fail_candidate(
                f"Biophysics: Compactness {candidate.compactness:.2f} outside "
                f"[{self.compactness_low}, {self.compactness_high}]"
            )
            candidate.biophysics_verdict = "fail_compactness"
            return candidate

        candidate.biophysics_verdict = "pass"
        return candidate
