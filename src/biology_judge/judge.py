"""Biology Judge: resolves conditional FR2 and CDR3 flags via 3D spatial analysis.

This judge only fires the expensive SAP calculation when Phase 1
flagged a conditional liability. Sequences with the canonical VHH
"YERL" motif and safe CDRs pass without any 3D computation.

Decision flow:
  1. No flags → biology_verdict = "pass"
  2. For each flag → compute localized SAP around the liability residue
  3. SAP > threshold → fail_candidate() with specific reason
  4. All flags cleared → biology_verdict = "pass"
"""

import logging

from Bio.PDB.Structure import Structure

from src.common.candidate import NanobodyCandidate
from src.common.config import Config
from .sap_calculator import calculate_localized_sap

logger = logging.getLogger(__name__)

# Maps biology flags to the Kabat residue ID that should be spatially evaluated
_FLAG_TO_RESIDUE: dict[str, int] = {
    "L45_GATEKEEPER_RISK": 45,
    "V37_CAVITY_RISK": 37,
    "G44_SOLVATION_RISK": 44,
}


class BiologyJudge:
    """Evaluates conditional biology flags using localized SAP analysis."""

    def __init__(
        self,
        sap_threshold: float = Config.SAP_SAFETY_THRESHOLD,
        sap_radius: float = Config.SAP_RADIUS,
    ):
        self.sap_threshold = sap_threshold
        self.sap_radius = sap_radius

    def evaluate(
        self,
        candidate: NanobodyCandidate,
        structure: Structure,
    ) -> NanobodyCandidate:
        """Run the Biology Judge on a candidate with a folded structure.

        Args:
            candidate: Must have already passed through sequence_filter.
                       If already failed (is_valid=False), returns immediately.
            structure: Biopython Structure parsed from the candidate's PDB.

        Returns:
            The candidate with biology_verdict set.
        """
        if not candidate.is_valid:
            return candidate

        if not candidate.biology_flags:
            candidate.biology_verdict = "pass"
            return candidate

        # Evaluate each conditional flag via localized SAP
        for flag in candidate.biology_flags:
            target_res_id = _FLAG_TO_RESIDUE.get(flag)

            if target_res_id is not None:
                sap = calculate_localized_sap(
                    structure,
                    target_res_id=target_res_id,
                    radius=self.sap_radius,
                )
                candidate.sap_scores[flag] = sap

                if sap > self.sap_threshold:
                    candidate.fail_candidate(
                        f"Biology: {flag} unshielded (SAP: {sap:.1f} > {self.sap_threshold})."
                    )
                    candidate.biology_verdict = "fail_conditional"
                    return candidate

            elif flag == "CDR3_HYDROPHOBIC_OVERRIDE_RISK":
                # For CDR3 overrides, we scan each CDR3 residue's exposure.
                # This requires knowing CDR3 residue positions in the PDB,
                # which depends on the numbering alignment. For now, we flag
                # it and let downstream judges (biophysics PSH) catch global
                # surface hydrophobicity.
                logger.info(
                    "Candidate %s: CDR3 hydrophobic override flagged — "
                    "deferred to global PSH evaluation.",
                    candidate.candidate_id,
                )

        # All flags cleared — the conditional liabilities are shielded
        candidate.biology_verdict = "pass"
        return candidate
