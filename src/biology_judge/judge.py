"""Biology Judge: resolves conditional FR2 and CDR3 flags via 3D spatial analysis.

This judge only fires the expensive SAP calculation when Phase 1
flagged a conditional liability. Sequences with the canonical VHH
"YERL" motif and safe CDRs pass without any 3D computation.

Independence contract: this judge always emits a ``biology_verdict``
regardless of ``candidate.is_valid``. ``is_valid`` is treated as a
downstream aggregate label, not a gate. The only case where the judge
defers is when Phase 1 already declared ``fail_absolute`` (ANARCI could
not parse the sequence) — there is nothing to SAP against.

Decision flow:
  1. Phase 1 fail_absolute already set → return as-is
  2. No flags → biology_verdict = "pass"
  3. For each flag → compute localized SAP around the liability residue
  4. SAP > threshold → fail_candidate() with specific reason
  5. All flags cleared → biology_verdict = "pass"
  6. SAP calculation error → biology_verdict = "error"
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
    "W47_BULKY_INDOLE_RISK": 47,
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
        chain_id: str = "A",
    ) -> NanobodyCandidate:
        """Run the Biology Judge on a candidate with a folded structure.

        Runs independently of any prior judge's verdict. ``is_valid`` is
        only used as an aggregate label downstream — it does not gate
        this judge. The single exception is a Phase 1 ``fail_absolute``
        verdict, which indicates ANARCI could not parse the sequence as
        an antibody; there are no Kabat positions to evaluate, so we
        return without overwriting it.

        Args:
            candidate: Candidate with biology_flags populated by Phase 1.
            structure: Biopython Structure parsed from the candidate's PDB.
            chain_id: Chain identifier of the nanobody in the PDB.
                      Defaults to ``"A"`` (TNP monomer convention).

        Returns:
            The candidate with biology_verdict set.
        """
        # Respect Phase 1 absolute rejects (ANARCI parse failure) —
        # there is no Kabat mapping to SAP against.
        if candidate.biology_verdict == "fail_absolute":
            return candidate

        if not candidate.biology_flags:
            candidate.biology_verdict = "pass"
            return candidate

        # Evaluate each conditional flag via localized SAP.  Wrapped in a
        # broad except so a malformed structure can't take down the run —
        # each judge emits a verdict no matter what.
        try:
            for flag in candidate.biology_flags:
                target_res_id = _FLAG_TO_RESIDUE.get(flag)

                if target_res_id is not None:
                    sap = calculate_localized_sap(
                        structure,
                        target_res_id=target_res_id,
                        radius=self.sap_radius,
                        chain_id=chain_id,
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
        except Exception:
            logger.error(
                "Candidate %s: biology SAP evaluation failed.",
                candidate.candidate_id,
                exc_info=True,
            )
            candidate.biology_verdict = "error"
            return candidate

        # All flags cleared — the conditional liabilities are shielded
        candidate.biology_verdict = "pass"
        return candidate
