"""Phase 1: 1D sequence annotation and deterministic pre-filtering.

Uses ANARCI (via abnumber) to align sequences to the Kabat numbering
scheme and applies the hallmark residue decision matrix:

- W47 (absolute):  Exposed bulky hydrophobe → immediate rejection.
- L45 (conditional): Gatekeeper risk → flagged for 3D SAP validation.
- V37 (conditional): Cavity risk → flagged for 3D SAP validation.
- G44 (conditional): Solvation risk → flagged for 3D SAP validation.
- CDR3 W/F (conditional): Hydrophobic override risk (W99 liability).
"""

import logging

from abnumber import Chain
from abnumber.exceptions import ChainParseError

from src.common.candidate import NanobodyCandidate
from src.common.config import Config

logger = logging.getLogger(__name__)


def annotate_and_filter(candidate: NanobodyCandidate) -> NanobodyCandidate:
    """Parse the raw sequence via ANARCI HMMs and apply deterministic rules.

    Modifies the candidate in-place: sets kabat_mapping, biology_flags,
    cdr3_sequence, and may call fail_candidate() for absolute violations.

    Args:
        candidate: A NanobodyCandidate with raw_sequence populated.

    Returns:
        The same candidate, annotated with flags or failed.
    """
    logger.debug(
        "Candidate %s: parsing %d-residue sequence (%.30s...)",
        candidate.candidate_id,
        len(candidate.raw_sequence),
        candidate.raw_sequence,
    )
    try:
        chain = Chain(
            candidate.raw_sequence,
            scheme="kabat",
            chain_type="H",
            assign_germline=False,
        )
    except ChainParseError as e:
        logger.warning(
            "Candidate %s: ANARCI parse failed — %s (seq length=%d, first 30: %.30s)",
            candidate.candidate_id, e, len(candidate.raw_sequence), candidate.raw_sequence,
        )
        candidate.fail_candidate(
            "Phase 1: Severe hallucination — sequence lacks immunoglobulin topology."
        )
        candidate.biology_verdict = "fail_absolute"
        return candidate

    # Extract hallmark positions and CDR3
    # abnumber.Chain uses bracket indexing; missing positions raise KeyError
    def _get_position(ch, pos_id: str):
        try:
            return ch[pos_id]
        except (KeyError, IndexError):
            return None

    pos_37 = _get_position(chain, "37")
    pos_44 = _get_position(chain, "44")
    pos_45 = _get_position(chain, "45")
    pos_47 = _get_position(chain, "47")
    cdr3_seq = chain.cdr3_seq

    candidate.kabat_mapping = {
        "37": pos_37, "44": pos_44, "45": pos_45, "47": pos_47,
    }
    candidate.cdr3_sequence = cdr3_seq

    # ── Absolute Rule: W47 ──
    # Exposed Trp at position 47 is almost universally fatal to colloidal
    # stability. The massive indole ring forces clathrate-like water ordering,
    # driving irreversible aggregation.
    if pos_47 == "W":
        candidate.fail_candidate(
            "Phase 1: Absolute liability — exposed W47 bulky hydrophobe detected."
        )
        candidate.biology_verdict = "fail_absolute"
        return candidate

    # ── Conditional Flags (require 3D SAP resolution) ──

    # L45: Loss of electrostatic repulsion (Arg→Leu) removes the
    # gatekeeper against homodimerization.
    if pos_45 == "L":
        candidate.biology_flags.append("L45_GATEKEEPER_RISK")

    # V37: Small aliphatic side chain leaves a structural cavity
    # on the former VL interface.
    if pos_37 == "V":
        candidate.biology_flags.append("V37_CAVITY_RISK")

    # G44: Loss of Glu solvation shell exposes adjacent hydrophobic
    # framework atoms.
    if pos_44 == "G":
        candidate.biology_flags.append("G44_SOLVATION_RISK")

    # CDR3 hydrophobic override: bulky hydrophobes (W, F) in CDR3
    # can nucleate aggregation independent of FR2 status (W99 liability).
    if cdr3_seq:
        for residue in cdr3_seq:
            if residue in Config.CDR3_HYDROPHOBIC_RESIDUES:
                candidate.biology_flags.append("CDR3_HYDROPHOBIC_OVERRIDE_RISK")
                break

    logger.info(
        "Candidate %s: pos47=%s, pos45=%s, pos37=%s, pos44=%s, flags=%s",
        candidate.candidate_id, pos_47, pos_45, pos_37, pos_44,
        candidate.biology_flags,
    )

    return candidate
