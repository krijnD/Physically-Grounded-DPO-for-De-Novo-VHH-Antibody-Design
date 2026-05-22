"""Physics Judge: evaluates thermodynamic viability via Rosetta residue-level energies.

Detects two classes of failure (Zhou et al., NeurIPS 2024 — AbDPO):

  1. **Steric clashes ("Physical Hallucinations")**
     E_Rep > 5.0 REU — atoms overlap in the predicted structure.
     Fast-fail gate: checked first because it is cheap to compute.

  2. **Non-binders ("Rocks")**
     CDR energy > -0.2 REU/residue — CDR residues do not contribute
     attractive (negative) energy in the bound complex, indicating a
     thermodynamically inert interface.  Computed as the mean Rosetta
     ``residue_total_energy`` across CDR residues per the AbDPO
     residue-level formulation, additionally normalized by CDR length
     for scope-invariance (CDR-H3-only or H1+H2+H3).
     Only evaluated if E_Rep passes (expensive scoring step).

Requires a **complex** PDB (nanobody + antigen) and chain identifiers.
This is unlike the Biology/Biophysics judges which operate on the
nanobody monomer alone.

Independence contract: this judge always emits a ``physics_verdict``
regardless of ``candidate.is_valid``. ``is_valid`` is treated as a
downstream aggregate label, not a gate. When there is no antigen chain
or no complex PDB we emit ``"skipped_no_antigen"`` (Rosetta cannot
evaluate an interface that does not exist), and on PyRosetta errors we
emit ``"error"`` — both explicit rather than silent None.

Decision flow:
  1. No complex PDB or no interface → physics_verdict = "skipped_no_antigen"
  2. Rosetta scoring raises → physics_verdict = "error"
  3. Non-physical CDR energy (|E_cdr| > 100 REU/residue) →
     physics_verdict = "skipped_scoring_failure" (structure prep
     couldn't resolve clashes; distinct from a legitimate weak-binder
     reject)
  4. E_Rep > 5.0 REU → fail_e_rep
  5. CDR energy > -0.2 REU/residue → fail_cdr_energy
  6. Both pass → physics_verdict = "pass"
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
        cdr_energy_reject: float = Config.CDR_ENERGY_PER_RES_REJECT,
        refinement_mode: str = "none",
    ):
        self.e_rep_reject = e_rep_reject
        self.cdr_energy_reject = cdr_energy_reject
        self.refinement_mode = refinement_mode

    def evaluate(
        self,
        candidate: NanobodyCandidate,
        complex_pdb_path: str | None = None,
        nanobody_chain_id: str = "H",
        interface: str | None = Config.ROSETTA_INTERFACE,
    ) -> NanobodyCandidate:
        """Run the Physics Judge on a nanobody–antigen complex.

        Runs independently of any prior judge's verdict. ``is_valid`` is
        only used as an aggregate label downstream — it does not gate
        this judge. If the required inputs (complex PDB + interface) are
        not available the judge emits ``"skipped_no_antigen"`` so the
        output parquet is self-describing.

        Args:
            candidate: Candidate to score. Results are written to
                       ``candidate.e_rep``, ``candidate.cdr_energy_per_res``,
                       and ``candidate.physics_verdict``.
            complex_pdb_path: Path to the complex PDB (nanobody + antigen),
                              or ``None`` to emit ``skipped_no_antigen``.
            nanobody_chain_id: Chain letter of the nanobody in the PDB.
            interface: PyRosetta interface string (e.g. ``"H_A"``), or
                       ``None`` to emit ``skipped_no_antigen``.

        Returns:
            The candidate with physics verdict set.
        """
        # Guard: complex PDB + interface must be available
        if (
            not complex_pdb_path
            or not interface
            or not Path(complex_pdb_path).exists()
        ):
            logger.info(
                "Candidate %s: no complex PDB / interface — "
                "physics_verdict = skipped_no_antigen.",
                candidate.candidate_id,
            )
            candidate.physics_verdict = "skipped_no_antigen"
            return candidate

        # Score the complex — wrapped in try/except because PyRosetta can
        # throw C++ exceptions on malformed structures.
        try:
            from src.physics_judge.rosetta_scorer import score_complex

            scores = score_complex(
                complex_pdb_path=complex_pdb_path,
                nanobody_chain_id=nanobody_chain_id,
                interface=interface,
                refinement_mode=self.refinement_mode,
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
        candidate.cdr_energy_per_res = scores.cdr_energy_per_res
        # Sub-residue side-chain decomposition (AbDPO App. B) — signal-only,
        # does not influence verdict. Fields are None when the E_Rep
        # fast-fail short-circuited scoring or Gly substitution failed.
        candidate.cdr_e_total_sidechain = scores.cdr_e_total_sidechain
        candidate.cdr_ag_e_nonrep_sidechain = scores.cdr_ag_e_nonrep_sidechain
        candidate.cdr_ag_e_rep_sidechain = scores.cdr_ag_e_rep_sidechain

        # ── Scoring-failure gate: non-physical CDR energy ──
        # The scorer sets this flag when |E_cdr| exceeds the pathological
        # threshold, meaning structure prep could not resolve clashes.
        # Treat as a scoring skip, not a weak-binder reject — otherwise
        # DPO pair selection treats unscored blowups as negatives.
        if scores.scoring_failed:
            logger.info(
                "Candidate %s: non-physical CDR energy detected — "
                "physics_verdict = skipped_scoring_failure.",
                candidate.candidate_id,
            )
            candidate.physics_verdict = "skipped_scoring_failure"
            return candidate

        # ── E_Rep: steric clash gate ──
        if candidate.e_rep > self.e_rep_reject:
            candidate.fail_candidate(
                f"Physics: E_Rep {candidate.e_rep:.3f} > "
                f"{self.e_rep_reject} REU (steric clash)"
            )
            candidate.physics_verdict = "fail_e_rep"
            return candidate

        # ── CDR energy: residue-level binding affinity check (AbDPO) ──
        if (
            candidate.cdr_energy_per_res is not None
            and candidate.cdr_energy_per_res > self.cdr_energy_reject
        ):
            candidate.fail_candidate(
                f"Physics: CDR energy {candidate.cdr_energy_per_res:.3f} > "
                f"{self.cdr_energy_reject} REU/residue (non-binder)"
            )
            candidate.physics_verdict = "fail_cdr_energy"
            return candidate

        candidate.physics_verdict = "pass"
        return candidate
