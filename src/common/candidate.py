from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NanobodyCandidate:
    """Central data structure tracking a de novo VHH sequence through
    the multi-judge evaluation pipeline.

    Mutable by design: each judge appends its verdict as the candidate
    flows through the pipeline. Serialized to Parquet at the end.
    """

    candidate_id: str
    raw_sequence: str

    # --- AAPR provenance (None for GT/SAbDab/ANDD candidates) ---
    # When this candidate comes from the AAPR sampler, these record the
    # parent GT it was generated from and the K-replicate index within
    # that GT. Required for downstream Pareto pair selection
    # (scripts/dpo/select_pareto_pairs.py groups by gt_complex_id to
    # find the winner for each loser). Always None for natural GT rows.
    gt_complex_id: Optional[str] = None
    sample_idx: Optional[int] = None

    # --- 1D Sequence Annotations (set by Phase 1 filter) ---
    kabat_mapping: dict = field(default_factory=dict)
    biology_flags: list[str] = field(default_factory=list)
    cdr3_sequence: Optional[str] = None

    # --- 3D Structural State ---
    pdb_filepath: Optional[str] = None
    complex_pdb_path: Optional[str] = None   # PDB containing nanobody + antigen
    nanobody_chain_id: Optional[str] = None  # chain ID of nanobody in the complex
    antigen_chain_ids: Optional[str] = None  # chain ID(s) of antigen (e.g. "A" or "AB")

    # --- Global Status ---
    is_valid: bool = True
    failure_reason: Optional[str] = None

    # --- Biology Judge Metrics ---
    sap_scores: dict[str, float] = field(default_factory=dict)
    biology_verdict: Optional[str] = None  # "pass", "fail_absolute", "fail_conditional"

    # --- Biophysics Judge Metrics (TNP) ---
    psh_score: Optional[float] = None
    ppc_score: Optional[float] = None
    pnc_score: Optional[float] = None
    compactness: Optional[float] = None
    cdr_length: Optional[int] = None
    cdr3_length: Optional[int] = None
    biophysics_verdict: Optional[str] = None

    # --- Physics Judge Metrics (Rosetta, AbDPO residue-level CDR energy) ---
    # Mean Rosetta total energy across CDR residues (REU/residue), per
    # Zhou et al. NeurIPS 2024 §3.2, normalized by N_CDR_residues for
    # scope-invariance. Replaces the legacy `delta_g` (which was full-
    # interface dG_separated and on a different scale than the AbDPO
    # threshold of -2.0 REU).
    cdr_energy_per_res: Optional[float] = None
    e_rep: Optional[float] = None
    physics_verdict: Optional[str] = None

    # --- Physics Judge: AbDPO Appendix B sub-residue side-chain decomp ---
    # Forward-looking DPO preference signal (not used by AAPR gates).
    # Mean REU/residue across CDR residues; computed by two-pose
    # differencing against a Gly-substituted backbone-only copy. See
    # `compute_cdr_sidechain_energies` for the exact form and the
    # relationship to AbDPO Eqs. 10–12. None if the Gly substitution
    # failed or if the E_Rep fast-fail short-circuited Physics scoring.
    cdr_e_total_sidechain: Optional[float] = None
    cdr_ag_e_nonrep_sidechain: Optional[float] = None
    cdr_ag_e_rep_sidechain: Optional[float] = None

    def fail_candidate(self, reason: str) -> None:
        """Transition the candidate to a terminal failure state."""
        self.is_valid = False
        self.failure_reason = reason

    def to_dict(self) -> dict:
        """Flat dictionary for Parquet serialization."""
        return {
            "candidate_id": self.candidate_id,
            "raw_sequence": self.raw_sequence,
            "cdr3_sequence": self.cdr3_sequence,
            "gt_complex_id": self.gt_complex_id,
            "sample_idx": self.sample_idx,
            "pdb_filepath": self.pdb_filepath,
            "complex_pdb_path": self.complex_pdb_path,
            "nanobody_chain_id": self.nanobody_chain_id,
            "antigen_chain_ids": self.antigen_chain_ids,
            "is_valid": self.is_valid,
            "failure_reason": self.failure_reason,
            # Biology
            "biology_flags": ",".join(self.biology_flags) if self.biology_flags else None,
            "biology_verdict": self.biology_verdict,
            "sap_scores": str(self.sap_scores) if self.sap_scores else None,
            # Biophysics
            "psh_score": self.psh_score,
            "ppc_score": self.ppc_score,
            "pnc_score": self.pnc_score,
            "compactness": self.compactness,
            "cdr_length": self.cdr_length,
            "cdr3_length": self.cdr3_length,
            "biophysics_verdict": self.biophysics_verdict,
            # Physics
            "cdr_energy_per_res": self.cdr_energy_per_res,
            "e_rep": self.e_rep,
            "physics_verdict": self.physics_verdict,
            # Physics: sub-residue side-chain decomp (AbDPO App. B)
            "cdr_e_total_sidechain": self.cdr_e_total_sidechain,
            "cdr_ag_e_nonrep_sidechain": self.cdr_ag_e_nonrep_sidechain,
            "cdr_ag_e_rep_sidechain": self.cdr_ag_e_rep_sidechain,
        }
