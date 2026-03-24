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

    # --- 1D Sequence Annotations (set by Phase 1 filter) ---
    kabat_mapping: dict = field(default_factory=dict)
    biology_flags: list[str] = field(default_factory=list)
    cdr3_sequence: Optional[str] = None

    # --- 3D Structural State ---
    pdb_filepath: Optional[str] = None

    # --- Global Status ---
    is_valid: bool = True
    failure_reason: Optional[str] = None

    # --- Biology Judge Metrics ---
    sap_scores: dict[str, float] = field(default_factory=dict)
    biology_verdict: Optional[str] = None  # "pass", "fail_absolute", "fail_conditional"

    # --- Biophysics Judge Metrics (TNP) ---
    psh_score: Optional[float] = None
    ppc_score: Optional[float] = None
    compactness: Optional[float] = None
    biophysics_verdict: Optional[str] = None

    # --- Physics Judge Metrics (Rosetta) ---
    delta_g: Optional[float] = None
    e_rep: Optional[float] = None
    physics_verdict: Optional[str] = None

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
            "pdb_filepath": self.pdb_filepath,
            "is_valid": self.is_valid,
            "failure_reason": self.failure_reason,
            # Biology
            "biology_flags": ",".join(self.biology_flags) if self.biology_flags else None,
            "biology_verdict": self.biology_verdict,
            "sap_scores": str(self.sap_scores) if self.sap_scores else None,
            # Biophysics
            "psh_score": self.psh_score,
            "ppc_score": self.ppc_score,
            "compactness": self.compactness,
            "biophysics_verdict": self.biophysics_verdict,
            # Physics
            "delta_g": self.delta_g,
            "e_rep": self.e_rep,
            "physics_verdict": self.physics_verdict,
        }
