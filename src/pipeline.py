"""Top-level pipeline orchestrator.

Runs VHH candidate sequences through the multi-judge evaluation:
  Phase 1: 1D sequence annotation + deterministic pre-filter
  Phase 2: 3D structure generation (placeholder — uses pre-folded PDBs)
  Phase 3: Parallel multi-judge evaluation (Biology → Biophysics → Physics)

Outputs a Parquet file with per-candidate verdicts for DPO pair construction.
"""

import logging
from pathlib import Path

import pandas as pd

from src.common.candidate import NanobodyCandidate
from src.common.pdb_utils import load_structure
from src.biology_judge.sequence_filter import annotate_and_filter
from src.biology_judge.judge import BiologyJudge

logger = logging.getLogger(__name__)

# Default paths relative to project root
STRUCTURES_DIR = Path("data/structures")
RESULTS_DIR = Path("data/results")


def run_pipeline(
    sequences: list[dict[str, str]],
    structures_dir: Path = STRUCTURES_DIR,
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Run the full evaluation pipeline on a list of sequences.

    Args:
        sequences: List of dicts with keys "candidate_id" and "raw_sequence".
                   Optionally "pdb_filepath" if the structure is pre-folded.
        structures_dir: Directory where PDB files are stored/expected.
        results_dir: Directory where the output Parquet will be written.

    Returns:
        DataFrame with one row per candidate and all judge verdicts.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    biology_judge = BiologyJudge()
    candidates: list[NanobodyCandidate] = []

    for seq_record in sequences:
        candidate = NanobodyCandidate(
            candidate_id=seq_record["candidate_id"],
            raw_sequence=seq_record["raw_sequence"],
            pdb_filepath=seq_record.get("pdb_filepath"),
        )

        # ── Phase 1: 1D Sequence Pre-filter ──
        annotate_and_filter(candidate)

        if not candidate.is_valid:
            # Absolute failure (e.g. W47) — skip folding entirely
            candidates.append(candidate)
            continue

        # ── Phase 2: Load 3D Structure ──
        # For now, expects pre-folded PDB files. NanoBodyBuilder2
        # integration will replace this with on-the-fly folding.
        pdb_path = _resolve_pdb_path(candidate, structures_dir)
        if pdb_path is None:
            # No structure available — can't run 3D judges
            logger.warning(
                "Candidate %s: no PDB found, skipping 3D evaluation.",
                candidate.candidate_id,
            )
            candidates.append(candidate)
            continue

        candidate.pdb_filepath = str(pdb_path)
        structure = load_structure(str(pdb_path), candidate.candidate_id)

        # ── Phase 3: Multi-Judge Evaluation ──
        # Biology Judge
        biology_judge.evaluate(candidate, structure)

        # Biophysics Judge (TNP) — placeholder
        # biophysics_judge.evaluate(candidate, structure)

        # Physics Judge (Rosetta) — placeholder
        # physics_judge.evaluate(candidate, structure)

        candidates.append(candidate)

    # Serialize results
    df = pd.DataFrame([c.to_dict() for c in candidates])
    output_path = results_dir / "judge_verdicts.parquet"
    df.to_parquet(output_path, index=False)
    logger.info("Wrote %d candidates to %s", len(df), output_path)

    return df


def _resolve_pdb_path(
    candidate: NanobodyCandidate,
    structures_dir: Path,
) -> Path | None:
    """Find the PDB file for a candidate.

    Checks in order:
      1. Explicit pdb_filepath on the candidate
      2. structures_dir / {candidate_id}.pdb
    """
    if candidate.pdb_filepath:
        p = Path(candidate.pdb_filepath)
        if p.exists():
            return p

    default = structures_dir / f"{candidate.candidate_id}.pdb"
    if default.exists():
        return default

    return None
