"""Top-level pipeline orchestrator.

Runs VHH candidate sequences through the multi-judge evaluation:
  Phase 1: 1D sequence annotation + deterministic pre-filter
  Phase 2: TNP batch (NanoBodyBuilder2 folding + biophysics metrics)
  Phase 3: Multi-judge evaluation (Biology, Biophysics, Physics)

TNP serves as both the folder and the biophysics analyzer, enforcing
the "Fold Once, Judge Many" architecture.  The PDB structures it
generates are shared with the Biology Judge (SAP) and Physics Judge
(Rosetta, future).

Judge independence
------------------
Each judge runs on every candidate and emits its own verdict. A
failure in one judge does NOT gate any other judge — ``is_valid`` is
a pure aggregate label built from individual verdicts, used downstream
for hard-negative DPO pair construction. The only legitimate skip is
"required input missing" (no antigen → physics skipped;
non-parseable sequence → TNP can't fold → biophysics skipped). In
those cases the judge emits an explicit ``skipped_*`` verdict rather
than silent None.

Outputs a Parquet file with one row per candidate, one column per
judge verdict.
"""

import logging
import time
from pathlib import Path

import pandas as pd

from src.common.candidate import NanobodyCandidate
from src.common.config import Config
from src.common.pdb_utils import load_structure
from src.biology_judge.sequence_filter import annotate_and_filter
from src.biology_judge.judge import BiologyJudge
from src.biophysics_judge.tnp_runner import run_tnp_batch
from src.biophysics_judge.judge import BiophysicsJudge
from src.physics_judge.judge import PhysicsJudge

logger = logging.getLogger(__name__)

# Default paths relative to project root
STRUCTURES_DIR = Path("data/structures")
RESULTS_DIR = Path("data/results")


def run_pipeline(
    sequences: list[dict[str, str]],
    structures_dir: Path = STRUCTURES_DIR,
    results_dir: Path = RESULTS_DIR,
    tnp_ncores: int = Config.TNP_NCORES,
) -> pd.DataFrame:
    """Run the full evaluation pipeline on a list of sequences.

    Args:
        sequences: List of dicts with keys "candidate_id" and "raw_sequence".
                   Optionally "pdb_filepath" if the structure is pre-folded.
        structures_dir: Directory where PDB files are stored/expected.
        results_dir: Directory where the output Parquet will be written.
        tnp_ncores: Number of CPU cores for TNP to use.

    Returns:
        DataFrame with one row per candidate and all judge verdicts.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[NanobodyCandidate] = []

    # ── Phase 1: 1D Sequence Pre-filter ──
    for seq_record in sequences:
        candidate = NanobodyCandidate(
            candidate_id=seq_record["candidate_id"],
            raw_sequence=seq_record["raw_sequence"],
            pdb_filepath=seq_record.get("pdb_filepath"),
            complex_pdb_path=seq_record.get("complex_pdb_path"),
            nanobody_chain_id=seq_record.get("nanobody_chain_id"),
            antigen_chain_ids=seq_record.get("antigen_chain_ids"),
        )
        annotate_and_filter(candidate)
        candidates.append(candidate)

    # Phase 1 parseable candidates are the ones TNP can fold.  Unparseable
    # sequences (ANARCI fail_absolute) still proceed to Phase 3 — they just
    # get "skipped_unannotated" / "skipped_no_tnp" verdicts from downstream
    # judges, so every candidate still has a complete row in the output.
    foldable = [
        c for c in candidates if c.biology_verdict != "fail_absolute"
    ]
    logger.info(
        "Phase 1 complete: %d/%d candidates proceed to folding "
        "(remaining %d will receive skipped verdicts).",
        len(foldable),
        len(candidates),
        len(candidates) - len(foldable),
    )

    # ── Phase 2: TNP Batch (fold + biophysics metrics) ──
    if foldable:
        tnp_output_dir = results_dir / "tnp_output"
        tnp_results = run_tnp_batch(
            sequences=[
                {"id": c.candidate_id, "sequence": c.raw_sequence}
                for c in foldable
            ],
            output_dir=tnp_output_dir,
            ncores=tnp_ncores,
        )

        # Populate candidates with TNP results.  If TNP produced nothing
        # for a candidate, its metrics stay None — the Biophysics Judge
        # will emit "skipped_no_tnp" for that row.
        for candidate in foldable:
            result = tnp_results.get(candidate.candidate_id)
            if result is None:
                logger.warning(
                    "Candidate %s: TNP produced no result — "
                    "biophysics/biology will report skipped verdicts.",
                    candidate.candidate_id,
                )
                continue

            candidate.psh_score = result.psh
            candidate.ppc_score = result.ppc
            candidate.pnc_score = result.pnc
            candidate.compactness = result.compactness
            candidate.cdr_length = result.cdr_length
            candidate.cdr3_length = result.cdr3_length

            # Use TNP-generated PDB (prefer over any pre-existing path)
            if result.pdb_path:
                candidate.pdb_filepath = result.pdb_path

    # ── Phase 3: Multi-Judge Evaluation (every judge, every candidate) ──
    biology_judge = BiologyJudge()
    biophysics_judge = BiophysicsJudge()
    physics_judge = PhysicsJudge()

    total = len(candidates)
    judge_start = time.time()
    durations: list[float] = []

    for idx, candidate in enumerate(candidates, 1):
        entry_start = time.time()

        # Biology Judge: localized SAP on conditional flags
        if candidate.biology_verdict == "fail_absolute":
            # Phase 1 already spoke — leave its verdict intact.
            pass
        elif candidate.pdb_filepath:
            structure = load_structure(
                candidate.pdb_filepath, candidate.candidate_id
            )
            chain_id = candidate.nanobody_chain_id or "A"
            biology_judge.evaluate(candidate, structure, chain_id=chain_id)
        else:
            # Can't SAP without a folded structure (TNP likely failed).
            logger.warning(
                "Candidate %s: no PDB — biology_verdict = skipped_no_structure.",
                candidate.candidate_id,
            )
            candidate.biology_verdict = "skipped_no_structure"

        # Biophysics Judge: threshold check on TNP metrics.  Judge itself
        # emits "skipped_no_tnp" when metrics are missing.
        biophysics_judge.evaluate(candidate)

        # Physics Judge: Rosetta energy decomposition (E_Rep + delta_G).
        # Always invoke — the judge emits "skipped_no_antigen" when it
        # lacks the inputs it needs.
        if candidate.complex_pdb_path and candidate.antigen_chain_ids:
            nb_chain = candidate.nanobody_chain_id or "H"
            # SAbDab uses "A | C | B" format; PyRosetta needs "ACB"
            ag_clean = candidate.antigen_chain_ids.replace(" ", "").replace("|", "")
            interface = f"{nb_chain}_{ag_clean}" if ag_clean else None
            physics_judge.evaluate(
                candidate,
                complex_pdb_path=candidate.complex_pdb_path,
                nanobody_chain_id=nb_chain,
                interface=interface,
            )
        else:
            physics_judge.evaluate(
                candidate,
                complex_pdb_path=None,
                interface=None,
            )

        # Progress tracking
        elapsed = time.time() - entry_start
        durations.append(elapsed)
        avg = sum(durations) / len(durations)
        remaining = avg * (total - idx)
        total_elapsed = time.time() - judge_start
        pct = idx / total * 100
        filled = int(30 * idx // total)
        bar = "█" * filled + "░" * (30 - filled)
        logger.info(
            "  %s %3.0f%% [%d/%d] %s | %.0fs elapsed | ~%.0fs remaining",
            bar, pct, idx, total, candidate.candidate_id,
            total_elapsed, remaining,
        )

    # ── Serialize results ──
    df = pd.DataFrame([c.to_dict() for c in candidates])
    output_path = results_dir / "judge_verdicts.parquet"
    try:
        df.to_parquet(output_path, index=False)
    except ImportError:
        output_path = output_path.with_suffix(".csv")
        df.to_csv(output_path, index=False)
        logger.warning("pyarrow not installed — wrote CSV instead.")
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
