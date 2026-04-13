#!/usr/bin/env python3
"""Sanity-test all three judges on SAbDab ground-truth nanobody data.

Loads SAbDab entries (nanobody+antigen complex PDBs), extracts sequences,
and runs Phase 1 (sequence filter) + all three judges directly.

With --run-tnp, also invokes TNP (Therapeutic Nanobody Profiler) to fold
sequences and compute biophysics metrics (PSH, PPC, PNC, Compactness),
enabling a full end-to-end sanity test of all judges including Biophysics.

Usage on Snellius:
    # Quick test (biology + physics only):
    python scripts/test_sabdab_judges.py \
        --tsv /projects/0/hpmlprjs/interns/krijn/sabdab_nano_dataset_IgLM/sabdab_nano_summary.tsv \
        --pdb-dir /projects/0/hpmlprjs/interns/krijn/sabdab_nano_dataset_IgLM/filtered_vhh_pdbs \
        --output data/results/sabdab_judge_test.parquet \
        --limit 5

    # Full test (all judges incl. biophysics via TNP):
    python scripts/test_sabdab_judges.py \
        --tsv /projects/0/hpmlprjs/interns/krijn/sabdab_nano_dataset_IgLM/sabdab_nano_summary.tsv \
        --pdb-dir /projects/0/hpmlprjs/interns/krijn/sabdab_nano_dataset_IgLM/filtered_vhh_pdbs \
        --output data/results/sabdab_judge_test.parquet \
        --run-tnp --ncores 4
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.candidate import NanobodyCandidate
from src.common.pdb_utils import load_structure
from src.common.sabdab_loader import load_sabdab_entries
from src.biology_judge.sequence_filter import annotate_and_filter
from src.biology_judge.judge import BiologyJudge
from src.biophysics_judge.judge import BiophysicsJudge
from src.biophysics_judge.tnp_runner import run_tnp_batch
from src.physics_judge.judge import PhysicsJudge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("test_sabdab_judges")


def _log_progress(
    idx: int, total: int, elapsed: float,
    durations: list[float], run_start: float,
) -> None:
    """Log a progress bar with timing estimates."""
    avg_per_entry = sum(durations) / len(durations)
    remaining = avg_per_entry * (total - idx)
    total_elapsed = time.time() - run_start
    pct = idx / total * 100
    bar_len = 30
    filled = int(bar_len * idx // total)
    bar = "█" * filled + "░" * (bar_len - filled)
    logger.info(
        "  ⏱  %s %3.0f%% [%d/%d] | %.0fs this entry | "
        "%.0fs elapsed | ~%.0fs remaining (%.1fs/entry avg)",
        bar, pct, idx, total, elapsed, total_elapsed,
        remaining, avg_per_entry,
    )


def _save_results(candidates: list, output_path: Path) -> None:
    """Save current results to disk (Parquet or CSV fallback)."""
    df = pd.DataFrame([c.to_dict() for c in candidates])
    try:
        df.to_parquet(output_path, index=False)
    except ImportError:
        df.to_csv(output_path.with_suffix(".csv"), index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Test all three judges on SAbDab nanobody data."
    )
    parser.add_argument(
        "--tsv",
        required=True,
        help="Path to sabdab_nano_summary.tsv",
    )
    parser.add_argument(
        "--pdb-dir",
        required=True,
        help="Directory containing filtered SAbDab PDB files",
    )
    parser.add_argument(
        "--output",
        default="data/results/sabdab_judge_test.parquet",
        help="Output Parquet path (default: data/results/sabdab_judge_test.parquet)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N entries (for quick testing)",
    )
    parser.add_argument(
        "--run-tnp",
        action="store_true",
        default=False,
        help="Run TNP folding to compute biophysics metrics (PSH, PPC, Compactness). "
             "Without this flag the Biophysics Judge is skipped.",
    )
    parser.add_argument(
        "--ncores",
        type=int,
        default=1,
        help="Number of CPU cores for TNP folding (default: 1)",
    )
    args = parser.parse_args()

    # Load SAbDab entries
    entries = load_sabdab_entries(args.tsv, args.pdb_dir)
    if not entries:
        logger.error("No valid SAbDab entries found. Check paths.")
        sys.exit(1)

    if args.limit:
        entries = entries[: args.limit]
        logger.info("Limited to %d entries for testing.", args.limit)

    # Initialize judges
    biology_judge = BiologyJudge()
    biophysics_judge = BiophysicsJudge()
    physics_judge = PhysicsJudge()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(entries)

    # ── Phase 1 (all entries): Sequence pre-filter ──
    logger.info("=" * 60)
    logger.info("PHASE 1: Sequence annotation (%d entries)", total)
    logger.info("=" * 60)
    all_candidates: list[NanobodyCandidate] = []
    for idx, entry in enumerate(entries, 1):
        candidate = NanobodyCandidate(
            candidate_id=entry["candidate_id"],
            raw_sequence=entry["raw_sequence"],
            pdb_filepath=entry.get("pdb_filepath"),
            complex_pdb_path=entry.get("complex_pdb_path"),
            nanobody_chain_id=entry.get("nanobody_chain_id"),
            antigen_chain_ids=entry.get("antigen_chain_ids"),
        )
        annotate_and_filter(candidate)
        all_candidates.append(candidate)
        logger.info(
            "  [%d/%d] %s — %s (flags: %s)",
            idx, total, candidate.candidate_id,
            "REJECTED" if not candidate.is_valid else "ok",
            candidate.biology_flags or "none",
        )

    phase1_survivors = [c for c in all_candidates if c.is_valid]
    logger.info(
        "Phase 1 complete: %d/%d proceed.", len(phase1_survivors), total
    )

    # ── Phase 2 (optional): TNP folding + biophysics metrics ──
    if args.run_tnp and phase1_survivors:
        logger.info("=" * 60)
        logger.info(
            "PHASE 2: TNP folding (%d sequences, ncores=%d)",
            len(phase1_survivors), args.ncores,
        )
        logger.info("=" * 60)
        logger.info("  Running TNP batch — this may take a while ...")
        tnp_start = time.time()

        tnp_output_dir = output_path.parent / "tnp_sabdab_output"
        tnp_results = run_tnp_batch(
            sequences=[
                {"id": c.candidate_id, "sequence": c.raw_sequence}
                for c in phase1_survivors
            ],
            output_dir=tnp_output_dir,
            ncores=args.ncores,
        )

        # Populate candidates with TNP metrics (but keep SAbDab PDB for
        # biology/physics — we want to judge the real crystal structure)
        tnp_matched = 0
        for candidate in phase1_survivors:
            result = tnp_results.get(candidate.candidate_id)
            if result is None:
                logger.warning(
                    "  %s: TNP produced no result.", candidate.candidate_id
                )
                continue
            candidate.psh_score = result.psh
            candidate.ppc_score = result.ppc
            candidate.pnc_score = result.pnc
            candidate.compactness = result.compactness
            candidate.cdr_length = result.cdr_length
            candidate.cdr3_length = result.cdr3_length
            tnp_matched += 1
            # NOTE: intentionally NOT overriding pdb_filepath — we use the
            # real SAbDab crystal structure for biology/physics judges.

        tnp_elapsed = time.time() - tnp_start
        logger.info(
            "  TNP complete: %d/%d sequences profiled in %.0fs (%.1fs/seq).",
            tnp_matched, len(phase1_survivors), tnp_elapsed,
            tnp_elapsed / max(len(phase1_survivors), 1),
        )
    elif not args.run_tnp:
        logger.info(
            "Skipping TNP folding (use --run-tnp to enable Biophysics Judge)."
        )

    # ── Phase 3: Multi-judge evaluation ──
    logger.info("=" * 60)
    logger.info("PHASE 3: Judge evaluation (%d entries)", total)
    logger.info("=" * 60)
    candidates: list[NanobodyCandidate] = []
    run_start = time.time()
    durations: list[float] = []

    for idx, candidate in enumerate(all_candidates, 1):
        entry_start = time.time()
        nb_chain = candidate.nanobody_chain_id or "H"
        ag_chains = candidate.antigen_chain_ids

        logger.info(
            "[%d/%d] %s (chain %s vs %s)",
            idx, total, candidate.candidate_id, nb_chain, ag_chains or "?",
        )

        if not candidate.is_valid:
            logger.info("  Skipped (failed Phase 1)")
            candidates.append(candidate)
            elapsed = time.time() - entry_start
            durations.append(elapsed)
            _log_progress(idx, total, elapsed, durations, run_start)
            if idx % 5 == 0 or idx == total:
                _save_results(candidates, output_path)
            continue

        # Biology Judge
        if candidate.pdb_filepath and Path(candidate.pdb_filepath).exists():
            structure = load_structure(candidate.pdb_filepath, candidate.candidate_id)
            biology_judge.evaluate(candidate, structure, chain_id=nb_chain)
            logger.info("  Biology: %s", candidate.biology_verdict or "skipped")
        else:
            logger.info("  Biology: skipped (no PDB)")

        if not candidate.is_valid:
            candidates.append(candidate)
            elapsed = time.time() - entry_start
            durations.append(elapsed)
            _log_progress(idx, total, elapsed, durations, run_start)
            if idx % 5 == 0 or idx == total:
                _save_results(candidates, output_path)
            continue

        # Biophysics Judge
        biophysics_judge.evaluate(candidate)
        logger.info(
            "  Biophysics: %s",
            candidate.biophysics_verdict or "skipped (no TNP metrics)",
        )

        if not candidate.is_valid:
            candidates.append(candidate)
            elapsed = time.time() - entry_start
            durations.append(elapsed)
            _log_progress(idx, total, elapsed, durations, run_start)
            if idx % 5 == 0 or idx == total:
                _save_results(candidates, output_path)
            continue

        # Physics Judge
        if candidate.complex_pdb_path and ag_chains:
            ag_clean = ag_chains.replace(" ", "").replace("|", "")
            interface = f"{nb_chain}_{ag_clean}"

            if nb_chain.islower() or any(c.islower() for c in ag_clean):
                logger.warning(
                    "  Physics: skipped — lowercase chain ID(s) in '%s' "
                    "(known PyRosetta crash risk)", interface,
                )
                candidate.physics_verdict = "error"
                candidate.is_valid = False
            else:
                physics_judge.evaluate(
                    candidate,
                    complex_pdb_path=candidate.complex_pdb_path,
                    nanobody_chain_id=nb_chain,
                    interface=interface,
                )
                verdict = candidate.physics_verdict or "skipped"
                e_rep_str = f"{candidate.e_rep:.3f}" if candidate.e_rep is not None else "N/A"
                dg_str = f"{candidate.delta_g:.3f}" if candidate.delta_g is not None else "N/A"
                logger.info("  Physics: %s (E_Rep=%s, dG=%s)", verdict, e_rep_str, dg_str)
        else:
            logger.info("  Physics: skipped (no complex PDB or no antigen chain info)")

        candidates.append(candidate)
        elapsed = time.time() - entry_start
        durations.append(elapsed)
        _log_progress(idx, total, elapsed, durations, run_start)

        # Save partial results every 5 entries (guards against C++ segfaults)
        if idx % 5 == 0 or idx == total:
            _save_results(candidates, output_path)
            logger.info("  [checkpoint] Saved %d/%d results.", idx, total)

    # Summarize results
    df = pd.DataFrame([c.to_dict() for c in candidates])

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY: %d entries processed", len(df))
    logger.info("-" * 60)

    # Phase 1 results
    absolute_fails = df[df["biology_verdict"] == "fail_absolute"]
    logger.info("Phase 1 absolute rejects: %d", len(absolute_fails))

    # Biology Judge
    bio_verdicts = df["biology_verdict"].value_counts()
    logger.info("Biology verdicts: %s", dict(bio_verdicts))

    # Biophysics
    bp_verdicts = df["biophysics_verdict"].value_counts(dropna=False)
    logger.info("Biophysics verdicts: %s", dict(bp_verdicts))
    if not args.run_tnp:
        logger.info("  (Biophysics skipped — rerun with --run-tnp for full test)")

    # Physics Judge
    phys_verdicts = df["physics_verdict"].value_counts(dropna=False)
    logger.info("Physics verdicts: %s", dict(phys_verdicts))

    # Overall pass/fail
    overall_valid = df["is_valid"].sum()
    logger.info(
        "Overall: %d pass, %d fail", overall_valid, len(df) - overall_valid
    )
    logger.info("=" * 60)

    # Final save
    _save_results(candidates, output_path)
    logger.info("Results written to %s", output_path)

    # Also print a concise table to stdout
    cols = [
        "candidate_id",
        "is_valid",
        "biology_verdict",
        "biophysics_verdict",
        "physics_verdict",
        "e_rep",
        "delta_g",
    ]
    print("\n" + df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
