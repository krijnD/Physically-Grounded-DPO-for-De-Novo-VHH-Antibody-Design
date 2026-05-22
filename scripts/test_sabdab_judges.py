#!/usr/bin/env python3
"""Sanity-test all three judges on SAbDab ground-truth nanobody data.

Loads SAbDab entries (nanobody+antigen complex PDBs), extracts sequences,
and runs Phase 1 (sequence filter) + all three judges directly.

With --score-biophysics, also runs theraprofnano metric functions (PSH,
PPC, PNC, Compactness) directly on the VHH chain of each input PDB —
no NanoBodyBuilder2 re-folding involved.

Usage on Snellius:
    # Quick test (biology + physics only):
    python scripts/test_sabdab_judges.py \
        --tsv /projects/0/hpmlprjs/interns/krijn/sabdab_nano_dataset_IgLM/sabdab_nano_summary.tsv \
        --pdb-dir /projects/0/hpmlprjs/interns/krijn/sabdab_nano_dataset_IgLM/filtered_vhh_pdbs \
        --output data/results/sabdab_judge_test.parquet \
        --limit 5

    # Full test (all judges including biophysics):
    python scripts/test_sabdab_judges.py \
        --tsv /projects/0/hpmlprjs/interns/krijn/sabdab_nano_dataset_IgLM/sabdab_nano_summary.tsv \
        --pdb-dir /projects/0/hpmlprjs/interns/krijn/sabdab_nano_dataset_IgLM/filtered_vhh_pdbs \
        --output data/results/sabdab_judge_test.parquet \
        --score-biophysics
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
from src.common.sabdab_loader import (
    load_sabdab_entries, load_andd_entries, load_aapr_entries, load_pdb_entries,
)
from src.biology_judge.sequence_filter import annotate_and_filter
from src.biology_judge.judge import BiologyJudge
from src.biophysics_judge.judge import BiophysicsJudge
from src.biophysics_judge.tnp_direct import score_pdb as score_biophysics_pdb
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
        default=None,
        help="Path to sabdab_nano_summary.tsv (SAbDab mode).",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to ANDD_VHH_with_structure.csv (ANDD mode) OR an "
             "AAPR candidate manifest CSV (auto-detected by presence of "
             "the `gt_complex_id` column). For AAPR mode `--pdb-dir` is "
             "ignored because per-row `complex_pdb_path` is in the CSV.",
    )
    parser.add_argument(
        "--pdb-dir",
        default=None,
        help="Directory containing PDB files. Required for SAbDab/ANDD "
             "modes; ignored for AAPR mode (per-row paths are in the CSV).",
    )
    parser.add_argument(
        "--chain",
        default="A",
        help="Nanobody chain ID — only used when neither --tsv nor --csv is "
             "provided (plain PDB directory mode, default: A).",
    )
    parser.add_argument(
        "--antigen-chain",
        default=None,
        help="Antigen chain ID — only used in plain PDB directory mode. "
             "Required for the Physics Judge to run (e.g. 'B').",
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
        "--score-biophysics",
        "--run-tnp",  # back-compat alias
        action="store_true",
        default=False,
        help="Score biophysics metrics (PSH, PPC, PNC, Compactness) directly "
             "on each PDB's VHH chain via theraprofnano. Without this flag "
             "the Biophysics Judge emits skipped_no_tnp.",
    )
    parser.add_argument(
        "--refinement-mode",
        choices=["none", "pack_cdrs", "full"],
        default="none",
        help="Physics Judge structure prep mode. 'none' (default, "
             "since the 2026-05-22 pilot showed pack_cdrs can degrade "
             "well-resolved crystals catastrophically) scores the PDB "
             "exactly as loaded. 'pack_cdrs' (~10–30s/entry) repacks "
             "CDR + ±2 framework shell side chains. 'full' (~5–6 min/"
             "entry) does full-complex repack + FastRelax on CDR loops. "
             "Use 'full' only for paranoid runs on raw crystal PDBs "
             "where you suspect framework clashes.",
    )
    args = parser.parse_args()

    # Load entries — four modes:
    #   1. SAbDab TSV (--tsv): needs --pdb-dir
    #   2. ANDD CSV (--csv without gt_complex_id col): needs --pdb-dir
    #   3. AAPR manifest CSV (--csv with gt_complex_id col): no --pdb-dir needed
    #   4. Plain PDB directory (no --tsv, no --csv): needs --pdb-dir
    if args.tsv:
        if args.pdb_dir is None:
            logger.error("--tsv mode requires --pdb-dir.")
            sys.exit(2)
        entries = load_sabdab_entries(args.tsv, args.pdb_dir)
    elif args.csv:
        # Peek the header to decide ANDD vs AAPR.
        with open(args.csv) as fh:
            header = fh.readline()
        if "gt_complex_id" in header:
            logger.info("Detected AAPR manifest (gt_complex_id present): %s", args.csv)
            entries = load_aapr_entries(args.csv)
        else:
            if args.pdb_dir is None:
                logger.error("ANDD CSV mode requires --pdb-dir.")
                sys.exit(2)
            logger.info("Loading ANDD entries from CSV: %s", args.csv)
            entries = load_andd_entries(args.csv, args.pdb_dir)
    else:
        if args.pdb_dir is None:
            logger.error("PDB-directory mode requires --pdb-dir.")
            sys.exit(2)
        logger.info(
            "No --tsv or --csv provided; loading PDB files directly "
            "(chain='%s', antigen='%s').",
            args.chain, args.antigen_chain or "none",
        )
        entries = load_pdb_entries(
            args.pdb_dir, chain_id=args.chain, antigen_chain_id=args.antigen_chain
        )
    if not entries:
        logger.error("No valid entries found. Check paths.")
        sys.exit(1)

    if args.limit:
        entries = entries[: args.limit]
        logger.info("Limited to %d entries for testing.", args.limit)

    # Initialize judges
    biology_judge = BiologyJudge()
    biophysics_judge = BiophysicsJudge()
    physics_judge = PhysicsJudge(refinement_mode=args.refinement_mode)
    logger.info("Physics refinement mode: %s", args.refinement_mode)

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
            gt_complex_id=entry.get("gt_complex_id"),
            sample_idx=entry.get("sample_idx"),
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

    # ── Phase 2 (optional): theraprofnano biophysics scoring ──
    # Scores the VHH chain of each input PDB directly (no NB2 re-fold).
    # Sets candidate.pdb_filepath to the extracted monomer so the
    # Biology Judge SAPs the same coordinates the biophysics metrics
    # were computed on.
    if args.score_biophysics and phase1_survivors:
        logger.info("=" * 60)
        logger.info(
            "PHASE 2: theraprofnano scoring (%d candidates)",
            len(phase1_survivors),
        )
        logger.info("=" * 60)
        score_start = time.time()
        monomer_dir = output_path.parent / "vhh_monomers"

        n_scored = 0
        for candidate in phase1_survivors:
            source_pdb = candidate.complex_pdb_path or candidate.pdb_filepath
            if not source_pdb:
                logger.warning(
                    "  %s: no PDB on candidate — skipping.",
                    candidate.candidate_id,
                )
                continue
            try:
                result = score_biophysics_pdb(
                    complex_pdb_path=source_pdb,
                    nanobody_chain_id=candidate.nanobody_chain_id or "H",
                    candidate_id=candidate.candidate_id,
                    sequence=candidate.raw_sequence,
                    output_dir=monomer_dir,
                )
            except Exception:
                logger.exception(
                    "  %s: biophysics scoring failed.",
                    candidate.candidate_id,
                )
                continue

            candidate.psh_score = result.psh
            candidate.ppc_score = result.ppc
            candidate.pnc_score = result.pnc
            candidate.compactness = result.compactness
            candidate.cdr_length = result.cdr_length
            candidate.cdr3_length = result.cdr3_length
            # Re-point Biology Judge at the extracted monomer.
            candidate.pdb_filepath = result.pdb_path
            n_scored += 1

        score_elapsed = time.time() - score_start
        logger.info(
            "  Biophysics scoring complete: %d/%d in %.0fs (%.1fs/cand).",
            n_scored, len(phase1_survivors), score_elapsed,
            score_elapsed / max(len(phase1_survivors), 1),
        )
    elif not args.score_biophysics:
        logger.info(
            "Skipping biophysics scoring "
            "(use --score-biophysics to enable Biophysics Judge)."
        )

    # ── Phase 3: Multi-judge evaluation ──
    logger.info("=" * 60)
    logger.info("PHASE 3: Judge evaluation (%d entries)", total)
    logger.info("=" * 60)
    candidates: list[NanobodyCandidate] = []
    run_start = time.time()
    durations: list[float] = []

    # Every judge runs on every candidate — no short-circuit.  Each judge
    # emits its own verdict (pass / fail_* / skipped_* / error); `is_valid`
    # is the aggregate label only.
    for idx, candidate in enumerate(all_candidates, 1):
        entry_start = time.time()
        nb_chain = candidate.nanobody_chain_id or "H"
        ag_chains = candidate.antigen_chain_ids

        logger.info(
            "[%d/%d] %s (chain %s vs %s)",
            idx, total, candidate.candidate_id, nb_chain, ag_chains or "?",
        )

        # ── Biology Judge ──
        if candidate.biology_verdict == "fail_absolute":
            # Phase 1 already spoke (ANARCI parse failure) — leave it.
            logger.info("  Biology: %s (from Phase 1)", candidate.biology_verdict)
        elif candidate.pdb_filepath and Path(candidate.pdb_filepath).exists():
            structure = load_structure(candidate.pdb_filepath, candidate.candidate_id)
            biology_judge.evaluate(candidate, structure, chain_id=nb_chain)
            logger.info("  Biology: %s", candidate.biology_verdict)
        else:
            candidate.biology_verdict = "skipped_no_structure"
            logger.info("  Biology: skipped_no_structure (no PDB)")

        # ── Biophysics Judge ── (always runs; judge emits skipped_no_tnp)
        biophysics_judge.evaluate(candidate)
        logger.info("  Biophysics: %s", candidate.biophysics_verdict)

        # ── Physics Judge ── (always runs; judge emits skipped_no_antigen)
        if candidate.complex_pdb_path and ag_chains:
            ag_clean = ag_chains.replace(" ", "").replace("|", "")
            interface = f"{nb_chain}_{ag_clean}" if ag_clean else None

            if nb_chain.islower() or any(c.islower() for c in ag_clean):
                logger.warning(
                    "  Physics: skipped — lowercase chain ID(s) in '%s' "
                    "(known PyRosetta crash risk)", interface,
                )
                candidate.physics_verdict = "error"
            else:
                physics_judge.evaluate(
                    candidate,
                    complex_pdb_path=candidate.complex_pdb_path,
                    nanobody_chain_id=nb_chain,
                    interface=interface,
                )
                e_rep_str = f"{candidate.e_rep:.3f}" if candidate.e_rep is not None else "N/A"
                ce_str = (
                    f"{candidate.cdr_energy_per_res:.3f}"
                    if candidate.cdr_energy_per_res is not None
                    else "N/A"
                )
                logger.info(
                    "  Physics: %s (E_Rep=%s, E_cdr=%s REU/res)",
                    candidate.physics_verdict, e_rep_str, ce_str,
                )
        else:
            # Call the judge so it sets the explicit verdict itself.
            physics_judge.evaluate(candidate, complex_pdb_path=None, interface=None)
            logger.info("  Physics: %s", candidate.physics_verdict)

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
    if not args.score_biophysics:
        logger.info(
            "  (Biophysics skipped — rerun with --score-biophysics for full test)"
        )

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
        "cdr_energy_per_res",
    ]
    print("\n" + df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
