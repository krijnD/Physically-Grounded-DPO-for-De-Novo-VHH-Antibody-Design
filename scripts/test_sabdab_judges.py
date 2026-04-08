#!/usr/bin/env python3
"""Test all three judges on SAbDab nanobody data — no TNP folding required.

Loads SAbDab entries (nanobody+antigen complex PDBs), extracts sequences,
and runs Phase 1 (sequence filter) + all three judges directly.

Biophysics Judge will be skipped (no TNP metrics) but is called to
verify it handles missing data gracefully.

Usage on Snellius:
    python scripts/test_sabdab_judges.py \
        --tsv /projects/0/hpmlprjs/interns/krijn/sabdab_nano_dataset_IgLM/sabdab_nano_summary.tsv \
        --pdb-dir /projects/0/hpmlprjs/interns/krijn/sabdab_nano_dataset_IgLM/filtered_vhh_pdbs \
        --output data/results/sabdab_judge_test.parquet \
        --limit 5
"""

import argparse
import logging
import sys
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
from src.physics_judge.judge import PhysicsJudge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("test_sabdab_judges")


def run_judges_on_entry(
    entry: dict,
    biology_judge: BiologyJudge,
    biophysics_judge: BiophysicsJudge,
    physics_judge: PhysicsJudge,
) -> NanobodyCandidate:
    """Run all pipeline phases on a single SAbDab entry."""

    candidate = NanobodyCandidate(
        candidate_id=entry["candidate_id"],
        raw_sequence=entry["raw_sequence"],
        pdb_filepath=entry.get("pdb_filepath"),
        complex_pdb_path=entry.get("complex_pdb_path"),
        nanobody_chain_id=entry.get("nanobody_chain_id"),
        antigen_chain_ids=entry.get("antigen_chain_ids"),
    )

    nb_chain = candidate.nanobody_chain_id or "H"
    ag_chains = candidate.antigen_chain_ids

    logger.info(
        "Processing %s (chain %s vs %s) — seq length %d",
        candidate.candidate_id,
        nb_chain,
        ag_chains or "?",
        len(candidate.raw_sequence),
    )

    # ── Phase 1: Sequence pre-filter (1D only) ──
    annotate_and_filter(candidate)
    logger.info(
        "  Phase 1: %s (flags: %s)",
        "REJECTED" if not candidate.is_valid else "ok",
        candidate.biology_flags or "none",
    )

    if not candidate.is_valid:
        return candidate

    # ── Biology Judge ──
    if candidate.pdb_filepath and Path(candidate.pdb_filepath).exists():
        structure = load_structure(candidate.pdb_filepath, candidate.candidate_id)
        biology_judge.evaluate(candidate, structure, chain_id=nb_chain)
        logger.info("  Biology: %s", candidate.biology_verdict or "skipped")
    else:
        logger.info("  Biology: skipped (no PDB)")

    if not candidate.is_valid:
        return candidate

    # ── Biophysics Judge (will skip — no TNP metrics) ──
    biophysics_judge.evaluate(candidate)
    logger.info("  Biophysics: %s", candidate.biophysics_verdict or "skipped (no TNP metrics)")

    if not candidate.is_valid:
        return candidate

    # ── Physics Judge ──
    if candidate.complex_pdb_path and ag_chains:
        # SAbDab uses "A | C | B" format; PyRosetta needs "ACB"
        ag_clean = ag_chains.replace(" ", "").replace("|", "")
        interface = f"{nb_chain}_{ag_clean}"
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
        logger.info(
            "  Physics: skipped (no complex PDB or no antigen chain info)"
        )

    return candidate


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

    # Run all judges on each entry
    candidates: list[NanobodyCandidate] = []
    for entry in entries:
        candidate = run_judges_on_entry(
            entry, biology_judge, biophysics_judge, physics_judge
        )
        candidates.append(candidate)

    # Summarize results
    df = pd.DataFrame([c.to_dict() for c in candidates])

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY: %d entries processed", len(df))
    logger.info("-" * 60)

    # Phase 1 results
    absolute_fails = df[df["biology_verdict"] == "fail_absolute"]
    logger.info("Phase 1 absolute rejects (W47): %d", len(absolute_fails))

    # Biology Judge
    bio_verdicts = df["biology_verdict"].value_counts()
    logger.info("Biology verdicts: %s", dict(bio_verdicts))

    # Biophysics (expect all None/skipped)
    bp_verdicts = df["biophysics_verdict"].value_counts(dropna=False)
    logger.info("Biophysics verdicts: %s", dict(bp_verdicts))

    # Physics Judge
    phys_verdicts = df["physics_verdict"].value_counts(dropna=False)
    logger.info("Physics verdicts: %s", dict(phys_verdicts))

    # Overall pass/fail
    overall_valid = df["is_valid"].sum()
    logger.info(
        "Overall: %d pass, %d fail", overall_valid, len(df) - overall_valid
    )
    logger.info("=" * 60)

    # Save results (Parquet preferred, CSV fallback if pyarrow missing)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(output_path, index=False)
        logger.info("Results written to %s", output_path)
    except ImportError:
        csv_path = output_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        logger.warning(
            "pyarrow not installed — wrote CSV instead: %s", csv_path
        )

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
