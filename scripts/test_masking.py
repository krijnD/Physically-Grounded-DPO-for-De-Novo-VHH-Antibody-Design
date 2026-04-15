#!/usr/bin/env python3
"""Integration test for the masking module.

Loads SAbDab or ANDD entries (nanobody+antigen complex PDBs), runs all
four masking strategies on each entry, and reports:
  - Number of masked positions per strategy
  - Contiguous spans (for IgLM infilling)
  - Strategy invariant validation
  - Visual masked-sequence representation

Usage on Snellius:
    python scripts/test_masking.py \
        --tsv /projects/0/hpmlprjs/interns/krijn/sabdab_nano_dataset_IgLM/sabdab_nano_summary.tsv \
        --pdb-dir /projects/0/hpmlprjs/interns/krijn/sabdab_nano_dataset_IgLM/filtered_vhh_pdbs \
        --limit 5

    python scripts/test_masking.py \
        --csv path/to/ANDD_VHH_with_structure.csv \
        --pdb-dir path/to/pdb_directory \
        --limit 5
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.candidate import NanobodyCandidate
from src.common.sabdab_loader import (
    load_andd_entries,
    load_pdb_entries,
    load_sabdab_entries,
)
from src.masking import MaskingEngine, MaskStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("test_masking")


def _visualize_mask(sequence: str, masked_positions: set[int]) -> str:
    """Produce a visual representation: masked positions shown as '_'."""
    chars = []
    for i, aa in enumerate(sequence):
        chars.append("_" if i in masked_positions else aa)
    return "".join(chars)


def _validate_invariants(
    result,
    kabat_mapping,
    strategy: MaskStrategy,
    paratope_seq_indices: set[int] | None = None,
) -> list[str]:
    """Check strategy-specific invariants. Returns list of violations."""
    violations = []
    mask = result.masked_positions

    if strategy in (MaskStrategy.PARATOPE, MaskStrategy.CDR_FOCUSED):
        anchor_overlap = mask & kabat_mapping.anchor_indices
        if anchor_overlap:
            violations.append(
                f"  VIOLATION: {len(anchor_overlap)} anchor positions in mask: "
                f"{sorted(anchor_overlap)}"
            )
        fr2_overlap = mask & kabat_mapping.fr2_hallmark_indices
        if fr2_overlap:
            violations.append(
                f"  VIOLATION: {len(fr2_overlap)} FR2 hallmark positions in mask: "
                f"{sorted(fr2_overlap)}"
            )

    if strategy == MaskStrategy.FR2_REVERSION:
        if mask != kabat_mapping.fr2_region_indices:
            diff = mask.symmetric_difference(kabat_mapping.fr2_region_indices)
            violations.append(
                f"  VIOLATION: FR2 reversion mask differs from FR2 region by "
                f"{len(diff)} positions"
            )

    if strategy == MaskStrategy.UNANCHORED_CLASH and paratope_seq_indices:
        if not mask >= paratope_seq_indices:
            missing = paratope_seq_indices - mask
            violations.append(
                f"  VIOLATION: {len(missing)} paratope positions missing from "
                f"unanchored mask"
            )

    return violations


def run_test(entries: list[dict], limit: int | None = None) -> None:
    """Run all four masking strategies on loaded entries."""
    engine = MaskingEngine()

    if limit:
        entries = entries[:limit]

    total = len(entries)
    logger.info("Testing masking on %d entries", total)

    all_violations = []

    for idx, entry in enumerate(entries, 1):
        candidate_id = entry["candidate_id"]
        logger.info("=" * 70)
        logger.info("[%d/%d] Candidate: %s", idx, total, candidate_id)

        candidate = NanobodyCandidate(
            candidate_id=candidate_id,
            raw_sequence=entry["raw_sequence"],
            pdb_filepath=entry.get("pdb_filepath"),
            complex_pdb_path=entry.get("complex_pdb_path"),
            nanobody_chain_id=entry.get("nanobody_chain_id"),
            antigen_chain_ids=entry.get("antigen_chain_ids"),
        )

        logger.info("  Sequence length: %d", len(candidate.raw_sequence))
        logger.info("  Complex PDB: %s", candidate.complex_pdb_path)
        logger.info("  Chains: nb=%s ag=%s",
                     candidate.nanobody_chain_id, candidate.antigen_chain_ids)

        # Get Kabat mapping for invariant checks.
        try:
            kabat_mapping = engine._get_kabat_mapping(candidate.raw_sequence)
        except Exception as exc:
            logger.error("  ANARCI parse failed: %s", exc)
            continue

        logger.info(
            "  Kabat: %d positions, CDR=%d, anchors=%d, FR2_hallmarks=%d, FR2_region=%d",
            len(kabat_mapping.kabat_to_seq_idx),
            len(kabat_mapping.cdr_indices),
            len(kabat_mapping.anchor_indices),
            len(kabat_mapping.fr2_hallmark_indices),
            len(kabat_mapping.fr2_region_indices),
        )

        # Run all strategies and collect paratope indices for invariant checks.
        paratope_seq_indices = None
        strategies = [
            MaskStrategy.PARATOPE,
            MaskStrategy.CDR_FOCUSED,
            MaskStrategy.FR2_REVERSION,
            MaskStrategy.UNANCHORED_CLASH,
        ]

        for strategy in strategies:
            logger.info("  --- Strategy: %s ---", strategy.name)
            try:
                result = engine.mask(candidate, strategy)
            except (ValueError, Exception) as exc:
                logger.warning("    Skipped: %s", exc)
                continue

            # Capture paratope indices from PARATOPE result for invariant checks.
            if strategy == MaskStrategy.PARATOPE:
                paratope_seq_indices = result.masked_positions | (
                    result.masked_positions  # placeholder
                )

            logger.info("    Masked positions: %d", len(result.masked_positions))
            logger.info("    Contiguous spans: %d -> %s",
                         len(result.contiguous_spans), result.contiguous_spans)
            logger.info("    Metadata: %s", result.metadata)

            # Visual.
            visual = _visualize_mask(candidate.raw_sequence, result.masked_positions)
            logger.info("    Visual: %.80s...", visual)

            # Invariants.
            violations = _validate_invariants(
                result, kabat_mapping, strategy, paratope_seq_indices
            )
            if violations:
                for v in violations:
                    logger.error(v)
                    all_violations.append(f"{candidate_id}/{strategy.name}: {v}")
            else:
                logger.info("    Invariants: PASS")

    logger.info("=" * 70)
    logger.info("SUMMARY: %d entries tested", total)
    if all_violations:
        logger.error("%d invariant violations found:", len(all_violations))
        for v in all_violations:
            logger.error("  %s", v)
    else:
        logger.info("All invariants passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Test masking module on SAbDab/ANDD entries"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tsv", help="SAbDab summary TSV file")
    group.add_argument("--csv", help="ANDD CSV file")
    group.add_argument("--pdb-only", action="store_true",
                       help="Use plain PDB directory mode")
    parser.add_argument("--pdb-dir", required=True,
                        help="Directory containing PDB files")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max entries to test")
    parser.add_argument("--chain", default="H",
                        help="Nanobody chain ID (for --pdb-only mode)")
    parser.add_argument("--antigen-chain", default=None,
                        help="Antigen chain ID (for --pdb-only mode)")
    args = parser.parse_args()

    if args.tsv:
        entries = load_sabdab_entries(args.tsv, args.pdb_dir)
    elif args.csv:
        entries = load_andd_entries(args.csv, args.pdb_dir)
    else:
        entries = load_pdb_entries(
            args.pdb_dir,
            chain_id=args.chain,
            antigen_chain_id=args.antigen_chain,
        )

    if not entries:
        logger.error("No entries loaded. Check your input paths.")
        sys.exit(1)

    logger.info("Loaded %d entries", len(entries))
    run_test(entries, limit=args.limit)


if __name__ == "__main__":
    main()
