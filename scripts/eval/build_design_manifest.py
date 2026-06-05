#!/usr/bin/env python3
"""Build an AAPR-format candidate manifest from a design-eval PDB tree.

Walks ``<pdb_root>/<entry_id>/<cdr>/sample_NNNN.pdb`` produced by
``scripts/diffab_ft/evaluate.py --mode design --save-pdbs ...``, parses
the heavy-chain sequence from each PDB, looks up antigen-chain metadata
from ``data/datasets/diffab_manifest.tsv``, and emits a CSV consumed by
``scripts/test_sabdab_judges.py`` in its AAPR mode (auto-detected via
the ``gt_complex_id`` column).

The ``candidate_id`` encodes ``<variant>__<test_set>__<entry>__<cdr>__s<NNNN>``
so that the master-parquet assembly (Brief 11 §3 Step 5) can decode
variant / test_set / entry_id / cdr / sample_idx back out without any
filename-inference hacks. ``gt_complex_id`` is the bare entry_id so the
AAPR-mode loader can find per-row metadata for scoring.

Usage::

    python scripts/eval/build_design_manifest.py \\
        --pdb-root      runs/vhh_ft/seed42_jfix/eval_test_pdbs \\
        --manifest-tsv  data/datasets/diffab_manifest.tsv \\
        --output        data/eval/manifests/seed42_jfix_oldtest.csv \\
        --variant-tag   seed42_jfix \\
        --test-set-tag  oldtest

Brief 11 (TNP + Rosetta re-report on final design samples) — Phase B.
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections import Counter
from pathlib import Path

from Bio.PDB import PDBParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("build_design_manifest")

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

SAMPLE_RE = re.compile(r"^sample_(\d+)\.pdb$")
EXPECTED_CDRS = {"H1", "H2", "H3"}


def _load_antigen_map(manifest_tsv: Path) -> dict[str, dict[str, str]]:
    """Parse diffab_manifest.tsv → {pdb_id: {Hchain, antigen_chain}}."""
    out: dict[str, dict[str, str]] = {}
    with open(manifest_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            out[row["pdb"]] = {
                "Hchain": row.get("Hchain", "") or "",
                "antigen_chain": row.get("antigen_chain", "") or "",
            }
    logger.info("Loaded antigen map for %d PDBs from %s", len(out), manifest_tsv)
    return out


def _extract_heavy_seq(pdb_path: Path, hchain_id: str, parser: PDBParser) -> str:
    """Read the heavy chain sequence from ``pdb_path``. Returns '' on miss."""
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = next(iter(structure))
    if hchain_id not in [c.id for c in model]:
        return ""
    return "".join(
        THREE_TO_ONE.get(res.get_resname(), "X")
        for res in model[hchain_id].get_residues()
        if res.id[0] == " "
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pdb-root", required=True, type=Path,
                    help="Design-eval PDB tree root: <root>/<entry>/<cdr>/sample_NNNN.pdb")
    ap.add_argument("--manifest-tsv", required=True, type=Path,
                    help="diffab_manifest.tsv for Hchain + antigen_chain lookup.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output AAPR-format CSV.")
    ap.add_argument("--variant-tag", required=True,
                    help="Variant label (e.g., seed42_jfix). Encoded in candidate_id.")
    ap.add_argument("--test-set-tag", required=True,
                    help="Test-set label (e.g., oldtest, newtest). Encoded in candidate_id.")
    args = ap.parse_args()

    if not args.pdb_root.exists():
        logger.error("PDB root not found: %s", args.pdb_root)
        return 2
    if not args.manifest_tsv.exists():
        logger.error("Manifest TSV not found: %s", args.manifest_tsv)
        return 2

    antigen_map = _load_antigen_map(args.manifest_tsv)
    parser = PDBParser(QUIET=True)

    rows: list[dict] = []
    n_skipped_no_chain = 0
    n_skipped_no_meta = 0

    for entry_dir in sorted(args.pdb_root.iterdir()):
        if not entry_dir.is_dir():
            continue
        entry_id = entry_dir.name  # e.g., "7f5h_C"
        pdb_id, _, hchain_from_entry = entry_id.partition("_")
        meta = antigen_map.get(pdb_id, {})
        hchain = meta.get("Hchain") or hchain_from_entry
        ag_chain = meta.get("antigen_chain", "")
        if not meta:
            n_skipped_no_meta += 1
            logger.warning("entry %s: PDB %s not in manifest TSV — using entry-derived Hchain '%s', ag_chain ''",
                           entry_id, pdb_id, hchain)

        for cdr_dir in sorted(entry_dir.iterdir()):
            if not cdr_dir.is_dir():
                continue
            cdr = cdr_dir.name
            if cdr not in EXPECTED_CDRS:
                logger.warning("Skipping unexpected CDR dir: %s", cdr_dir)
                continue
            for pdb_path in sorted(cdr_dir.glob("sample_*.pdb")):
                m = SAMPLE_RE.match(pdb_path.name)
                if not m:
                    continue
                sample_idx = int(m.group(1))
                seq = _extract_heavy_seq(pdb_path, hchain, parser)
                if not seq:
                    n_skipped_no_chain += 1
                    logger.warning(
                        "entry %s cdr %s sample %d: heavy chain '%s' missing in %s",
                        entry_id, cdr, sample_idx, hchain, pdb_path,
                    )
                    continue
                candidate_id = (
                    f"{args.variant_tag}__{args.test_set_tag}__"
                    f"{entry_id}__{cdr}__s{sample_idx:04d}"
                )
                rows.append({
                    "candidate_id":      candidate_id,
                    "gt_complex_id":     entry_id,
                    "sample_idx":        sample_idx,
                    "raw_sequence":      seq,
                    "complex_pdb_path":  str(pdb_path.resolve()),
                    "nanobody_chain_id": hchain,
                    "antigen_chain_ids": ag_chain,
                    "mask_strategy":     f"SINGLE_CDR_{cdr}",
                    "cdrs_masked":       cdr,
                    "temperature":       "",
                    "checkpoint_id":     args.variant_tag,
                    "seed":              42,
                })

    if not rows:
        logger.error("No rows produced from %s — check pdb_root + manifest.", args.pdb_root)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(
        "Wrote %d rows to %s (skipped %d for missing chain, %d for missing meta)",
        len(rows), args.output, n_skipped_no_chain, n_skipped_no_meta,
    )

    cdr_counts = Counter(r["candidate_id"].split("__")[3] for r in rows)
    entry_counts = Counter(r["gt_complex_id"] for r in rows)
    logger.info("CDR breakdown: %s", dict(cdr_counts))
    logger.info(
        "Entries: %d, samples/entry min=%d max=%d (expect 12 for 3 CDRs × 4 samples)",
        len(entry_counts), min(entry_counts.values()), max(entry_counts.values()),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
