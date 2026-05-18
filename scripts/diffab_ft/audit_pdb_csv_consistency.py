#!/usr/bin/env python3
"""Audit each manifest entry for sequence consistency between the curated
ANDD CSV and the actual PDB structure on disk.

Background: the curated CSV column ``Ab/Nano H_Chain AA`` is the
construct-level VHH sequence. The PDB ATOM records may differ — flexible
loops can be unresolved (Biopython's PPBuilder drops them), expression
tags may or may not be in the structure, chain IDs may differ, and the
crystallized construct sometimes carries 1-2 point mutations vs the
canonical sequence.

DiffAb trains on (sequence, structure) pairs. Large CSV/PDB drift means
the model would learn the wrong sequence-structure association. This
script tallies the drift so you can decide whether to proceed to a
12-18h finetune.

Output buckets:
  * missing CSV sequence       — manifest row exists but CSV has no AA
  * missing/unreadable PDB     — file absent or chain not extractable
  * length mismatch (>5 aa)    — significant truncation or extra residues
  * seq mismatch (same length) — point mutations in the construct

Decision rule (suggested):
  * <10 length mismatches  → ignore, proceed
  * 10-30                  → glance at the first 10, check for a systematic cause
  * >30                    → pause; ~7%+ of dataset has structural drift

Usage:
    python scripts/diffab_ft/audit_pdb_csv_consistency.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.sabdab_loader import extract_chain_sequence  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--curated-csv",
        default="/projects/0/hpmlprjs/interns/krijn/ANDD_VHH_curated_diffab.csv",
        type=Path,
        help="Curated ANDD CSV (provides Ab/Nano H_Chain AA).",
    )
    p.add_argument(
        "--manifest-tsv",
        default="data/datasets/diffab_manifest.tsv",
        type=Path,
        help="DiffAb manifest TSV (output of prepare_manifest.py).",
    )
    p.add_argument(
        "--pdb-dir",
        default="/projects/0/hpmlprjs/interns/krijn/ANDD_nano_dataset_IgLM/VHH_structures_post_diffab",
        type=Path,
        help="Directory containing PDB files (one per pdb_id, lowercase).",
    )
    p.add_argument(
        "--len-tolerance",
        type=int,
        default=5,
        help="Length-diff (aa) treated as 'mismatch' (default: 5).",
    )
    p.add_argument(
        "--max-print",
        type=int,
        default=10,
        help="How many mismatching entries to print per bucket (default: 10).",
    )
    args = p.parse_args()

    for path in (args.curated_csv, args.manifest_tsv, args.pdb_dir):
        if not path.exists():
            print(f"ERROR: not found: {path}", file=sys.stderr)
            return 1

    csv = pd.read_csv(args.curated_csv)
    mani = pd.read_csv(args.manifest_tsv, sep="\t")

    seq_lookup: dict[tuple[str, str], str] = {
        (str(r["PDB_ID"]).strip().lower(), str(r["H_Chain Auth Asym ID"]).strip()):
            str(r["Ab/Nano H_Chain AA"]).strip()
        for _, r in csv.iterrows()
        if pd.notna(r.get("Ab/Nano H_Chain AA"))
    }

    n_missing_pdb = 0
    n_missing_csv_seq = 0
    n_len_mismatch = 0
    n_seq_mismatch = 0
    n_ok = 0
    len_mismatches: list[tuple[str, str, int, int]] = []
    seq_mismatches: list[tuple[str, str, int]] = []

    for _, row in mani.iterrows():
        pdb_id = str(row["pdb"]).strip().lower()
        h_chain = str(row["Hchain"]).strip()
        csv_seq = seq_lookup.get((pdb_id, h_chain))

        if csv_seq is None:
            n_missing_csv_seq += 1
            continue

        pdb_path = args.pdb_dir / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            pdb_path = args.pdb_dir / f"{pdb_id.upper()}.pdb"
            if not pdb_path.exists():
                n_missing_pdb += 1
                continue

        pdb_seq = extract_chain_sequence(str(pdb_path), h_chain)
        if not pdb_seq:
            n_missing_pdb += 1
            continue

        len_diff = abs(len(csv_seq) - len(pdb_seq))
        if len_diff > args.len_tolerance:
            n_len_mismatch += 1
            len_mismatches.append((pdb_id, h_chain, len(csv_seq), len(pdb_seq)))
        elif csv_seq != pdb_seq:
            n_seq_mismatch += 1
            # Count residue-level diffs (over the common prefix length)
            min_len = min(len(csv_seq), len(pdb_seq))
            n_residue_diffs = sum(
                1 for a, b in zip(csv_seq[:min_len], pdb_seq[:min_len]) if a != b
            )
            seq_mismatches.append((pdb_id, h_chain, n_residue_diffs))
        else:
            n_ok += 1

    # ── Report ──────────────────────────────────────────────────────
    total = len(mani)
    print("=" * 60)
    print(f"PDB ↔ CSV consistency audit (len-tolerance = {args.len_tolerance} aa)")
    print("=" * 60)
    print(f"manifest rows:                {total}")
    print(f"  fully consistent:           {n_ok:4d}  ({100*n_ok/total:5.1f}%)")
    print(f"  missing CSV sequence:       {n_missing_csv_seq:4d}  "
          f"({100*n_missing_csv_seq/total:5.1f}%)")
    print(f"  missing/unreadable PDB:     {n_missing_pdb:4d}  "
          f"({100*n_missing_pdb/total:5.1f}%)")
    print(f"  length mismatch (>{args.len_tolerance} aa):     {n_len_mismatch:4d}  "
          f"({100*n_len_mismatch/total:5.1f}%)")
    print(f"  seq mismatch (same length): {n_seq_mismatch:4d}  "
          f"({100*n_seq_mismatch/total:5.1f}%)")

    if len_mismatches:
        print(f"\nFirst {min(args.max_print, len(len_mismatches))} length mismatches:")
        for pdb, h, cl, pl in len_mismatches[: args.max_print]:
            print(f"  {pdb}_{h}: csv_len={cl}  pdb_len={pl}  (Δ={cl-pl:+d})")
    if seq_mismatches:
        print(f"\nFirst {min(args.max_print, len(seq_mismatches))} same-length seq mismatches:")
        for pdb, h, n_diff in seq_mismatches[: args.max_print]:
            print(f"  {pdb}_{h}: {n_diff} residue diff{'s' if n_diff != 1 else ''}")

    # ── Decision hint ──────────────────────────────────────────────
    print("\n" + "-" * 60)
    if n_len_mismatch + n_seq_mismatch < 10:
        print("OK low drift — safe to proceed to Phase 2 (cluster_split.py).")
        return 0
    elif n_len_mismatch + n_seq_mismatch < 30:
        print("MODERATE drift — glance at the listing above. If the cause looks "
              "systematic (e.g. all cryo-EM unresolved loops), proceed.")
        return 0
    else:
        print("HIGH drift — pause and investigate before training. "
              "Consider tightening curation or excluding the worst entries.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
