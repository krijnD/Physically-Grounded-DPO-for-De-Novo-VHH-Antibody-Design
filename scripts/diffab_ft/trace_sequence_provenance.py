#!/usr/bin/env python3
"""Decide whether the bad PDB↔CSV sequences come from the upstream
ANDD Excel or from a filtering bug in our data-prep pipeline.

For each "bad" entry flagged by probe_chain_id_mismatch.py (i.e. the
curated CSV's Ab/Nano H_Chain AA doesn't match any chain in the PDB),
walk back through every intermediate CSV in the pipeline:

    Excel  →  ANDD_VHH_with_structure.csv
           →  ANDD_VHH_with_structure_post_diffab.csv
           →  ANDD_VHH_curated_diffab.csv

If the sequence is IDENTICAL at every stage → filtering is clean,
the Excel itself has wrong sequence-to-PDB mappings (NOT FIXABLE).

If the sequence CHANGES between two stages → that filter introduced
the bug (FIXABLE by fixing the script).

Usage:
    python scripts/diffab_ft/trace_sequence_provenance.py
    python scripts/diffab_ft/trace_sequence_provenance.py --pdb-ids 7ept 7b2m
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def percent_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return 100 * sum(1 for x, y in zip(a[:n], b[:n]) if x == y) / n


def all_pdb_chains(pdb_path: Path) -> dict[str, str]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    ppb = PPBuilder()
    out: dict[str, str] = {}
    for model in structure:
        for chain in model:
            seqs = [str(pp.get_sequence()) for pp in ppb.build_peptides(chain)]
            if seqs:
                out[chain.id] = "".join(seqs)
        break
    return out


def lookup_seq(df: pd.DataFrame, pdb_id: str, seq_col: str) -> str | None:
    """Return the heavy-chain sequence for a PDB ID from a DataFrame, or None."""
    if "PDB_ID" not in df.columns or seq_col not in df.columns:
        return None
    match = df[df["PDB_ID"].astype(str).str.lower() == pdb_id.lower()]
    if len(match) == 0:
        return None
    seq = match.iloc[0][seq_col]
    return str(seq).strip() if pd.notna(seq) else None


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    base = "/projects/0/hpmlprjs/interns/krijn/ANDD_nano_dataset_IgLM"
    p.add_argument("--excel",
                   default=f"{base}/Antibody and Nanobody Design Dataset (ANDD)_v2.xlsx",
                   type=Path)
    p.add_argument("--with-structure-csv",
                   default=f"{base}/ANDD_VHH_with_structure.csv",
                   type=Path)
    p.add_argument("--post-cutoff-csv",
                   default=f"{base}/ANDD_VHH_with_structure_post_diffab.csv",
                   type=Path)
    p.add_argument("--curated-csv",
                   default=f"{base}/ANDD_VHH_curated_diffab.csv",
                   type=Path)
    p.add_argument("--pdb-dir",
                   default=f"{base}/VHH_structures_post_diffab",
                   type=Path)
    p.add_argument(
        "--pdb-ids", nargs="*", default=None,
        help="Specific PDB IDs to trace. Default: auto-detect "
             "from probe_chain_id_mismatch (uses representative bad entries).",
    )
    p.add_argument(
        "--seq-col", default="Ab/Nano H_Chain AA",
        help="Sequence column name (default: 'Ab/Nano H_Chain AA').",
    )
    args = p.parse_args()

    # ── Pick PDB IDs to trace ──────────────────────────────────────
    if args.pdb_ids:
        pdb_ids = [s.lower() for s in args.pdb_ids]
        print(f"Tracing user-specified PDB IDs: {pdb_ids}")
    else:
        # Sensible defaults: representative known-bad entries from
        # the probe run that ID'd this issue.
        pdb_ids = ["7ept", "7b2m", "7b2p", "7f5g", "7nbb", "7qbd"]
        print(f"Tracing default representative bad entries: {pdb_ids}")

    # ── Load all stage CSVs (lazy: only what exists) ──────────────
    stages: list[tuple[str, pd.DataFrame | None]] = []

    if args.excel.exists():
        excel = pd.read_excel(args.excel)
        stages.append(("Excel (source)", excel))
    else:
        print(f"WARN: Excel not found at {args.excel}", file=sys.stderr)
        stages.append(("Excel (source)", None))

    for label, path in [
        ("ANDD_VHH_with_structure.csv", args.with_structure_csv),
        ("ANDD_VHH_with_structure_post_diffab.csv", args.post_cutoff_csv),
        ("ANDD_VHH_curated_diffab.csv (final)", args.curated_csv),
    ]:
        if path.exists():
            stages.append((label, pd.read_csv(path)))
        else:
            print(f"WARN: not found: {path}", file=sys.stderr)
            stages.append((label, None))

    # ── Trace each PDB ID through the pipeline ─────────────────────
    print("\n" + "=" * 78)
    print("Per-PDB sequence trace")
    print("=" * 78)

    any_filter_change = False
    any_excel_pdb_mismatch = False

    for pdb_id in pdb_ids:
        print(f"\n── PDB {pdb_id.upper()} ─────────────────────────────────────")
        seqs_seen: list[tuple[str, str]] = []
        prev_seq: str | None = None
        for label, df in stages:
            if df is None:
                print(f"  {label:50s}  [skipped — file missing]")
                continue
            seq = lookup_seq(df, pdb_id, args.seq_col)
            if seq is None:
                print(f"  {label:50s}  [not present in this stage]")
                continue
            tag = ""
            if prev_seq is not None:
                if seq == prev_seq:
                    tag = " (same as prev)"
                else:
                    tag = " ⚠ CHANGED from prev stage"
                    any_filter_change = True
            print(f"  {label:50s}  len={len(seq):3d}  hash={hash(seq) & 0xFFFFFF:06x}{tag}")
            seqs_seen.append((label, seq))
            prev_seq = seq

        # Compare Excel seq vs actual PDB chains
        if seqs_seen and args.pdb_dir.exists():
            excel_label, excel_seq = seqs_seen[0]
            pdb_path = args.pdb_dir / f"{pdb_id}.pdb"
            if pdb_path.exists():
                chains = all_pdb_chains(pdb_path)
                best_chain, best_pid = "", 0.0
                for cid, cseq in chains.items():
                    pid = percent_identity(excel_seq, cseq)
                    if pid > best_pid:
                        best_chain, best_pid = cid, pid
                print(f"  PDB chains in {pdb_id}.pdb: {sorted(chains.keys())}")
                print(f"  Best chain match for Excel sequence: "
                      f"chain '{best_chain}' at {best_pid:.1f}% identity")
                if best_pid < 80:
                    any_excel_pdb_mismatch = True
                    print(f"  → Excel sequence does NOT appear in PDB {pdb_id}")
            else:
                print(f"  PDB {pdb_id}.pdb not found in {args.pdb_dir}")

    # ── Verdict ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if any_filter_change:
        print("FILTER BUG: at least one PDB had its sequence CHANGE between stages.")
        print("→ The filtering pipeline introduced the mismatch. Inspect the marked")
        print("  stage above to find the bug. Re-running the corrected filter would")
        print("  recover the data.")
        return 1
    elif any_excel_pdb_mismatch:
        print("EXCEL SOURCE ISSUE: the sequence is consistent at every filter stage,")
        print("but the Excel-provided sequence doesn't appear in the actual PDB.")
        print("→ The ANDD Excel itself has wrong sequence-to-PDB mappings for these")
        print("  entries. Re-running curation will NOT fix it. Options:")
        print("    (a) drop these entries (clean_manifest.py default)")
        print("    (b) re-source sequences from PDB SEQRES directly")
        print("    (c) check whether ANDD published an erratum or a newer version")
        return 2
    else:
        print("UNEXPECTED: sequences trace cleanly AND match the PDB. The original")
        print("audit findings may be a measurement artifact — re-check audit logic.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
