#!/usr/bin/env python3
"""Count distinct PDB-ATOM-derived sequences at the curated H-chain.

DiffAb trains on PDB ATOM-record sequences, not on the curated CSV's
Ab/Nano H_Chain AA. So the meaningful question for the broken seed42
finetune is not "how many CSV-distinct sequences are there" (220 per
the calibration doc) but "how many PDB-distinct VHH sequences did the
model actually see during training".

This script:
  1. Walks every manifest row.
  2. Opens the PDB and reads the ATOM-record sequence of the chain
     listed in Hchain (the chain DiffAb extracts for training).
  3. Counts unique sequences, reports the redundancy factor, and
     prints the top duplicated PDB-sequences (true structural
     duplicates, not metadata duplicates).

Decision shape:
  * PDB diversity ≈ 220  → CSV-based dedup is correct; many real
                            duplicates; the dedup plan applies as-is.
  * PDB diversity ≈ 400+ → CSV-based dedup is misleading; previous
                            seed42 saw diverse data; dedup harmless
                            but not critical; refinetune may be
                            unnecessary.
  * Somewhere between   → partial recovery; the right dedup key is
                            pdb_atom_sequence, not raw_sequence.

Usage:
    python scripts/diffab_ft/audit_pdb_atom_diversity.py
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def chain_atom_sequence(pdb_path: Path, chain_id: str) -> str:
    """Return the PDB ATOM-record sequence of one chain (joined across
    polypeptide breaks). Empty string if chain absent."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    ppb = PPBuilder()
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                seqs = [str(pp.get_sequence()) for pp in ppb.build_peptides(chain)]
                return "".join(seqs)
        break
    return ""


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--manifest-tsv",
        default="data/datasets/diffab_manifest.tsv",
        type=Path,
    )
    p.add_argument(
        "--pdb-dir",
        default="/projects/0/hpmlprjs/interns/krijn/ANDD_nano_dataset_IgLM/VHH_structures_post_diffab",
        type=Path,
    )
    p.add_argument(
        "--curated-csv",
        default="/projects/0/hpmlprjs/interns/krijn/ANDD_nano_dataset_IgLM/ANDD_VHH_curated_diffab.csv",
        type=Path,
        help="For optional CSV-vs-PDB diversity comparison.",
    )
    args = p.parse_args()

    mani = pd.read_csv(args.manifest_tsv, sep="\t")
    csv = pd.read_csv(args.curated_csv) if args.curated_csv.exists() else None

    seq_lookup_csv: dict[tuple[str, str], str] = {}
    if csv is not None:
        seq_lookup_csv = {
            (str(r["PDB_ID"]).strip().lower(), str(r["H_Chain Auth Asym ID"]).strip()):
                str(r["Ab/Nano H_Chain AA"]).strip()
            for _, r in csv.iterrows()
            if pd.notna(r.get("Ab/Nano H_Chain AA"))
        }

    pdb_seq_to_entries: dict[str, list[str]] = defaultdict(list)
    csv_seq_to_entries: dict[str, list[str]] = defaultdict(list)
    n_skipped = 0

    print(f"Extracting PDB ATOM sequences for {len(mani)} manifest entries...")
    for i, row in mani.iterrows():
        pdb_id = str(row["pdb"]).strip().lower()
        h_chain = str(row["Hchain"]).strip()
        entry_id = f"{pdb_id}_{h_chain}"
        pdb_path = args.pdb_dir / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            n_skipped += 1
            continue
        try:
            pdb_seq = chain_atom_sequence(pdb_path, h_chain)
        except Exception as exc:
            print(f"  WARN {pdb_id}_{h_chain}: {exc}", file=sys.stderr)
            n_skipped += 1
            continue
        if not pdb_seq:
            n_skipped += 1
            continue
        pdb_seq_to_entries[pdb_seq].append(entry_id)
        csv_seq = seq_lookup_csv.get((pdb_id, h_chain))
        if csv_seq:
            csv_seq_to_entries[csv_seq].append(entry_id)
        if (i + 1) % 50 == 0:
            print(f"  ...processed {i+1}/{len(mani)}")

    n_total = len(mani) - n_skipped
    n_unique_pdb = len(pdb_seq_to_entries)
    n_unique_csv = len(csv_seq_to_entries)

    print("\n" + "=" * 60)
    print("PDB-ATOM-sequence diversity at curated H-chain")
    print("=" * 60)
    print(f"manifest rows processed:           {n_total}")
    print(f"skipped (missing PDB or chain):    {n_skipped}")
    print()
    print(f"unique PDB-ATOM sequences:         {n_unique_pdb}")
    print(f"  → PDB redundancy factor:         {n_total / max(n_unique_pdb, 1):.2f}x")
    print(f"unique CSV sequences (for comparison): {n_unique_csv}")
    print(f"  → CSV redundancy factor:         {n_total / max(n_unique_csv, 1):.2f}x")
    print()

    # ── Top duplicated PDB sequences ────────────────────────────────
    pdb_sizes = sorted(((len(v), seq, v) for seq, v in pdb_seq_to_entries.items()),
                       key=lambda t: -t[0])
    n_with_dups = sum(1 for sz, _, _ in pdb_sizes if sz > 1)
    n_in_dup_groups = sum(sz for sz, _, _ in pdb_sizes if sz > 1)
    print(f"PDB sequences with >1 entry:       {n_with_dups}")
    print(f"entries in any PDB-dup group:      {n_in_dup_groups}  "
          f"({100*n_in_dup_groups/max(n_total,1):.1f}%)")

    print(f"\nTop 10 most-replicated PDB sequences:")
    print(f"  {'size':>5}  {'CDR3 tail (last 25aa)':25s}  examples")
    for sz, seq, ents in pdb_sizes[:10]:
        print(f"  {sz:5d}  {seq[-25:]:25s}  {ents[:3]}")

    # ── CSV vs PDB cluster agreement ────────────────────────────────
    if csv_seq_to_entries:
        # For each CSV-defined cluster, how many distinct PDB sequences does it contain?
        csv_cluster_pdb_diversity = []
        for csv_seq, ents in csv_seq_to_entries.items():
            if len(ents) <= 1:
                continue
            pdb_seqs_in_cluster = set()
            for eid in ents:
                # Find the pdb_seq for this entry
                for ps, es in pdb_seq_to_entries.items():
                    if eid in es:
                        pdb_seqs_in_cluster.add(ps)
                        break
            csv_cluster_pdb_diversity.append((len(ents), len(pdb_seqs_in_cluster), csv_seq))

        csv_cluster_pdb_diversity.sort(key=lambda t: -t[0])
        print(f"\nCSV-defined dup clusters: how many PDB sequences do they actually span?")
        print(f"  (if all entries in a CSV-cluster have the same PDB seq → true duplicate")
        print(f"   if entries span many PDB seqs → mislabeled, NOT real duplicates)")
        print(f"\n  {'CSV-cluster size':>16s}  {'distinct PDB seqs in cluster':>30s}  verdict")
        for csv_n, pdb_n, _ in csv_cluster_pdb_diversity[:10]:
            verdict = ("TRUE duplicates" if pdb_n == 1
                       else f"mislabeled ({pdb_n} different molecules)")
            print(f"  {csv_n:16d}  {pdb_n:30d}  {verdict}")

    # ── Verdict ────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    if n_unique_pdb >= 400:
        print("HIGH PDB diversity (~400+). The previous seed42 finetune saw very")
        print("diverse training data despite CSV metadata redundancy. The 'duplication")
        print("problem' was largely a CSV-side artifact. Re-finetune may be optional.")
    elif n_unique_pdb >= 300:
        print("MODERATE PDB diversity (300-400). Some real duplication, but less")
        print("severe than CSV suggested. Dedup-by-pdb_atom_sequence is the right key;")
        print("dedup-by-CSV-raw_sequence over-aggressively drops mislabeled-but-distinct")
        print("entries.")
    else:
        print("LOW PDB diversity (<300). Real structural duplication exists. Dedup")
        print("is still needed. PDB-based and CSV-based dedup will give similar")
        print("answers for these heavy-redundancy entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
