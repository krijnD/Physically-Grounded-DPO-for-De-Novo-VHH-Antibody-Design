#!/usr/bin/env python3
"""Probe major PDB↔CSV sequence mismatches by checking every chain.

The Phase 1 audit (audit_pdb_csv_consistency.py) flagged ~50 manifest
entries where the PDB chain at the curated H_Chain Auth Asym ID is
essentially a different protein than the CSV says (100+ residue diffs
out of ~120). This script tests whether the actual VHH is under a
different chain ID — i.e. an ANDD-curation bug — by scanning all
chains in each suspect PDB and reporting the best sequence match.

For each "major mismatch" entry:
  1. List every chain ID in the PDB file
  2. Extract the sequence of each
  3. Compare to the CSV's Ab/Nano H_Chain AA
  4. Report the best match (chain ID, % identity, length)

Output buckets:
  * "curated correct"        — best match IS the curated chain
  * "swapped to other chain" — best match is a different chain
                                (likely a curation bug)
  * "no good match in PDB"   — no chain in the PDB matches the CSV
                                (PDB and CSV are about different molecules)

Usage:
    python scripts/diffab_ft/probe_chain_id_mismatch.py
    python scripts/diffab_ft/probe_chain_id_mismatch.py --max-entries 5
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def all_chain_sequences(pdb_path: Path) -> dict[str, str]:
    """Return {chain_id: sequence_from_ATOM_records} for every chain."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    ppb = PPBuilder()
    out: dict[str, str] = {}
    for model in structure:
        for chain in model:
            cid = chain.id
            # PPBuilder may return multiple polypeptides per chain (chain
            # breaks); concatenate them with '' (no joiner) to get the
            # full chain-level sequence ignoring resolved-residue gaps.
            seqs = [str(pp.get_sequence()) for pp in ppb.build_peptides(chain)]
            if seqs:
                out[cid] = "".join(seqs)
        break  # only first model
    return out


def percent_identity(a: str, b: str) -> tuple[float, int]:
    """Return (% identity over shared prefix, shared length)."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0
    matches = sum(1 for x, y in zip(a[:n], b[:n]) if x == y)
    return 100 * matches / n, n


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--curated-csv",
        default="/projects/0/hpmlprjs/interns/krijn/ANDD_nano_dataset_IgLM/ANDD_VHH_curated_diffab.csv",
        type=Path,
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
        "--major-mismatch-threshold",
        type=float,
        default=30.0,
        help="Min %% residue diff vs CSV (over shared prefix) to be 'major'. Default 30.",
    )
    p.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Only probe first N major-mismatch entries (debug). Default: all.",
    )
    p.add_argument(
        "--good-match-threshold",
        type=float,
        default=95.0,
        help="Min %% identity to call a chain a 'good match' to CSV. Default 95.",
    )
    args = p.parse_args()

    csv = pd.read_csv(args.curated_csv)
    mani = pd.read_csv(args.manifest_tsv, sep="\t")
    seq_lookup = {
        (str(r["PDB_ID"]).strip().lower(), str(r["H_Chain Auth Asym ID"]).strip()):
            str(r["Ab/Nano H_Chain AA"]).strip()
        for _, r in csv.iterrows()
        if pd.notna(r.get("Ab/Nano H_Chain AA"))
    }

    # ── Identify major-mismatch entries ─────────────────────────────
    suspects: list[tuple[str, str, str]] = []  # (pdb, h_chain, csv_seq)
    for _, row in mani.iterrows():
        pdb_id = str(row["pdb"]).strip().lower()
        h_chain = str(row["Hchain"]).strip()
        csv_seq = seq_lookup.get((pdb_id, h_chain))
        if csv_seq is None:
            continue
        pdb_path = args.pdb_dir / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            continue
        try:
            chains = all_chain_sequences(pdb_path)
        except Exception as exc:
            print(f"WARN failed to parse {pdb_id}: {exc}", file=sys.stderr)
            continue
        pdb_seq_at_curated_chain = chains.get(h_chain, "")
        if not pdb_seq_at_curated_chain:
            suspects.append((pdb_id, h_chain, csv_seq))
            continue
        pid, _ = percent_identity(csv_seq, pdb_seq_at_curated_chain)
        if (100 - pid) > args.major_mismatch_threshold:
            suspects.append((pdb_id, h_chain, csv_seq))

    print(f"Found {len(suspects)} major-mismatch entries to probe "
          f"(>{args.major_mismatch_threshold}% diff at curated chain).")

    if args.max_entries is not None:
        suspects = suspects[: args.max_entries]
        print(f"Probing first {len(suspects)} (--max-entries).")

    # ── Probe each suspect: find best-matching chain in PDB ─────────
    n_curated_correct = 0
    n_swapped = 0
    n_no_good_match = 0
    swap_examples: list[tuple[str, str, str, float]] = []
    swap_target_chains: dict[str, int] = {}

    for pdb_id, h_chain, csv_seq in suspects:
        pdb_path = args.pdb_dir / f"{pdb_id}.pdb"
        chains = all_chain_sequences(pdb_path)

        # Score every chain
        ranked = sorted(
            ((cid, *percent_identity(csv_seq, seq), len(seq))
             for cid, seq in chains.items()),
            key=lambda t: -t[1],  # by % identity descending
        )

        best_cid, best_pid, best_shared, best_len = ranked[0]

        if best_pid >= args.good_match_threshold and best_cid == h_chain:
            n_curated_correct += 1
        elif best_pid >= args.good_match_threshold and best_cid != h_chain:
            n_swapped += 1
            swap_examples.append((pdb_id, h_chain, best_cid, best_pid))
            swap_target_chains[best_cid] = swap_target_chains.get(best_cid, 0) + 1
        else:
            n_no_good_match += 1
            # Show top candidate for diagnosis
            swap_examples.append((pdb_id, h_chain, f"{best_cid}?", best_pid))

    # ── Report ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Chain-ID swap probe (good-match threshold = {args.good_match_threshold}%)")
    print("=" * 60)
    total = len(suspects)
    print(f"Major-mismatch entries probed:     {total}")
    print(f"  curated chain ID is correct:     {n_curated_correct:4d}  "
          f"({100*n_curated_correct/max(total,1):5.1f}%)")
    print(f"  best match on a DIFFERENT chain: {n_swapped:4d}  "
          f"({100*n_swapped/max(total,1):5.1f}%)  ← likely curation bug")
    print(f"  no chain in PDB matches CSV:     {n_no_good_match:4d}  "
          f"({100*n_no_good_match/max(total,1):5.1f}%)  ← deeper data issue")

    if swap_target_chains:
        print(f"\nWhich chain ID does the VHH actually live under, when curated wrong?")
        for cid, count in sorted(swap_target_chains.items(), key=lambda t: -t[1]):
            print(f"  chain '{cid}': {count} entries")

    if swap_examples:
        print(f"\nFirst 15 examples (pdb, curated_chain, best_match_chain, % identity):")
        for pdb_id, curated, best, pid in swap_examples[:15]:
            arrow = "→" if best != curated else "="
            print(f"  {pdb_id}: curated={curated} {arrow} actual_best={best} "
                  f"({pid:5.1f}%)")

    print("\n" + "-" * 60)
    if n_swapped >= total * 0.5:
        print("DIAGNOSIS: systematic chain-ID swap in the curated CSV.")
        print("ACTION: re-curate H_Chain Auth Asym ID from the PDB before training.")
    elif n_no_good_match >= total * 0.5:
        print("DIAGNOSIS: the curated CSV sequences don't appear in these PDBs at all.")
        print("ACTION: investigate whether ANDD pulled sequences from a different source.")
    elif n_curated_correct >= total * 0.5:
        print("UNEXPECTED: most curated chain IDs are correct after all.")
        print("ACTION: re-examine why the audit reported them as major mismatches.")
    else:
        print("MIXED picture. Inspect the examples above to triage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
