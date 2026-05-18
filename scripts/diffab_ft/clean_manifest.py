#!/usr/bin/env python3
"""Produce a sequence-consistent version of diffab_manifest.tsv.

The Phase 1 audit (audit_pdb_csv_consistency.py) and chain-ID probe
(probe_chain_id_mismatch.py) revealed that ~27% of the manifest has
CSV sequences that don't match the corresponding PDB chains. Of those:

  * ~21 entries: the CSV sequence DOES appear in the PDB, just under a
    different chain ID — fixable by patching the manifest's Hchain.
  * ~104 entries: the CSV sequence appears in NO chain of the PDB. The
    upstream ANDD Excel mismapped a sequence to the wrong PDB ID; not
    recoverable without re-sourcing sequences. Drop these.

This script does both: emits a new manifest TSV with chain IDs patched
where possible and bad entries removed entirely. The original manifest
is left untouched.

For an entry where we patch Hchain, we also re-verify that the OLD chain
was an antigen partner (i.e. not also a VHH that should have been kept)
and adjust antigen_chain if the OLD H-chain wasn't already listed as
antigen. This keeps the (VHH, antigen) topology coherent.

Outputs:
  - data/datasets/diffab_manifest_clean.tsv   (default)
  - data/datasets/manifest_cleanup_report.csv (per-row decisions)

Usage:
    python scripts/diffab_ft/clean_manifest.py
    python scripts/diffab_ft/clean_manifest.py --good-match-threshold 90
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
    """Return {chain_id: sequence_from_ATOM_records} for every chain in
    the first model of the PDB file."""
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


def percent_identity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return 100 * sum(1 for x, y in zip(a[:n], b[:n]) if x == y) / n


def best_matching_chain(
    chains: dict[str, str], csv_seq: str
) -> tuple[str, float]:
    """Return (chain_id, % identity) of the chain whose sequence best
    matches csv_seq. Returns ('', 0.0) if no chains at all."""
    if not chains:
        return "", 0.0
    ranked = sorted(
        ((cid, percent_identity(csv_seq, seq)) for cid, seq in chains.items()),
        key=lambda t: -t[1],
    )
    return ranked[0]


def _split_pipe(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [c.strip() for c in str(value).split("|") if c.strip()]


def _join_pipe(chains: list[str]) -> str:
    return " | ".join(chains)


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
        "--good-match-threshold",
        type=float,
        default=95.0,
        help="Min %% identity for a chain to be accepted as the VHH (default 95).",
    )
    p.add_argument(
        "--out-manifest",
        default="data/datasets/diffab_manifest_clean.tsv",
        type=Path,
    )
    p.add_argument(
        "--out-report",
        default="data/datasets/manifest_cleanup_report.csv",
        type=Path,
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

    # ── Walk manifest, classify each entry ────────────────────────
    kept_rows: list[dict] = []
    report_rows: list[dict] = []
    n_kept_asis = 0
    n_patched = 0
    n_dropped = 0
    n_skipped = 0  # missing CSV, missing PDB

    for _, row in mani.iterrows():
        pdb_id = str(row["pdb"]).strip().lower()
        h_chain = str(row["Hchain"]).strip()
        antigen_chains = _split_pipe(row.get("antigen_chain"))
        csv_seq = seq_lookup.get((pdb_id, h_chain))

        if csv_seq is None:
            n_skipped += 1
            report_rows.append(
                {"pdb": pdb_id, "curated_chain": h_chain, "action": "skip_no_csv",
                 "best_chain": "", "best_pid": 0.0, "patched_chain": h_chain}
            )
            kept_rows.append(row.to_dict())  # keep — wasn't part of audit
            continue

        pdb_path = args.pdb_dir / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            n_skipped += 1
            report_rows.append(
                {"pdb": pdb_id, "curated_chain": h_chain, "action": "skip_no_pdb",
                 "best_chain": "", "best_pid": 0.0, "patched_chain": h_chain}
            )
            kept_rows.append(row.to_dict())
            continue

        try:
            chains = all_chain_sequences(pdb_path)
        except Exception as exc:
            print(f"WARN parse fail {pdb_id}: {exc}", file=sys.stderr)
            n_skipped += 1
            kept_rows.append(row.to_dict())
            continue

        # 1. If curated chain already matches well → keep as-is
        curated_seq = chains.get(h_chain, "")
        curated_pid = percent_identity(csv_seq, curated_seq) if curated_seq else 0.0
        if curated_pid >= args.good_match_threshold:
            n_kept_asis += 1
            report_rows.append(
                {"pdb": pdb_id, "curated_chain": h_chain, "action": "keep_asis",
                 "best_chain": h_chain, "best_pid": round(curated_pid, 1),
                 "patched_chain": h_chain}
            )
            kept_rows.append(row.to_dict())
            continue

        # 2. Try to find a better chain
        best_cid, best_pid = best_matching_chain(chains, csv_seq)
        if best_cid and best_pid >= args.good_match_threshold:
            # Patch Hchain to the best match
            patched = row.to_dict()
            patched["Hchain"] = best_cid
            # Antigen update: if the OLD H-chain wasn't already listed as
            # antigen, add it (since the OLD chain is presumably a real
            # partner of the now-correct VHH chain). And remove the NEW
            # H-chain from antigen list if it was there.
            new_antigen = [c for c in antigen_chains if c != best_cid]
            if h_chain not in new_antigen:
                new_antigen.append(h_chain)
            patched["antigen_chain"] = _join_pipe(new_antigen)
            patched["antigen_type"] = _join_pipe(["protein"] * len(new_antigen))
            n_patched += 1
            report_rows.append(
                {"pdb": pdb_id, "curated_chain": h_chain, "action": "patch_chain",
                 "best_chain": best_cid, "best_pid": round(best_pid, 1),
                 "patched_chain": best_cid}
            )
            kept_rows.append(patched)
            continue

        # 3. No good match anywhere → drop
        n_dropped += 1
        report_rows.append(
            {"pdb": pdb_id, "curated_chain": h_chain, "action": "drop_no_match",
             "best_chain": best_cid, "best_pid": round(best_pid, 1),
             "patched_chain": ""}
        )

    # ── Write outputs ─────────────────────────────────────────────
    out_df = pd.DataFrame(kept_rows, columns=mani.columns)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_manifest, sep="\t", index=False)

    rep_df = pd.DataFrame(report_rows)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    rep_df.to_csv(args.out_report, index=False)

    # ── Report ─────────────────────────────────────────────────────
    total = len(mani)
    print("=" * 60)
    print("Manifest cleanup")
    print("=" * 60)
    print(f"input manifest rows:       {total}")
    print(f"  kept as-is:              {n_kept_asis:4d}  ({100*n_kept_asis/total:5.1f}%)")
    print(f"  Hchain patched:          {n_patched:4d}  ({100*n_patched/total:5.1f}%)")
    print(f"  dropped (no match):      {n_dropped:4d}  ({100*n_dropped/total:5.1f}%)")
    print(f"  skipped audit (no input):{n_skipped:4d}  ({100*n_skipped/total:5.1f}%)")
    print(f"output manifest rows:      {len(out_df)}")
    print(f"\nWrote: {args.out_manifest}")
    print(f"Wrote: {args.out_report}  (per-row decisions; useful for thesis writeup)")

    if n_patched:
        print(f"\nPatched entries (first 10):")
        patched_examples = [r for r in report_rows if r["action"] == "patch_chain"][:10]
        for r in patched_examples:
            print(f"  {r['pdb']}: H={r['curated_chain']} → {r['patched_chain']}  ({r['best_pid']}% identity)")

    if n_dropped:
        print(f"\nDropped entries (first 10):")
        dropped_examples = [r for r in report_rows if r["action"] == "drop_no_match"][:10]
        for r in dropped_examples:
            print(f"  {r['pdb']}: curated H={r['curated_chain']}, best identity anywhere = {r['best_pid']}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
