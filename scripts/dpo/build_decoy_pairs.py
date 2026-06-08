#!/usr/bin/env python3
"""Brief 17 §7 — swap GT winner PDBs for decoy winner PDBs in the pair pool.

Takes the floor pair parquet (1492 pairs, winner = X-ray GT) and rewrites
the ``winner_pdb_path`` + ``winner_candidate_id`` columns to point at the
decoy PDBs produced by ``sample_decoy_winners.py``. All other columns
are preserved — in particular ``axes_winner``, which still holds the
GT's judge scores. The brief's rationale (Brief 17 §7):

    The decoy is a structural surrogate for the same biological winner,
    so the GT's judge scores remain semantically correct.

Two provenance columns are added for audit:

  * ``original_winner_pdb_path``      — the X-ray crystal PDB path before swap.
  * ``original_winner_candidate_id``  — the original candidate_id (the GT one).
  * ``winner_provenance``             — sentinel "decoy_t{T}" so downstream
                                        loaders can recognise decoy pairs.

CLI
---
::

    python scripts/dpo/build_decoy_pairs.py \\
        --pairs-parquet  data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs.parquet \\
        --decoy-dir      data/aapr/ftseed42_jfix_trainval_K8_20260525/decoys_t10/pdbs \\
        --t-decoy        10 \\
        --output         data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_decoy_t10.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pairs-parquet", required=True, type=Path,
                    help="Source pair parquet whose winners get swapped.")
    ap.add_argument("--decoy-dir", required=True, type=Path,
                    help="Directory containing the decoy PDBs "
                         "(one per unique GT, named "
                         "<gt_complex_id>__decoy_t<T>.pdb).")
    ap.add_argument("--t-decoy", type=int, default=10,
                    help="Decoy depth used to construct PDB filenames "
                         "and the candidate_id suffix.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output parquet path. REQUIRED to avoid "
                         "overwriting the original pairs parquet.")
    ap.add_argument("--strict", action="store_true",
                    help="If set, abort when any decoy PDB is missing. "
                         "Default: warn and drop pairs whose decoy is missing.")
    args = ap.parse_args()

    for p in (args.pairs_parquet, args.decoy_dir):
        if not p.exists():
            print(f"ERROR: required input not found: {p}", file=sys.stderr)
            return 2

    df = pd.read_parquet(args.pairs_parquet)
    n_in = len(df)
    print(f"Read {n_in} pairs from {args.pairs_parquet}")
    print(f"  columns:        {list(df.columns)}")
    print(f"  unique GTs:     {df['gt_complex_id'].nunique()}")

    decoy_dir = args.decoy_dir.resolve()
    t_decoy = int(args.t_decoy)

    def decoy_path(gt_id: str) -> str:
        return str(decoy_dir / f"{gt_id}__decoy_t{t_decoy}.pdb")

    df["original_winner_pdb_path"] = df["winner_pdb_path"]
    df["original_winner_candidate_id"] = df["winner_candidate_id"]
    df["winner_pdb_path"] = df["gt_complex_id"].astype(str).apply(decoy_path)
    df["winner_candidate_id"] = (
        df["gt_complex_id"].astype(str) + f"__decoy_t{t_decoy}"
    )
    df["winner_provenance"] = f"decoy_t{t_decoy}"

    # Sanity: every decoy PDB referenced must exist on disk.
    unique_paths = df["winner_pdb_path"].unique().tolist()
    missing = [p for p in unique_paths if not Path(p).exists()]
    if missing:
        msg = (
            f"{len(missing)}/{len(unique_paths)} decoy PDBs missing under "
            f"{decoy_dir} (sample: {missing[:5]})"
        )
        if args.strict:
            print(f"ERROR: {msg}", file=sys.stderr)
            return 3
        print(f"WARNING: {msg}", file=sys.stderr)
        before = len(df)
        df = df[df["winner_pdb_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
        dropped = before - len(df)
        print(f"  dropped {dropped} pair(s) whose decoy PDB was missing.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(f"\nWrote {len(df)} pairs with decoy winners → {args.output}")
    print(f"  unique decoys: {df['winner_candidate_id'].nunique()}")
    print(f"  winner_provenance: {df['winner_provenance'].iloc[0] if len(df) else '<empty>'}")
    print("  first 3 decoy paths:")
    for p in df["winner_pdb_path"].head(3).tolist():
        print(f"    {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
