#!/usr/bin/env python3
"""Check 6: do the GT PDB files actually contain Chothia insertion codes?

D1 (code dive) showed that the parser + CDR-labelling code WOULD preserve
insertion-coded residues if they were present (the 7uny outlier from
Check 5, with H3_n=15, proves this). So the H3=8 we see for 19/20 GTs
either reflects truly-short H3 loops or upstream renumbering.

This check parses raw PDB files (bypassing the LMDB) and reports, for
20 random GTs:
  - How many residues exist in the heavy chain at integer resseq 95-102
  - Of those, how many have non-blank insertion codes
  - The actual insertion-code distribution

If most GTs have icode='' for all residues, the source PDBs are
sequentially-numbered (no insertions) — there's no parser bug, the model
is just trained on short H3 loops because that's what the data has.

If many GTs DO have insertion codes but Check 5 still showed H3=8, then
something between PDB parsing and LMDB write is dropping them, and we'd
have a real bug.
"""
from __future__ import annotations
import sys
import random
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "diffab"))

from Bio.PDB import PDBParser  # type: ignore
from diffab.datasets import get_dataset
from diffab.utils.misc import load_config
import src.diffab_ft.datasets  # noqa: F401


def main() -> int:
    config_path = PROJECT_ROOT / "configs/dpo/vhh_dpo.yml"
    config, _ = load_config(str(config_path))
    pairs = pd.read_parquet(PROJECT_ROOT / config.dpo.pair_parquet)
    pdb_dir = Path(config.dataset.train.pdb_dir)
    print(f"Inspecting raw GT PDBs from: {pdb_dir}")

    print("Building base dataset for entry metadata...")
    base_dataset = get_dataset(config.dataset.train)
    pdb_to_entry = {e["pdbcode"]: e for e in base_dataset.sabdab_entries
                    if e["id"] in set(base_dataset.db_ids or [])}

    random.seed(42)
    sample_gts = (pairs["gt_complex_id"].drop_duplicates()
                  .sample(n=20, random_state=42).tolist())

    print(f"\n=== Check 6: insertion codes in raw GT PDBs (n=20) ===\n")
    print(f"{'gt_id':<10} {'H':<3} {'n_in_95_102':>11} {'n_with_icode':>13} "
          f"{'icode_set':<30} {'h3_resseq_icode'}")

    n_with_insertions = 0
    for gt_id in sample_gts:
        entry = pdb_to_entry.get(gt_id)
        if entry is None:
            print(f"{gt_id:<10} not in LMDB")
            continue
        H = entry.get("H_chain", "H")
        pdb_path = pdb_dir / f"{gt_id}.pdb"
        if not pdb_path.exists():
            print(f"{gt_id:<10} PDB missing: {pdb_path}")
            continue

        parser = PDBParser(QUIET=True)
        try:
            struct = parser.get_structure("x", str(pdb_path))
        except Exception as e:
            print(f"{gt_id:<10} parse error: {e!r}")
            continue

        try:
            chain = next(struct.get_models())[H]
        except KeyError:
            print(f"{gt_id:<10} no H chain {H}")
            continue

        # All residues at integer resseq in 95-102
        residues = [r for r in chain
                    if r.id[0] == " " and 95 <= int(r.id[1]) <= 102]
        n_total = len(residues)
        icodes = [r.id[2] for r in residues]
        icode_nonblank = [c for c in icodes if c != " " and c != ""]
        h3_pos = [(int(r.id[1]), r.id[2] if r.id[2] != " " else "")
                  for r in residues]
        h3_str = ",".join(f"{n}{i}" for n, i in h3_pos)

        if icode_nonblank:
            n_with_insertions += 1

        icode_set = sorted(set(icodes))
        icode_repr = repr([c if c != " " else "_" for c in icode_set])[:30]

        print(f"{gt_id:<10} {H:<3} {n_total:>11} {len(icode_nonblank):>13} "
              f"{icode_repr:<30} {h3_str}")

    print(f"\nGTs with at least one insertion-coded H3 residue: "
          f"{n_with_insertions}/20")
    print("\nInterpretation:")
    print("  If most GTs have insertion codes → labeling pipeline is broken, fix it.")
    print("  If most GTs have NO insertion codes → source PDBs really have short H3,")
    print("    the parser is fine, and improving π_ref needs more data not a code fix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
