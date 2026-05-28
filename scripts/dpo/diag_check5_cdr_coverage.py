#!/usr/bin/env python3
"""Check 5: how much of each CDR is the cdr_flag actually capturing?

Check 2 showed n_cdr ≈ 20 for almost every pair — too few for a typical
VHH's full H1+H2+H3 (~27-42 residues). Suspicion: the handoff-flagged
CDR3-labeling bug shrinks the DPO mask, so the DPO loss is being
computed on a truncated CDR region (mostly H1+H2, partial H3).

For 20 random pairs from the LMDB:
  - Decompose cdr_flag by value (1=H1, 2=H2, 3=H3 in DiffAb convention)
  - Report per-CDR residue count
  - Compute residue range for each CDR (first/last position in the heavy chain)
  - Compare to canonical Chothia expectations:
        H1: positions 26-32  (7 residues, sometimes extending)
        H2: positions 52-56  (5 residues, sometimes extending)
        H3: positions 95-102 (varies, 5-25 residues in VHH)

If H3 length is consistently ~3-8 residues, that confirms the labeling
truncation flagged in the handoff.
"""
from __future__ import annotations
import sys
import random
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "diffab"))

from diffab.datasets import get_dataset  # noqa: E402
from diffab.utils.misc import load_config  # noqa: E402
import src.diffab_ft.datasets  # noqa: E402, F401


def main() -> int:
    config_path = PROJECT_ROOT / "configs/dpo/vhh_dpo.yml"
    config, _ = load_config(str(config_path))
    print("Building base dataset...")
    base_dataset = get_dataset(config.dataset.train)

    pairs = pd.read_parquet(PROJECT_ROOT / config.dpo.pair_parquet)
    pdb_to_entry = {e["pdbcode"]: e for e in base_dataset.sabdab_entries
                    if e["id"] in set(base_dataset.db_ids or [])}

    random.seed(42)
    sample_gts = (pairs["gt_complex_id"].drop_duplicates()
                  .sample(n=20, random_state=42).tolist())

    print("\n=== Check 5: cdr_flag coverage by CDR (n=20 unique GTs) ===\n")
    print(f"{'gt_id':<10} {'H_chain':<7} {'len':>4} "
          f"{'H1_n':>5} {'H1_resseq':<14} "
          f"{'H2_n':>5} {'H2_resseq':<14} "
          f"{'H3_n':>5} {'H3_resseq':<14} "
          f"{'total':>5}")

    summary = {1: [], 2: [], 3: []}
    for gt_id in sample_gts:
        entry = pdb_to_entry.get(gt_id)
        if entry is None:
            print(f"{gt_id:<10} not in LMDB")
            continue
        struct = base_dataset.get_structure(entry["id"])
        heavy = struct.get("heavy")
        if heavy is None:
            print(f"{gt_id:<10} no heavy chain")
            continue
        cdr_flag = heavy["cdr_flag"].cpu().numpy()
        resseq = heavy.get("resseq")
        resseq_arr = resseq.cpu().numpy() if resseq is not None else None
        H = str(entry.get("H_chain", "H"))

        counts = {}
        ranges = {}
        for cdr in (1, 2, 3):
            idx = np.where(cdr_flag == cdr)[0]
            counts[cdr] = len(idx)
            if len(idx) and resseq_arr is not None:
                ranges[cdr] = f"{int(resseq_arr[idx[0]])}-{int(resseq_arr[idx[-1]])}"
            else:
                ranges[cdr] = "—"
            summary[cdr].append(len(idx))

        total = counts[1] + counts[2] + counts[3]
        print(f"{gt_id:<10} {H:<7} {len(cdr_flag):>4} "
              f"{counts[1]:>5} {ranges[1]:<14} "
              f"{counts[2]:>5} {ranges[2]:<14} "
              f"{counts[3]:>5} {ranges[3]:<14} "
              f"{total:>5}")

    print(f"\nSummary across {len(summary[1])} GTs:")
    for cdr in (1, 2, 3):
        if summary[cdr]:
            a = np.array(summary[cdr])
            print(f"  H{cdr}:  median={int(np.median(a))}  min={a.min()}  max={a.max()}  "
                  f"all_counts={sorted(set(a.tolist()))}")
    print("\nCanonical Chothia expectation for VHH:")
    print("  H1: 7-12 residues (positions 26-35 ish)")
    print("  H2: 5-9  residues (positions 50-58 ish)")
    print("  H3: 5-25 residues (positions 95-102+ — highly variable in VHH)")
    print("\nIf H3 median is < 6 — confirms the CDR3-labeling truncation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
