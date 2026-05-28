#!/usr/bin/env python3
"""Check 2: are AAPR losers meaningfully different from GT winners at CDRs?

If AAPR samples are too close to the GT, the DPO loss has no signal to learn.
If they're wildly different (random), there's no continuous "preference" signal either —
the model just learns "GT vs random noise".

For 20 random pairs:
  - Load winner from the LMDB (so we get cdr_flag from preprocessing)
  - Parse loser PDB with the same parser
  - Verify residue counts match
  - At CDR positions (cdr_flag > 0): compute
       * sequence identity (winner_aa == loser_aa)
       * CA-RMSD between winner and loser positions
  - Break out per CDR (H1, H2, H3) if cdr_flag carries that info, else aggregate.

Output: per-pair line + summary distribution.
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
from diffab.utils.protein.constants import BBHeavyAtom  # noqa: E402

import src.diffab_ft.datasets  # noqa: E402, F401  (registry side effect)
from src.diffab_ft.datasets.vhh_andd import (  # noqa: E402
    DEFAULT_HEAVY_MAX_RESSEQ,
    _preprocess_vhh_structure,
)

# 0..19 → 1-letter code (DiffAb's int encoding; matches AA constants module)
AA_INT_TO_LETTER = "ACDEFGHIKLMNPQRSTVWY"


def aa_to_letter(idx: int) -> str:
    if 0 <= idx < 20:
        return AA_INT_TO_LETTER[idx]
    return "X"


def main() -> int:
    config_path = PROJECT_ROOT / "configs/dpo/vhh_dpo.yml"
    config, _ = load_config(str(config_path))
    pairs_path = PROJECT_ROOT / config.dpo.pair_parquet
    pairs = pd.read_parquet(pairs_path)
    print(f"Loaded {len(pairs)} pairs")

    print("Building base dataset (LMDB winner source)...")
    base_dataset = get_dataset(config.dataset.train)
    pdb_to_entry = {e["pdbcode"]: e for e in base_dataset.sabdab_entries
                    if e["id"] in set(base_dataset.db_ids or [])}

    random.seed(42)
    sample = pairs.sample(n=min(20, len(pairs)), random_state=42).reset_index(drop=True)

    print("\n=== Check 2: winner vs loser CDR difference (n=20) ===\n")
    print(f"{'gt_id':<10} {'n_cdr':>5} {'seq_id%':>7} {'CA_rmsd_Å':>10} "
          f"{'n_diff_aa':>9} {'winner_cdr_seq':<35} {'loser_cdr_seq':<35}")

    all_seq_ids, all_rmsds, all_n_cdr = [], [], []
    for _, row in sample.iterrows():
        gt_id = str(row["gt_complex_id"])
        loser_path = Path(str(row["loser_pdb_path"]))
        if not loser_path.is_absolute():
            loser_path = PROJECT_ROOT / loser_path
        entry = pdb_to_entry.get(gt_id)
        if entry is None:
            print(f"{gt_id:<10} NOT IN LMDB")
            continue

        winner = base_dataset.get_structure(entry["id"])
        loser = _preprocess_vhh_structure(
            {"id": f"loser__{loser_path.stem}", "entry": entry, "pdb_path": str(loser_path)},
            DEFAULT_HEAVY_MAX_RESSEQ,
        )
        if loser is None:
            print(f"{gt_id:<10} loser parse failed")
            continue

        wh, lh = winner.get("heavy"), loser.get("heavy")
        if wh is None or lh is None:
            print(f"{gt_id:<10} missing heavy")
            continue
        if wh["aa"].size(0) != lh["aa"].size(0):
            print(f"{gt_id:<10} length mismatch w={wh['aa'].size(0)} l={lh['aa'].size(0)}")
            continue

        cdr_mask = (wh["cdr_flag"] > 0)
        n_cdr = int(cdr_mask.sum().item())
        if n_cdr == 0:
            print(f"{gt_id:<10} no CDR residues flagged")
            continue

        w_aa = wh["aa"][cdr_mask].cpu().numpy()
        l_aa = lh["aa"][cdr_mask].cpu().numpy()
        n_diff = int((w_aa != l_aa).sum())
        seq_id_pct = 100.0 * (n_cdr - n_diff) / n_cdr

        # CA = atom index BBHeavyAtom.CA
        w_ca = wh["pos_heavyatom"][cdr_mask, BBHeavyAtom.CA].cpu().numpy()
        l_ca = lh["pos_heavyatom"][cdr_mask, BBHeavyAtom.CA].cpu().numpy()
        rmsd = float(np.sqrt(((w_ca - l_ca) ** 2).sum(axis=-1).mean()))

        w_seq = "".join(aa_to_letter(int(x)) for x in w_aa)
        l_seq = "".join(aa_to_letter(int(x)) for x in l_aa)

        all_seq_ids.append(seq_id_pct)
        all_rmsds.append(rmsd)
        all_n_cdr.append(n_cdr)

        print(f"{gt_id:<10} {n_cdr:>5} {seq_id_pct:>6.1f}% {rmsd:>9.2f} {n_diff:>9} "
              f"{w_seq[:35]:<35} {l_seq[:35]:<35}")

    if all_seq_ids:
        arr_seq = np.array(all_seq_ids)
        arr_rmsd = np.array(all_rmsds)
        arr_n = np.array(all_n_cdr)
        print(f"\nSummary (n={len(arr_seq)}):")
        print(f"  n_cdr_residues:   median={np.median(arr_n):.0f}  range=[{arr_n.min()}, {arr_n.max()}]")
        print(f"  seq identity %:   median={np.median(arr_seq):.1f}  "
              f"q10={np.percentile(arr_seq, 10):.1f}  q90={np.percentile(arr_seq, 90):.1f}  "
              f"min={arr_seq.min():.1f}  max={arr_seq.max():.1f}")
        print(f"  CA RMSD (Å):      median={np.median(arr_rmsd):.2f}  "
              f"q10={np.percentile(arr_rmsd, 10):.2f}  q90={np.percentile(arr_rmsd, 90):.2f}  "
              f"min={arr_rmsd.min():.2f}  max={arr_rmsd.max():.2f}")
        print("\nReadability:")
        print("  ~100% seq id + ~0 RMSD → AAPR is reproducing the GT → no signal.")
        print("  ~5% seq id + huge RMSD → AAPR is random → only learning GT-vs-noise.")
        print("  20-60% seq id + 1-5Å RMSD → real, learnable variation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
