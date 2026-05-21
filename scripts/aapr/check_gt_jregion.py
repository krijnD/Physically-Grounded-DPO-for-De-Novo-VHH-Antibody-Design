#!/usr/bin/env python3
"""Diagnose Hypothesis B from docs/aapr_masking_research_context.md.

Question: are the 14 NB2-unfoldable GTs from the seed42_dedup canary
actually truncated at the C-terminal J-region in the dataset itself?

Loads each GT via the same path the AAPR sampler uses
(``VHHANDDDataset.get_structure``), filters to the heavy chain via the
manifest's ``Hchain`` field, and prints the C-terminal of the heavy
chain sequence plus a verdict on whether the canonical J-anchor
``WG[QH]G[T/S][QHP]VTV`` is present.

Interpretation:
  * If all 14 GT C-terminals show full anchor (``WGQGTQVTVS``) →
    Hypothesis B is OUT; the sampler is producing the truncation, not
    the dataset.
  * If GT C-terminals show ``WGQ`` / ``DYW`` / ``CNT`` etc. (truncated)
    → Hypothesis B is IN; the sampler is correctly reproducing the
    truncated input.

Usage on Snellius::

    source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate
    cd ~/Physically-Grounded-DPO-for-De-Novo-VHH-Antibody-Design
    python scripts/aapr/check_gt_jregion.py

To check all test-split GTs (not just the 14 failing)::

    python scripts/aapr/check_gt_jregion.py --all
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Project paths — match scripts/aapr/sample_candidates.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "diffab"))

from diffab.datasets import get_dataset  # noqa: E402
from diffab.utils.misc import load_config  # noqa: E402

# Side-effect: registers ``vhh_andd`` so get_dataset can resolve it.
import src.diffab_ft.datasets  # noqa: E402, F401

from Bio.PDB import Polypeptide  # noqa: E402


# 14 GTs that lost all 8/8 AAPR samples to NB2 in the seed42_dedup canary.
# Source: docs/aapr_masking_research_context.md (2026-05-20 analysis).
FAILING_GTS = [
    "7q6c_K", "7qbf_B", "7sk7_K", "7vfa_D", "7vke_B", "7wd2_C",
    "7xrp_B", "7zlg_K", "8acf_K", "8cy6_D", "8gsi_F", "8qot_B",
    "8r61_C", "8u4v_K",
]

# Canonical VHH J-anchor; relaxed to allow the two known atypical variants
# (e.g. 8fcz / 8pyr W103-substitution). Hits anywhere in the last ~15 AA.
ANCHOR_RE = re.compile(r"WG[A-Z]G[A-Z]{1,4}VTV", re.IGNORECASE)
STRICT_ANCHOR_RE = re.compile(r"WG[QH]GT[QHP]VTVSS", re.IGNORECASE)


def heavy_seq_from_structure(struct: dict) -> tuple[str, int, int]:
    """Extract heavy-chain 1-letter sequence + (min_resseq, max_resseq).

    The raw LMDB structure is the dict returned by
    ``preprocess_sabdab_structure``: keys ``heavy``, ``heavy_seqmap``,
    ``light``, ``antigen``, etc. The heavy sub-dict has ``chain_id``,
    ``resseq``, ``aa``, ``pos_heavyatom``. The Hchain manifest filter
    is unnecessary here because ``parsed['heavy']`` is already
    heavy-only (the parser was given only ``model[H_chain]``).
    """
    heavy = struct["heavy"]
    if heavy is None:
        raise ValueError("structure has no 'heavy' sub-dict")
    aa = heavy["aa"]
    resseq = heavy["resseq"]
    aa_iter = aa.tolist() if hasattr(aa, "tolist") else list(aa)
    resseq_iter = resseq.tolist() if hasattr(resseq, "tolist") else list(resseq)
    chars = [Polypeptide.index_to_one(int(a)) for a in aa_iter]
    return "".join(chars), min(resseq_iter), max(resseq_iter)


def classify(seq: str) -> str:
    """One-word verdict for the C-terminal."""
    tail = seq[-20:]
    if STRICT_ANCHOR_RE.search(tail):
        return "FULL_ANCHOR"
    if ANCHOR_RE.search(tail):
        return "ATYPICAL_ANCHOR"
    return "TRUNCATED"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/diffab_ft/vhh_ft.yml",
        help="Path to the DiffAb fine-tune config (uses dataset.val block).",
    )
    parser.add_argument(
        "--split", default="test",
        help="Which split to load (default: test).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Check every GT in the split, not just the 14 known-failing ones.",
    )
    args = parser.parse_args()

    config, _ = load_config(args.config)
    ds_block = config.dataset.val
    ds_block.split = args.split
    # Strip transforms — we want the raw structure dict, not the masked one.
    ds_block.transform = None
    dataset = get_dataset(ds_block)

    targets = list(dataset.ids_in_split) if args.all else FAILING_GTS

    print(f"Checking {len(targets)} GTs (split={args.split})")
    print(f"{'GT':<10} {'len':>4}  {'resseq':<11}  verdict           C-terminal (last 25)")
    print("-" * 95)

    counts = {"FULL_ANCHOR": 0, "ATYPICAL_ANCHOR": 0, "TRUNCATED": 0,
              "MISSING_FROM_SPLIT": 0, "ERROR": 0}

    for gt_id in targets:
        if gt_id not in dataset.ids_in_split:
            print(f"{gt_id:<10}  -    MISSING_FROM_SPLIT")
            counts["MISSING_FROM_SPLIT"] += 1
            continue
        try:
            struct = dataset.get_structure(gt_id)
            seq, min_rs, max_rs = heavy_seq_from_structure(struct)
            verdict = classify(seq)
            counts[verdict] += 1
            resseq_range = f"{min_rs}-{max_rs}"
            print(f"{gt_id:<10} {len(seq):>4}  {resseq_range:<11}  {verdict:<16}  ...{seq[-25:]}")
        except Exception as exc:  # noqa: BLE001
            print(f"{gt_id:<10}  -    -            ERROR  ({exc.__class__.__name__}: {exc})")
            counts["ERROR"] += 1

    print("-" * 95)
    print("Summary:")
    for k, v in counts.items():
        if v:
            print(f"  {k:<22} {v:>3}")

    # Interpretation hint
    if not args.all and counts["TRUNCATED"] >= 10:
        print("\n>>> Hypothesis B SUPPORTED: most failing GTs are truncated "
              "at the dataset level. Fix 2 (mask J-protection) won't help "
              "these GTs because there is no J-region to preserve.")
    elif not args.all and counts["TRUNCATED"] <= 2:
        print("\n>>> Hypothesis B REJECTED: GT C-terminals show the J-anchor. "
              "The sampler is producing the truncation. Investigate "
              "Hypothesis A (H3 mask boundary) next.")
    elif not args.all:
        print("\n>>> MIXED result: some GTs truncated, others not. Inspect "
              "row-by-row to decide which fix applies per-GT.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
