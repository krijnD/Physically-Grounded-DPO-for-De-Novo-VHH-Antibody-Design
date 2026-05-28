#!/usr/bin/env python3
"""D2: inventory available VHH data that could extend the fine-tune pool.

The current FT uses ~242 entries (VHH-antigen complexes with ground-truth
annotations). The hypothesis is that this is too small to produce a
competent π_ref. Before deciding to expand, count what's actually
available beyond the 242 GTs.

What this checks:
  1. The current manifest (data/datasets/diffab_manifest.tsv) — total
     rows, how many made it into the LMDB, how many got filtered out.
  2. Any SAbDab summary files lying around that have additional VHHs
     we haven't used.
  3. The ANDD dataset directory at the IgLM-cutoff path — count PDB
     files, see whether any have associated antigens.

Output: structured summary numbers that let us decide whether
"expanded FT" is feasible.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ANDD_BASE = Path("/projects/0/hpmlprjs/interns/krijn/ANDD_nano_dataset_IgLM")


def main() -> int:
    print("=" * 70)
    print("D2 — Data inventory for expanded VHH fine-tuning")
    print("=" * 70)

    # 1. Current manifest
    manifest = PROJECT_ROOT / "data/datasets/diffab_manifest.tsv"
    print(f"\n[1] Current manifest: {manifest}")
    if manifest.exists():
        df = pd.read_csv(manifest, sep="\t")
        print(f"  rows: {len(df)}")
        print(f"  columns: {list(df.columns)}")
        print(f"  example row:\n    {df.iloc[0].to_dict()}")
        if "Hchain" in df.columns:
            print(f"  has H_chain: {df['Hchain'].notna().sum()}")
        if "antigen_chain" in df.columns:
            print(f"  has antigen_chain: "
                  f"{df['antigen_chain'].notna().sum()}")
        if "antigen_type" in df.columns:
            print(f"  antigen_type counts:")
            print(df["antigen_type"].value_counts(dropna=False).head(10).to_string())
    else:
        print(f"  MISSING")

    # 2. Splits JSON — see how many entries are in train/val/test
    splits = PROJECT_ROOT / "data/datasets/clustering/cluster_splits.json"
    print(f"\n[2] Cluster splits: {splits}")
    if splits.exists():
        import json
        with open(splits) as f:
            data = json.load(f)
        for split_name, ids in data.get("splits", {}).items():
            print(f"  {split_name}: {len(ids)} entries")
        print(f"  cluster_assignments: {len(data.get('cluster_assignments', {}))}")
    else:
        print("  MISSING")

    # 3. ANDD PDB directory — count files and check for antigens
    print(f"\n[3] ANDD PDB directory: {ANDD_BASE}")
    if ANDD_BASE.exists():
        for sub in sorted(ANDD_BASE.iterdir()):
            if sub.is_dir():
                n_pdb = sum(1 for _ in sub.glob("*.pdb"))
                n_cif = sum(1 for _ in sub.glob("*.cif"))
                print(f"  {sub.name}/  PDB={n_pdb}  CIF={n_cif}")
    else:
        print("  MISSING — check path")

    # 4. SAbDab summary if it exists
    sabdab_candidates = [
        PROJECT_ROOT / "data/sabdab/sabdab_summary_all.tsv",
        PROJECT_ROOT / "data/raw/sabdab_summary_all.tsv",
        Path("/projects/0/hpmlprjs/interns/krijn/sabdab/sabdab_summary_all.tsv"),
    ]
    print(f"\n[4] SAbDab summary files (looking in standard locations)")
    for p in sabdab_candidates:
        if p.exists():
            df = pd.read_csv(p, sep="\t")
            print(f"  FOUND: {p}  rows={len(df)}")
            if "Hchain" in df.columns:
                has_h = df["Hchain"].notna().sum()
                print(f"    has Hchain: {has_h}")
            if "Lchain" in df.columns:
                no_l = df["Lchain"].isna().sum()
                print(f"    no Lchain (VHH candidates): {no_l}")
            if "antigen_chain" in df.columns:
                has_ag = df["antigen_chain"].notna().sum()
                no_ag = df["antigen_chain"].isna().sum()
                print(f"    has antigen: {has_ag}   no antigen: {no_ag}")
        else:
            print(f"  not at: {p}")

    # 5. Any other curation artifacts that might tell us what was filtered
    curation_artifacts = [
        PROJECT_ROOT / "data/datasets/diffab_manifest_raw.tsv",
        PROJECT_ROOT / "data/datasets/filter_log.txt",
        PROJECT_ROOT / "scripts/datasets/curate_andd.py",
        PROJECT_ROOT / "scripts/datasets/prepare_manifest.py",
    ]
    print(f"\n[5] Curation artifacts (to understand what was filtered out)")
    for p in curation_artifacts:
        if p.exists():
            stat = p.stat()
            print(f"  exists: {p}  size={stat.st_size}")
        else:
            print(f"  not at: {p}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
