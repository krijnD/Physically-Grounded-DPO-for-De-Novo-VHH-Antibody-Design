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

    # 4. SAbDab nanobody summary (correct path: sabdab_nano_dataset_IgLM)
    SABDAB_NANO_DIR = Path("/projects/0/hpmlprjs/interns/krijn/sabdab_nano_dataset_IgLM")
    SABDAB_NANO_TSV = SABDAB_NANO_DIR / "sabdab_nano_summary.tsv"
    SABDAB_NANO_PDBS = SABDAB_NANO_DIR / "filtered_vhh_pdbs"
    print(f"\n[4] SAbDab nanobody dataset: {SABDAB_NANO_DIR}")
    if SABDAB_NANO_TSV.exists():
        df = pd.read_csv(SABDAB_NANO_TSV, sep="\t")
        print(f"  summary TSV: {SABDAB_NANO_TSV}  rows={len(df)}")
        print(f"  columns: {list(df.columns)[:20]}")
        # Try common SAbDab column names
        for col_h in ("Hchain", "Hchain_id", "H_chain", "heavy_chain"):
            if col_h in df.columns:
                print(f"  has {col_h}: {df[col_h].notna().sum()}")
                break
        for col_l in ("Lchain", "L_chain", "light_chain"):
            if col_l in df.columns:
                no_l = df[col_l].isna().sum()
                print(f"  no {col_l} (true VHH candidates): {no_l}")
                break
        for col_ag in ("antigen_chain", "antigen_chains", "Agchain"):
            if col_ag in df.columns:
                has_ag = df[col_ag].notna().sum()
                no_ag = df[col_ag].isna().sum()
                print(f"  WITH antigen ({col_ag}): {has_ag}")
                print(f"  WITHOUT antigen: {no_ag}")
                break
        for col_at in ("antigen_type", "Agtype"):
            if col_at in df.columns:
                print(f"  {col_at} value_counts (top 10):")
                print(df[col_at].value_counts(dropna=False).head(10).to_string())
                break
        # Unique PDB codes
        for col_pdb in ("pdb", "pdb_id", "PDB"):
            if col_pdb in df.columns:
                print(f"  unique PDBs ({col_pdb}): {df[col_pdb].nunique()}")
                break
    else:
        print(f"  TSV not found: {SABDAB_NANO_TSV}")
    if SABDAB_NANO_PDBS.exists():
        n_pdbs = sum(1 for _ in SABDAB_NANO_PDBS.glob("*.pdb"))
        n_cifs = sum(1 for _ in SABDAB_NANO_PDBS.glob("*.cif"))
        print(f"  filtered_vhh_pdbs/  PDB={n_pdbs}  CIF={n_cifs}")
    else:
        print(f"  PDB dir not found: {SABDAB_NANO_PDBS}")

    # 5. INDI dataset — large sequence database, possibly folded by ESMFold
    INDI_BASE = Path("/projects/0/hpmlprjs/interns/krijn/INDI_dataset")
    print(f"\n[5] INDI dataset: {INDI_BASE}")
    if INDI_BASE.exists():
        for sub in sorted(INDI_BASE.iterdir())[:20]:
            if sub.is_dir():
                # Count up to a sane upper bound to avoid stalling on huge dirs
                n = 0
                for _ in sub.iterdir():
                    n += 1
                    if n > 5000:
                        n = ">5000"
                        break
                print(f"  {sub.name}/  ~{n} entries")
            else:
                print(f"  {sub.name}  ({sub.stat().st_size} bytes)")
    else:
        print(f"  not found")

    # 6. ESMFold output dir — possibly folded VHH structures
    ESMFOLD_BASE = Path("/projects/0/hpmlprjs/interns/krijn/ESMFold_Snellius")
    print(f"\n[6] ESMFold output: {ESMFOLD_BASE}")
    if ESMFOLD_BASE.exists():
        for sub in sorted(ESMFOLD_BASE.iterdir())[:20]:
            if sub.is_dir():
                n_pdb = sum(1 for _ in sub.glob("**/*.pdb"))
                print(f"  {sub.name}/  PDB={n_pdb}")
            else:
                print(f"  {sub.name}")
    else:
        print(f"  not found")

    # 7. Any other curation artifacts that might tell us what was filtered
    curation_artifacts = [
        PROJECT_ROOT / "scripts/datasets",
        PROJECT_ROOT / "data scripts",  # noted in your dir listing
        PROJECT_ROOT / "data/datasets",
    ]
    print(f"\n[7] Curation artifacts / scripts that built the current dataset")
    for p in curation_artifacts:
        if p.exists() and p.is_dir():
            print(f"  {p}/")
            for child in sorted(p.iterdir())[:10]:
                print(f"    {child.name}")
        elif p.exists():
            print(f"  {p}  ({p.stat().st_size} bytes)")
        else:
            print(f"  not at: {p}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
