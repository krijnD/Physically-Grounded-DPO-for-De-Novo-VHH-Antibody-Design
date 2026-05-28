#!/usr/bin/env python3
"""Check 1: are the AAPR loser PDBs structurally well-formed?

For 20 random pairs:
  - Can the PDB be parsed by Biopython?
  - Does the heavy chain have the expected residue count (matches GT)?
  - Are all backbone atoms (N, CA, C, O) present at every residue?
  - Are all residue types valid 20-AA?
  - Are CA-CA distances reasonable (<5Å between consecutive residues)?

Output: per-pair OK/ISSUE line + summary counts.
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

from Bio.PDB import PDBParser  # type: ignore
from diffab.datasets import get_dataset  # noqa: E402
from diffab.utils.misc import load_config  # noqa: E402
import src.diffab_ft.datasets  # noqa: E402, F401  (registry side effect)

THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}


def inspect_pdb(pdb_path: Path, heavy_chain_id: str) -> dict:
    """Return well-formedness diagnostics for one PDB's heavy chain."""
    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("x", str(pdb_path))
    except Exception as e:
        return {"ok": False, "error": f"parse failed: {e!r}"}

    model = next(struct.get_models())
    if heavy_chain_id not in model:
        return {"ok": False, "error": f"heavy_chain {heavy_chain_id!r} missing; have {[c.id for c in model]}"}
    chain = model[heavy_chain_id]
    residues = [r for r in chain if r.id[0] == " "]
    n_res = len(residues)

    missing_atoms = 0
    invalid_residues = 0
    seq = []
    for r in residues:
        aa = THREE_TO_ONE.get(r.get_resname().upper())
        if aa is None:
            invalid_residues += 1
            seq.append("X")
        else:
            seq.append(aa)
        for atom in ("N", "CA", "C", "O"):
            if atom not in r:
                missing_atoms += 1

    ca_coords = np.array(
        [r["CA"].coord for r in residues if "CA" in r], dtype=float
    )
    if len(ca_coords) > 1:
        ca_dists = np.linalg.norm(ca_coords[1:] - ca_coords[:-1], axis=1)
        breaks = int(np.sum(ca_dists > 5.0))
        cd_min, cd_med, cd_max = ca_dists.min(), float(np.median(ca_dists)), ca_dists.max()
    else:
        breaks, cd_min, cd_med, cd_max = -1, float("nan"), float("nan"), float("nan")

    ok = (missing_atoms == 0 and invalid_residues == 0 and breaks == 0)
    return {
        "ok": ok, "n_res": n_res,
        "missing_backbone_atoms": missing_atoms,
        "invalid_residues": invalid_residues,
        "breaks_gt_5A": breaks,
        "ca_min": float(cd_min), "ca_med": cd_med, "ca_max": float(cd_max),
        "sequence": "".join(seq),
    }


def main() -> int:
    pairs_path = PROJECT_ROOT / "data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs.parquet"
    pairs = pd.read_parquet(pairs_path)
    print(f"Loaded {len(pairs)} pairs")

    # Use the same entry source the PairDataset uses (robust to manifest schema
    # changes). build the pdbcode → heavy-chain map from sabdab_entries.
    config_path = PROJECT_ROOT / "configs/dpo/vhh_dpo.yml"
    config, _ = load_config(str(config_path))
    print("Building base dataset to get heavy-chain IDs...")
    base_dataset = get_dataset(config.dataset.train)
    live_ids = set(base_dataset.db_ids or [])
    pdb_to_H = {}
    for entry in base_dataset.sabdab_entries:
        if entry.get("id") in live_ids:
            pdb_to_H[entry["pdbcode"]] = str(entry.get("H_chain", "H"))
    print(f"Entry map covers {len(pdb_to_H)} PDB codes")

    random.seed(42)
    sample = pairs.sample(n=min(20, len(pairs)), random_state=42).reset_index(drop=True)

    print("\n=== Check 1: AAPR loser PDB well-formedness (n=20) ===\n")
    print(f"{'gt_id':<10} {'H':<3} {'status':<7} {'n_res':>5} {'miss':>4} {'badAA':>5} "
          f"{'brk':>3} {'ca_min/med/max':<18}")
    n_ok = 0
    issues = []
    for _, row in sample.iterrows():
        gt_id = str(row["gt_complex_id"])
        loser_path = Path(str(row["loser_pdb_path"]))
        if not loser_path.is_absolute():
            loser_path = PROJECT_ROOT / loser_path
        H = pdb_to_H.get(gt_id, "H")
        if not loser_path.exists():
            print(f"{gt_id:<10} {H:<3} MISSING {str(loser_path)}")
            issues.append((gt_id, "file not found"))
            continue
        d = inspect_pdb(loser_path, H)
        if "error" in d:
            print(f"{gt_id:<10} {H:<3} ERROR   {d['error']}")
            issues.append((gt_id, d["error"]))
            continue
        status = "OK" if d["ok"] else "ISSUE"
        if d["ok"]:
            n_ok += 1
        else:
            issues.append((gt_id, d))
        print(f"{gt_id:<10} {H:<3} {status:<7} {d['n_res']:>5} "
              f"{d['missing_backbone_atoms']:>4} {d['invalid_residues']:>5} "
              f"{d['breaks_gt_5A']:>3} "
              f"{d['ca_min']:.2f}/{d['ca_med']:.2f}/{d['ca_max']:.2f}")

    print(f"\nSummary: {n_ok}/20 well-formed")
    if issues:
        print("First 5 issues:")
        for gt, info in issues[:5]:
            print(f"  {gt}: {info if isinstance(info, str) else {k: v for k, v in info.items() if k != 'sequence'}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
