"""
scRMSD dispatcher (Brief 12) — folds each generated PDB's heavy-chain sequence
with ABodyBuilder2 (NanoBodyBuilder2), Kabsch-aligns the predicted backbone to
the generated backbone on framework Cα atoms, and computes per-CDR Cα RMSD.

Persists one parquet shard per array task; merge handled downstream in Step 4.

Usage:
    python scripts/eval/run_scrmsd_array.py \\
        --pdb-roots <root1> [<root2> ...] \\
        --output    data/eval/scrmsd_design_samples.parquet \\
        --array-task-id    $SLURM_ARRAY_TASK_ID \\
        --array-task-count 32

Path convention (must match Brief 11's eval-PDB tree):
    <variant_dir>/eval_<testset>_pdbs/<entry>/<cdr>/sample_NNNN.pdb
"""
import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# CDR position windows confirmed against Brief 11 PDB sample (Brief 12 Step 1c).
# Convention: integer resseq under the project's author-numbered VHH parser.
CDR_WINDOWS = {"H1": (26, 32), "H2": (52, 56), "H3": (95, 102)}


def extract_heavy_seq(pdb_path):
    """Return (heavy_chain_id, sequence string) from a design PDB."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", pdb_path)
    for chain in struct.get_chains():
        seq = "".join(
            AA3.get(r.get_resname(), "X")
            for r in chain.get_residues()
            if r.id[0] == " " and r.get_resname() in AA3
        )
        if 100 <= len(seq) <= 160:
            return chain.id, seq
    return None, None


def get_ca_coords(pdb_path, chain_id, resseq_range):
    """Return (N, 3) Cα coord array for residues with resseq in [lo, hi]."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", pdb_path)
    coords = []
    for chain in struct.get_chains():
        if chain.id != chain_id:
            continue
        for r in chain.get_residues():
            if r.id[0] != " ":
                continue
            resseq = r.id[1]
            if resseq_range[0] <= resseq <= resseq_range[1] and "CA" in r:
                coords.append(r["CA"].get_coord())
    return np.array(coords) if coords else np.zeros((0, 3))


def framework_ca(pdb_path, chain_id):
    """Cα coords on framework only (exclude H1/H2/H3 windows)."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", pdb_path)
    coords = []
    for chain in struct.get_chains():
        if chain.id != chain_id:
            continue
        for r in chain.get_residues():
            if r.id[0] != " ":
                continue
            resseq = r.id[1]
            in_cdr = any(lo <= resseq <= hi for (lo, hi) in CDR_WINDOWS.values())
            if not in_cdr and "CA" in r:
                coords.append(r["CA"].get_coord())
    return np.array(coords) if coords else np.zeros((0, 3))


def kabsch(P, Q):
    """Optimal rotation+translation mapping P → Q (both (N,3) arrays)."""
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = Q.mean(0) - R @ P.mean(0)
    return R, t


def compute_scrmsd(generated_pdb, predicted_pdb, gen_chain_id, pred_chain_id="H"):
    """Return dict of per-CDR scRMSD after framework Kabsch alignment."""
    gen_fw = framework_ca(generated_pdb, gen_chain_id)
    pred_fw = framework_ca(predicted_pdb, pred_chain_id)
    out = {
        "fw_atoms_gen": int(len(gen_fw)),
        "fw_atoms_pred": int(len(pred_fw)),
    }
    if len(gen_fw) == 0 or len(pred_fw) == 0 or len(gen_fw) != len(pred_fw):
        out.update({
            "H1": np.nan, "H2": np.nan, "H3": np.nan,
            "error": f"fw_atom_mismatch_{len(gen_fw)}_vs_{len(pred_fw)}",
        })
        return out
    R, t = kabsch(pred_fw, gen_fw)
    for cdr, win in CDR_WINDOWS.items():
        gen_cdr = get_ca_coords(generated_pdb, gen_chain_id, win)
        pred_cdr = get_ca_coords(predicted_pdb, pred_chain_id, win)
        out[f"{cdr}_atoms_gen"] = int(len(gen_cdr))
        out[f"{cdr}_atoms_pred"] = int(len(pred_cdr))
        if len(gen_cdr) == 0 or len(pred_cdr) == 0 or len(gen_cdr) != len(pred_cdr):
            out[cdr] = np.nan
            continue
        pred_cdr_aligned = (R @ pred_cdr.T).T + t
        rmsd = float(np.sqrt(((gen_cdr - pred_cdr_aligned) ** 2).sum(1).mean()))
        out[cdr] = rmsd
    out["error"] = None
    return out


def parse_path(pdb_path):
    """Parse .../<variant_dir>/eval_<testset>_pdbs/<entry>/<cdr>/sample_NNNN.pdb."""
    parts = Path(pdb_path).parts
    sample_file = parts[-1]
    cdr_dir = parts[-2]
    entry = parts[-3]
    eval_dir = parts[-4]
    variant_dir = parts[-5]
    test_set = eval_dir.replace("eval_", "").replace("_pdbs", "")
    sample = int(sample_file.replace("sample_", "").replace(".pdb", ""))
    return {
        "variant": variant_dir,
        "test_set": test_set,
        "entry": entry,
        "cdr": cdr_dir,
        "sample": sample,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-roots", nargs="+", required=True,
                    help="Root dirs containing eval_<testset>_pdbs/ trees.")
    ap.add_argument("--output", required=True,
                    help="Base output path; per-task shard is "
                         "<output>_task<id>.parquet")
    ap.add_argument("--array-task-id", type=int, default=0)
    ap.add_argument("--array-task-count", type=int, default=1)
    args = ap.parse_args()

    all_pdbs = []
    for root in args.pdb_roots:
        all_pdbs.extend(
            sorted(glob.glob(f"{root}/**/sample_*.pdb", recursive=True))
        )
    print(f"Found {len(all_pdbs)} PDBs across {len(args.pdb_roots)} root(s)",
          flush=True)

    my_pdbs = [
        p for i, p in enumerate(all_pdbs)
        if i % args.array_task_count == args.array_task_id
    ]
    print(f"Task {args.array_task_id}/{args.array_task_count} → "
          f"{len(my_pdbs)} PDBs", flush=True)
    if not my_pdbs:
        print("Empty shard; exiting cleanly.")
        return

    from ImmuneBuilder import NanoBodyBuilder2
    t0 = time.time()
    predictor = NanoBodyBuilder2()
    print(f"Predictor loaded in {time.time() - t0:.1f}s", flush=True)

    rows = []
    debug_first = True
    for i, pdb in enumerate(my_pdbs):
        try:
            meta = parse_path(pdb)
        except (IndexError, ValueError) as e:
            print(f"PATH PARSE FAIL {pdb}: {e}", flush=True)
            continue

        try:
            chain_id, seq = extract_heavy_seq(pdb)
            if not seq:
                rows.append({
                    **meta, "pdb_path": pdb,
                    "scrmsd_H1": np.nan, "scrmsd_H2": np.nan, "scrmsd_H3": np.nan,
                    "error": "no_heavy_chain",
                })
                continue
            tmp_pred = f"/tmp/abb2_pred_{args.array_task_id}_{i}.pdb"
            result = predictor.predict({"H": seq})
            result.save(tmp_pred)
            sc = compute_scrmsd(pdb, tmp_pred, chain_id, pred_chain_id="H")
            if debug_first:
                print(f"DEBUG first PDB: {pdb}", flush=True)
                print(f"  gen_chain={chain_id}, seq_len={len(seq)}", flush=True)
                print(f"  fw_atoms gen={sc['fw_atoms_gen']} "
                      f"pred={sc['fw_atoms_pred']}", flush=True)
                print(f"  per-CDR: H1={sc.get('H1')} "
                      f"H2={sc.get('H2')} H3={sc.get('H3')}", flush=True)
                debug_first = False
            row = {
                **meta, "pdb_path": pdb,
                "scrmsd_H1": sc.get("H1"),
                "scrmsd_H2": sc.get("H2"),
                "scrmsd_H3": sc.get("H3"),
                "fw_atoms_gen": sc.get("fw_atoms_gen"),
                "fw_atoms_pred": sc.get("fw_atoms_pred"),
                "error": sc.get("error"),
            }
            rows.append(row)
            try:
                os.unlink(tmp_pred)
            except OSError:
                pass
            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(my_pdbs)}] processed", flush=True)
        except Exception as e:  # noqa: BLE001
            rows.append({
                **meta, "pdb_path": pdb,
                "scrmsd_H1": np.nan, "scrmsd_H2": np.nan, "scrmsd_H3": np.nan,
                "error": str(e)[:200],
            })

    out_df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    shard_path = args.output.replace(
        ".parquet", f"_task{args.array_task_id}.parquet"
    )
    out_df.to_parquet(shard_path)
    print(f"Wrote {len(out_df)} rows to {shard_path}", flush=True)


if __name__ == "__main__":
    main()
