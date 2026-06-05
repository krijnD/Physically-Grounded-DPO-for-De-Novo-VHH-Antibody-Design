"""
scRMSD dispatcher (Brief 12) — folds each generated PDB's heavy-chain sequence
with ABodyBuilder2 (NanoBodyBuilder2), Kabsch-aligns the predicted backbone to
the generated backbone on framework Cα atoms, and computes per-CDR Cα RMSD.

Persists one parquet shard per array task; merge handled downstream in Step 4.

Alignment strategy:
    Both PDBs encode the *same* 114-ish-residue VHH sequence (the predictor is
    folding the sequence we extracted from the generated PDB). They use
    DIFFERENT numbering conventions, though: generated PDBs use sequential
    resseq 1..N; ABodyBuilder2 emits a canonical-numbered output with gaps.
    Matching residues by resseq therefore fails. Instead, we match by chain
    iteration order — the i-th standard residue of the generated chain
    corresponds to the i-th standard residue of the predicted chain by
    construction. CDR boundary indices are derived from the generated PDB's
    resseq (where the H1=26-32 / H2=52-56 / H3=95-102 windows are known to
    work — Brief 12 Step 1c), then applied as positional indices to both
    PDBs. `seq_identity_pct` is emitted as a sanity column — should be 100%
    on every row.

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

# CDR position windows on the GENERATED PDB's sequential resseq numbering.
# Confirmed against a Brief 11 PDB in Step 1c (H1=7, H2=5, H3=8 residues).
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


def get_chain_residues(pdb_path, chain_id):
    """Return ordered list of standard residues (resname ∈ AA3) on chain."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", pdb_path)
    for chain in struct.get_chains():
        if chain.id != chain_id:
            continue
        return [
            r for r in chain.get_residues()
            if r.id[0] == " " and r.get_resname() in AA3
        ]
    return []


def derive_cdr_indices_from_generated(gen_residues):
    """Map CDR_WINDOWS (sequential resseq on generated PDB) → chain-order indices."""
    cdr_idx = {"H1": [], "H2": [], "H3": []}
    for i, r in enumerate(gen_residues):
        resseq = r.id[1]
        for cdr, (lo, hi) in CDR_WINDOWS.items():
            if lo <= resseq <= hi:
                cdr_idx[cdr].append(i)
                break
    return cdr_idx


def find_pred_offset_in_gen(gen_residues, pred_residues):
    """Find offset where pred-sequence matches as substring of gen-sequence.

    ABodyBuilder2 uses ANARCI to detect the V-domain envelope and silently
    trims residues outside it (typically a small C-terminal tail past the J
    motif). When that happens, pred is a contiguous substring of gen at some
    offset; the i-th pred residue is the predicted fold of the (i+offset)-th
    gen residue. Returns offset (≥0) or None on no match.
    """
    if len(gen_residues) < len(pred_residues):
        return None
    gen_seq = "".join(AA3[r.get_resname()] for r in gen_residues)
    pred_seq = "".join(AA3[r.get_resname()] for r in pred_residues)
    idx = gen_seq.find(pred_seq)
    return idx if idx >= 0 else None


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


def compute_scrmsd(generated_pdb, predicted_pdb, gen_chain_id,
                   pred_chain_id="H"):
    """Per-CDR scRMSD via chain-order-index alignment.

    Returns a dict with: n_gen_res, n_pred_res, seq_identity_pct,
    fw_atoms, H1_atoms, H2_atoms, H3_atoms, scrmsd_H1, scrmsd_H2,
    scrmsd_H3, error.
    """
    gen_res = get_chain_residues(generated_pdb, gen_chain_id)
    pred_res = get_chain_residues(predicted_pdb, pred_chain_id)
    n_gen, n_pred = len(gen_res), len(pred_res)

    out = {
        "n_gen_res": n_gen, "n_pred_res": n_pred,
        "gen_trim_offset": 0, "gen_trim_n_dropped": 0,
        "seq_identity_pct": np.nan,
        "fw_atoms": 0,
        "H1_atoms": 0, "H2_atoms": 0, "H3_atoms": 0,
        "scrmsd_H1": np.nan, "scrmsd_H2": np.nan, "scrmsd_H3": np.nan,
        "error": None,
    }

    if n_gen == 0 or n_pred == 0:
        out["error"] = f"empty_chain_gen{n_gen}_pred{n_pred}"
        return out

    # Handle ANARCI's V-domain trim (pred is a substring of gen, usually
    # C-terminal trim by 1-4 residues). Falls back to count-equality otherwise.
    if n_gen != n_pred:
        offset = find_pred_offset_in_gen(gen_res, pred_res)
        if offset is None:
            out["error"] = f"seq_no_substring_match_{n_gen}_vs_{n_pred}"
            return out
        out["gen_trim_offset"] = offset
        out["gen_trim_n_dropped"] = n_gen - n_pred
        gen_res = gen_res[offset: offset + n_pred]
        n_gen = len(gen_res)

    # Sequence identity sanity (should be 100% — same sequence input/output)
    seq_match = sum(
        1 for i in range(n_gen)
        if gen_res[i].get_resname() == pred_res[i].get_resname()
    )
    out["seq_identity_pct"] = round(100 * seq_match / n_gen, 1)

    cdr_idx = derive_cdr_indices_from_generated(gen_res)
    cdr_set = set().union(*cdr_idx.values())

    fw_indices = [
        i for i in range(n_gen)
        if i not in cdr_set
        and "CA" in gen_res[i]
        and "CA" in pred_res[i]
    ]
    out["fw_atoms"] = len(fw_indices)
    if not fw_indices:
        out["error"] = "no_framework_ca"
        return out

    gen_fw = np.array([gen_res[i]["CA"].get_coord() for i in fw_indices])
    pred_fw = np.array([pred_res[i]["CA"].get_coord() for i in fw_indices])
    R, t = kabsch(pred_fw, gen_fw)

    for cdr in ["H1", "H2", "H3"]:
        valid = [
            i for i in cdr_idx[cdr]
            if "CA" in gen_res[i] and "CA" in pred_res[i]
        ]
        out[f"{cdr}_atoms"] = len(valid)
        if not valid:
            continue
        gen_cdr = np.array([gen_res[i]["CA"].get_coord() for i in valid])
        pred_cdr = np.array([pred_res[i]["CA"].get_coord() for i in valid])
        pred_cdr_aligned = (R @ pred_cdr.T).T + t
        rmsd = float(np.sqrt(((gen_cdr - pred_cdr_aligned) ** 2).sum(1).mean()))
        out[f"scrmsd_{cdr}"] = rmsd

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
                    "n_gen_res": 0, "n_pred_res": 0,
                    "gen_trim_offset": 0, "gen_trim_n_dropped": 0,
                    "seq_identity_pct": np.nan,
                    "fw_atoms": 0,
                    "H1_atoms": 0, "H2_atoms": 0, "H3_atoms": 0,
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
                print(f"  n_gen_res={sc['n_gen_res']} n_pred_res={sc['n_pred_res']} "
                      f"gen_trim_offset={sc['gen_trim_offset']} "
                      f"gen_trim_n_dropped={sc['gen_trim_n_dropped']} "
                      f"seq_identity_pct={sc['seq_identity_pct']}", flush=True)
                print(f"  fw_atoms={sc['fw_atoms']} "
                      f"H1_atoms={sc['H1_atoms']} H2_atoms={sc['H2_atoms']} "
                      f"H3_atoms={sc['H3_atoms']}", flush=True)
                print(f"  scrmsd: H1={sc['scrmsd_H1']} H2={sc['scrmsd_H2']} "
                      f"H3={sc['scrmsd_H3']}", flush=True)
                debug_first = False
            row = {**meta, "pdb_path": pdb, **sc}
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
                "n_gen_res": 0, "n_pred_res": 0,
                "seq_identity_pct": np.nan,
                "fw_atoms": 0,
                "H1_atoms": 0, "H2_atoms": 0, "H3_atoms": 0,
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
