"""
CAAR + EpiF1 dispatcher (Brief 13).

Vendored from CHIMERA-Bench (Mansoor et al. 2026, MIT licence;
https://github.com/mansoor181/chimera-bench commit a49d85d):
  - caar()                 : evaluation/metrics.py L27-33
  - compute_epitope_mask() : evaluation/metrics.py L152-171  (8.0 A Ca-Ca)
  - epitope_metrics()      : evaluation/metrics.py L174-206
GT paratope/epitope labelling follows data/annotate.py L88-147
(4.5 A atom-pair NeighborSearch, config.contact_cutoff = 4.5).

Per design PDB:
  1. Parse path to (variant, test_set, entry_id, cdr, sample).
  2. Load GT PDB from gt_pdb_map[entry_id]. Identify VHH (single chain
     length 100-160) and antigen (all other polymer chains) on each side.
  3. Build GT paratope_mask over the masked CDR's resseq window using
     4.5 A heavy-atom NeighborSearch on the GT structure.
  4. Build GT epitope_set as set of (ag_chain, resseq) tuples whose
     atoms come within 4.5 A of any GT-CDR atom.
  5. Build design seq / design epitope_set (using 8.0 A Ca-Ca per
     ChimeraBench convention for predicted-side epitope).
  6. CAAR  = AAR of design vs GT at paratope_mask positions.
  7. EpiF1 = F1 of design-epitope vs GT-epitope sets (by (chain, resseq)).

Shardable: pass --array-task-id / --array-task-count.

Usage:
    python scripts/eval/run_caar_epif1_array.py \\
        --design-pdb-roots runs/vhh_ft/seed42_jfix/eval_test_pdbs \\
        --gt-pdb-map data/eval/gt_pdb_map.json \\
        --output data/eval/caar_epif1.parquet \\
        --array-task-id 0 --array-task-count 32
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, NeighborSearch

# ChimeraBench config.py L63: contact_cutoff = 4.5 A (atom-pair, for GT labels)
CONTACT_THRESHOLD_A = 4.5
# ChimeraBench metrics.py L153/187: cutoff = 8.0 A (Ca-Ca, for predicted epitope)
CA_CONTACT_THRESHOLD_A = 8.0

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
       "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
       "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
       "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}

# Chothia-ish CDR windows used by the author-numbered VHH pipeline
# (handoff §10 / Brief 02 Check 5/6).
CDR_WINDOWS = {"H1": (26, 32), "H2": (52, 56), "H3": (95, 102)}


def _classify_chains(structure):
    """Return (vhh_chain_id, list_of_antigen_chain_ids).

    VHH = single polymer chain of length 100-160;
    antigen = other polymer chains (non-empty).
    """
    vhh, antigens = None, []
    for chain in structure.get_chains():
        residues = [r for r in chain.get_residues() if r.id[0] == " "]
        n = len(residues)
        if 100 <= n <= 160 and vhh is None:
            vhh = chain.id
        elif n > 0:
            antigens.append(chain.id)
    return vhh, antigens


def _cdr_residues(structure, vhh_chain_id, cdr):
    """List of residues in the VHH chain within the integer CDR window."""
    win = CDR_WINDOWS[cdr]
    out = []
    for chain in structure.get_chains():
        if chain.id != vhh_chain_id:
            continue
        for r in chain.get_residues():
            if r.id[0] != " ":
                continue
            if win[0] <= r.id[1] <= win[1]:
                out.append(r)
    return out


def _ag_atoms(structure, ag_chain_ids):
    """Flat list of all antigen-chain atoms (for NeighborSearch)."""
    out = []
    for chain in structure.get_chains():
        if chain.id in ag_chain_ids:
            for r in chain.get_residues():
                if r.id[0] != " ":
                    continue
                out.extend(list(r.get_atoms()))
    return out


def _ag_residues(structure, ag_chain_ids):
    """List of (chain_id, residue) tuples for antigen residues."""
    out = []
    for chain in structure.get_chains():
        if chain.id in ag_chain_ids:
            for r in chain.get_residues():
                if r.id[0] == " ":
                    out.append((chain.id, r))
    return out


def _gt_contacts_at_cdr(gt_struct, vhh_chain_id, ag_chain_ids, cdr):
    """ChimeraBench-style 4.5 A atom-pair contacts restricted to the CDR window.

    Returns:
      paratope_resseqs : set of CDR resseqs (in the VHH chain) with any
                         atom <= 4.5 A from any antigen atom in the GT.
      epitope_set      : set of (ag_chain, resseq) for antigen residues
                         contacted by CDR atoms.
      gt_cdr_seq_dict  : {resseq: aa1} mapping over the CDR window
                         (for downstream CAAR alignment).
    """
    cdr_residues = _cdr_residues(gt_struct, vhh_chain_id, cdr)
    ag_atoms = _ag_atoms(gt_struct, ag_chain_ids)
    if not cdr_residues or not ag_atoms:
        return set(), set(), {}
    ns = NeighborSearch(ag_atoms)
    paratope_resseqs = set()
    epitope_set = set()
    for cdr_res in cdr_residues:
        for cdr_atom in cdr_res.get_atoms():
            nearby = ns.search(cdr_atom.get_coord(),
                               CONTACT_THRESHOLD_A, "A")
            for ag_atom in nearby:
                ag_res = ag_atom.get_parent()
                ag_chain = ag_res.get_parent().id
                paratope_resseqs.add(cdr_res.id[1])
                epitope_set.add((ag_chain, ag_res.id[1]))
    gt_cdr_seq_dict = {
        r.id[1]: AA3.get(r.get_resname(), "X") for r in cdr_residues
    }
    return paratope_resseqs, epitope_set, gt_cdr_seq_dict


def _design_cdr_seq_dict(dsg_struct, vhh_chain_id, cdr):
    """{resseq: aa1} over the design CDR window."""
    cdr_residues = _cdr_residues(dsg_struct, vhh_chain_id, cdr)
    return {r.id[1]: AA3.get(r.get_resname(), "X") for r in cdr_residues}


def _design_epitope_set(dsg_struct, vhh_chain_id, ag_chain_ids, cdr):
    """8.0 A Ca-Ca: predicted CDR Ca atoms vs antigen Ca atoms.

    Returns set of (ag_chain, resseq) tuples.
    (ChimeraBench compute_epitope_mask convention — metrics.py L152-171.)
    """
    cdr_residues = _cdr_residues(dsg_struct, vhh_chain_id, cdr)
    ag_residues_pairs = _ag_residues(dsg_struct, ag_chain_ids)
    if not cdr_residues or not ag_residues_pairs:
        return set()
    cdr_ca = []
    for r in cdr_residues:
        if "CA" in r:
            cdr_ca.append(np.array(r["CA"].get_coord()))
    if not cdr_ca:
        return set()
    cdr_ca = np.stack(cdr_ca)
    out = set()
    for ag_chain, ag_res in ag_residues_pairs:
        if "CA" not in ag_res:
            continue
        ag_ca = np.array(ag_res["CA"].get_coord())
        d = np.linalg.norm(cdr_ca - ag_ca[None, :], axis=1).min()
        if d <= CA_CONTACT_THRESHOLD_A:
            out.add((ag_chain, ag_res.id[1]))
    return out


def _compute_metrics(gt_pdb, design_pdb, cdr):
    parser = PDBParser(QUIET=True)
    try:
        gt_struct = parser.get_structure("gt", gt_pdb)
        dsg_struct = parser.get_structure("dsg", design_pdb)
    except Exception as e:
        return {"error": f"parse_fail:{str(e)[:80]}"}

    gt_vhh, gt_ag = _classify_chains(gt_struct)
    dsg_vhh, dsg_ag = _classify_chains(dsg_struct)
    if not gt_vhh or not dsg_vhh:
        return {"error": "vhh_chain_missing"}
    if not gt_ag or not dsg_ag:
        return {"error": "antigen_chain_missing"}

    paratope_resseqs, gt_epi_set, gt_cdr_aas = _gt_contacts_at_cdr(
        gt_struct, gt_vhh, gt_ag, cdr
    )
    dsg_cdr_aas = _design_cdr_seq_dict(dsg_struct, dsg_vhh, cdr)
    dsg_epi_set = _design_epitope_set(dsg_struct, dsg_vhh, dsg_ag, cdr)

    # CAAR — restricted to GT paratope positions; align by resseq
    if not paratope_resseqs:
        caar_val, caar_n = float("nan"), 0
    else:
        matched = sum(
            1 for p in paratope_resseqs
            if gt_cdr_aas.get(p, "X") == dsg_cdr_aas.get(p, "X")
        )
        caar_val = 100.0 * matched / len(paratope_resseqs)
        caar_n = len(paratope_resseqs)

    # EpiF1 — compare by (chain, resseq); DiffAb preserves antigen
    # chain IDs and resseqs as conditioning, so this is well-aligned.
    overlap = gt_epi_set & dsg_epi_set
    if not gt_epi_set and not dsg_epi_set:
        precision, recall, f1 = float("nan"), float("nan"), float("nan")
    elif not gt_epi_set or not dsg_epi_set:
        precision, recall, f1 = 0.0, 0.0, 0.0
    else:
        precision = len(overlap) / len(dsg_epi_set)
        recall = len(overlap) / len(gt_epi_set)
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

    return {
        "caar": caar_val,
        "caar_n_positions": caar_n,
        "epif1": f1,
        "epif1_precision": precision,
        "epif1_recall": recall,
        "gt_paratope_n": len(paratope_resseqs),
        "gt_epitope_n": len(gt_epi_set),
        "design_epitope_n": len(dsg_epi_set),
        "overlap_n": len(overlap),
        "error": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design-pdb-roots", nargs="+", required=True,
                    help="One or more root dirs containing variant/eval_*_pdbs/ trees.")
    ap.add_argument("--gt-pdb-map", required=True,
                    help="JSON {entry_id: gt_pdb_path}")
    ap.add_argument("--output", required=True, help="Output parquet path.")
    ap.add_argument("--array-task-id", type=int, default=0)
    ap.add_argument("--array-task-count", type=int, default=1)
    args = ap.parse_args()

    gt_map = json.loads(Path(args.gt_pdb_map).read_text())
    pdbs = []
    for root in args.design_pdb_roots:
        pdbs.extend(sorted(glob.glob(f"{root}/**/sample_*.pdb", recursive=True)))
    my_pdbs = [p for i, p in enumerate(pdbs)
               if i % args.array_task_count == args.array_task_id]
    print(f"Task {args.array_task_id}/{args.array_task_count} → {len(my_pdbs)} of {len(pdbs)} PDBs")

    rows = []
    for i, pdb in enumerate(my_pdbs):
        parts = Path(pdb).parts
        try:
            cdr = parts[-2]
            entry = parts[-3]
            eval_dir = parts[-4]
            variant = parts[-5]
            test_set = eval_dir.replace("eval_", "").replace("_pdbs", "")
            sample = int(parts[-1].replace("sample_", "").replace(".pdb", ""))
        except (IndexError, ValueError) as e:
            rows.append({"pdb_path": pdb, "error": f"path_parse:{e}"})
            continue

        gt_pdb = gt_map.get(entry)
        if not gt_pdb:
            rows.append({"pdb_path": pdb, "variant": variant, "test_set": test_set,
                         "entry_id": entry, "cdr": cdr, "sample": sample,
                         "error": "gt_pdb_not_in_map"})
            continue

        metrics = _compute_metrics(gt_pdb, pdb, cdr)
        rows.append({"pdb_path": pdb, "variant": variant, "test_set": test_set,
                     "entry_id": entry, "cdr": cdr, "sample": sample, **metrics})
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(my_pdbs)}] processed")

    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    shard = args.output.replace(".parquet", f"_task{args.array_task_id}.parquet")
    df.to_parquet(shard)
    print(f"Wrote {len(df)} rows → {shard}")


if __name__ == "__main__":
    main()
