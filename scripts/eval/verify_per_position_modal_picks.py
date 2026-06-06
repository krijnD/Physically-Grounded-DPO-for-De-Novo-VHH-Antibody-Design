"""
Verification — does Brief 13's per_position_modal_picks_all.parquet
actually read the H3 CDR, or does it read FR3 framework?

Thesis writer flagged a discrepancy: Brief 13's parquet says H3 modal
motif K-P-E-D-T-A-V-Y at "positions 95-102" for every model variant,
but aggregating gen_seq (the H3 CDR string used by the AAR
computation) gives modal motif Y-C-A-A-A-G-G-G at positions 0-7.
KPEDTAVY appears in raw_sequence at linear positions 85-92 (the end
of FR3, just before the conserved Cys at ~93).

This script reproduces both analyses side by side, then inspects one
design PDB residue-by-residue to find where gen_seq's H3 sequence
actually lives. Output is a verdict block the orchestrator can use to
decide whether to backup+regenerate the parquet.

Run from the campaign repo root:
    cd Physically-Grounded-DPO-for-De-Novo-VHH-Antibody-Design
    source .venv/bin/activate
    python scripts/eval/verify_per_position_modal_picks.py
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser

PROJECT_ROOT = Path(__file__).resolve().parents[2]
THESIS_ROOT = PROJECT_ROOT.parent / "master-thesis"
PARQUET_BRIEF13 = PROJECT_ROOT / "data/eval/per_position_modal_picks_all.parquet"
MASTER_PARQUET  = PROJECT_ROOT / "data/eval/design_samples_master.parquet"
CSV_DIR         = THESIS_ROOT / "data/eval/per_sample_csvs"

# Design PDB mirrored locally as a Brief-12 artifact:
DESIGN_PDB_SAMPLE = PROJECT_ROOT / "data/eval/fig12c_inputs/7n9v_J_H3_sample_0001.pdb"
GT_PDB_SAMPLE     = PROJECT_ROOT / "data/eval/fig12c_inputs/7n9v.pdb"

# Diagnostic slice — the writer's headline cell
SLICE_VARIANT = "seed42_jfix"
SLICE_TEST    = "oldtest"
SLICE_CDR     = "H3"
SLICE_EXPECTED_N_GEN = 116           # 29 entries × 4 samples

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
       "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
       "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
       "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def header(text):
    print("\n" + "═" * 72)
    print(text)
    print("═" * 72)


# ════════════════════════════════════════════════════════════════════════
# STAGE A — read the parquet's claim
# ════════════════════════════════════════════════════════════════════════
def stage_a():
    header("STAGE A — Brief 13's parquet says (seed42_jfix × oldtest × H3)")
    if not PARQUET_BRIEF13.exists():
        sys.exit(f"FATAL: {PARQUET_BRIEF13} missing")
    p = pd.read_parquet(PARQUET_BRIEF13)
    sub = p[(p["variant"] == SLICE_VARIANT)
            & (p["test_set"] == SLICE_TEST)
            & (p["cdr"] == SLICE_CDR)].sort_values("position")
    print(f"Rows for the slice: {len(sub)}")
    cols = ["position", "n_gt", "n_gen", "gt_modal_aa", "gt_modal_freq",
            "gen_modal_aa", "gen_modal_freq"]
    print(sub[cols].to_string(index=False))
    gen_motif = "".join(sub.sort_values("position")["gen_modal_aa"].fillna("?"))
    gt_motif  = "".join(sub.sort_values("position")["gt_modal_aa"].fillna("?"))
    print(f"\nParquet gen modal motif (positions 95→102): {gen_motif}")
    print(f"Parquet GT  modal motif (positions 95→102): {gt_motif}")
    return gen_motif, gt_motif


# ════════════════════════════════════════════════════════════════════════
# STAGE B — gen_seq position-by-position aggregation
# ════════════════════════════════════════════════════════════════════════
def stage_b():
    header("STAGE B — gen_seq[k] modal pick from per-sample CSV")
    csv = CSV_DIR / f"{SLICE_VARIANT}_{SLICE_TEST}.csv"
    if not csv.exists():
        sys.exit(f"FATAL: {csv} missing")
    df = pd.read_csv(csv)
    print(f"CSV total rows: {len(df)}; CDRs: {df['cdr'].value_counts().to_dict()}")

    sub = df[df["cdr"] == SLICE_CDR].copy()
    print(f"\nH3 rows: {len(sub)} (expected ~{SLICE_EXPECTED_N_GEN})")

    # Length distribution
    sub["gen_len"]    = sub["gen_seq"].astype(str).str.len()
    sub["native_len"] = sub["native_seq"].astype(str).str.len()
    print("\ngen_seq length distribution:")
    print(sub["gen_len"].value_counts().sort_index().to_string())
    print("\nnative_seq length distribution:")
    print(sub["native_len"].value_counts().sort_index().to_string())

    # Use the modal length as the canonical L
    L = int(sub["native_len"].mode().iloc[0])
    print(f"\nCanonical CDR length L = {L} (native modal); using gen samples with len==L")
    sub_L = sub[sub["gen_len"] == L]
    print(f"  n samples used: {len(sub_L)}")

    print(f"\nPer-position modal AA on gen_seq[0..{L-1}]:")
    print(f"{'k':>3} | {'gen_modal':<10}| {'gen_top3':<35} | "
          f"{'native_modal':<13}| native_top3")
    gen_motif_parts, gt_motif_parts = [], []
    for k in range(L):
        gen_col = [s[k] for s in sub_L["gen_seq"]]
        native_col = [s[k] for s in sub_L["native_seq"] if len(str(s)) > k]
        gc = Counter(gen_col).most_common(3)
        nc = Counter(native_col).most_common(3)
        gen_modal_aa, gen_modal_n = gc[0]
        gt_modal_aa,  gt_modal_n  = nc[0]
        gen_motif_parts.append(gen_modal_aa)
        gt_motif_parts.append(gt_modal_aa)
        gen_top3_s = ", ".join(f"{a}={n/len(gen_col):.2f}" for a, n in gc)
        nat_top3_s = ", ".join(f"{a}={n/len(native_col):.2f}" for a, n in nc)
        print(f"{k:>3} | {gen_modal_aa} {gen_modal_n/len(gen_col):>5.2%}  | "
              f"{gen_top3_s:<35} | "
              f"{gt_modal_aa} {gt_modal_n/len(native_col):>5.2%}  | {nat_top3_s}")
    gen_motif = "".join(gen_motif_parts)
    gt_motif = "".join(gt_motif_parts)
    print(f"\ngen_seq H3 modal motif (positions 0→{L-1}): {gen_motif}")
    print(f"native_seq H3 modal motif (positions 0→{L-1}): {gt_motif}")
    return gen_motif, gt_motif


# ════════════════════════════════════════════════════════════════════════
# STAGE C — search raw_sequence for KPEDTAVY (writer's FR3 hypothesis)
# ════════════════════════════════════════════════════════════════════════
def stage_c_substring():
    header("STAGE C-1 — substring 'KPEDTAVY' search in raw_sequence")
    if not MASTER_PARQUET.exists():
        sys.exit(f"FATAL: {MASTER_PARQUET} missing")
    m = pd.read_parquet(MASTER_PARQUET)
    sub = m[(m["variant"] == SLICE_VARIANT)
            & (m["test_set"] == SLICE_TEST)
            & (m["cdr"] == SLICE_CDR)]
    print(f"n samples in slice: {len(sub)}")
    n_total = len(sub)
    n_KPEDTAVY = sub["raw_sequence"].astype(str).str.contains("KPEDTAVY", na=False).sum()
    print(f"raw_sequence contains 'KPEDTAVY' as substring: "
          f"{n_KPEDTAVY}/{n_total} ({100*n_KPEDTAVY/n_total:.1f}%)")
    n_PEDTAVY = sub["raw_sequence"].astype(str).str.contains("PEDTAVY", na=False).sum()
    print(f"raw_sequence contains 'PEDTAVY' as substring: "
          f"{n_PEDTAVY}/{n_total} ({100*n_PEDTAVY/n_total:.1f}%)")
    if "cdr3_sequence" in sub.columns:
        n_cdr3_kpedtavy = sub["cdr3_sequence"].astype(str).str.contains(
            "KPEDTAVY", na=False).sum()
        print(f"cdr3_sequence contains 'KPEDTAVY': "
              f"{n_cdr3_kpedtavy}/{n_total} ({100*n_cdr3_kpedtavy/n_total:.1f}%)")

    # Find where PEDTAVY sits in raw_sequence (linear position)
    print("\nLinear position of 'PEDTAVY' inside raw_sequence (first 5 samples):")
    for i, raw in enumerate(sub["raw_sequence"].astype(str).head(5)):
        idx = raw.find("PEDTAVY")
        print(f"  sample[{i}] raw_sequence len={len(raw)}, "
              f"PEDTAVY at linear position {idx}: ...{raw[max(0,idx-3):idx+10]}...")

    # Distribution of PEDTAVY linear positions across all samples
    indices = (sub["raw_sequence"].astype(str)
               .apply(lambda s: s.find("PEDTAVY"))
               .replace(-1, pd.NA).dropna())
    if len(indices):
        print(f"\nLinear position of PEDTAVY: "
              f"min={int(indices.min())}, max={int(indices.max())}, "
              f"median={int(indices.median())}, mode={int(indices.mode().iloc[0])}, "
              f"n found={len(indices)}/{len(sub)}")


# ════════════════════════════════════════════════════════════════════════
# STAGE C-2 — inspect one design PDB residue by residue
# ════════════════════════════════════════════════════════════════════════
def stage_c_pdb():
    header("STAGE C-2 — design PDB residue-by-residue (7n9v_J_H3_sample_0001)")
    if not DESIGN_PDB_SAMPLE.exists():
        print(f"SKIP: {DESIGN_PDB_SAMPLE} missing (need Brief 12 fig12c_inputs)")
        return
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("design", DESIGN_PDB_SAMPLE)
    chains = list(s.get_chains())
    print(f"Chains: {[(c.id, sum(1 for r in c.get_residues() if r.id[0]==' ')) for c in chains]}")

    # entry_id = 7n9v_J → VHH chain hint = 'J'
    vhh_chain_id = "J"
    chain = next((c for c in chains if c.id == vhh_chain_id), None)
    if chain is None:
        # Fall back to longest polymer chain in [100, 160]
        for c in chains:
            n = sum(1 for r in c.get_residues() if r.id[0] == " ")
            if 100 <= n <= 160:
                chain = c
                vhh_chain_id = c.id
                break
    if chain is None:
        print("FATAL: no VHH chain found")
        return
    print(f"VHH chain: {vhh_chain_id}")

    # Build (resseq → aa1) mapping
    seq_by_resseq = {}
    for r in chain.get_residues():
        if r.id[0] != " ":
            continue
        seq_by_resseq[r.id[1]] = AA3.get(r.get_resname(), "X")
    resseqs = sorted(seq_by_resseq)
    print(f"VHH chain residues: resseq min={min(resseqs)}, "
          f"max={max(resseqs)}, n={len(resseqs)}")

    # Show the resseq range 80-110 to see FR3 / H3 boundary
    print("\nResidues at resseq 80→110 (one per line, resseq aa1):")
    for rs in range(80, 111):
        if rs in seq_by_resseq:
            print(f"  {rs:>3}  {seq_by_resseq[rs]}")
        else:
            print(f"  {rs:>3}  -")

    # Find the conserved Cys (it should be at FR3/H3 boundary, ~position 92)
    # And the "WGQG" or "WGKG" of FR4 (after H3)
    print("\nFR4 start (WGQ/WGK/WGR motif) inside chain — find its resseq:")
    seq_str_with_resseq = sorted(seq_by_resseq.items())
    seq_str = "".join(aa for _, aa in seq_str_with_resseq)
    for motif in ["WGQG", "WGKG", "WGRG", "WGQGT", "WGKGT"]:
        idx = seq_str.find(motif)
        if idx >= 0:
            resseq_of_motif = seq_str_with_resseq[idx][0]
            print(f"  {motif} at chain-index {idx} → starting resseq {resseq_of_motif}")
            break

    # Also: search for KPEDTAVY in this PDB's chain sequence and report its
    # resseq range
    for query in ["KPEDTAVY", "PEDTAVY", "EPEDTAVY", "RPEDTAVY"]:
        idx = seq_str.find(query)
        if idx >= 0:
            resseqs_for_query = [seq_str_with_resseq[idx + k][0]
                                 for k in range(len(query))]
            print(f"\n'{query}' found at chain-index {idx} "
                  f"→ resseqs {resseqs_for_query[0]}..{resseqs_for_query[-1]}")
            break

    # Cross-check: find this sample's gen_seq from the CSV and locate it in
    # the PDB sequence
    csv = CSV_DIR / f"{SLICE_VARIANT}_{SLICE_TEST}.csv"
    df = pd.read_csv(csv)
    row = df[(df["entry_id"] == "7n9v_J")
             & (df["cdr"] == "H3")
             & (df["sample"] == 0)]
    if not len(row):
        print("\nNo CSV row for 7n9v_J / H3 / sample 0 — skipping cross-check")
        return
    gen_seq = row.iloc[0]["gen_seq"]
    print(f"\nCSV row 7n9v_J H3 sample 0: gen_seq = {gen_seq!r} (len {len(gen_seq)})")
    idx = seq_str.find(gen_seq)
    if idx >= 0:
        resseqs_for_gen = [seq_str_with_resseq[idx + k][0]
                           for k in range(len(gen_seq))]
        print(f"  → found inside PDB chain sequence at chain-index {idx}, "
              f"resseqs {resseqs_for_gen[0]}..{resseqs_for_gen[-1]}")
    else:
        print("  → NOT found as substring in PDB chain sequence "
              "(suspect numbering / parsing issue)")


# ════════════════════════════════════════════════════════════════════════
# Verdict assembly
# ════════════════════════════════════════════════════════════════════════
def verdict(parquet_gen, parquet_gt, gen_seq_gen, gen_seq_gt):
    header("VERDICT")
    print(f"\nParquet's claimed seed42_jfix×oldtest×H3 model motif:  {parquet_gen}")
    print(f"gen_seq-aggregated H3 model motif (writer's method):    {gen_seq_gen}")
    print()
    if parquet_gen.upper() == gen_seq_gen.upper():
        print("BOTH analyses produce the SAME motif → REFUTE bug claim. "
              "Brief 13's parquet correctly reflects gen_seq aggregation.")
    else:
        print("DIFFERENT motifs → CONFIRM bug.")
        print("Brief 13's parquet does NOT reflect the H3 CDR sequence "
              "the AAR computation uses.")
        print("\nLikely root cause (stage C above shows the source): "
              "the dispatcher reads PDB residues at resseq in CDR_WINDOWS[H3] "
              "= (95, 102), but the design PDB's resseq 95-102 sits in FR3 "
              "framework (or a shifted position), not the H3 CDR.")


def main():
    parquet_gen, parquet_gt = stage_a()
    gen_seq_gen, gen_seq_gt = stage_b()
    stage_c_substring()
    stage_c_pdb()
    verdict(parquet_gen, parquet_gt, gen_seq_gen, gen_seq_gt)


if __name__ == "__main__":
    main()
