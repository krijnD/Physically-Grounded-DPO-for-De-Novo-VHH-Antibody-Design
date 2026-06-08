"""
Brief 15 Track 1 — Phase A.

Rebuild per_position_modal_picks_all.parquet directly from the eval CSVs'
`gen_seq` / `native_seq` columns, bypassing PDB parsing entirely.

Why: the v1 script (compute_per_position_modal_picks.py) loaded design PDBs
from the master parquet's `pdb_filepath` column, which points to the IMGT-
renumbered judged-chunk PDBs at
  data/eval/judged_chunks/all_variants/vhh_monomers/<v>__<t>__<e>__<c>__s<n>.pdb
and applied a hard-coded CDR_WINDOWS = {"H3": (95, 102)} resseq slice. On
those IMGT-numbered PDBs, resseq 95-102 lands on the conserved
FR3 framework `KPEDTAVY` motif — NOT the H3 CDR. The "antigen-conditional
mode collapse onto KPEDTAVY" headline from Brief 13 §8.2 is an artefact of
that slicing bug.

The eval CSVs at
  runs/<variant_dir>/<eval_design_csv>
already store the model's CDR design as a sequence string in `gen_seq`
(and the ground-truth CDR as `native_seq`). These are the same columns
the AAR computation uses, so reading them directly is the ground-truth
aggregation.

Output schema mirrors the v1 parquet for direct diff-ability, with the
key semantic change that `position` is CDR-relative 0..L-1 (NOT raw
resseq).

Extra columns added:
  slicing_convention : str — "gen_seq_v2" provenance tag
  canonical_length   : int — mode of native_seq length for this slice

Run:
    python scripts/eval/compute_per_position_modal_picks_v2.py
        [--master data/eval/design_samples_master.parquet]
        [--output data/eval/per_position_modal_picks_all.parquet]
        [--keep-v1-backup data/eval/per_position_modal_picks_all.backup_pre_track1.parquet]

Designed for the campaign root; one-shot on the login node, ~30 s.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Map (variant_label_in_master, test_set_label_in_master) → relative CSV path.
# These are the canonical eval CSVs persisted by the design-eval driver in
# Briefs 06 / 07b — also matches the EVAL_CSV_MAP used by
# scripts/eval/verify_per_position_modal_picks.py.
EVAL_CSV_MAP = {
    ("seed42_jfix",         "oldtest"): "runs/vhh_ft/seed42_jfix/eval_test_design.csv",
    ("floor_pi_theta",      "oldtest"): "runs/dpo/dpo_seqonly_filtered/eval_test_design.csv",
    ("expanded_pi_ref",     "oldtest"): "runs/vhh_ft/seed42_jfix_expanded/eval_oldtest_design.csv",
    ("expanded_pi_ref",     "newtest"): "runs/vhh_ft/seed42_jfix_expanded/eval_newtest_design.csv",
    ("expanded_pi_theta",   "oldtest"): "runs/dpo/dpo_seqonly_filtered_expanded/eval_oldtest_design.csv",
    ("expanded_pi_theta",   "newtest"): "runs/dpo/dpo_seqonly_filtered_expanded/eval_newtest_design.csv",
    # Brief 16 β-sweep — corrective ablation; OLD test only.
    ("floor_pi_theta_b0005","oldtest"): "runs/dpo/floor_dpo_beta0005/eval_test_design.csv",
    ("floor_pi_theta_b05",  "oldtest"): "runs/dpo/floor_dpo_beta05/eval_test_design.csv",
    # Brief 18 IPO β-sweep — robustness baseline; OLD test for floor runs,
    # OLD+NEW for the expanded run.
    ("ipo_floor_beta0005",  "oldtest"): "runs/dpo/ipo_seqonly_floor_beta0005/eval_test_design.csv",
    ("ipo_floor_beta05",    "oldtest"): "runs/dpo/ipo_seqonly_floor_beta05/eval_test_design.csv",
    ("ipo_floor_beta5",     "oldtest"): "runs/dpo/ipo_seqonly_floor_beta5/eval_test_design.csv",
    ("ipo_expanded_beta05", "oldtest"): "runs/dpo/ipo_seqonly_expanded_beta05/eval_oldtest_design.csv",
    ("ipo_expanded_beta05", "newtest"): "runs/dpo/ipo_seqonly_expanded_beta05/eval_newtest_design.csv",
}

EXPECTED_COLS = {"cdr", "entry_id", "native_seq", "gen_seq", "sample"}


def load_eval_csvs(project_root: Path) -> pd.DataFrame:
    """Load + concatenate all eval CSVs, tagging with (variant, test_set)."""
    frames = []
    print("─" * 64)
    print("Loading eval CSVs:")
    for (variant, test_set), rel in EVAL_CSV_MAP.items():
        path = project_root / rel
        if not path.exists():
            print(f"  MISSING  {variant:<22} {test_set:<8} {rel}")
            continue
        df = pd.read_csv(path)
        missing = EXPECTED_COLS - set(df.columns)
        if missing:
            sys.exit(f"FATAL: CSV {rel} missing columns: {missing}")
        df = df[list(EXPECTED_COLS)].copy()
        df["variant"] = variant
        df["test_set"] = test_set
        frames.append(df)
        print(f"  OK       {variant:<22} {test_set:<8} n={len(df):<5} {rel}")
    if not frames:
        sys.exit("FATAL: no eval CSVs loaded; check campaign root.")
    out = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows loaded: {len(out)}")
    return out


def per_position_aggregate(df: pd.DataFrame, max_len: int = 25) -> pd.DataFrame:
    """For each (variant × test_set × cdr × position 0..L-1), compute modal AAs.

    GT is deduplicated by entry_id (each test entry contributes one
    native_seq per CDR). Generated is one row per sample (K=4 per entry
    per CDR).
    """
    rows = []
    grouped = df.groupby(["variant", "test_set", "cdr"], sort=True)
    print("\n─" * 8)
    print("Per-(variant × test × cdr) sample counts:")
    print(f"{'variant':<22} {'test':<8} {'cdr':<4} {'n_entries':>10} {'n_samples':>10} {'L_native_mode':>14}")
    for (variant, test_set, cdr), sub in grouped:
        gt_unique = sub.drop_duplicates(subset=["entry_id"])
        gt_seqs = [s for s in gt_unique["native_seq"].astype(str).tolist()
                   if s and s.lower() != "nan"]
        gen_seqs = [s for s in sub["gen_seq"].astype(str).tolist()
                    if s and s.lower() != "nan"]
        if not gt_seqs and not gen_seqs:
            continue
        canonical_length = int(pd.Series([len(s) for s in gt_seqs]).mode().iloc[0]) \
            if gt_seqs else int(pd.Series([len(s) for s in gen_seqs]).mode().iloc[0])
        print(f"{variant:<22} {test_set:<8} {cdr:<4} "
              f"{len(gt_seqs):>10} {len(gen_seqs):>10} {canonical_length:>14}")
        for pos in range(max_len):
            gt_col = [s[pos] for s in gt_seqs if pos < len(s)]
            gen_col = [s[pos] for s in gen_seqs if pos < len(s)]
            if not gt_col and not gen_col:
                continue
            gt_aa, gt_freq = (None, None)
            if gt_col:
                gt_count = Counter(gt_col)
                top = gt_count.most_common(1)[0]
                gt_aa = top[0]
                gt_freq = top[1] / len(gt_col)
            gen_aa, gen_freq = (None, None)
            if gen_col:
                gen_count = Counter(gen_col)
                top = gen_count.most_common(1)[0]
                gen_aa = top[0]
                gen_freq = top[1] / len(gen_col)
            modals_match = (
                gt_aa is not None and gen_aa is not None and gt_aa == gen_aa
            )
            modal_gap_pp = (
                None if gt_freq is None or gen_freq is None
                else 100.0 * (gen_freq - gt_freq)
            )
            rows.append({
                "variant": variant,
                "test_set": test_set,
                "cdr": cdr,
                "position": pos,
                "n_gt": len(gt_col),
                "n_gen": len(gen_col),
                "gt_modal_aa": gt_aa,
                "gt_modal_freq": gt_freq,
                "gen_modal_aa": gen_aa,
                "gen_modal_freq": gen_freq,
                "modals_match": modals_match,
                "modal_gap_pp": modal_gap_pp,
                "slicing_convention": "gen_seq_v2",
                "canonical_length": canonical_length,
            })
    out = pd.DataFrame(rows).sort_values(
        ["variant", "test_set", "cdr", "position"]
    ).reset_index(drop=True)
    return out


def print_diagnostic_slice(out: pd.DataFrame) -> None:
    """Print the writer's canonical sanity slice — seed42_jfix × oldtest × H3."""
    print("\n" + "═" * 72)
    print("DIAGNOSTIC SLICE: seed42_jfix × oldtest × H3 (writer's check)")
    print("═" * 72)
    slc = out[
        (out["variant"] == "seed42_jfix")
        & (out["test_set"] == "oldtest")
        & (out["cdr"] == "H3")
    ].sort_values("position")
    cols = ["position", "n_gt", "n_gen",
            "gt_modal_aa", "gt_modal_freq",
            "gen_modal_aa", "gen_modal_freq", "modal_gap_pp"]
    fmt = slc[cols].copy()
    fmt["gt_modal_freq"] = fmt["gt_modal_freq"].astype(float).round(3)
    fmt["gen_modal_freq"] = fmt["gen_modal_freq"].astype(float).round(3)
    fmt["modal_gap_pp"] = fmt["modal_gap_pp"].astype(float).round(1)
    print(fmt.to_string(index=False))
    gen_motif = "".join(slc["gen_modal_aa"].fillna("?").astype(str).tolist())
    gt_motif = "".join(slc["gt_modal_aa"].fillna("?").astype(str).tolist())
    print(f"\nReconstructed v2 gen modal motif: {gen_motif}")
    print(f"Reconstructed v2 GT  modal motif: {gt_motif}")
    if "KPEDTAVY" in gen_motif:
        print("\n⚠ KPEDTAVY appears in v2 reconstruction — fix is INCOMPLETE.")
    else:
        print("\n✓ KPEDTAVY does NOT appear in v2 reconstruction — fix is consistent with the writer's hypothesis.")


def print_modal_match_summary(out: pd.DataFrame) -> None:
    """Per (variant × test × cdr): how many positions match GT modal."""
    print("\n" + "═" * 72)
    print("MODAL-MATCH SUMMARY (v2): % positions where gen-modal == GT-modal")
    print("═" * 72)
    grouped = out.groupby(["variant", "test_set", "cdr"])["modals_match"].agg(
        ["sum", "count"]
    )
    grouped["pct"] = 100 * grouped["sum"] / grouped["count"]
    print(grouped.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".",
                    help="Campaign repo root (default: cwd)")
    ap.add_argument("--master", default="data/eval/design_samples_master.parquet")
    ap.add_argument("--output",
                    default="data/eval/per_position_modal_picks_all.parquet",
                    help="Overwrites the v1 parquet at this path.")
    ap.add_argument("--keep-v1-backup", action="store_true",
                    help="If set, copies the existing output to .backup_pre_track1.parquet "
                         "before overwriting (Brief 15 already pre-backs up; default off).")
    ap.add_argument("--max-len", type=int, default=25,
                    help="Maximum CDR-relative position to scan (default 25 — H3 max length 19).")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    df = load_eval_csvs(root)

    out_df = per_position_aggregate(df, max_len=args.max_len)
    print(f"\nv2 per-position rows: {len(out_df)}")

    output_path = (root / args.output).resolve()
    if args.keep_v1_backup and output_path.exists():
        bak = output_path.with_suffix(".backup_pre_track1_v2write.parquet")
        if not bak.exists():
            shutil.copy(output_path, bak)
            print(f"Backup written: {bak}")
        else:
            print(f"Backup already exists (skipping): {bak}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path)
    print(f"v2 parquet written: {output_path}")

    print_diagnostic_slice(out_df)
    print_modal_match_summary(out_df)

    print("\n" + "═" * 72)
    print("DONE — Phase A (per_position v2) complete.")
    print("═" * 72)


if __name__ == "__main__":
    main()
