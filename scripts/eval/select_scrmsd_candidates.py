"""
Brief 12 Step 6 — select PyMOL-render candidates for fig12c.

Reads the design master parquet (already augmented with scrmsd by
join_scrmsd_into_master.py) and prints multiple candidate slices:

1. H3 length distribution across the chosen (variant, test_set) slice
2. Top-K rows by lowest scrmsd (regardless of H3 length) — global "best"
3. Top-K rows by highest scrmsd (regardless of H3 length) — global "worst"
4. Short-H3 successes (shortest H3 with scrmsd < 2 Å)
5. Long-H3 failures (longest H3 with scrmsd > the --fail-threshold)

These four slices give the orchestrator multiple options to pick from
when defining the "short-H3 success vs long-H3 failure" PyMOL overlay
pair. The selection criterion intentionally couples H3 length and
scRMSD, since the campaign's data-property thesis predicts the
length-vs-scRMSD coupling.

Defaults target expanded_pi_theta × oldtest × H3 (Brief 12's headline
variant on the apples-to-apples test split).

Usage:
    python scripts/eval/select_scrmsd_candidates.py
    python scripts/eval/select_scrmsd_candidates.py --variant floor_pi_theta
    python scripts/eval/select_scrmsd_candidates.py --test-set newtest --top-k 10
"""
import argparse
from pathlib import Path

import pandas as pd


DEFAULT_MASTER = "data/eval/design_samples_master.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=DEFAULT_MASTER)
    ap.add_argument("--variant", default="expanded_pi_theta")
    ap.add_argument("--test-set", default="oldtest")
    ap.add_argument("--cdr", default="H3")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--success-threshold", type=float, default=2.0,
                    help="scRMSD threshold (Å) for 'success' in the short-H3 slice")
    ap.add_argument("--fail-threshold", type=float, default=4.0,
                    help="scRMSD threshold (Å) for 'failure' in the long-H3 slice")
    args = ap.parse_args()

    master_path = Path(args.master)
    if not master_path.exists():
        raise SystemExit(f"FATAL: master not found at {master_path}")
    master = pd.read_parquet(master_path)
    print(f"[load] master {master_path}: {len(master)} rows, "
          f"{len(master.columns)} columns")

    sub = master[
        (master["variant"] == args.variant)
        & (master["test_set"] == args.test_set)
        & (master["cdr"] == args.cdr)
    ].copy()
    print(f"[slice] variant={args.variant} × test_set={args.test_set} × "
          f"cdr={args.cdr} → {len(sub)} rows")
    if not len(sub):
        raise SystemExit("FATAL: empty slice")

    # H3 length: prefer cdr3_length column; fall back to cdr3_sequence.str.len()
    if "cdr3_length" in sub.columns and sub["cdr3_length"].notna().any():
        sub["h3_len"] = sub["cdr3_length"]
    elif "cdr3_sequence" in sub.columns and sub["cdr3_sequence"].notna().any():
        sub["h3_len"] = sub["cdr3_sequence"].str.len()
    else:
        sub["h3_len"] = pd.NA

    sub_valid = sub.dropna(subset=["scrmsd"]).copy()
    print(f"[scrmsd] non-null: {len(sub_valid)}/{len(sub)}")

    print(f"\n=== H3 length distribution ({args.variant} × "
          f"{args.test_set} × {args.cdr}) ===")
    print(sub_valid["h3_len"].value_counts().sort_index().to_string())

    cols_show = [
        c for c in [
            "entry_id", "sample", "candidate_id",
            "h3_len", "scrmsd",
            "scrmsd_H1", "scrmsd_H2", "scrmsd_H3",
            "cdr3_sequence",
            "pdb_filepath", "complex_pdb_path",
        ] if c in sub_valid.columns
    ]

    print(f"\n=== TOP {args.top_k}: lowest scRMSD (global best, any H3 length) ===")
    print(sub_valid.nsmallest(args.top_k, "scrmsd")[cols_show].to_string(index=False))

    print(f"\n=== BOTTOM {args.top_k}: highest scRMSD (global worst, any H3 length) ===")
    print(sub_valid.nlargest(args.top_k, "scrmsd")[cols_show].to_string(index=False))

    short_success = (
        sub_valid[sub_valid["scrmsd"] < args.success_threshold]
        .sort_values(["h3_len", "scrmsd"], ascending=[True, True])
        .head(args.top_k)
    )
    print(f"\n=== SHORT-H3 SUCCESS (scRMSD < {args.success_threshold} Å, "
          f"shortest H3 first) — Fig 12.C panel-1 candidates ===")
    print(short_success[cols_show].to_string(index=False) if len(short_success)
          else "  (no rows match)")

    long_failure = (
        sub_valid[sub_valid["scrmsd"] > args.fail_threshold]
        .sort_values(["h3_len", "scrmsd"], ascending=[False, False])
        .head(args.top_k)
    )
    print(f"\n=== LONG-H3 FAILURE (scRMSD > {args.fail_threshold} Å, "
          f"longest H3 first) — Fig 12.C panel-2 candidates ===")
    print(long_failure[cols_show].to_string(index=False) if len(long_failure)
          else "  (no rows match)")

    print(f"\n=== Joint H3-length × scRMSD bin counts ===")
    sub_valid["len_bin"] = pd.cut(
        sub_valid["h3_len"], bins=[0, 7, 9, 11, 13, 30], right=True,
        labels=["≤7", "8-9", "10-11", "12-13", "≥14"],
    )
    sub_valid["scrmsd_bin"] = pd.cut(
        sub_valid["scrmsd"], bins=[0, 1, 2, 4, 8, 30], right=True,
        labels=["<1 Å", "1-2 Å", "2-4 Å", "4-8 Å", ">8 Å"],
    )
    pivot = pd.crosstab(sub_valid["len_bin"], sub_valid["scrmsd_bin"])
    print(pivot.to_string())


if __name__ == "__main__":
    main()
