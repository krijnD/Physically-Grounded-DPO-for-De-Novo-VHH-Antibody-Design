"""
Step 5 of Brief 13: merge CAAR + EpiF1 shards + join into the master parquet.

Mirrors the join_scrmsd_into_master.py pattern (Brief 12 §4):
1. Merge {shard-base}_task*.parquet into a consolidated {shard-base}.parquet.
2. Inspect master + shard variant / test_set vocabularies; refuse to overwrite
   master if they disagree without --variant-map / --test-set-map remapping.
3. Optional remap of shard variant / test_set names.
4. Backup master → {master}.backup_pre13.parquet (refuses to clobber existing
   backup unless --force-backup).
5. Left-join master ← shards on (variant, test_set, entry_id, cdr, sample).
6. Refuse to write the joined master if join coverage < --min-coverage.
7. Print per-(variant × test × cdr) summary tables of CAAR + EpiF1.

Each shard row already holds the CDR-specific CAAR + EpiF1 (the dispatcher
only computes the masked-CDR metrics per PDB), so no "active" picking is
needed — unlike the scRMSD join which selects scrmsd_<cdr> per row.

Usage:
    # 1) Inspect first
    python scripts/eval/join_caar_epif1_into_master.py --inspect-only

    # 2) Real join
    python scripts/eval/join_caar_epif1_into_master.py

    # 3) Relax coverage gate (e.g. master has variants not in the shards)
    python scripts/eval/join_caar_epif1_into_master.py --min-coverage 0.85
"""
import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

JOIN_KEYS = ["variant", "test_set", "entry_id", "cdr", "sample"]

METRIC_COLS = [
    "caar", "caar_n_positions",
    "epif1", "epif1_precision", "epif1_recall",
    "gt_paratope_n", "gt_epitope_n", "design_epitope_n", "overlap_n",
    "gt_vhh_chain", "design_vhh_chain", "gt_ag_chains", "design_ag_chains",
]


def load_inline_or_file(arg):
    if arg is None:
        return None
    if arg.startswith("@"):
        return json.loads(Path(arg[1:]).read_text())
    return json.loads(arg)


def merge_shards(shard_base: str) -> pd.DataFrame:
    pattern = f"{shard_base}_task*.parquet"
    shards = sorted(glob.glob(pattern))
    print(f"[merge] {len(shards)} shards matching {pattern}")
    if not shards:
        sys.exit(f"FATAL: no shards found matching {pattern}")
    dfs = [pd.read_parquet(s) for s in shards]
    df = pd.concat(dfs, ignore_index=True)
    print(f"[merge] consolidated rows: {len(df)}")
    return df


def report_quality(df: pd.DataFrame):
    print("\n=== Shard quality ===")
    if "error" in df.columns:
        n_err = df["error"].notna().sum()
        print(f"  error rows: {n_err} ({100 * n_err / len(df):.2f}%)")
        if n_err:
            print(df[df["error"].notna()]["error"].value_counts().head(10).to_string())

    print("\n  Per-CDR NaN rates + ranges:")
    for cdr, g in df.groupby("cdr"):
        n = len(g)
        n_zero_par = (g["gt_paratope_n"] == 0).sum()
        caar_ok = g["caar"].dropna()
        epif1_ok = g["epif1"].dropna()
        print(f"    {cdr}: n={n}, gt_paratope==0: {n_zero_par} ({100*n_zero_par/n:.1f}%)")
        if len(caar_ok):
            print(f"        CAAR  non-NaN n={len(caar_ok)}, mean={caar_ok.mean():.2f}, "
                  f"median={caar_ok.median():.2f}")
        if len(epif1_ok):
            print(f"        EpiF1 non-NaN n={len(epif1_ok)}, mean={epif1_ok.mean():.3f}, "
                  f"median={epif1_ok.median():.3f}")


def inspect_vocabs(master, shards):
    print("\n=== Vocabulary comparison ===")
    m_var = sorted(master["variant"].unique().tolist())
    s_var = sorted(shards["variant"].unique().tolist())
    print(f"  master variants ({len(m_var)}): {m_var}")
    print(f"  shard  variants ({len(s_var)}): {s_var}")
    m_test = sorted(master["test_set"].unique().tolist())
    s_test = sorted(shards["test_set"].unique().tolist())
    print(f"  master test_set ({len(m_test)}): {m_test}")
    print(f"  shard  test_set ({len(s_test)}): {s_test}")
    print(f"  master rows: {len(master)}; shard rows: {len(shards)}")
    vocab_ok = (set(s_var).issubset(set(m_var))
                and set(s_test).issubset(set(m_test)))
    print(f"  vocabularies align: {vocab_ok}")
    return vocab_ok


def summarize_joined(joined: pd.DataFrame):
    print("\n=== Per-(variant × test_set × cdr) CAAR + EpiF1 summary ===")
    sub = joined.dropna(subset=["caar"])
    grp_caar = sub.groupby(["variant", "test_set", "cdr"])["caar"].agg(["count", "mean", "median"])
    grp_caar.columns = ["n", "mean_caar", "median_caar"]
    grp_caar = grp_caar.round(2)

    sub = joined.dropna(subset=["epif1"])
    grp_epi = sub.groupby(["variant", "test_set", "cdr"])["epif1"].agg(["count", "mean", "median"])
    grp_epi.columns = ["n", "mean_epif1", "median_epif1"]
    grp_epi = grp_epi.round(3)

    print("\nCAAR:")
    print(grp_caar.to_string())
    print("\nEpiF1:")
    print(grp_epi.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-base", default="data/eval/caar_epif1",
                    help="Shards live at <shard-base>_task*.parquet; "
                         "consolidated written to <shard-base>.parquet")
    ap.add_argument("--master", default="data/eval/design_samples_master.parquet")
    ap.add_argument("--variant-map", default=None,
                    help="Inline JSON or @file.json mapping shard variant → master variant")
    ap.add_argument("--test-set-map", default=None,
                    help="Inline JSON or @file.json mapping shard test_set → master test_set")
    ap.add_argument("--min-coverage", type=float, default=1.0,
                    help="Minimum fraction of master rows that must get a CAAR/EpiF1 "
                         "row after join (default 1.0). Refuses to overwrite master "
                         "below this.")
    ap.add_argument("--force-backup", action="store_true",
                    help="Overwrite an existing backup file if present.")
    ap.add_argument("--inspect-only", action="store_true",
                    help="Report quality + vocab; don't write the master.")
    args = ap.parse_args()

    consolidated_path = f"{args.shard_base}.parquet"

    shards = merge_shards(args.shard_base)
    report_quality(shards)

    Path(consolidated_path).parent.mkdir(parents=True, exist_ok=True)
    shards.to_parquet(consolidated_path)
    print(f"\n[consolidated] wrote {len(shards)} rows → {consolidated_path}")

    if not Path(args.master).exists():
        sys.exit(f"FATAL: master parquet not found: {args.master}")
    master = pd.read_parquet(args.master)
    print(f"[master] loaded {len(master)} rows from {args.master}")
    print(f"[master] columns ({len(master.columns)}): {list(master.columns)}")

    variant_map = load_inline_or_file(args.variant_map)
    test_set_map = load_inline_or_file(args.test_set_map)
    if variant_map:
        shards["variant"] = shards["variant"].map(lambda v: variant_map.get(v, v))
        print(f"\n[remap] applied variant_map: {variant_map}")
    if test_set_map:
        shards["test_set"] = shards["test_set"].map(lambda v: test_set_map.get(v, v))
        print(f"[remap] applied test_set_map: {test_set_map}")

    vocab_ok = inspect_vocabs(master, shards)

    if args.inspect_only:
        print("\n[inspect-only] exiting without writing master.")
        return

    if not vocab_ok:
        sys.exit("FATAL: vocabularies don't align. Pass --variant-map / --test-set-map "
                 "to remap, or run with --inspect-only to diagnose.")

    # Build the join slice + rename error col to avoid collision with master
    keep_cols = JOIN_KEYS + [c for c in METRIC_COLS if c in shards.columns]
    if "error" in shards.columns:
        keep_cols.append("error")
    shards_j = shards[keep_cols].copy()
    if "error" in shards_j.columns:
        shards_j = shards_j.rename(columns={"error": "caar_epif1_error"})

    # Avoid column collisions
    overlap = set(shards_j.columns) - set(JOIN_KEYS)
    collisions = overlap.intersection(set(master.columns))
    if collisions:
        print(f"\n[warn] dropping pre-existing columns in master that would "
              f"collide with shard columns: {sorted(collisions)}")
        master = master.drop(columns=list(collisions))

    print(f"\n=== Joining on {JOIN_KEYS} ===")
    joined = master.merge(shards_j, on=JOIN_KEYS, how="left")
    print(f"  master rows: {len(master)} → joined rows: {len(joined)}")

    # Coverage: a master row is "covered" if it got SOME caar/epif1 column populated
    # (gt_paratope_n is always set on a successful shard row, even when CAAR is NaN
    # due to gt_paratope_n == 0 — that's the right coverage marker).
    has_shard = joined["gt_paratope_n"].notna() if "gt_paratope_n" in joined.columns else None
    if has_shard is None:
        sys.exit("FATAL: gt_paratope_n missing after join — shard didn't carry it.")
    coverage = has_shard.mean()
    print(f"  shard coverage: {has_shard.sum()}/{len(joined)} ({100 * coverage:.1f}%)")

    if coverage < args.min_coverage:
        missing = joined[~has_shard]
        if len(missing):
            print("\n[diagnose] sample of master rows that didn't get a shard row:")
            print(missing[JOIN_KEYS].head(10).to_string(index=False))
            print(f"\n  variants in missing: {missing['variant'].value_counts().to_dict()}")
            print(f"  test_sets in missing: {missing['test_set'].value_counts().to_dict()}")
        sys.exit(f"FATAL: coverage {coverage:.1%} below --min-coverage "
                 f"{args.min_coverage:.1%}. Refusing to overwrite master.")

    backup = f"{args.master}.backup_pre13.parquet"
    if Path(backup).exists() and not args.force_backup:
        sys.exit(f"FATAL: backup already exists at {backup}; pass "
                 f"--force-backup to overwrite. Refusing to clobber.")
    shutil.copy(args.master, backup)
    print(f"\n[backup] copied master → {backup}")

    joined.to_parquet(args.master)
    print(f"[write] updated master at {args.master} "
          f"({len(joined)} rows, {len(joined.columns)} columns)")

    summarize_joined(joined)


if __name__ == "__main__":
    main()
