"""
Step 4 of Brief 12: merge scRMSD shards + join into Brief 11 master parquet.

Sequence:
1. Merge {shard-base}_task*.parquet shards into a consolidated
   {shard-base}.parquet (saved unconditionally — useful as a standalone scRMSD
   reference even if the master join fails).
2. Inspect master + shard variant / test_set vocabularies. If they disagree
   AND no --variant-map / --test-set-map is provided, print the disagreement
   and EXIT non-zero (refuse to silently corrupt the master with all-NaN).
3. Apply optional name remappings on the shards (--variant-map / --test-set-map
   are inline JSON or @file.json).
4. Compute scrmsd_active per shard row = scrmsd_{row.cdr}.
5. Backup master to {master}.backup_pre12.parquet (refuses to overwrite an
   existing backup unless --force-backup).
6. Left-join master ← shards on (variant, test_set, entry_id, cdr, sample),
   preserving all master rows. Report match coverage; refuse to overwrite
   master if coverage < --min-coverage (default 1.0).
7. Write the joined master back to its original path.

Usage:
    # 1) Dry-run inspection
    python scripts/eval/join_scrmsd_into_master.py --inspect-only

    # 2) Joining once vocabularies are aligned
    python scripts/eval/join_scrmsd_into_master.py \\
        --variant-map '{"seed42_jfix": "seed42_jfix_pi_ref"}'

    # 3) Force a known-good join with relaxed coverage (e.g. master has more
    #    rows than scRMSD covers — usually pretrained DiffAb that wasn't folded)
    python scripts/eval/join_scrmsd_into_master.py --min-coverage 0.85
"""
import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SHARD_KEYS_DEFAULT = ["variant", "test_set", "entry", "cdr", "sample"]
MASTER_KEYS_DEFAULT = ["variant", "test_set", "entry_id", "cdr", "sample"]


def load_inline_or_file(arg):
    """Accept either inline JSON or @path/to/file.json."""
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
    print("\n=== scRMSD shard quality ===")
    if "error" in df.columns:
        n_err = df["error"].notna().sum()
        print(f"  error rows: {n_err} ({100 * n_err / len(df):.1f}%)")
        if n_err > 0:
            print("  top error reasons:")
            print(df[df["error"].notna()]["error"].value_counts().head(10).to_string())
    for cdr in ["H1", "H2", "H3"]:
        col = f"scrmsd_{cdr}"
        if col not in df.columns:
            continue
        nan = df[col].isna().sum()
        finite = df[col].dropna()
        if len(finite):
            print(f"  {col}: NaN {nan} ({100 * nan / len(df):.1f}%); "
                  f"mean {finite.mean():.2f} Å; "
                  f"% <2 Å {(finite < 2.0).mean() * 100:.1f}%; "
                  f"% <4 Å {(finite < 4.0).mean() * 100:.1f}%")
        else:
            print(f"  {col}: ALL NaN")
    if "seq_identity_pct" in df.columns:
        below = (df["seq_identity_pct"] < 100).sum()
        print(f"  seq_identity_pct: < 100 on {below} rows (alignment correctness check)")
    if "gen_trim_n_dropped" in df.columns:
        trim = df["gen_trim_n_dropped"].fillna(0)
        print(f"  gen_trim_n_dropped: rows with trim>0: {(trim > 0).sum()}; "
              f"distribution: {trim.value_counts().sort_index().to_dict()}")


def inspect_vocabs(master: pd.DataFrame, shards: pd.DataFrame,
                   master_keys, shard_keys):
    print("\n=== Vocabulary comparison ===")
    m_variant = sorted(master["variant"].unique().tolist())
    s_variant = sorted(shards["variant"].unique().tolist())
    print(f"  master variants ({len(m_variant)}): {m_variant}")
    print(f"  shard  variants ({len(s_variant)}): {s_variant}")
    m_test = sorted(master["test_set"].unique().tolist())
    s_test = sorted(shards["test_set"].unique().tolist())
    print(f"  master test_set ({len(m_test)}): {m_test}")
    print(f"  shard  test_set ({len(s_test)}): {s_test}")
    print(f"  master keys: {master_keys}")
    print(f"  shard  keys: {shard_keys}")
    print(f"  master rows: {len(master)}; shard rows: {len(shards)}")
    vocab_ok = (set(s_variant).issubset(set(m_variant))
                and set(s_test).issubset(set(m_test)))
    print(f"  vocabularies align: {vocab_ok}")
    return vocab_ok


def compute_scrmsd_active(shards: pd.DataFrame) -> pd.DataFrame:
    def _pick(r):
        cdr = r.get("cdr")
        if cdr in ("H1", "H2", "H3"):
            return r.get(f"scrmsd_{cdr}")
        return np.nan
    shards = shards.copy()
    shards["scrmsd_active"] = shards.apply(_pick, axis=1)
    return shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-base",
                    default="data/eval/scrmsd_design_samples",
                    help="Shard parquets live at <shard-base>_task*.parquet; "
                         "consolidated written to <shard-base>.parquet")
    ap.add_argument("--master",
                    default="data/eval/design_samples_master.parquet")
    ap.add_argument("--shard-keys", nargs=5, default=SHARD_KEYS_DEFAULT)
    ap.add_argument("--master-keys", nargs=5, default=MASTER_KEYS_DEFAULT)
    ap.add_argument("--variant-map", default=None,
                    help="Inline JSON or @file.json mapping shard variant → master variant")
    ap.add_argument("--test-set-map", default=None,
                    help="Inline JSON or @file.json mapping shard test_set → master test_set")
    ap.add_argument("--min-coverage", type=float, default=1.0,
                    help="Minimum fraction of master rows that must get a scrmsd "
                         "after join (default 1.0). Refuses to overwrite master below this.")
    ap.add_argument("--force-backup", action="store_true",
                    help="Overwrite an existing backup file if present")
    ap.add_argument("--inspect-only", action="store_true",
                    help="Print quality + vocabularies; don't write the master")
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

    vocab_ok = inspect_vocabs(master, shards, args.master_keys, args.shard_keys)

    if args.inspect_only:
        print("\n[inspect-only] exiting without writing master.")
        return

    if not vocab_ok:
        sys.exit("FATAL: vocabularies don't align. Pass --variant-map / --test-set-map "
                 "to remap, or run with --inspect-only to diagnose.")

    shards = compute_scrmsd_active(shards)

    # Rename shard keys to align with master before join
    rename_keys = dict(zip(args.shard_keys, args.master_keys))
    shards_j = shards.rename(columns=rename_keys)

    # Keep all useful shard columns (debugging metadata + per-CDR scRMSD + active)
    keep_cols = list(args.master_keys) + [
        "scrmsd_H1", "scrmsd_H2", "scrmsd_H3", "scrmsd_active",
        "gen_trim_offset", "gen_trim_n_dropped", "seq_identity_pct",
        "fw_atoms", "n_gen_res", "n_pred_res",
        "H1_atoms", "H2_atoms", "H3_atoms",
        "error",
    ]
    keep_cols = [c for c in keep_cols if c in shards_j.columns]
    shards_j = shards_j[keep_cols].rename(columns={"error": "scrmsd_error"})

    # Avoid column collisions on join
    overlap = set(shards_j.columns) - set(args.master_keys)
    collisions = overlap.intersection(set(master.columns))
    if collisions:
        print(f"\n[warn] dropping pre-existing columns in master that would "
              f"collide with shard columns: {sorted(collisions)}")
        master = master.drop(columns=list(collisions))

    print(f"\n=== Joining on {args.master_keys} ===")
    joined = master.merge(shards_j, on=args.master_keys, how="left")
    print(f"  master rows: {len(master)} → joined rows: {len(joined)}")

    if "scrmsd_active" in joined.columns:
        joined = joined.rename(columns={"scrmsd_active": "scrmsd"})
    coverage = joined["scrmsd"].notna().mean() if "scrmsd" in joined.columns else 0.0
    print(f"  scrmsd coverage: {joined['scrmsd'].notna().sum()}/{len(joined)} "
          f"({100 * coverage:.1f}%)")

    if coverage < args.min_coverage:
        # Diagnose the misses
        print("\n[diagnose] sample of master rows that didn't get scrmsd:")
        missing = joined[joined["scrmsd"].isna()]
        if len(missing):
            print(missing[args.master_keys].head(10).to_string(index=False))
            print(f"\n  variants in missing: "
                  f"{missing['variant'].value_counts().to_dict()}")
            print(f"  test_sets in missing: "
                  f"{missing['test_set'].value_counts().to_dict()}")
        sys.exit(f"FATAL: coverage {coverage:.1%} below --min-coverage "
                 f"{args.min_coverage:.1%}. Refusing to overwrite master.")

    backup = f"{args.master}.backup_pre12.parquet"
    if Path(backup).exists() and not args.force_backup:
        sys.exit(f"FATAL: backup already exists at {backup}; pass "
                 f"--force-backup to overwrite. Refusing to clobber.")
    shutil.copy(args.master, backup)
    print(f"\n[backup] copied master → {backup}")

    joined.to_parquet(args.master)
    print(f"[write] updated master at {args.master} "
          f"({len(joined)} rows, {len(joined.columns)} columns)")

    print("\n=== Joined master summary ===")
    print(f"  per-(variant, test_set, cdr) % designable (scRMSD < 2 Å):")
    grp = (
        joined.dropna(subset=["scrmsd"])
        .assign(designable=lambda d: (d["scrmsd"] < 2.0).astype(int))
        .groupby(["variant", "test_set", "cdr"])
        .agg(n=("scrmsd", "size"),
             mean_scrmsd=("scrmsd", "mean"),
             pct_designable=("designable", "mean"))
    )
    grp["pct_designable"] = (grp["pct_designable"] * 100).round(1)
    grp["mean_scrmsd"] = grp["mean_scrmsd"].round(2)
    print(grp.to_string())


if __name__ == "__main__":
    main()
