"""Brief 23 -- Snellius-side manifest leakage check.

Read-only audit of the 927-row expanded fine-tune manifest against the 29-entry
shared floor holdout and the 83-entry expanded holdout. Uses MMseqs2 easy-search
with the same parameters that built the floor clustering (min-seq-id 0.7,
coverage 0.8, cov-mode 0). Answers:

  (a) Arithmetic: |expanded| = |floor| + n_added; how does n_added compare to
      the writer's claim (312 ANDD-rescue + 130 SAbDab unique = 442)?
  (b) Leakage: how many added entries would have clustered with a shared-holdout
      entry's floor cluster? Same for the 83-entry expanded holdout.

Inputs (paths passed via argparse so nothing is environment-dependent):
- expanded manifest TSV (required; filename TBD by Krijn at run time)
- floor manifest TSV (default: data/datasets/diffab_manifest.tsv)
- floor cluster TSV + rep fasta + concat_cdrs fasta (Brief 05 outputs)
- expanded concat_cdrs fasta (Brief 05 expanded clustering output)

Outputs (under --out-dir, default tmp_brief23/):
- added_concat_cdrs.fasta             # not committed (large, regenerable)
- added_to_floor_reps.m8              # not committed (large, regenerable)
- expanded_manifest_cluster_assignments.csv   # COMMITTED -- load-bearing
- manifest_arithmetic_summary.txt             # COMMITTED -- feeds the deliverable

The script is single-purpose. No flags beyond paths; no behavior toggles.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

SHARED_HOLDOUT_29 = [
    "7f5h_C", "7n9v_J", "7ndf_C", "7ph3_C", "7ph4_C", "7q6c_K", "7qbf_B",
    "7qia_C", "7r74_B", "7sk7_K", "7vfa_D", "7vke_B", "7vq0_D", "7wd2_C",
    "7xrp_B", "7zlg_K", "8acf_K", "8cy6_D", "8elq_B", "8fcz_C", "8gsi_F",
    "8hbg_E", "8oud_D", "8pyr_D", "8qot_B", "8r61_C", "8tb7_N", "8u4v_K",
    "8wo4_G",
]

EXPANDED_HOLDOUT_83 = [
    "1kxt_B", "2p4a_B", "2vyr_E", "3cfi_C", "4krp_B", "4lgs_B", "4nbz_D",
    "4nc0_B", "4p2c_G", "4y7m_B", "5e7f_A", "5f1o_B", "5lhr_B", "5mwn_D",
    "5my6_B", "5mzv_D", "5nbl_E", "5o2u_D", "5vak_B", "5van_B", "5vaq_B",
    "6f5g_B", "6fuz_N", "6fv0_F", "6ir1_B", "6oca_D", "6ze1_B", "6zrv_B",
    "7aqy_D", "7f5h_C", "7jkm_K", "7my2_H", "7n9v_J", "7ndf_C", "7pa5_B",
    "7ph2_D", "7ph3_C", "7ph4_C", "7q6c_K", "7qbf_B", "7qbg_E", "7qia_C",
    "7r74_B", "7sak_B", "7sk7_K", "7sp6_B", "7sp8_B", "7spa_B", "7th3_B",
    "7vfa_D", "7vke_B", "7vq0_D", "7wd2_C", "7wn1_C", "7x7e_B", "7xrp_B",
    "7zkw_C", "7zlg_K", "7zxu_A", "8acf_K", "8bb7_D", "8bev_B", "8cii_C",
    "8cy6_D", "8cyd_D", "8dfl_E", "8ee2_C", "8elq_B", "8fcz_C", "8gsi_F",
    "8h5t_B", "8hbg_E", "8oud_D", "8pjp_C", "8pyr_D", "8q6k_N", "8qot_B",
    "8r61_C", "8snc_B", "8t7h_C", "8tb7_N", "8u4v_K", "8wo4_G",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--floor-manifest", default="data/datasets/diffab_manifest.tsv")
    p.add_argument("--expanded-manifest", required=True,
                   help="path to the 927-row expanded manifest TSV")
    p.add_argument("--floor-cluster-tsv",
                   default="data/datasets/clustering/cluster_result_cluster.tsv")
    p.add_argument("--floor-rep-fasta",
                   default="data/datasets/clustering/cluster_result_rep_seq.fasta")
    p.add_argument("--floor-concat-fasta",
                   default="data/datasets/clustering/concat_cdrs.fasta")
    p.add_argument("--expanded-concat-fasta", required=True,
                   help="path to the 927-row concat_cdrs fasta (Brief 05 expanded clustering output)")
    p.add_argument("--out-dir", default="tmp_brief23")
    p.add_argument("--mmseqs-bin", default="mmseqs")
    return p.parse_args()


def ensure_entry_id(df):
    if "entry_id" in df.columns:
        return df
    # Floor manifest convention from Brief 05: <pdb_lower>_<Hchain>
    df["entry_id"] = df["pdb"].astype(str).str.lower() + "_" + df["Hchain"].astype(str)
    return df


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Read manifests --
    floor = ensure_entry_id(pd.read_csv(args.floor_manifest, sep="\t"))
    exp = ensure_entry_id(pd.read_csv(args.expanded_manifest, sep="\t"))
    print(f"Floor manifest: {len(floor)} rows")
    print(f"Expanded manifest: {len(exp)} rows")

    if len(exp) != 927:
        print(f"WARNING: expected 927 rows in expanded manifest, got {len(exp)}.")
        print("         Continuing -- the actual count is what gets reconciled in the deliverable.")

    floor_ids = set(floor["entry_id"])
    exp_ids = set(exp["entry_id"])
    added_ids_sorted = sorted(exp_ids - floor_ids)  # sort for reproducibility
    added_ids = set(added_ids_sorted)
    dropped_ids = floor_ids - exp_ids
    overlap_ids = floor_ids & exp_ids

    print(f"Floor IDs: {len(floor_ids)}")
    print(f"Expanded IDs: {len(exp_ids)}")
    print(f"Added IDs: {len(added_ids)}")
    print(f"Floor IDs preserved in expanded: {len(overlap_ids)}")
    print(f"Floor IDs dropped in expanded: {len(dropped_ids)}")

    # -- Reconcile arithmetic --
    source_col = next((c for c in ("source", "source_dataset", "origin") if c in exp.columns), None)
    if source_col:
        added_rows = exp[exp["entry_id"].isin(added_ids)]
        source_counts = added_rows[source_col].value_counts().to_dict()
        id_to_source = dict(zip(added_rows["entry_id"], added_rows[source_col]))
        print(f"Added entries by source ({source_col}):")
        for k, v in source_counts.items():
            print(f"  {k}: {v}")
    else:
        source_counts = {"unknown": len(added_ids)}
        id_to_source = {aid: "unknown" for aid in added_ids}
        print("No source column on the expanded manifest; downstream needs another way "
              "to partition rescue vs SAbDab. Tagging all added entries as 'unknown'.")

    arith_path = out_dir / "manifest_arithmetic_summary.txt"
    writer_claim = 465 + 312 + 130  # =907
    with open(arith_path, "w") as fh:
        fh.write(f"Floor manifest:    {len(floor_ids)} entries\n")
        fh.write(f"Expanded manifest: {len(exp_ids)} entries\n")
        fh.write(f"Added:             {len(added_ids)} entries\n")
        fh.write(f"Dropped:           {len(dropped_ids)} entries (must be 0; current = {len(dropped_ids)})\n")
        fh.write(f"Overlap floor∩exp: {len(overlap_ids)} entries\n")
        fh.write("\nAdded entries by source:\n")
        for k, v in source_counts.items():
            fh.write(f"  {k}: {v}\n")
        fh.write(f"\nWriter's claim in thesis §3.2.1: 465 + 312 + 130 = {writer_claim}; "
                 f"actual expanded manifest = {len(exp_ids)}.\n")
        fh.write(f"Writer-side gap: {len(exp_ids) - writer_claim} "
                 "(positive = thesis under-counts; negative = thesis over-counts).\n")
    print(f"Wrote {arith_path}")

    # -- Extract concat_cdrs for added entries --
    from Bio import SeqIO

    if not Path(args.expanded_concat_fasta).exists():
        print(f"ERROR: --expanded-concat-fasta does not exist: {args.expanded_concat_fasta}")
        print("       Brief 05 produced this during expanded clustering; Krijn must locate it.")
        sys.exit(1)

    all_recs = {r.id: r for r in SeqIO.parse(args.expanded_concat_fasta, "fasta")}
    print(f"Loaded {len(all_recs)} expanded concat_cdrs from {args.expanded_concat_fasta}")

    missing = [aid for aid in added_ids_sorted if aid not in all_recs]
    if missing:
        print(f"WARNING: {len(missing)}/{len(added_ids)} added IDs have no concat_cdrs entry")
        print(f"         (first 5 missing: {missing[:5]})")
    added_recs = [all_recs[aid] for aid in added_ids_sorted if aid in all_recs]
    added_fa = out_dir / "added_concat_cdrs.fasta"
    SeqIO.write(added_recs, added_fa, "fasta")
    print(f"Wrote {added_fa} ({len(added_recs)} records)")

    # -- Run MMseqs2 search (added → floor reps) --
    work = out_dir / "mmseqs_work"
    if work.exists():
        shutil.rmtree(work)  # mmseqs is picky about stale tmp dirs
    work.mkdir(parents=True)
    m8 = out_dir / "added_to_floor_reps.m8"

    cmd = [
        args.mmseqs_bin, "easy-search",
        str(added_fa), args.floor_rep_fasta,
        str(m8), str(work),
        "--min-seq-id", "0.7", "-c", "0.8", "--cov-mode", "0",
        "--format-output", "query,target,pident,evalue",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"MMseqs2 stderr: {result.stderr[:2000]}")
        sys.exit(1)
    print(f"Wrote {m8}")

    # -- Build per-entry cluster assignments + leakage flags --
    hits = pd.read_csv(m8, sep="\t", names=["query", "target", "pident", "evalue"])
    print(f"MMseqs2 hits: {len(hits)} rows (queries with ≥1 hit: {hits['query'].nunique()})")

    best = (hits.sort_values("pident", ascending=False)
                 .groupby("query").first().reset_index())

    cluster_tsv = pd.read_csv(args.floor_cluster_tsv, sep="\t", names=["cluster_rep", "member"])
    rep_to_cid = {rep: i for i, rep in enumerate(cluster_tsv["cluster_rep"].unique())}
    cluster_members = cluster_tsv.groupby("cluster_rep")["member"].apply(set).to_dict()

    shared_set = set(SHARED_HOLDOUT_29)
    exp_set = set(EXPANDED_HOLDOUT_83)
    shared_cluster_ids = {rep_to_cid[r] for r, m in cluster_members.items() if m & shared_set}
    expanded_cluster_ids = {rep_to_cid[r] for r, m in cluster_members.items() if m & exp_set}

    rows = []
    for aid in added_ids_sorted:
        src = id_to_source.get(aid, "unknown")
        if aid not in all_recs:
            rows.append({
                "entry_id": aid, "source": src,
                "cluster_id": "NO_CDR_DATA",
                "nearest_holdout_entry_in_cluster": None,
                "pct_identity_to_nearest_holdout": None,
                "is_shared_holdout_cluster": False,
                "is_expanded_holdout_cluster": False,
            })
            continue
        hit = best[best["query"] == aid]
        if len(hit) == 0:
            # No floor rep within 70 % / cov 80 % -- this added entry forms its own cluster.
            rows.append({
                "entry_id": aid, "source": src,
                "cluster_id": f"singleton_{aid}",
                "nearest_holdout_entry_in_cluster": None,
                "pct_identity_to_nearest_holdout": None,
                "is_shared_holdout_cluster": False,
                "is_expanded_holdout_cluster": False,
            })
            continue
        target = hit["target"].iloc[0]
        cid = rep_to_cid[target]
        members = cluster_members[target]
        shared_in = sorted(members & shared_set)
        exp_in = sorted(members & exp_set)
        nearest = shared_in[0] if shared_in else (exp_in[0] if exp_in else None)
        # mmseqs pident is already a percentage (0-100) in the easy-search output.
        pident = float(hit["pident"].iloc[0]) if nearest else None
        rows.append({
            "entry_id": aid, "source": src,
            "cluster_id": cid,
            "nearest_holdout_entry_in_cluster": nearest,
            "pct_identity_to_nearest_holdout": pident,
            "is_shared_holdout_cluster": cid in shared_cluster_ids,
            "is_expanded_holdout_cluster": cid in expanded_cluster_ids,
        })

    df_out = pd.DataFrame(rows)
    csv_path = out_dir / "expanded_manifest_cluster_assignments.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df_out)} rows)")

    # -- Summary stats --
    n_added_shared = int(df_out["is_shared_holdout_cluster"].sum())
    n_added_expanded = int(df_out["is_expanded_holdout_cluster"].sum())
    shared_rows = df_out[df_out["is_shared_holdout_cluster"]]
    expanded_rows = df_out[df_out["is_expanded_holdout_cluster"]]
    n_shared_contaminated = int(shared_rows["cluster_id"].nunique())
    n_expanded_contaminated = int(expanded_rows["cluster_id"].nunique())
    max_id_shared = shared_rows["pct_identity_to_nearest_holdout"].max() if len(shared_rows) else None
    max_id_expanded = expanded_rows["pct_identity_to_nearest_holdout"].max() if len(expanded_rows) else None

    print()
    print("=== LEAKAGE SUMMARY ===")
    print(f"  n_added_in_shared_cluster        : {n_added_shared}")
    print(f"  n_added_in_expanded_cluster      : {n_added_expanded}")
    print(f"  n_shared_clusters_contaminated   : {n_shared_contaminated}")
    print(f"  n_expanded_clusters_contaminated : {n_expanded_contaminated}")
    print(f"  max_pct_id_added_to_shared       : {max_id_shared}")
    print(f"  max_pct_id_added_to_expanded     : {max_id_expanded}")

    # Append leakage summary into the arithmetic file so a single cat call covers both.
    with open(arith_path, "a") as fh:
        fh.write("\n=== LEAKAGE SUMMARY ===\n")
        fh.write(f"  n_added_in_shared_cluster        : {n_added_shared}\n")
        fh.write(f"  n_added_in_expanded_cluster      : {n_added_expanded}\n")
        fh.write(f"  n_shared_clusters_contaminated   : {n_shared_contaminated}\n")
        fh.write(f"  n_expanded_clusters_contaminated : {n_expanded_contaminated}\n")
        fh.write(f"  max_pct_id_added_to_shared       : {max_id_shared}\n")
        fh.write(f"  max_pct_id_added_to_expanded     : {max_id_expanded}\n")


if __name__ == "__main__":
    main()
