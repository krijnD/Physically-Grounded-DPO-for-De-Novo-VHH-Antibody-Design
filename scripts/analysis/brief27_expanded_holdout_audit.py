"""Brief 27 -- Expanded-holdout MMseqs2 audit.

Are there additional train↔test leakage cases under the EXPANDED re-clustering,
beyond the 3 Brief 23 found against the FLOOR clustering?

Brief 23 checked added training entries vs floor 70 %-identity cluster
boundaries (3 leakage cases). It did NOT check the 83-entry expanded holdout
against the full expanded-train pool under the expanded re-clustering. MMseqs2
with --min-seq-id 0.7 -c 0.8 doesn't guarantee that all 100 %-identity
sequences land in the same cluster, so Brief 05's "cluster integrity preserved"
assertion doesn't rule out additional high-identity train↔test pairs.

Method: MMseqs2 easy-search of expanded-train concat-CDRs (built from floor
concat_cdrs + Brief 23's added concat_cdrs, subset to splits.train) against the
83-entry expanded-holdout concat-CDRs, same parameters as the floor clustering
(--min-seq-id 0.7 -c 0.8 --cov-mode 0). Flag hits at ≥85 % identity
(matching Brief 23's gate); cross-reference the train-side id against the 3
already-known Brief 23 cases to separate KNOWN from NEW.

Inputs (on Snellius):
- floor concat_cdrs              data/datasets/clustering/concat_cdrs.fasta (465 entries)
- added concat_cdrs (Brief 23)   tmp_brief23/added_concat_cdrs.fasta (~462 entries)
                                 OR fallback: --combined-curated-csv to rebuild via abnumber
- expanded cluster splits        data/datasets/clustering/cluster_splits_expanded.json
                                 (verify exact filename on Snellius)

Outputs (under --out-dir, default tmp_brief27/):
- expanded_train_concat_cdrs.fasta     (the ~922 expanded-train pool)   -- not committed
- expanded_holdout_concat_cdrs.fasta   (83 entries)                     -- not committed
- expanded_train_vs_holdout.m8         (MMseqs2 raw hits ≥70 %)         -- not committed
- expanded_holdout_leakage_summary.csv (hits ≥85 %, flagged + tagged)   -- COMMITTED
- audit_summary.txt                    (human-readable one-page)        -- COMMITTED
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

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
assert len(EXPANDED_HOLDOUT_83) == 83

# Brief 23 deliverable §"Leakage" table -- the 3 added training entries that
# clustered with floor-holdout members at ≥85 % CDR identity. These are the
# TRAIN-side ids (the contaminating entries).
BRIEF23_LEAKED_TRAIN = {"7jkm_K", "7o31_X", "5o2u_D"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--floor-fasta",
                   default="data/datasets/clustering/concat_cdrs.fasta",
                   help="floor concat_cdrs (Brief 05 output, 465 entries)")
    p.add_argument("--added-fasta",
                   default="tmp_brief23/added_concat_cdrs.fasta",
                   help="Brief 23 added concat_cdrs (~462 entries). If missing, "
                        "pass --combined-curated-csv to rebuild via abnumber.")
    p.add_argument("--combined-curated-csv",
                   default="data/raw/curate_full/combined_curated.csv",
                   help="fallback source for the added CDRs if --added-fasta is absent.")
    p.add_argument("--expanded-splits",
                   default="data/datasets/clustering/cluster_splits_expanded.json",
                   help="cluster split JSON for the expanded re-clustering. "
                        "If filename differs, check `ls data/datasets/clustering/*expanded*.json`.")
    p.add_argument("--scheme", default="chothia",
                   help="abnumber numbering scheme for the curated-CSV fallback.")
    p.add_argument("--out-dir", default="tmp_brief27")
    p.add_argument("--mmseqs-bin", default="mmseqs")
    p.add_argument("--threshold", type=float, default=0.85,
                   help="Identity threshold (0-1) for FLAGGING hits (matches Brief 23's "
                        "gate). Raw .m8 retains everything ≥70 % for completeness.")
    return p.parse_args()


def load_splits_train_test(path):
    """Robust load of cluster_splits_expanded.json. Accept either nested under
    'splits' (Brief 05 cluster_split.py convention) or top-level keys.
    """
    with open(path) as fh:
        data = json.load(fh)
    if "splits" in data:
        s = data["splits"]
        train = set(s.get("train", []))
        test = set(s.get("test", []))
    else:
        train = set(data.get("train", []))
        test = set(data.get("test", []))
    if not train or not test:
        raise ValueError(
            f"Could not locate train/test member lists in {path}. "
            f"Top-level keys: {list(data.keys())}"
        )
    return train, test


def extract_concat_cdrs(seq, scheme="chothia"):
    """Mirror of scripts/diffab_ft/cluster_split.py::extract_concat_cdrs."""
    from abnumber import Chain as AbnumberChain
    from abnumber.exceptions import ChainParseError
    try:
        chain = AbnumberChain(seq, scheme=scheme)
    except (ChainParseError, ValueError):
        return None
    if chain.chain_type != "H":
        return None
    cdr1 = chain.cdr1_seq or ""
    cdr2 = chain.cdr2_seq or ""
    cdr3 = chain.cdr3_seq or ""
    if not (cdr1 and cdr2 and cdr3):
        return None
    return cdr1 + cdr2 + cdr3


def build_added_cdrs_from_curated(curated_csv_path, scheme):
    """Read combined_curated.csv -> {entry_id: cdr_concat_seq}. Fallback when
    tmp_brief23/added_concat_cdrs.fasta is absent."""
    cur = pd.read_csv(curated_csv_path)
    needed = {"PDB_ID", "H_Chain Auth Asym ID", "Ab/Nano H_Chain AA"}
    missing = needed - set(cur.columns)
    if missing:
        raise ValueError(f"--combined-curated-csv missing required columns: {missing}")
    out = {}
    n_no_seq = n_no_cdr = 0
    for _, row in cur.iterrows():
        pdb = str(row["PDB_ID"]).strip().lower()
        hch = str(row["H_Chain Auth Asym ID"]).strip()
        seq = str(row["Ab/Nano H_Chain AA"]).strip()
        if not seq or seq.lower() == "nan":
            n_no_seq += 1
            continue
        cdrs = extract_concat_cdrs(seq, scheme=scheme)
        if cdrs is None:
            n_no_cdr += 1
            continue
        out[f"{pdb}_{hch}"] = cdrs
    print(f"Curated CSV: {len(cur)} rows; extracted CDRs for {len(out)} entries "
          f"(skipped {n_no_seq} no-seq, {n_no_cdr} CDR-extraction failed).")
    return out


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -- Load expanded splits --
    splits_path = Path(args.expanded_splits)
    if not splits_path.exists():
        print(f"ERROR: --expanded-splits {splits_path} not found.")
        print(f"       Try: ls data/datasets/clustering/*expanded* 2>/dev/null")
        sys.exit(1)
    train_ids, test_ids = load_splits_train_test(splits_path)
    print(f"Expanded train: {len(train_ids)} entries")
    print(f"Expanded test:  {len(test_ids)} entries")

    # -- Cross-check the hardcoded 83-entry list against splits.test (§6 gate 3) --
    missing_from_splits = sorted(set(EXPANDED_HOLDOUT_83) - test_ids)
    if missing_from_splits:
        # Surface but don't abort -- the hardcoded list comes from per_sample_csvs
        # and may diverge from cluster_splits convention. The audit treats the
        # hardcoded list as ground truth.
        print(f"WARNING: {len(missing_from_splits)}/83 hardcoded holdout entries are NOT "
              f"in cluster_splits_expanded.json test set.")
        print(f"         First 5 missing: {missing_from_splits[:5]}")
        print(f"         Continuing -- audit uses the hardcoded 83-entry list as ground truth.")
    else:
        print(f"OK: all 83 hardcoded holdout entries are a subset of splits.test.")

    # -- Load expanded concat_cdrs (floor + Brief 23 added, or curated fallback) --
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    all_cdrs = {}
    floor_path = Path(args.floor_fasta)
    if not floor_path.exists():
        print(f"ERROR: --floor-fasta {floor_path} not found.")
        sys.exit(1)
    for r in SeqIO.parse(floor_path, "fasta"):
        all_cdrs[r.id] = str(r.seq)
    print(f"Loaded {len(all_cdrs)} floor concat_cdrs from {floor_path}")

    added_path = Path(args.added_fasta)
    if added_path.exists():
        n_before = len(all_cdrs)
        for r in SeqIO.parse(added_path, "fasta"):
            all_cdrs[r.id] = str(r.seq)
        print(f"Loaded {len(all_cdrs) - n_before} added concat_cdrs from {added_path}")
    elif args.combined_curated_csv and Path(args.combined_curated_csv).exists():
        print(f"--added-fasta {added_path} not present; falling back to "
              f"--combined-curated-csv {args.combined_curated_csv}.")
        added_dict = build_added_cdrs_from_curated(args.combined_curated_csv, args.scheme)
        n_before = len(all_cdrs)
        for eid, seq in added_dict.items():
            all_cdrs[eid] = seq
        print(f"Added {len(all_cdrs) - n_before} new concat_cdrs from curated CSV.")
    else:
        print(f"ERROR: neither --added-fasta ({added_path}) nor --combined-curated-csv "
              f"({args.combined_curated_csv}) is readable.")
        sys.exit(1)

    # -- Build train + test fastas --
    train_recs = [SeqRecord(Seq(all_cdrs[eid]), id=eid, description="")
                  for eid in sorted(train_ids) if eid in all_cdrs]
    test_recs = [SeqRecord(Seq(all_cdrs[eid]), id=eid, description="")
                 for eid in EXPANDED_HOLDOUT_83 if eid in all_cdrs]

    train_missing_cdr = [eid for eid in train_ids if eid not in all_cdrs]
    test_missing_cdr = [eid for eid in EXPANDED_HOLDOUT_83 if eid not in all_cdrs]
    print(f"Train fasta records: {len(train_recs)} of {len(train_ids)} expanded-train members "
          f"(missing CDRs for {len(train_missing_cdr)})")
    print(f"Test  fasta records: {len(test_recs)} of {len(EXPANDED_HOLDOUT_83)} hardcoded holdout entries "
          f"(missing CDRs for {len(test_missing_cdr)})")
    if test_missing_cdr:
        print(f"  Missing test entries: {test_missing_cdr}")

    if not train_recs or not test_recs:
        print(f"ERROR: empty fasta on one side (train={len(train_recs)}, test={len(test_recs)}). "
              f"MMseqs2 would fail.")
        sys.exit(1)

    train_fa = out / "expanded_train_concat_cdrs.fasta"
    test_fa = out / "expanded_holdout_concat_cdrs.fasta"
    SeqIO.write(train_recs, train_fa, "fasta")
    SeqIO.write(test_recs, test_fa, "fasta")
    print(f"Wrote {train_fa}")
    print(f"Wrote {test_fa}")

    # -- MMseqs2 search: expanded-train -> 83-entry holdout --
    work = out / "mmseqs_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    m8 = out / "expanded_train_vs_holdout.m8"

    cmd = [
        args.mmseqs_bin, "easy-search",
        str(train_fa), str(test_fa),
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

    # -- Analyze hits --
    # MMseqs easy-search returns pident as a percentage (0-100), confirmed by
    # Brief 23's script (see brief23_manifest_leakage.py line 338 comment).
    if m8.stat().st_size == 0:
        hits = pd.DataFrame(columns=["train_id", "holdout_id", "pident", "evalue"])
    else:
        hits = pd.read_csv(m8, sep="\t",
                           names=["train_id", "holdout_id", "pident", "evalue"])
    print(f"MMseqs2 hits (≥70 %): {len(hits)} rows")

    hits_flagged = hits[hits["pident"] >= args.threshold * 100].copy()
    hits_flagged["is_brief23_known"] = hits_flagged["train_id"].isin(BRIEF23_LEAKED_TRAIN)
    hits_flagged = (hits_flagged
                    .sort_values("pident", ascending=False)
                    .reset_index(drop=True))
    csv_path = out / "expanded_holdout_leakage_summary.csv"
    hits_flagged.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(hits_flagged)} rows at ≥{args.threshold*100:.0f} %)")

    # -- Summary --
    n_flagged = len(hits_flagged)
    n_known = int(hits_flagged["is_brief23_known"].sum())
    n_new = n_flagged - n_known
    distinct_holdout = hits_flagged["holdout_id"].nunique() if n_flagged else 0
    max_pident = float(hits_flagged["pident"].max()) if n_flagged else 0.0

    lines = [
        "=== Expanded-holdout MMseqs2 audit (Brief 27) ===",
        f"Threshold: ≥{args.threshold*100:.0f} % CDR identity",
        f"MMseqs2 params: --min-seq-id 0.7 -c 0.8 --cov-mode 0",
        "",
        f"Expanded splits: train {len(train_ids)}, test {len(test_ids)}",
        f"Hardcoded 83-entry holdout vs splits.test: "
        f"{83 - len(missing_from_splits)}/83 match"
        + (f" ({len(missing_from_splits)} mismatched)" if missing_from_splits else ""),
        f"Train fasta records: {len(train_recs)} of {len(train_ids)} expanded-train members",
        f"Test  fasta records: {len(test_recs)} of {len(EXPANDED_HOLDOUT_83)} hardcoded holdout entries",
        "",
        f"Total flagged train→holdout pairs: {n_flagged}",
        f"  Brief 23-known training entries: {n_known}",
        f"  NEW (not in Brief 23):           {n_new}",
        f"Distinct expanded-holdout entries contaminated: {distinct_holdout}",
        f"Max % identity among flagged pairs: {max_pident:.1f}%",
        "",
        "Detail (sorted by % identity desc):",
    ]
    if n_flagged > 0:
        for _, row in hits_flagged.iterrows():
            tag = "(known)" if row["is_brief23_known"] else "(NEW)"
            lines.append(
                f"  {row['train_id']:>10s} → {row['holdout_id']:>10s}  "
                f"{row['pident']:>5.1f}%  {tag}"
            )
    else:
        lines.append("  None.")

    txt = "\n".join(lines)
    print()
    print(txt)
    (out / "audit_summary.txt").write_text(txt + "\n")
    print(f"\nWrote {out / 'audit_summary.txt'}")


if __name__ == "__main__":
    main()
