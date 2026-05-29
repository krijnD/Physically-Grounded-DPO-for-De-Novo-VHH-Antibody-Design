"""Relaxed-filter SAbDab nano downloader for the expanded-FT campaign.

Differences from fetch_nano.py:
  - Drop the date filter (all years).
  - Resolution: download the <= 3.0 A superset; brief 05 picks the final
    threshold empirically. NMR / no-resolution entries are also kept,
    tagged in download_manifest_ft.csv so brief 05 can decide.
  - Require antigen_chain (FT is antigen-conditioned).
  - Skip-if-exists preserved; polite 0.2 s delay preserved.

Run from the directory containing sabdab_nano_summary.tsv (e.g.
/projects/0/hpmlprjs/interns/krijn/sabdab_nano_dataset_IgLM/).
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

SUMMARY_TSV = "sabdab_nano_summary.tsv"
TARGET_DIR = "filtered_vhh_pdbs"
MANIFEST_CSV = "download_manifest_ft.csv"
FAILED_TXT = "download_failed_ft.txt"
RES_THRESH = 3.0       # download ceiling; brief 05 may tighten to <=2.5
KEEP_NMR = True        # include NaN-resolution (NMR / unknown) entries
REQUIRE_ANTIGEN = True # FT is antigen-conditioned
SLEEP_SEC = 0.2


def bucket_for(res):
    if pd.isna(res):
        return "NMR/unknown"
    if res <= 2.5:
        return "<=2.5"
    if res <= 3.0:
        return "2.5-3.0"
    return ">3.0"  # excluded


def main() -> int:
    df = pd.read_csv(SUMMARY_TSV, sep="\t")
    n_rows = len(df)
    n_unique = df["pdb"].nunique()
    print(f"Loaded {n_rows} rows / {n_unique} unique PDBs from {SUMMARY_TSV}")

    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y", errors="coerce")
    df["resolution"] = pd.to_numeric(df["resolution"], errors="coerce")

    # --- filters (applied row-wise; dedup to PDB after) ---
    if REQUIRE_ANTIGEN:
        before = df["pdb"].nunique()
        df = df[df["antigen_chain"].notna()].copy()
        after = df["pdb"].nunique()
        print(f"  antigen_chain notna:          {before} -> {after} unique PDBs (-{before - after})")

    if KEEP_NMR:
        df = df[(df["resolution"] <= RES_THRESH) | df["resolution"].isna()].copy()
    else:
        df = df[df["resolution"] <= RES_THRESH].copy()
    after_res = df["pdb"].nunique()
    print(f"  resolution <= {RES_THRESH} or NMR:    -> {after_res} unique PDBs")

    df["bucket"] = df["resolution"].apply(bucket_for)

    # Dedup by PDB code: keep best-resolution row per PDB (NaN res sorted last)
    df_sorted = df.sort_values(by="resolution", na_position="last")
    target = df_sorted.drop_duplicates(subset="pdb", keep="first").reset_index(drop=True)
    print(f"  after dedup by PDB code:      {len(target)} download candidates\n")

    bucket_counts = target["bucket"].value_counts().to_dict()
    print(f"Bucket counts (download candidates): {bucket_counts}\n")

    os.makedirs(TARGET_DIR, exist_ok=True)

    rows = []
    n_already = n_downloaded = n_failed = 0

    for _, row in target.iterrows():
        pdb_id = str(row["pdb"]).lower()
        file_path = os.path.join(TARGET_DIR, f"{pdb_id}.pdb")

        if os.path.exists(file_path):
            status = "already_present"
            n_already += 1
        else:
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(resp.content)
                    status = "downloaded"
                    n_downloaded += 1
                    if n_downloaded % 50 == 0:
                        print(f"    .. {n_downloaded} downloaded so far")
                else:
                    status = f"http_{resp.status_code}"
                    n_failed += 1
                    print(f"    FAIL {pdb_id}: status {resp.status_code}")
            except Exception as e:
                status = f"err:{type(e).__name__}"
                n_failed += 1
                print(f"    ERR  {pdb_id}: {e}")
            time.sleep(SLEEP_SEC)

        rows.append({
            "pdb": pdb_id,
            "resolution": row["resolution"],
            "method": row.get("method", ""),
            "date": row["date"].date().isoformat() if pd.notna(row["date"]) else "",
            "Hchain": row.get("Hchain", ""),
            "antigen_chain": row["antigen_chain"],
            "antigen_type": row.get("antigen_type", ""),
            "bucket": row["bucket"],
            "status": status,
        })

    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(MANIFEST_CSV, index=False)

    failed_ids = [r["pdb"] for r in rows
                  if r["status"] not in ("downloaded", "already_present")]
    Path(FAILED_TXT).write_text("\n".join(failed_ids) + ("\n" if failed_ids else ""))

    # --- final summary ---
    print("\n===== DOWNLOAD SUMMARY =====")
    print(f"Total candidates (post-filter): {len(target)}")
    print(f"  already present: {n_already}")
    print(f"  downloaded:      {n_downloaded}")
    print(f"  failed:          {n_failed}")
    print("\nBucket breakdown (post-filter, by status):")
    print(f"  {'bucket':>12}  {'total':>5}  {'on_disk':>7}  {'failed':>6}")
    for b in ["<=2.5", "2.5-3.0", "NMR/unknown"]:
        sub = manifest_df[manifest_df["bucket"] == b]
        on_disk = sub["status"].isin(["downloaded", "already_present"]).sum()
        failed = (~sub["status"].isin(["downloaded", "already_present"])).sum()
        print(f"  {b:>12}  {len(sub):>5}  {on_disk:>7}  {failed:>6}")

    print(f"\nManifest written: {MANIFEST_CSV}")
    print(f"Failed list:      {FAILED_TXT} ({len(failed_ids)} ids)")
    print(f"PDB dir:          {TARGET_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
