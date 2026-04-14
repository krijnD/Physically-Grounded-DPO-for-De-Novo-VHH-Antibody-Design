import argparse
import csv
import json
import os
import time
import requests
from datetime import datetime

RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"
BATCH_SIZE = 50


def extract_real_pdb_ids(csv_path: str) -> list[str]:
    pdb_ids = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb = row["PDB_ID"].strip()
            pred = row["Predicted_or_Not"].strip()
            if pdb and pred == "real":
                pdb_ids.add(pdb.upper())
    return sorted(pdb_ids)


def fetch_deposition_dates(pdb_ids: list[str]) -> dict[str, str | None]:
    results = {}
    total_batches = (len(pdb_ids) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(pdb_ids), BATCH_SIZE):
        batch = pdb_ids[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"Fetching batch {batch_num}/{total_batches} ({len(batch)} PDB IDs)...")

        ids_str = json.dumps(batch)
        query = f"""
        {{
          entries(entry_ids: {ids_str}) {{
            rcsb_id
            rcsb_accession_info {{
              initial_release_date
            }}
          }}
        }}
        """

        try:
            response = requests.post(
                RCSB_GRAPHQL_URL,
                json={"query": query},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            entries = data.get("data", {}).get("entries", []) or []
            for entry in entries:
                pdb_id = entry["rcsb_id"]
                release_date = (
                    entry.get("rcsb_accession_info", {}) or {}
                ).get("initial_release_date")
                results[pdb_id] = release_date
        except Exception as e:
            print(f"  Error on batch {batch_num}: {e}")
            for pdb_id in batch:
                if pdb_id not in results:
                    results[pdb_id] = None

        time.sleep(0.2)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fetch RCSB deposition dates for real ANDD VHH PDB structures."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the CSV file containing PDB_ID and Predicted_or_Not columns.",
    )
    parser.add_argument(
        "--cutoff",
        default="2022-01-01",
        help="Training data cutoff date (YYYY-MM-DD). PDBs on or after this date are safe. "
             "Default: 2022-01-01 (IgLM). Example for ESM: --cutoff 2020-05-01",
    )
    parser.add_argument(
        "--label",
        default="post_cutoff",
        help="Column name for the boolean flag in the output CSV. "
             "Default: 'post_cutoff'. Example: --label post_iglm",
    )
    args = parser.parse_args()

    input_path = args.input
    cutoff = datetime.strptime(args.cutoff, "%Y-%m-%d")
    label = args.label
    output_path = os.path.join(os.path.dirname(input_path), "andd_real_deposition_dates.csv")

    print(f"Reading PDB IDs from: {input_path}")
    print(f"Training cutoff date: {args.cutoff}  (flag column: '{label}')")
    pdb_ids = extract_real_pdb_ids(input_path)
    print(f"Found {len(pdb_ids)} unique real PDB IDs.\n")

    date_map = fetch_deposition_dates(pdb_ids)

    # Write output CSV
    post_iglm_count = 0
    pre_iglm_count = 0
    failed_count = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pdb_id", "deposition_date", label])

        for pdb_id in pdb_ids:
            raw_date = date_map.get(pdb_id)
            if raw_date is None:
                writer.writerow([pdb_id, "", ""])
                failed_count += 1
                continue

            # RCSB returns ISO 8601, e.g. "2022-03-15T00:00:00+0000"
            dep_date = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            dep_date_str = dep_date.strftime("%Y-%m-%d")
            post_cutoff = dep_date.replace(tzinfo=None) >= cutoff

            writer.writerow([pdb_id, dep_date_str, post_cutoff])
            if post_cutoff:
                post_iglm_count += 1
            else:
                pre_iglm_count += 1

    print(f"\nOutput written to: {output_path}")
    print(f"  Total PDB IDs:              {len(pdb_ids)}")
    print(f"  Successfully fetched:       {len(pdb_ids) - failed_count}")
    print(f"  Failed / missing:           {failed_count}")
    print(f"  Post-cutoff ({args.cutoff}): {post_iglm_count}  <- safe to use")
    print(f"  Pre-cutoff  ({args.cutoff}):  {pre_iglm_count}  <- potential contamination")


if __name__ == "__main__":
    main()
