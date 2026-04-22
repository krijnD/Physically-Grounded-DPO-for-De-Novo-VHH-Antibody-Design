import argparse
import csv
import os
import shutil

def load_safe_pdb_ids(dates_csv: str, label: str) -> set[str]:
    """Return PDB IDs where the cutoff label column is True."""
    safe = set()
    with open(dates_csv, newline="") as f:
        reader = csv.DictReader(f)
        if label not in reader.fieldnames:
            raise ValueError(
                f"Column '{label}' not found in {dates_csv}. "
                f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            if row[label].strip() == "True":
                safe.add(row["pdb_id"].strip().upper())
    return safe


def subset_structures(safe_ids: set[str], structures_dir: str, output_dir: str) -> tuple[int, int]:
    """Copy PDB files for safe_ids into output_dir. Returns (copied, missing)."""
    os.makedirs(output_dir, exist_ok=True)
    copied, missing = 0, 0

    for pdb_id in sorted(safe_ids):
        src = os.path.join(structures_dir, f"{pdb_id}.pdb")
        dst = os.path.join(output_dir, f"{pdb_id}.pdb")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"  [WARNING] Structure not found: {pdb_id}.pdb")
            missing += 1

    return copied, missing


def subset_metadata_csv(safe_ids: set[str], metadata_csv: str, output_csv: str) -> int:
    """Write a filtered version of the metadata CSV keeping only safe_ids. Returns row count."""
    written = 0
    with open(metadata_csv, newline="") as fin, open(output_csv, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["PDB_ID"].strip().upper() in safe_ids:
                writer.writerow(row)
                written += 1
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Create a post-cutoff subset of VHH structures and metadata."
    )
    parser.add_argument(
        "--dates-csv",
        required=True,
        help="Path to andd_real_deposition_dates.csv (output of fetch_deposition_dates.py)",
    )
    parser.add_argument(
        "--structures-dir",
        required=True,
        help="Path to directory containing VHH PDB files (e.g. VHH_structures/)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to copy safe PDB files into (will be created if needed)",
    )
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="(Optional) Path to ANDD_VHH_with_structure.csv to also produce a filtered metadata CSV",
    )
    parser.add_argument(
        "--label",
        default="post_iglm",
        help="Boolean column in dates CSV to filter on (default: post_iglm)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="(Optional) Path for the filtered metadata CSV. Defaults to "
             "ANDD_VHH_with_structure_post_cutoff.csv next to --metadata-csv.",
    )
    args = parser.parse_args()

    # --- Load safe PDB IDs ---
    print(f"Reading deposition dates from: {args.dates_csv}")
    safe_ids = load_safe_pdb_ids(args.dates_csv, args.label)
    print(f"PDB IDs marked safe ('{args.label}' == True): {len(safe_ids)}\n")

    # --- Copy structures ---
    print(f"Copying structures from: {args.structures_dir}")
    print(f"Output directory:        {args.output_dir}")
    copied, missing = subset_structures(safe_ids, args.structures_dir, args.output_dir)
    print(f"  Copied:  {copied}")
    print(f"  Missing: {missing}")

    # --- Filter metadata CSV (optional) ---
    if args.metadata_csv:
        out_csv = args.output_csv or os.path.join(
            os.path.dirname(args.metadata_csv),
            "ANDD_VHH_with_structure_post_cutoff.csv",
        )
        print(f"\nFiltering metadata CSV: {args.metadata_csv}")
        rows = subset_metadata_csv(safe_ids, args.metadata_csv, out_csv)
        print(f"  Rows written: {rows}")
        print(f"  Output:       {out_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
