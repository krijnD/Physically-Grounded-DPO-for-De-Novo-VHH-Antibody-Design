#!/usr/bin/env python3
"""Split a CSV (with header) into N approximately-equal chunks.

Used to fan out the Physics Judge across a Slurm job array — one chunk
per array task. Each chunk preserves the header so it can be passed
directly to ``scripts/test_sabdab_judges.py --csv``.

Usage:
    python scripts/judges/slurm/split_csv.py \
        --input  /projects/0/hpmlprjs/interns/krijn/ANDD_nano_dataset_IgLM/ANDD_VHH_curated_diffab.csv \
        --n-chunks 32 \
        --output-dir data/results/judges_chunks

Produces:
    data/results/judges_chunks/chunk_00.csv
    data/results/judges_chunks/chunk_01.csv
    ...
    data/results/judges_chunks/chunk_31.csv
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("split_csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument(
        "--n-chunks", type=int, required=True, help="Number of output chunks"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory; created if missing",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input CSV not found: %s", input_path)
        sys.exit(1)

    if args.n_chunks < 1:
        logger.error("--n-chunks must be >= 1, got %d", args.n_chunks)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read header + all rows.
    with input_path.open(newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            logger.error("Input CSV is empty: %s", input_path)
            sys.exit(1)
        rows = list(reader)

    n_rows = len(rows)
    logger.info(
        "Input: %s — %d data rows, header: %s", input_path, n_rows, header
    )

    if n_rows == 0:
        logger.error("Input CSV has a header but no data rows.")
        sys.exit(1)

    # Even distribution: floor-divide and spread the remainder across
    # the first (n_rows % n_chunks) chunks. Earlier versions used ceiling
    # division (chunk_size = ceil(n_rows / n_chunks)), which produced
    # n_chunks-1 actual chunks when n_chunks evenly divided into the
    # ceiling — e.g. --n-chunks 32 on 465 rows gave 31 chunks because
    # 15*31 = 465 exactly, so the 32nd task always failed.
    n_chunks = min(args.n_chunks, n_rows)
    if n_chunks < args.n_chunks:
        logger.warning(
            "Requested %d chunks but only %d data rows — producing %d chunks.",
            args.n_chunks, n_rows, n_chunks,
        )

    base_size, remainder = divmod(n_rows, n_chunks)
    # First `remainder` chunks get an extra row.
    chunk_sizes = [base_size + 1] * remainder + [base_size] * (n_chunks - remainder)
    assert sum(chunk_sizes) == n_rows, (sum(chunk_sizes), n_rows)
    assert len(chunk_sizes) == n_chunks

    width = max(2, len(str(n_chunks - 1)))
    written = 0
    cursor = 0
    for i, size in enumerate(chunk_sizes):
        start = cursor
        end = cursor + size
        chunk_path = output_dir / f"chunk_{i:0{width}d}.csv"
        with chunk_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows[start:end])
        logger.info(
            "Wrote %s (%d rows: %d..%d)",
            chunk_path, end - start, start, end - 1,
        )
        written += 1
        cursor = end

    logger.info(
        "Done — %d chunks written to %s (%d rows total).",
        written, output_dir, n_rows,
    )


if __name__ == "__main__":
    main()
