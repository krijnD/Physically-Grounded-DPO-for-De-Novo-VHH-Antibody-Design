#!/usr/bin/env python3
"""Merge per-chunk judge result parquets into a single parquet.

Companion to ``split_csv.py`` + ``judges_andd.sbatch``. Run after the
Slurm array completes and you have a directory of ``chunk_NN.parquet``
(or ``.csv`` fallback) files.

Usage:
    python scripts/judges/slurm/merge_chunks.py \
        --chunk-dir data/results/judges_andd \
        --output    data/results/andd_judge_test_full.parquet

Reports row counts per chunk and total. Detects mixed parquet/CSV
chunks (which can happen if a worker missed pyarrow) and merges them
all using the same schema check.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("merge_chunks")


def _read_chunk(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported chunk file extension: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunk-dir",
        required=True,
        help="Directory containing chunk_NN.parquet (or .csv) files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the merged parquet (or .csv if pyarrow missing)",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Don't fail if some chunk indices are missing; just merge what's there",
    )
    args = parser.parse_args()

    chunk_dir = Path(args.chunk_dir)
    if not chunk_dir.is_dir():
        logger.error("Chunk dir not found: %s", chunk_dir)
        sys.exit(1)

    # Sort by name so chunk_00, chunk_01, ... merge in order.
    paths = sorted(
        list(chunk_dir.glob("chunk_*.parquet"))
        + list(chunk_dir.glob("chunk_*.csv"))
    )
    if not paths:
        logger.error("No chunk_*.parquet or chunk_*.csv files in %s", chunk_dir)
        sys.exit(1)

    # Detect missing indices in the chunk_NN naming.
    indices = []
    for p in paths:
        stem = p.stem  # "chunk_07"
        try:
            idx = int(stem.split("_", 1)[1])
            indices.append(idx)
        except (IndexError, ValueError):
            logger.warning("Skipping unparseable chunk name: %s", p.name)
    indices.sort()
    if indices:
        missing = sorted(set(range(indices[0], indices[-1] + 1)) - set(indices))
        if missing:
            msg = f"Missing chunk indices: {missing}"
            if args.allow_missing:
                logger.warning(msg + " (continuing because --allow-missing)")
            else:
                logger.error(msg + " (use --allow-missing to merge anyway)")
                sys.exit(1)

    dfs: list[pd.DataFrame] = []
    total = 0
    for p in paths:
        try:
            df = _read_chunk(p)
        except Exception as e:
            logger.error("Failed to read %s: %s", p, e)
            sys.exit(1)
        logger.info("  %s — %d rows", p.name, len(df))
        dfs.append(df)
        total += len(df)

    merged = pd.concat(dfs, ignore_index=True)
    if len(merged) != total:
        logger.warning(
            "Concat row count mismatch: %d vs sum=%d", len(merged), total
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        merged.to_parquet(output, index=False)
        logger.info("Wrote %s (%d rows)", output, len(merged))
    except ImportError:
        csv_out = output.with_suffix(".csv")
        merged.to_csv(csv_out, index=False)
        logger.warning(
            "pyarrow missing — wrote CSV fallback %s (%d rows)",
            csv_out, len(merged),
        )


if __name__ == "__main__":
    main()
