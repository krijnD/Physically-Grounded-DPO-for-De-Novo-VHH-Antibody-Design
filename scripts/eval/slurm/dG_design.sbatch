#!/bin/bash
# Snellius SLURM array: compute InterfaceAnalyzerMover ΔG on the same
# chunked design manifest the main judges array consumes (Brief 11 §3
# Step 4, stretch axis).
#
# Reads chunk CSVs (AAPR-format from build_design_manifest.py), runs
# scripts/eval/compute_interface_dG.py per chunk, writes per-chunk
# parquets keyed by candidate_id. Master assembly joins these onto the
# main scored parquet on candidate_id.
#
# Submit alongside (or after) the main judges array — they are
# independent, both read from the same chunk dir, both write to
# disjoint output dirs.
#
# Usage:
#   sbatch --export=ALL,\
#          CHUNK_DIR=data/eval/chunks/all_variants,\
#          OUT_DIR=data/eval/dG_chunks/all_variants \
#          scripts/eval/slurm/dG_design.sbatch

#SBATCH --job-name=dG_design
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-31

set -uo pipefail
module purge
module load 2025 2024 gompi/2024a HMMER/3.4-gompi-2024a
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate
set -e

CHUNK_DIR="${CHUNK_DIR:?CHUNK_DIR must be set via --export=ALL,CHUNK_DIR=...}"
OUT_DIR="${OUT_DIR:?OUT_DIR must be set via --export=ALL,OUT_DIR=...}"

mkdir -p "$OUT_DIR" logs/slurm

TASK_ID=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
CSV_FILE="$CHUNK_DIR/chunk_${TASK_ID}.csv"
OUT_PARQUET="$OUT_DIR/chunk_${TASK_ID}.parquet"

if [[ ! -f "$CSV_FILE" ]]; then
    echo "ERROR: chunk CSV not found: $CSV_FILE" >&2
    exit 1
fi

echo "JobID:       ${SLURM_JOB_ID:-n/a}"
echo "ArrayJob:    ${SLURM_ARRAY_JOB_ID:-n/a}"
echo "ArrayTask:   ${SLURM_ARRAY_TASK_ID:-n/a}"
echo "Node:        $(hostname)"
echo "Chunk CSV:   $CSV_FILE"
echo "Output:      $OUT_PARQUET"
echo "Started:     $(date -Iseconds)"

python scripts/eval/compute_interface_dG.py \
    --input-csv      "$CSV_FILE" \
    --output-parquet "$OUT_PARQUET"

echo "Finished:    $(date -Iseconds)"
