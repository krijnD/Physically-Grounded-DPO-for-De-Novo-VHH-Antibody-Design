#!/bin/bash
# Snellius SLURM array: ChimeraBench CAAR + EpiF1 on all persisted design
# PDBs across the 4 model variants × 2 test sets (Brief 13 §3 Step 4).
#
# Each task processes a 1-in-N stride of all PDBs found under the
# --design-pdb-roots globs, writing a per-task shard parquet. Merge with
# scripts/eval/build_master_parquet.py or a similar wrapper after the
# array completes.
#
# Submit:
#   sbatch scripts/eval/slurm/caar_epif1_design.sbatch
#
# Optionally override the GT-PDB map or output path via --export:
#   sbatch --export=ALL,GT_MAP=...,OUTPUT=... \
#          scripts/eval/slurm/caar_epif1_design.sbatch

#SBATCH --job-name=caar_epif1
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

GT_MAP="${GT_MAP:-data/eval/gt_pdb_map.json}"
OUTPUT="${OUTPUT:-data/eval/caar_epif1.parquet}"

mkdir -p "$(dirname "$OUTPUT")" logs/slurm

echo "JobID:       ${SLURM_JOB_ID:-n/a}"
echo "ArrayJob:    ${SLURM_ARRAY_JOB_ID:-n/a}"
echo "ArrayTask:   ${SLURM_ARRAY_TASK_ID:-n/a}"
echo "Node:        $(hostname)"
echo "GT map:      $GT_MAP"
echo "Output:      $OUTPUT"
echo "Started:     $(date -Iseconds)"

python scripts/eval/run_caar_epif1_array.py \
    --design-pdb-roots \
        runs/vhh_ft/seed42_jfix/eval_test_pdbs \
        runs/dpo/dpo_seqonly_filtered/eval_test_pdbs \
        runs/vhh_ft/seed42_jfix_expanded/eval_oldtest_pdbs \
        runs/vhh_ft/seed42_jfix_expanded/eval_newtest_pdbs \
        runs/dpo/dpo_seqonly_filtered_expanded/eval_oldtest_pdbs \
        runs/dpo/dpo_seqonly_filtered_expanded/eval_newtest_pdbs \
    --gt-pdb-map       "$GT_MAP" \
    --output           "$OUTPUT" \
    --array-task-id    "$SLURM_ARRAY_TASK_ID" \
    --array-task-count 32

echo "Finished:    $(date -Iseconds)"
