#!/bin/bash
# Snellius SLURM array: ABodyBuilder2 scRMSD on all persisted design PDBs
# across the 4 model variants × 2 test sets (Brief 12 + Brief 15 Track 1).
#
# Each task processes a 1-in-N stride of all PDBs found under --pdb-roots,
# writing a per-task shard parquet. Merge in a follow-up step.
#
# Submit:
#   sbatch scripts/eval/slurm/scrmsd_design.sbatch
#
# Override the output path via --export (placed BEFORE the script path):
#   sbatch --export=ALL,OUTPUT=data/eval/scrmsd_v2.parquet \
#          scripts/eval/slurm/scrmsd_design.sbatch

#SBATCH --job-name=scrmsd_v2
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

OUTPUT="${OUTPUT:-data/eval/scrmsd_design_samples.parquet}"

mkdir -p "$(dirname "$OUTPUT")" logs/slurm

echo "JobID:       ${SLURM_JOB_ID:-n/a}"
echo "ArrayJob:    ${SLURM_ARRAY_JOB_ID:-n/a}"
echo "ArrayTask:   ${SLURM_ARRAY_TASK_ID:-n/a}"
echo "Node:        $(hostname)"
echo "Output:      $OUTPUT"
echo "Started:     $(date -Iseconds)"

python scripts/eval/run_scrmsd_array.py \
    --pdb-roots \
        runs/vhh_ft/seed42_jfix/eval_test_pdbs \
        runs/dpo/dpo_seqonly_filtered/eval_test_pdbs \
        runs/vhh_ft/seed42_jfix_expanded/eval_oldtest_pdbs \
        runs/vhh_ft/seed42_jfix_expanded/eval_newtest_pdbs \
        runs/dpo/dpo_seqonly_filtered_expanded/eval_oldtest_pdbs \
        runs/dpo/dpo_seqonly_filtered_expanded/eval_newtest_pdbs \
    --output           "$OUTPUT" \
    --array-task-id    "$SLURM_ARRAY_TASK_ID" \
    --array-task-count 32

echo "Finished:    $(date -Iseconds)"
