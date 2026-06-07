#!/bin/bash
# Brief 16 §4.7 — v2 scRMSD dispatcher for β=0.5 sweep variant.
# OLD test only (n=29). Reads design PDBs persisted by §4.5 design-eval at
# runs/dpo/floor_dpo_beta05/eval_test_pdbs.
#
# Submit AFTER §4.5 design-eval (eval_beta05_oldtest) completes.

#SBATCH --job-name=scrmsd_beta05
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-3

set -uo pipefail
module purge
module load 2025 2024 gompi/2024a HMMER/3.4-gompi-2024a
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate
set -e

OUTPUT="${OUTPUT:-data/eval/scrmsd_beta05.parquet}"

mkdir -p "$(dirname "$OUTPUT")" logs/slurm

echo "JobID:       ${SLURM_JOB_ID:-n/a}"
echo "ArrayJob:    ${SLURM_ARRAY_JOB_ID:-n/a}"
echo "ArrayTask:   ${SLURM_ARRAY_TASK_ID:-n/a}"
echo "Node:        $(hostname)"
echo "Output:      $OUTPUT"
echo "Started:     $(date -Iseconds)"

python scripts/eval/run_scrmsd_array.py \
    --pdb-roots \
        runs/dpo/floor_dpo_beta05/eval_test_pdbs \
    --output           "$OUTPUT" \
    --array-task-id    "$SLURM_ARRAY_TASK_ID" \
    --array-task-count 4

echo "Finished:    $(date -Iseconds)"
