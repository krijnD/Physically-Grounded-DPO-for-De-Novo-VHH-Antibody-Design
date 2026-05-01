#!/bin/bash
# Snellius SLURM script: Arm B fine-tune (Zenodo ANDD-init → our 191-cluster VHH set).
#
# DO NOT submit until Phase 1 has run:
#     python scripts/diffab_ft/check_zenodo_checkpoint.py \
#         --config third_party/diffab/configs/train/codesign_multicdrs.yml \
#         --checkpoint data/checkpoints/andd_init/checkpoint.pt
# and reported VERDICT: LOAD_OK or LOAD_OK_NONSTRICT. If it reports
# LOAD_FAIL, this submission will silently load garbage with strict=False.

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --job-name=diffab_ft_arm_b
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate

mkdir -p logs/slurm

echo "JobID:    ${SLURM_JOB_ID:-n/a}"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader || true)"
echo "Commit:   $(git rev-parse HEAD 2>/dev/null || echo n/a)"
echo "Branch:   $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
echo "Started:  $(date -Iseconds)"

python scripts/diffab_ft/train.py \
    configs/diffab_ft/arm_b_andd.yml \
    --output-dir runs/arm_b \
    --run-name   arm_b_seed42 \
    --num-workers 8 \
    --wandb-project vhh-diffab-ft \
    --wandb-run-name arm_b_seed42 \
    --wandb-group   arm_b_andd_init \
    --wandb-mode    offline

echo "Finished: $(date -Iseconds)"
