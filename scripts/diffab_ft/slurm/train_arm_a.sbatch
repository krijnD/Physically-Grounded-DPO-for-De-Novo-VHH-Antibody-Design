#!/bin/bash
# Snellius SLURM script: Arm A fine-tune (luost26/DiffAb → our 191-cluster VHH set).
#
# Submit from the project root:
#     sbatch scripts/diffab_ft/slurm/train_arm_a.sbatch
#
# Outputs:
#   logs/slurm/diffab_ft_arm_a_<jobid>.out  — combined stdout/stderr.
#   runs/arm_a/arm_a_seed42/                — per-run artifacts (configs,
#                                              checkpoints, log.txt).
#
# Wall time budget: ~12-18h on a single A100 for 20K iters with our
# settings; 24h leaves headroom for warm-up I/O and the LMDB build on
# the first run.

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --job-name=diffab_ft_arm_a
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

# ── Modules + venv ──────────────────────────────────────────────────
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate

# Make sure logs/slurm exists (sbatch fails before the job script runs
# otherwise, but harmless on re-runs).
mkdir -p logs/slurm

# ── Provenance ──────────────────────────────────────────────────────
echo "JobID:    ${SLURM_JOB_ID:-n/a}"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader || true)"
echo "Commit:   $(git rev-parse HEAD 2>/dev/null || echo n/a)"
echo "Branch:   $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
echo "Started:  $(date -Iseconds)"

# ── Run ─────────────────────────────────────────────────────────────
python scripts/diffab_ft/train.py \
    configs/diffab_ft/arm_a_sabdab.yml \
    --output-dir runs/arm_a \
    --run-name   arm_a_seed42 \
    --num-workers 8 \
    --wandb-project vhh-diffab-ft \
    --wandb-run-name arm_a_seed42 \
    --wandb-group   arm_a_sabdab_init \
    --wandb-mode    offline   # safer on compute nodes; sync after with `wandb sync`

echo "Finished: $(date -Iseconds)"
