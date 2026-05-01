#!/bin/bash
# Snellius SLURM script: VHH fine-tune (luost26/DiffAb → 191-cluster VHH set).
#
# Submit from the project root:
#     sbatch scripts/diffab_ft/slurm/train.sbatch
#
# Outputs:
#   logs/slurm/diffab_ft_<jobid>.out  — combined stdout/stderr.
#   runs/vhh_ft/seed42/               — per-run artifacts (configs,
#                                        checkpoints, log.txt).
#
# Wall time budget: ~12-18h on a single A100 for 20K iters with our
# settings; 24h leaves headroom for warm-up I/O and the LMDB build on
# the first run.
#
# Seed sweep: copy this file to train_seed123.sbatch / train_seed2026.sbatch
# and change the seed in the config + run-name. The seed-stability table
# is the thesis ablation (no second arm).

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --job-name=diffab_ft
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
    configs/diffab_ft/vhh_ft.yml \
    --output-dir runs/vhh_ft \
    --run-name   seed42 \
    --num-workers 8 \
    --wandb-project vhh-diffab-ft \
    --wandb-run-name seed42 \
    --wandb-group   vhh_ft \
    --wandb-mode    offline   # safer on compute nodes; sync after with `wandb sync`

echo "Finished: $(date -Iseconds)"
