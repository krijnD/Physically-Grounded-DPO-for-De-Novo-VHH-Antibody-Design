#!/bin/bash
# Snellius SLURM script — Brief 17 §8: per-channel iter-0 reference loss.
#
# Recomputes L_w_ref / L_l_ref per channel (rot, pos, seq) on the decoy
# pair pool. Output drives the §9 GATE check: pass requires
# |reward_rot.mean()| ≤ 0.1 AND |reward_pos.mean()| ≤ 0.1, with
# implicit-reward = -T · (L_w_ref - L_l_ref).
#
# Wall-time budget: ~20 min on A100 (1492 pairs / batch 4 = 373 batches
# × ~3 s/batch = ~19 min; matches diag_lwref_distribution.py).
#
# Submit from the project root, passing pi-ref / pairs / out as CLI:
#   sbatch scripts/dpo/slurm/diag_per_channel_reward.sbatch \
#       --pi-ref-checkpoint runs/vhh_ft/seed42_jfix/checkpoints/best_ema.pt \
#       --pairs-parquet     data/aapr/.../dpo/pairs_decoy_t10.parquet \
#       --output            data/aapr/.../dpo/lwref_per_channel_decoy_t10.parquet \
#       --t-eval            50 \
#       --device            cuda

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --job-name=diag_pc_reward
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

# ── Modules + venv ──────────────────────────────────────────────────
module purge
module load 2025 2024 gompi/2024a HMMER/3.4-gompi-2024a
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate

mkdir -p logs/slurm

# ── Provenance ──────────────────────────────────────────────────────
echo "JobID:    ${SLURM_JOB_ID:-n/a}"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader || true)"
echo "Commit:   $(git rev-parse HEAD 2>/dev/null || echo n/a)"
echo "Branch:   $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
echo "Started:  $(date -Iseconds)"
echo "Args:     $*"

# ── Run ─────────────────────────────────────────────────────────────
# All flags are forwarded to the Python script. See script docstring for
# the expected combination.
python scripts/dpo/diag_per_channel_reward.py "$@"

echo "Finished: $(date -Iseconds)"
