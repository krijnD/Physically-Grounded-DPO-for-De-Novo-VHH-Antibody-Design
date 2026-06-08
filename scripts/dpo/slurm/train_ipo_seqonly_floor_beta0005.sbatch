#!/bin/bash
# Brief 18 — HEADLINE run: IPO at β=0.005 on the FLOOR pair pool.
#
# Hypothesis: IPO's bounded-quadratic objective escapes the Isoleucine-
# homopolymer collapse that DPO at this same β produced in Brief 16
# (99.7% IIII...III generation, scRMSD 700-1450 Å, 0% designability).
#
# Run alongside the other 3 IPO sbatches; no inter-job dependencies.

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --job-name=ipo_floor_beta0005
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate
mkdir -p logs/slurm

echo "JobID: ${SLURM_JOB_ID:-n/a}  Node: $(hostname)"
echo "Commit: $(git rev-parse HEAD 2>/dev/null || echo n/a)"
echo "Started: $(date -Iseconds)"

python scripts/dpo/train_dpo.py \
    configs/dpo/vhh_ipo_seqonly_floor_beta0005.yml \
    --output-dir runs/dpo \
    --run-name   ipo_seqonly_floor_beta0005 \
    --num-workers 4 \
    --wandb-project vhh-dpo \
    --wandb-run-name ipo_seqonly_floor_beta0005 \
    --wandb-group   vhh_ipo_brief18 \
    --wandb-mode    online

echo "Finished: $(date -Iseconds)"
