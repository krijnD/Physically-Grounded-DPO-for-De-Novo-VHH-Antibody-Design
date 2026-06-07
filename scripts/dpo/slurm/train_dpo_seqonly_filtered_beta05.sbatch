#!/bin/bash
# Seq-only DPO on the ref-margin-filtered pair pool.
#
# Hypothesis composing two findings (handoff + diag results 2026-05-26):
#   1. seq-only avoids the TF32/no_grad iter-1 gap that destabilises the
#      full-channel residue DPO loss. Empirically best val of all 5 prior
#      runs (12.2697 < 12.3240 abdpo_match < 12.3572 old residue).
#   2. ~35-40% of original pairs have ref_margin < 0 — π_ref already
#      prefers loser. Filtering them out removes the contradictory
#      signal that pulls π_θ away from π_ref over training.
#
# Together: clean signal (seq channel) on clean pairs (π_ref-consistent).
#
# Patience extended to 30 because seqonly's prior best was at iter 900;
# with a cleaner pool we want headroom to learn further before stopping.

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --job-name=floor_dpo_beta05
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
    configs/dpo/vhh_dpo_seqonly_filtered_beta05.yml \
    --output-dir runs/dpo \
    --run-name   floor_dpo_beta05 \
    --num-workers 4 \
    --wandb-project vhh-dpo \
    --wandb-run-name floor_dpo_beta05 \
    --wandb-group   vhh_dpo_beta_sweep \
    --wandb-mode    online

echo "Finished: $(date -Iseconds)"
