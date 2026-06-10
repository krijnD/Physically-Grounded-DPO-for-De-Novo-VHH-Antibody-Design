#!/bin/bash
# Brief 22 E1-B: matched seq-only GT-vs-decoy control on the floor's 928 IDs.
#
# Same 928 pair IDs as the floor pi_theta (E1-A, run-name
# dpo_seqonly_filtered, beta=0.05, val 12.02). Only the winner type is
# swapped: GT crystal -> t=1 decoy. Closes the channel-scope confound
# Brief 17 left open (its decoy intervention ran on all-channel +
# partial-window-filtered, 658 pairs). E1-B runs seq-only + the FULL
# 928, matching E1-A's setup exactly except for the winner type.
#
# Fixed budget for the diagnostic: max_iters=1000, early_stop_patience=0,
# val_freq=50, seed=42. TF32-on / no-AMP (existing trainer regime).
#
# Pair pool: pairs_decoy_t1_floor928.parquet (EX-3 Brief 22 output;
# 928 rows x 13 cols, winner_provenance="decoy_t1").
# Loss: seq-only loss_weights {rot:0, pos:0, seq:1}. beta_dpo: 0.05.

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --job-name=brief22_e1
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --comment="Brief 22 E1-B: matched seq-only GT-vs-decoy control on the floor's 928 IDs"

set -euo pipefail
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate
mkdir -p logs/slurm

echo "JobID: ${SLURM_JOB_ID:-n/a}  Node: $(hostname)"
echo "Commit: $(git rev-parse HEAD 2>/dev/null || echo n/a)"
echo "Started: $(date -Iseconds)"

PAIR_PARQUET=data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_decoy_t1_floor928.parquet
if [[ ! -f "$PAIR_PARQUET" ]]; then
    echo "ERROR: $PAIR_PARQUET not found — run EX-3 first." >&2
    exit 1
fi

python scripts/dpo/train_dpo.py \
    configs/dpo/vhh_dpo_seqonly_decoy_t1_floor928_brief22.yml \
    --output-dir runs/dpo \
    --run-name   brief22_e1_decoyt1_seqonly_beta05 \
    --num-workers 4 \
    --wandb-project vhh-dpo \
    --wandb-run-name brief22_e1_decoyt1_seqonly_beta05 \
    --wandb-group   brief22_membership_diagnostic \
    --wandb-mode    online

echo "Finished: $(date -Iseconds)"
