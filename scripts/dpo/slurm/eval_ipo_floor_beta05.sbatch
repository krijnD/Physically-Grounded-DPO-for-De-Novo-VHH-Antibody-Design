#!/bin/bash
# Brief 18 §8 — design eval for ipo_seqonly_floor_beta05 (OLD test, 29 entries).

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --job-name=eval_ipo_floor_beta05
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate
mkdir -p logs/slurm

CKPT=runs/dpo/ipo_seqonly_floor_beta05/checkpoints/best_ema.pt
CFG=configs/diffab_ft/vhh_ft.yml

echo "JobID: ${SLURM_JOB_ID:-n/a}  Node: $(hostname)"
echo "Started: $(date -Iseconds)"
[ -f "$CKPT" ] || { echo "ERROR: missing $CKPT"; exit 1; }

python scripts/diffab_ft/evaluate.py \
    --checkpoint   "$CKPT" \
    --config       "$CFG" \
    --split        test \
    --mode         design \
    --num-samples  4 \
    --output       runs/dpo/ipo_seqonly_floor_beta05/eval_test_design.json

echo "--- Summary ---"
cat runs/dpo/ipo_seqonly_floor_beta05/eval_test_design.json
echo "Finished: $(date -Iseconds)"
