#!/bin/bash
# Brief 18 §8 — design eval for ipo_seqonly_expanded_beta05.
# Two passes (mirrors Brief 07b convention):
#   Pass 1: OLD test (vhh_ft.yml splits, 29 entries) → eval_oldtest_design.json
#   Pass 2: NEW test (vhh_ft_expanded.yml splits, 83 entries) → eval_newtest_design.json
# Wallclock: pass-1 ~30 min (29 × 4 × 3), pass-2 ~90 min (83 × 4 × 3).

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --job-name=eval_ipo_expanded_beta05
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate
mkdir -p logs/slurm

CKPT=runs/dpo/ipo_seqonly_expanded_beta05/checkpoints/best_ema.pt
CFG_OLD=configs/diffab_ft/vhh_ft.yml
CFG_NEW=configs/diffab_ft/vhh_ft_expanded.yml

echo "JobID: ${SLURM_JOB_ID:-n/a}  Node: $(hostname)"
echo "Started: $(date -Iseconds)"
[ -f "$CKPT" ] || { echo "ERROR: missing $CKPT"; exit 1; }
[ -f "$CFG_OLD" ] || { echo "ERROR: missing $CFG_OLD"; exit 1; }
[ -f "$CFG_NEW" ] || { echo "ERROR: missing $CFG_NEW"; exit 1; }

echo ""
echo "================ Pass 1: OLD test (vhh_ft.yml, 29 entries) ================"
python scripts/diffab_ft/evaluate.py \
    --checkpoint   "$CKPT" \
    --config       "$CFG_OLD" \
    --split        test \
    --mode         design \
    --num-samples  4 \
    --output       runs/dpo/ipo_seqonly_expanded_beta05/eval_oldtest_design.json

echo ""
echo "================ Pass 2: NEW test (vhh_ft_expanded.yml, 83 entries) ================"
python scripts/diffab_ft/evaluate.py \
    --checkpoint   "$CKPT" \
    --config       "$CFG_NEW" \
    --split        test \
    --mode         design \
    --num-samples  4 \
    --output       runs/dpo/ipo_seqonly_expanded_beta05/eval_newtest_design.json

echo ""
echo "================ Summary ================"
echo "--- OLD test ---"
cat runs/dpo/ipo_seqonly_expanded_beta05/eval_oldtest_design.json
echo ""
echo "--- NEW test ---"
cat runs/dpo/ipo_seqonly_expanded_beta05/eval_newtest_design.json
echo ""
echo "Finished: $(date -Iseconds)"
