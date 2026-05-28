#!/bin/bash
# E2: Compare seed42_dedup to seed42_jfix on the same test split.
#
# Per memory: seed42_dedup (2026-05-19) achieved "H3 AAR +11.4pp vs
# pretrained, healthy". seed42_jfix (current π_ref) is at H3 AAR ~25%
# — essentially pretrained-quality. Hypothesis: the J-anchor fix
# regressed sampling quality somehow.
#
# Caveat: seed42_dedup was trained on the pre-J-anchor LMDB
# (heavy_max_resseq=113). Evaluating it on the new LMDB (max_resseq=150)
# means it sees a few residues it wasn't trained on. If errors appear,
# we'll re-eval on the original dedup LMDB and note the caveat.

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_piref_dedup
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate
mkdir -p logs/slurm runs/vhh_ft/seed42_dedup

CKPT=runs/vhh_ft/seed42_dedup/checkpoints/best_ema.pt
CFG=configs/diffab_ft/vhh_ft.yml

echo "JobID: ${SLURM_JOB_ID:-n/a}  Node: $(hostname)"
echo "Started: $(date -Iseconds)"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: seed42_dedup checkpoint not found at $CKPT"
    echo "Available checkpoints under runs/vhh_ft/:"
    find runs/vhh_ft -maxdepth 3 -name "best_ema.pt" 2>/dev/null
    exit 1
fi

echo ""
echo "================ ELBO eval ================"
python scripts/diffab_ft/evaluate.py \
    --checkpoint "$CKPT" \
    --config     "$CFG" \
    --split      test \
    --mode       elbo \
    --output     runs/vhh_ft/seed42_dedup/eval_test_elbo.json

echo ""
echo "================ design eval ================"
python scripts/diffab_ft/evaluate.py \
    --checkpoint   "$CKPT" \
    --config       "$CFG" \
    --split        test \
    --mode         design \
    --num-samples  4 \
    --output       runs/vhh_ft/seed42_dedup/eval_test_design.json

echo ""
echo "================ Side-by-side ================"
echo "--- seed42_jfix design (already evaluated) ---"
cat runs/vhh_ft/seed42_jfix/eval_test_design.json 2>/dev/null
echo ""
echo "--- seed42_dedup design (just now) ---"
cat runs/vhh_ft/seed42_dedup/eval_test_design.json 2>/dev/null

echo "Finished: $(date -Iseconds)"
