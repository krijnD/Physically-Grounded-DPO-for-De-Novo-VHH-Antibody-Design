#!/bin/bash
# Verify π_ref (seed42_jfix) standalone quality before launching the
# filtered+seqonly DPO canary.
#
# Why now (2026-05-26 diagnostic round):
#   The handoff doc, Check 4 from the broad diagnostic sweep, and our
#   own audit all agree: best fine-tune val 0.7316 is healthy, but no
#   AAR/RMSD on the test split was ever measured. Before spending
#   another DPO training cycle, confirm π_ref actually samples
#   competent CDRs. If H3 AAR is < ~15% or RMSD > ~5Å, DPO has nothing
#   to refine — fix π_ref first.
#
# Two passes:
#   1. ELBO eval on test split (cheap, ~1 min). Sanity-check the
#      checkpoint loads and val loss is in the same ballpark.
#   2. Design eval on test split — per-CDR (H1/H2/H3) AAR + CA-RMSD,
#      4 samples per entry. ~30-60 min walltime.

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_piref_jfix
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate
mkdir -p logs/slurm runs/vhh_ft/seed42_jfix

echo "JobID: ${SLURM_JOB_ID:-n/a}  Node: $(hostname)"
echo "Commit: $(git rev-parse HEAD 2>/dev/null || echo n/a)"
echo "Started: $(date -Iseconds)"

CKPT=runs/vhh_ft/seed42_jfix/checkpoints/best_ema.pt
CFG=configs/diffab_ft/vhh_ft.yml

echo ""
echo "================ PASS 1: ELBO eval (test split) ================"
python scripts/diffab_ft/evaluate.py \
    --checkpoint "$CKPT" \
    --config     "$CFG" \
    --split      test \
    --mode       elbo \
    --output     runs/vhh_ft/seed42_jfix/eval_test_elbo.json

echo ""
echo "================ PASS 2: design eval (test split) ================"
python scripts/diffab_ft/evaluate.py \
    --checkpoint   "$CKPT" \
    --config       "$CFG" \
    --split        test \
    --mode         design \
    --num-samples  4 \
    --save-pdbs    runs/vhh_ft/seed42_jfix/eval_test_pdbs \
    --output       runs/vhh_ft/seed42_jfix/eval_test_design.json

echo ""
echo "================ Summary ================"
echo "--- ELBO ---"
cat runs/vhh_ft/seed42_jfix/eval_test_elbo.json
echo ""
echo "--- DESIGN (AAR + RMSD per CDR) ---"
cat runs/vhh_ft/seed42_jfix/eval_test_design.json

echo ""
echo "Finished: $(date -Iseconds)"
