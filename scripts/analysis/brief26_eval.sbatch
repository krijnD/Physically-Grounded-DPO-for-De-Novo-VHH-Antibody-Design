#!/bin/bash
# Brief 26 -- leakage-corrected ELBO re-eval on the 29-entry shared holdout.
#
# Re-evaluates seed42_jfix (anchor) and seed42_jfix_expanded (expanded) on the
# SAME 29 entries used for the original SQ1 comparison (Brief 06). The 29 list
# is the "old test" split surfaced in
# master-thesis/data/eval/per_sample_csvs/floor_pi_theta_oldtest.csv.
#
# Submit from the project root:
#     sbatch scripts/analysis/brief26_eval.sbatch
#
# Wall budget: ~15 min for 29 entries x 2 models x 8 forward-passes on an A100.

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:45:00
#SBATCH --job-name=brief26_elbo
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate

mkdir -p logs/slurm tmp_brief26

echo "JobID:    ${SLURM_JOB_ID:-n/a}"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader || true)"
echo "Commit:   $(git rev-parse HEAD 2>/dev/null || echo n/a)"
echo "Started:  $(date -Iseconds)"

CONFIG=configs/diffab_ft/vhh_ft_expanded.yml
ANCHOR_CKPT=runs/vhh_ft/seed42_jfix/checkpoints/best_ema.pt
EXPANDED_CKPT=runs/vhh_ft/seed42_jfix_expanded/checkpoints/best_ema.pt

# Pre-flight: confirm both checkpoints exist.
for entry in "anchor:$ANCHOR_CKPT" "expanded:$EXPANDED_CKPT"; do
    name="${entry%%:*}"
    path="${entry#*:}"
    if [ ! -f "$path" ]; then
        echo "ERROR: $name checkpoint not found: $path" >&2
        exit 1
    fi
done
echo "Both checkpoints found."

python scripts/analysis/brief26_leakage_corrected_elbo.py \
    --anchor-checkpoint "$ANCHOR_CKPT" \
    --expanded-checkpoint "$EXPANDED_CKPT" \
    --config "$CONFIG" \
    --device cuda \
    --n-repeats 8 \
    --seed 42 \
    --out-dir tmp_brief26

echo "Finished: $(date -Iseconds)"
echo "--- tmp_brief26/leakage_corrected_summary.txt ---"
cat tmp_brief26/leakage_corrected_summary.txt
