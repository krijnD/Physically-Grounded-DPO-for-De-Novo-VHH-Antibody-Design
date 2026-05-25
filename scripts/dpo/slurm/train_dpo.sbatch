#!/bin/bash
# Snellius SLURM script: Diffusion-DPO on the seed42_jfix π_ref + the
# 192-pair canary (ftseed42_jfix_K8_20260525).
#
# Background:
#   * π_ref was trained 2026-05-22 on the J-anchor-fixed LMDB (heavy
#     chain max_resseq=150). See docs/post_jfix_dpo_handoff.md.
#   * Pair pool comes from scripts/dpo/select_pareto_pairs.py on the
#     AAPR-scored parquet. 192 valid (winner=GT, loser=AAPR sample)
#     pairs across 25 test-split GTs.
#   * This script runs DPO for 5000 iters, batch_size=4, β=0.1, with
#     EMA on π_θ and early-stop on val DPO loss.
#
# Submit from the project root:
#     sbatch scripts/dpo/slurm/train_dpo.sbatch
#
# Outputs:
#   logs/slurm/dpo_<jobid>.out          — combined stdout/stderr.
#   runs/dpo/dpo_seed42_jfix_canary/    — per-run artifacts (config,
#                                          checkpoints, log.txt).
#
# Wall time budget:
#   Per-iter cost: 4 forward passes (π_θ on w/l, π_ref on w/l) at
#   ~0.5s/forward = ~2s/step. 5000 steps × 2s = ~3h compute. Add I/O
#   and val passes → 4-6h actual. 12h walltime leaves headroom for
#   slow disk and the val pool's CPU-bound loser parsing.

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --job-name=vhh_dpo
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

# ── Modules + venv ──────────────────────────────────────────────────
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate

mkdir -p logs/slurm

# ── Provenance ──────────────────────────────────────────────────────
echo "JobID:    ${SLURM_JOB_ID:-n/a}"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader || true)"
echo "Commit:   $(git rev-parse HEAD 2>/dev/null || echo n/a)"
echo "Branch:   $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
echo "Started:  $(date -Iseconds)"

# ── Pre-flight checks ───────────────────────────────────────────────
# Hard-fail before the GPU clock starts if our inputs are missing.
# These mirror the assumptions in configs/dpo/vhh_dpo.yml.

PI_REF=runs/vhh_ft/seed42_jfix/checkpoints/best_ema.pt
PAIR_PARQUET=data/aapr/ftseed42_jfix_K8_20260525/dpo/pairs.parquet

if [[ ! -f "$PI_REF" ]]; then
    echo "ERROR: π_ref checkpoint not found at $PI_REF" >&2
    echo "       (rebuild from seed42_jfix fine-tune; see" >&2
    echo "        scripts/diffab_ft/slurm/train_seed42_jfix.sbatch)" >&2
    exit 1
fi
if [[ ! -f "$PAIR_PARQUET" ]]; then
    echo "ERROR: pair parquet not found at $PAIR_PARQUET" >&2
    echo "       (rebuild from scripts/dpo/select_pareto_pairs.py)" >&2
    exit 1
fi

# Quick sanity-print of pair pool size; non-blocking — we want to see
# in the log whether the parquet looks like 192 pairs or someone
# accidentally pointed us at a 5-pair stub.
python3 -c "
import pandas as pd
df = pd.read_parquet('$PAIR_PARQUET')
print(f'pair parquet: {len(df)} pairs across {df.gt_complex_id.nunique()} GTs')
"

# ── Run ─────────────────────────────────────────────────────────────
# W&B online; falls back to offline-then-sync if the compute node
# can't reach api.wandb.ai. Group 'vhh_dpo' so DPO runs appear in
# their own pane in the dashboard, separate from the fine-tune runs
# under 'vhh_ft'.
python scripts/dpo/train_dpo.py \
    configs/dpo/vhh_dpo.yml \
    --output-dir runs/dpo \
    --run-name   dpo_seed42_jfix_canary \
    --num-workers 4 \
    --wandb-project vhh-dpo \
    --wandb-run-name dpo_seed42_jfix_canary \
    --wandb-group   vhh_dpo \
    --wandb-mode    online

echo "Finished: $(date -Iseconds)"
