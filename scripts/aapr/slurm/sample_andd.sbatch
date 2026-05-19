#!/bin/bash
# Snellius SLURM script: AAPR loser-pool generation (multi-CDR DiffAb sampling).
#
# Mirrors scripts/diffab_ft/slurm/train.sbatch (same partition, modules, venv).
# Forks scripts/diffab_ft/evaluate.py's sampling path — see
# scripts/aapr/sample_candidates.py and docs/aapr_generation_context.md.
#
# Submit from project root:
#     sbatch scripts/aapr/slurm/sample_andd.sbatch                  # canary
#     sbatch --export=RUN_NAME=full_K8,MAX_ENTRIES= ...             # full
#
# Wall time: scales linearly in (num GT entries × num_samples) ÷ per-replicate-time.
# Per-replicate wall time on a single A100 is measured in the canary and
# logged to stdout. Set --time= conservatively; resubmit with a tighter
# budget once the canary numbers are known.
#
# Canary defaults: 10 entries × K=8 = 80 candidates. Total wall time
# should be < 30 min on a single A100. Full run (465 × K=8 = 3720
# candidates) at ~10s/replicate → ~10 GPU-hours. Use --time=12:00:00
# for the first full run, --time=6:00:00 once timing is known.
#
# Output layout (under --output-dir, defaults to data/aapr/):
#   <run_name>/
#     pdbs/<entry_id>/sample_<i:04d>.pdb   (K per entry)
#     candidates.csv                       (one row per (GT × sample))
#     run_metadata.json                    (checkpoint, seed, K, ...)

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:00:00         # canary budget; bump to 12h for full run
#SBATCH --job-name=aapr_sample
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

# ── Modules + venv ──────────────────────────────────────────────────
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate

mkdir -p logs/slurm

# ── Run parameters (override via sbatch --export=KEY=VAL,...) ────────
# Default: canary on the test split, 10 entries × K=8 = 80 candidates.
CHECKPOINT="${CHECKPOINT:-/projects/0/hpmlprjs/interns/krijn/checkpoints/diffab/codesign_multicdrs.pt}"
CONFIG="${CONFIG:-configs/diffab_ft/vhh_ft.yml}"
SPLIT="${SPLIT:-test}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
MAX_ENTRIES="${MAX_ENTRIES-10}"           # set to '' (empty) for full split; use `-` not `:-` so an explicit empty value is preserved
OUTPUT_DIR="${OUTPUT_DIR:-data/aapr}"
RUN_NAME="${RUN_NAME:-canary_K${NUM_SAMPLES}_$(date +%Y%m%d)}"
SEED="${SEED:-42}"
MANIFEST_TSV="${MANIFEST_TSV:-data/datasets/diffab_manifest.tsv}"

# ── Provenance ──────────────────────────────────────────────────────
echo "JobID:        ${SLURM_JOB_ID:-n/a}"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader || true)"
echo "Commit:       $(git rev-parse HEAD 2>/dev/null || echo n/a)"
echo "Branch:       $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
echo "Started:      $(date -Iseconds)"
echo "---"
echo "Checkpoint:   $CHECKPOINT"
echo "Config:       $CONFIG"
echo "Split:        $SPLIT"
echo "Num samples:  $NUM_SAMPLES"
echo "Max entries:  ${MAX_ENTRIES:-(full split)}"
echo "Run name:     $RUN_NAME"
echo "Seed:         $SEED"

# ── Run ─────────────────────────────────────────────────────────────
EXTRA_ARGS=()
if [ -n "$MAX_ENTRIES" ]; then
    EXTRA_ARGS+=(--max-entries "$MAX_ENTRIES")
fi

python scripts/aapr/sample_candidates.py \
    --checkpoint "$CHECKPOINT" \
    --config     "$CONFIG" \
    --split      "$SPLIT" \
    --num-samples "$NUM_SAMPLES" \
    --output-dir "$OUTPUT_DIR" \
    --run-name   "$RUN_NAME" \
    --manifest-tsv "$MANIFEST_TSV" \
    --device     cuda \
    --seed       "$SEED" \
    "${EXTRA_ARGS[@]}"

echo "Finished:     $(date -Iseconds)"
