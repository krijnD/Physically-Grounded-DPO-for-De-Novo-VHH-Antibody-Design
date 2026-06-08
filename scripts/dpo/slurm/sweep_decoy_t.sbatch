#!/bin/bash
# Brief 17 §9.1 — parameterized decoy-t sweep job.
#
# Runs the full three-step pipeline for a single t value:
#   1. sample_decoy_winners.py   → data/aapr/.../decoys_t${T}/pdbs/*.pdb
#   2. build_decoy_pairs.py      → data/aapr/.../dpo/pairs_decoy_t${T}.parquet
#   3. diag_per_channel_reward.py → data/aapr/.../dpo/lwref_per_channel_decoy_t${T}.parquet
#
# Set T_DECOY via env or --t-decoy CLI; defaults to 10. Example:
#   sbatch --export=T_DECOY=7 scripts/dpo/slurm/sweep_decoy_t.sbatch
#   sbatch --export=T_DECOY=8 scripts/dpo/slurm/sweep_decoy_t.sbatch
#   sbatch --export=T_DECOY=9 scripts/dpo/slurm/sweep_decoy_t.sbatch
#
# Each job: ~5 min decoy gen + ~5s pair rebuild + ~6 min diag ≈ 12 min on A100.
# Three jobs in parallel finish in ~12 min total wallclock.

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --job-name=decoy_sweep
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

# ── Parse T_DECOY (env or --t-decoy CLI) ──────────────────────────
T_DECOY="${T_DECOY:-10}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --t-decoy)
            T_DECOY="$2"; shift 2;;
        *)
            echo "Unknown arg: $1" >&2; exit 64;;
    esac
done

if ! [[ "$T_DECOY" =~ ^[0-9]+$ ]] || (( T_DECOY < 1 )) || (( T_DECOY > 100 )); then
    echo "ERROR: T_DECOY must be an integer in [1, 100]; got '$T_DECOY'" >&2
    exit 1
fi

echo "T_DECOY = $T_DECOY"

# ── Modules + venv ─────────────────────────────────────────────────
module purge
module load 2025 2024 gompi/2024a HMMER/3.4-gompi-2024a
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate

mkdir -p logs/slurm

# ── Provenance ─────────────────────────────────────────────────────
echo "JobID:    ${SLURM_JOB_ID:-n/a}"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader || true)"
echo "Commit:   $(git rev-parse HEAD 2>/dev/null || echo n/a)"
echo "Branch:   $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
echo "Started:  $(date -Iseconds)"

# ── Paths ──────────────────────────────────────────────────────────
PI_REF=runs/vhh_ft/seed42_jfix/checkpoints/best_ema.pt
CONFIG=configs/diffab_ft/vhh_ft.yml
PAIR_PARQUET=data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs.parquet
OUT_DIR=data/aapr/ftseed42_jfix_trainval_K8_20260525/decoys_t${T_DECOY}
PAIR_DECOY=data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_decoy_t${T_DECOY}.parquet
LWREF_DECOY=data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/lwref_per_channel_decoy_t${T_DECOY}.parquet

# ── Pre-flight ─────────────────────────────────────────────────────
for path in "$PI_REF" "$PAIR_PARQUET" "$CONFIG"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: required input not found: $path" >&2
        exit 1
    fi
done

# ── Step 1/3: decoy generation ────────────────────────────────────
echo
echo "=== [t=${T_DECOY}] Step 1/3: sample_decoy_winners.py ==="
python scripts/dpo/sample_decoy_winners.py \
    --pi-ref-checkpoint "$PI_REF" \
    --config            "$CONFIG" \
    --pairs-parquet     "$PAIR_PARQUET" \
    --output-dir        "$OUT_DIR" \
    --t-decoy           "$T_DECOY" \
    --base-seed         42 \
    --device            cuda

n_pdbs=$(ls "$OUT_DIR/pdbs" 2>/dev/null | wc -l)
echo "Decoy pool: $n_pdbs PDBs in $OUT_DIR/pdbs"

# ── Step 2/3: rebuild pair pool ────────────────────────────────────
echo
echo "=== [t=${T_DECOY}] Step 2/3: build_decoy_pairs.py ==="
python scripts/dpo/build_decoy_pairs.py \
    --pairs-parquet  "$PAIR_PARQUET" \
    --decoy-dir      "$OUT_DIR/pdbs" \
    --t-decoy        "$T_DECOY" \
    --output         "$PAIR_DECOY"

# ── Step 3/3: per-channel ref-margin diagnostic ───────────────────
echo
echo "=== [t=${T_DECOY}] Step 3/3: diag_per_channel_reward.py ==="
python scripts/dpo/diag_per_channel_reward.py \
    --pi-ref-checkpoint "$PI_REF" \
    --pairs-parquet     "$PAIR_DECOY" \
    --output            "$LWREF_DECOY" \
    --t-eval            50 \
    --device            cuda

echo
echo "Sweep job t=${T_DECOY} complete."
echo "Finished: $(date -Iseconds)"
