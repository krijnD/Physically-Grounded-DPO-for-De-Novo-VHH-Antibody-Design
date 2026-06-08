#!/bin/bash
# Snellius SLURM script — Brief 17 §6: noise-matched decoy winners.
#
# Forward-diffuses each GT in the floor pair pool to t=10 then reverse-
# denoises with π_ref (best_ema.pt from seed42_jfix) back to t=0. The
# resulting PDBs replace the X-ray crystal winners in the DPO pair pool,
# killing the iter-0 "real vs synthetic" shortcut that forces seq-only
# DPO. ~10 s/GT × 188 GTs ≈ 30 min wall-clock on one A100.
#
# Outputs:
#   data/aapr/ftseed42_jfix_trainval_K8_20260525/decoys_t10/
#     ├── pdbs/        — packed PDBs, one per unique GT (decoys)
#     ├── pdbs_raw/    — backbone-only DiffAb output (pre-packing)
#     ├── decoys_manifest.csv
#     └── run_metadata.json
#
# Submit from the project root:
#     sbatch scripts/dpo/slurm/sample_decoy_winners.sbatch

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --job-name=decoy_t10
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

# ── Modules + venv ──────────────────────────────────────────────────
module purge
module load 2025 2024 gompi/2024a HMMER/3.4-gompi-2024a
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate

mkdir -p logs/slurm

# ── Provenance ──────────────────────────────────────────────────────
echo "JobID:    ${SLURM_JOB_ID:-n/a}"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader || true)"
echo "Commit:   $(git rev-parse HEAD 2>/dev/null || echo n/a)"
echo "Branch:   $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
echo "Started:  $(date -Iseconds)"

# ── Pre-flight ─────────────────────────────────────────────────────
PI_REF=runs/vhh_ft/seed42_jfix/checkpoints/best_ema.pt
PAIR_PARQUET=data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs.parquet
CONFIG=configs/diffab_ft/vhh_ft.yml
OUT_DIR=data/aapr/ftseed42_jfix_trainval_K8_20260525/decoys_t10

for path in "$PI_REF" "$PAIR_PARQUET" "$CONFIG"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: required input not found: $path" >&2
        exit 1
    fi
done

python3 -c "
import pandas as pd
df = pd.read_parquet('$PAIR_PARQUET')
print(f'pair parquet: {len(df)} pairs across {df.gt_complex_id.nunique()} unique GTs')
"

# ── Run ─────────────────────────────────────────────────────────────
python scripts/dpo/sample_decoy_winners.py \
    --pi-ref-checkpoint "$PI_REF" \
    --config            "$CONFIG" \
    --pairs-parquet     "$PAIR_PARQUET" \
    --output-dir        "$OUT_DIR" \
    --t-decoy           10 \
    --base-seed         42 \
    --device            cuda

echo "Decoy pool: $(ls "$OUT_DIR/pdbs" 2>/dev/null | wc -l) PDBs in $OUT_DIR/pdbs"
echo "Finished: $(date -Iseconds)"
