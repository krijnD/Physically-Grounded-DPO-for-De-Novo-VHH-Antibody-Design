#!/bin/bash
# Snellius SLURM array: run biology + biophysics + physics judges on a
# chunked design-eval candidate manifest (Brief 11, Phase B).
#
# Reuses ``scripts/test_sabdab_judges.py`` in AAPR mode (auto-detected
# via the ``gt_complex_id`` column in the input CSV — populated by
# ``scripts/eval/build_design_manifest.py``). Locked thresholds from
# ``src/common/config.py``; ``--refinement-mode none`` (campaign-wide
# invariant per the AbDPO §E.1 calibration position).
#
# Workflow:
#
#   # 1. Build the merged manifest CSV with all 6 (variant × test_set)
#   #    tuples concatenated (candidate_id encodes the breakdown).
#   #    See the launcher block in the brief reply for the full pattern.
#   for spec in seed42_jfix:oldtest:runs/vhh_ft/seed42_jfix/eval_test_pdbs \
#               floor_pi_theta:oldtest:runs/dpo/dpo_seqonly_filtered/eval_test_pdbs \
#               ... ; do
#       IFS=':' read -r v t r <<< "$spec"
#       python scripts/eval/build_design_manifest.py \
#           --pdb-root "$r" --manifest-tsv data/datasets/diffab_manifest.tsv \
#           --output "data/eval/manifests/${v}_${t}.csv" \
#           --variant-tag "$v" --test-set-tag "$t"
#   done
#   # ... and concat to data/eval/manifests/all_variants.csv
#
#   # 2. Split into 32 chunks.
#   python scripts/judges/slurm/split_csv.py \
#       --input data/eval/manifests/all_variants.csv \
#       --n-chunks 32 \
#       --output-dir data/eval/chunks/all_variants
#
#   # 3. Submit this sbatch with CHUNK_DIR + OUT_DIR overrides.
#   sbatch --export=ALL,\
#          CHUNK_DIR=data/eval/chunks/all_variants,\
#          OUT_DIR=data/eval/judged_chunks/all_variants \
#          scripts/eval/slurm/judges_design.sbatch
#
#   # 4. Merge per-chunk parquets.
#   python scripts/judges/slurm/merge_chunks.py \
#       --chunk-dir data/eval/judged_chunks/all_variants \
#       --output    data/eval/design_samples_judged_all.parquet

#SBATCH --job-name=judges_design
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --partition=rome
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-31

# Defer -e past env setup (Brief 06 deviation log — bashrc returns
# non-zero in non-interactive shells under set -e).
set -uo pipefail
module purge
module load 2025 2024 gompi/2024a HMMER/3.4-gompi-2024a
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate
set -e

CHUNK_DIR="${CHUNK_DIR:?CHUNK_DIR must be set via --export=ALL,CHUNK_DIR=...}"
OUT_DIR="${OUT_DIR:?OUT_DIR must be set via --export=ALL,OUT_DIR=...}"

mkdir -p "$OUT_DIR" logs/slurm

TASK_ID=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
CSV_FILE="$CHUNK_DIR/chunk_${TASK_ID}.csv"
OUT_PARQUET="$OUT_DIR/chunk_${TASK_ID}.parquet"

if [[ ! -f "$CSV_FILE" ]]; then
    echo "ERROR: chunk CSV not found: $CSV_FILE" >&2
    exit 1
fi

echo "JobID:       ${SLURM_JOB_ID:-n/a}"
echo "ArrayJob:    ${SLURM_ARRAY_JOB_ID:-n/a}"
echo "ArrayTask:   ${SLURM_ARRAY_TASK_ID:-n/a}"
echo "Node:        $(hostname)"
echo "Commit:      $(git rev-parse HEAD 2>/dev/null || echo n/a)"
echo "Chunk CSV:   $CSV_FILE"
echo "Output:      $OUT_PARQUET"
echo "Started:     $(date -Iseconds)"

# AAPR mode auto-detected via the gt_complex_id column. --run-tnp wires
# the biophysics scorer (theraprofnano on the VHH chain extracted from
# each PDB; no NB2 re-fold). --refinement-mode none is the campaign-wide
# invariant from src/common/config.py.
python scripts/test_sabdab_judges.py \
    --csv "$CSV_FILE" \
    --output "$OUT_PARQUET" \
    --refinement-mode none \
    --run-tnp

echo "Finished:    $(date -Iseconds)"
