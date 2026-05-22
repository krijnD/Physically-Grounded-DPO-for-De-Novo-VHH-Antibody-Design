#!/bin/bash
# Snellius SLURM array job: run all three judges on the curated ANDD set.
#
# Workflow (do this from the project root, in order):
#
#   # 1. Split the curated CSV into N chunks (matches --array=0-N-1 below).
#   python scripts/judges/slurm/split_csv.py \
#       --input /projects/0/hpmlprjs/interns/krijn/ANDD_nano_dataset_IgLM/ANDD_VHH_curated_diffab.csv \
#       --n-chunks 32 \
#       --output-dir data/results/judges_chunks
#
#   # 2. Submit the array (one task per chunk; SLURM_ARRAY_TASK_ID
#   #    selects the chunk).
#   sbatch scripts/judges/slurm/judges_andd.sbatch
#
#   # 3. Once all array tasks finish, merge the per-chunk parquets:
#   python scripts/judges/slurm/merge_chunks.py \
#       --chunk-dir data/results/judges_andd \
#       --output    data/results/andd_judge_test_full.parquet
#
# Outputs:
#   logs/slurm/judges_andd_<jobid>_<task>.out — per-task stdout/stderr.
#   data/results/judges_andd/chunk_<NN>.parquet — per-chunk results.

#SBATCH --job-name=judges_andd
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --partition=rome
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-31

set -euo pipefail

# ── Modules + venv ──────────────────────────────────────────────────
# Adjust to whichever venv carries PyRosetta, ANARCI, NanoBodyBuilder2.
# The DPO venv may not include PyRosetta — confirm with `which pyrosetta`
# inside the venv before launching the array.
module purge
module load 2025 2024
module load Python/3.12.3-GCCcore-13.3.0
module load gompi/2024a
module load HMMER/3.4-gompi-2024a
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate


# ── Paths (overridable via sbatch --export=KEY=VAL) ─────────────────
# Defaults are GT-scoring. For AAPR scoring, override:
#   sbatch --array=0-N-1 \
#          --export=ALL,CHUNK_DIR=/path/to/aapr/chunks,OUT_DIR=/path/to/aapr/scored \
#          scripts/judges/slurm/judges_andd.sbatch
# In AAPR mode test_sabdab_judges.py auto-detects via the gt_complex_id
# column and uses each row's complex_pdb_path, so PDB_DIR is ignored.
CHUNK_DIR="${CHUNK_DIR:-data/results/judges_chunks}"
OUT_DIR="${OUT_DIR:-data/results/judges_andd}"
PDB_DIR="${PDB_DIR:-/projects/0/hpmlprjs/interns/krijn/ANDD_nano_dataset_IgLM/VHH_structures_post_diffab}"

mkdir -p "$OUT_DIR" logs/slurm

TASK_ID=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
CSV_FILE="$CHUNK_DIR/chunk_${TASK_ID}.csv"
OUT_PARQUET="$OUT_DIR/chunk_${TASK_ID}.parquet"

if [[ ! -f "$CSV_FILE" ]]; then
    echo "ERROR: chunk CSV not found: $CSV_FILE" >&2
    echo "Did you run split_csv.py with the right --n-chunks?" >&2
    exit 1
fi

# ── Provenance ──────────────────────────────────────────────────────
echo "JobID:       ${SLURM_JOB_ID:-n/a}"
echo "ArrayJobID:  ${SLURM_ARRAY_JOB_ID:-n/a}"
echo "ArrayTask:   ${SLURM_ARRAY_TASK_ID:-n/a}"
echo "Node:        $(hostname)"
echo "Commit:      $(git rev-parse HEAD 2>/dev/null || echo n/a)"
echo "Chunk CSV:   $CSV_FILE"
echo "Output:      $OUT_PARQUET"
echo "Started:     $(date -Iseconds)"

# ── Run ─────────────────────────────────────────────────────────────
# Refinement mode: `none` — supersedes the original `full` choice.
# The 2026-05-22 pilot (scripts/judges/pilot_refinement_compare.py) on
# 10 GTs at resolution ≤ 3 Å showed pack_cdrs catastrophically degrades
# well-resolved crystals in ~30% of cases (worst: 7f5g 1.75 Å, CDR-
# energy jumped −0.5 → +37 REU/res). FastRelax (`full`) additionally
# moves backbones, making thresholds inapplicable to as-generated AAPR
# outputs. With `none` we trust the deposited GT data and apply the
# identical no-refinement pipeline to AAPR scoring, so the thresholds
# and the scores live in the same metric space.
# Wall-time: ~1-2 s/row (no Rosetta refinement), so 15 rows/chunk
# finishes in <1 min of physics time. The TNP+Biology overhead now
# dominates; 12h --time stays well over budget.
python scripts/test_sabdab_judges.py \
    --csv "$CSV_FILE" \
    --pdb-dir "$PDB_DIR" \
    --output "$OUT_PARQUET" \
    --refinement-mode none \
    --run-tnp --ncores 4

echo "Finished:    $(date -Iseconds)"
