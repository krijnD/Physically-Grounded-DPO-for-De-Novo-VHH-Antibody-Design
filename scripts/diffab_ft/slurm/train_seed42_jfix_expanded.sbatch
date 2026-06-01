#!/bin/bash
# Snellius SLURM script: VHH fine-tune on the EXPANDED pool
# (ANDD + SAbDab nano, 927 manifest rows → 911 LMDB entries).
#
# Same recipe as train_seed42_jfix.sbatch — only the data changes
# (manifest, pdb_dir, processed_dir, splits_path swapped in
# configs/diffab_ft/vhh_ft_expanded.yml). Hyperparameters, init
# checkpoint, J-anchor cap (heavy_max_resseq default = 150 in
# src/diffab_ft/datasets/vhh_andd.py) are byte-identical.
#
# Brief: docs/executor_briefs/06_ft_run_expanded.md
# Plan : docs/expanded_ft_handoff.md
# State: docs/expanded_ft_progress.md
#
# Submit from the project root:
#     sbatch scripts/diffab_ft/slurm/train_seed42_jfix_expanded.sbatch
#
# Outputs:
#   logs/slurm/diffab_ft_expanded_<jobid>.out
#   runs/vhh_ft/seed42_jfix_expanded/  (configs, checkpoints, log.txt)
#
# Wall-time budget: ~10-12h expected (vs seed42_jfix's ~6h on 242
# entries); 14h leaves head-room for warm-up I/O on the larger LMDB.

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --job-name=diffab_ft_expanded
#SBATCH --output=logs/slurm/%x_%j.out

# Defer `set -e` until after env setup — module/venv ops can produce
# benign non-zero exits, and we want the slurm log to show what's
# happening before strict-mode tripwires fire.
set -uo pipefail

echo "=== sbatch boot: $(hostname) $(date -Iseconds) JobID=${SLURM_JOB_ID:-n/a} ==="

# ── Env setup (inlined from `start_project` alias) ─────────────────
# Aliases don't expand in non-interactive shells, so we replicate the
# alias body verbatim. Source: ~/.bashrc on Snellius login node
# (DPO venv = Python 3.13.5; modules unchanged from Brief 05's build).
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate
module purge
module load 2025 2024 gompi/2024a HMMER/3.4-gompi-2024a
export PATH="/projects/0/hpmlprjs/interns/krijn/tools/mmseqs/bin:$PATH"

mkdir -p logs/slurm runs/vhh_ft/seed42_jfix_expanded

# Strict mode for the rest of the script.
set -e

# ── Provenance ─────────────────────────────────────────────────────
echo "JobID:    ${SLURM_JOB_ID:-n/a}"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader || true)"
echo "Commit:   $(git rev-parse HEAD 2>/dev/null || echo n/a)"
echo "Branch:   $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
echo "Python:   $(python --version 2>&1)"
echo "Started:  $(date -Iseconds)"

# ── Pre-flight: LMDB freshness (cheap, catches the main "stale data"
#    failure mode for a 12h job) ────────────────────────────────────
python - <<'PY'
import sys, lmdb, os
LMDB = 'data/processed/vhh_ft_expanded/structures.lmdb'
# Brief 05 created this as a single .lmdb file (not a directory) →
# use subdir=False.
is_subdir = os.path.isdir(LMDB)
env = lmdb.open(LMDB, readonly=True, lock=False, subdir=is_subdir)
with env.begin() as txn:
    n = txn.stat()['entries']
print(f'LMDB {LMDB} entries={n} (subdir={is_subdir})')
if n < 800:
    print(f'ERROR: LMDB looks empty/partial (n={n}, expected ≥800).', file=sys.stderr)
    sys.exit(1)
PY

# ── Pre-flight: expanded splits JSON ─────────────────────────────────
python - <<'PY'
import json, sys
SPLITS = 'data/datasets/clustering/cluster_splits_expanded.json'
d = json.load(open(SPLITS))
splits = d.get('splits', {})
n_train = len(splits.get('train', []))
n_val   = len(splits.get('val',   []))
n_test  = len(splits.get('test',  []))
print(f'splits: train={n_train}, val={n_val}, test={n_test}')
if not (n_train >= 300 and n_test >= 50):
    print('ERROR: expanded splits look wrong (expected 463/54/86).', file=sys.stderr)
    sys.exit(1)
PY

# ── Run ────────────────────────────────────────────────────────────
# W&B online mode so the live dashboard link prints in the first
# minute of the job. If the compute node can't reach api.wandb.ai,
# W&B falls back to offline mode automatically.
python scripts/diffab_ft/train.py \
    configs/diffab_ft/vhh_ft_expanded.yml \
    --output-dir runs/vhh_ft \
    --run-name   seed42_jfix_expanded \
    --num-workers 8 \
    --wandb-project vhh-diffab-ft \
    --wandb-run-name seed42_jfix_expanded \
    --wandb-group   vhh_ft \
    --wandb-mode    online

echo "Finished: $(date -Iseconds)"
