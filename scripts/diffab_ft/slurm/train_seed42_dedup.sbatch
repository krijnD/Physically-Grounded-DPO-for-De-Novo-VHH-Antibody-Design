#!/bin/bash
# Snellius SLURM script: VHH fine-tune on the DEDUPLICATED VHH split.
#
# Background: the original seed42 run (scripts/diffab_ft/slurm/train.sbatch)
# was trained on cluster_splits.json from commit ad8c65f, which derived
# from the full 465-entry manifest with no sequence-level dedup.
# Forensic audit (see docs/threshold_calibration_context.md §8.4 and the
# repo-local concat_cdrs.fasta) showed:
#   * 465 entries → 205 unique CDR-concat sequences (56% redundancy)
#   * one CDR appears 147 times, ALL in the train split
#   * effective training redundancy: 2.41×
# The model in runs/vhh_ft/seed42/ massively overweighted the few
# heavily-replicated nanobodies. This sbatch trains the corrected
# version, on splits regenerated with `--dedupe-by raw_sequence`
# (default) and bumped val ratio (0.75/0.15/0.10 instead of 0.80/0.10/0.10
# so the early-stop signal isn't ~13 unique molecules of noise).
#
# Submit from the project root:
#     sbatch scripts/diffab_ft/slurm/train_seed42_dedup.sbatch
#
# Outputs:
#   logs/slurm/diffab_ft_dedup_<jobid>.out  — combined stdout/stderr.
#   runs/vhh_ft/seed42_dedup/               — per-run artifacts (configs,
#                                              checkpoints, log.txt).
#
# Wall time budget: ~12-18h on a single A100 for 20K iters with our
# settings; 24h leaves headroom for warm-up I/O.
#
# NB: the LMDB at data/processed/vhh_ft/ from the original seed42 run
# is STALE after the PDB/CSV cleanup, because clean_manifest.py
# patched the Hchain column for ~21 entries (their LMDB keys
# `{pdb}_{Hchain}` no longer match). The DataLoader will silently
# skip patched entries. Delete data/processed/vhh_ft/ before launching
# this job so the LMDB rebuilds from the clean manifest. This adds
# ~30 min to the first iter but is one-time.
#
# The broken seed42 run (runs/vhh_ft/seed42/) is intentionally left in
# place for side-by-side comparison; do not delete until the dedup run
# has converged and the AAR/RMSD numbers on test_antigen_disjoint are
# captured for the thesis ablation table.

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --job-name=diffab_ft_dedup
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

# ── Modules + venv ──────────────────────────────────────────────────
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate

# Make sure logs/slurm exists (sbatch fails before the job script runs
# otherwise, but harmless on re-runs).
mkdir -p logs/slurm

# ── Provenance ──────────────────────────────────────────────────────
echo "JobID:    ${SLURM_JOB_ID:-n/a}"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader || true)"
echo "Commit:   $(git rev-parse HEAD 2>/dev/null || echo n/a)"
echo "Branch:   $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
echo "Started:  $(date -Iseconds)"

# Pre-flight: (1) clean manifest must exist (PDB/CSV consistency
# fix), (2) splits JSON must be the deduped version regenerated from
# the clean manifest. Either failure means the user skipped a step
# upstream — abort before burning 12-18h of A100 time.
python -c "
import json, sys
from pathlib import Path

mani_clean = Path('data/datasets/diffab_manifest_clean.tsv')
if not mani_clean.exists():
    print(f'ERROR: clean manifest missing: {mani_clean}', file=sys.stderr)
    print('Run scripts/diffab_ft/clean_manifest.py first.', file=sys.stderr)
    sys.exit(1)
import pandas as pd
n_mani = len(pd.read_csv(mani_clean, sep='\t'))
print(f'clean manifest: {n_mani} rows')

d = json.load(open('data/datasets/clustering/cluster_splits.json'))
n_members = d['params'].get('n_members', 0)
dedupe_by = d['params'].get('dedupe_by', None)
print(f'splits: n_members={n_members}, dedupe_by={dedupe_by!r}')
if dedupe_by is None or n_members > 300:
    print('ERROR: splits look like the pre-dedup (broken seed42) version.', file=sys.stderr)
    print('Re-run scripts/diffab_ft/cluster_split.py --dedupe-by raw_sequence', file=sys.stderr)
    print('  --manifest-tsv data/datasets/diffab_manifest_clean.tsv', file=sys.stderr)
    sys.exit(1)
"

# ── Run ─────────────────────────────────────────────────────────────
python scripts/diffab_ft/train.py \
    configs/diffab_ft/vhh_ft.yml \
    --output-dir runs/vhh_ft \
    --run-name   seed42_dedup \
    --num-workers 8 \
    --wandb-project vhh-diffab-ft \
    --wandb-run-name seed42_dedup \
    --wandb-group   vhh_ft \
    --wandb-mode    offline   # safer on compute nodes; sync after with `wandb sync`

echo "Finished: $(date -Iseconds)"
