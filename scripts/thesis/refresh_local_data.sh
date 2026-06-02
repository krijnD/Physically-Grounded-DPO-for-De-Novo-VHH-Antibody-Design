#!/usr/bin/env bash
# Idempotent local-data refresh from Snellius. Run from repo root.
set -euo pipefail

SNEL=snel_gen
REMOTE=Physically-Grounded-DPO-for-De-Novo-VHH-Antibody-Design

# Parquets + PNGs from both AAPR run dirs (no PDBs, no LMDBs)
for d in \
    data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo \
    data/aapr/ftseed42_jfix_expanded_trainval_K8_20260601/dpo ; do
    mkdir -p "$d"
    rsync -avz --progress \
        --include='*.parquet' --include='*.png' --exclude='*' \
        "$SNEL:$REMOTE/$d/" "$d/"
done

# Scored parquets at the AAPR run root
for d in \
    data/aapr/ftseed42_jfix_trainval_K8_20260525 \
    data/aapr/ftseed42_jfix_expanded_trainval_K8_20260601 ; do
    rsync -avz "$SNEL:$REMOTE/$d/scored.parquet" "$d/scored.parquet"
done

# Design eval JSONs + CSVs
for d in \
    runs/dpo/dpo_seqonly_filtered \
    runs/dpo/dpo_seqonly_filtered_expanded \
    runs/vhh_ft/seed42_jfix_expanded \
    runs/vhh_ft/seed42_jfix ; do
    mkdir -p "$d"
    rsync -avz --include='eval*' --include='*.json' --include='*.csv' --exclude='*' \
        "$SNEL:$REMOTE/$d/" "$d/" 2>&1 | tail -3
done

# W&B exports
mkdir -p data/wandb_exports
rsync -avz "$SNEL:$REMOTE/data/wandb_exports/" data/wandb_exports/

echo "Refresh complete."
