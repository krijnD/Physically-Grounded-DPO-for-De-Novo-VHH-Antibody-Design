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
    runs/vhh_ft/seed42_jfix \
    runs/dpo/floor_dpo_beta0005 \
    runs/dpo/floor_dpo_beta05 \
    runs/dpo/ipo_seqonly_floor_beta0005 \
    runs/dpo/ipo_seqonly_floor_beta05 \
    runs/dpo/ipo_seqonly_floor_beta5 \
    runs/dpo/ipo_seqonly_expanded_beta05 \
    runs/dpo/dpo_allchannel_decoy_t1_beta0005 \
    runs/dpo/dpo_allchannel_decoy_t1_beta05 \
    runs/dpo/dpo_allchannel_decoy_t1_beta5 ; do
    mkdir -p "$d"
    rsync -avz --include='eval*' --include='*.json' --include='*.csv' --exclude='*' \
        "$SNEL:$REMOTE/$d/" "$d/" 2>&1 | tail -3
done

# W&B exports
mkdir -p data/wandb_exports
rsync -avz "$SNEL:$REMOTE/data/wandb_exports/" data/wandb_exports/

# ── Brief 11 (Phase B) — design-sample developability eval ────────────
# Master parquet + intermediates (under data/eval/, not gitignored but
# regenerable so kept out of git). The master parquet is the only one
# the local plotter needs; pull the others if you want to recompute
# the master locally with different thresholds.
mkdir -p data/eval
for f in design_samples_master.parquet \
         design_samples_judged_all.parquet \
         design_samples_dG_all.parquet \
         caar_epif1.parquet \
         per_position_modal_picks_all.parquet \
         gt_pdb_map.json \
         beta_sweep_comparison.parquet \
         caar_epif1_beta0005.parquet \
         caar_epif1_beta05.parquet \
         scrmsd_beta0005.parquet \
         scrmsd_beta05.parquet \
         per_position_modal_picks_brief18.parquet \
         per_position_modal_picks_brief17.parquet ; do
    rsync -avz "$SNEL:$REMOTE/data/eval/$f" "data/eval/$f"
done

# ── Brief 17 — decoy-winners + all-channel DPO deliverable ───────────
# Pull the deliverable + the bathtub figure + the machine-readable
# sweep CSV. docs/ is gitignored, so rsync is the only path off Snellius.
mkdir -p docs/executor_briefs
rsync -avz \
    "$SNEL:$REMOTE/docs/executor_briefs/17_decoy_winners_deliverable.md" \
    docs/executor_briefs/17_decoy_winners_deliverable.md 2>/dev/null \
    || echo "  (17_decoy_winners_deliverable.md not on Snellius yet — use the local Mac copy)"

mkdir -p data/results
rsync -avz "$SNEL:$REMOTE/data/results/decoy_t_sweep.csv" data/results/decoy_t_sweep.csv 2>/dev/null \
    || echo "  (decoy_t_sweep.csv not present on Snellius — run tabulate_decoy_sweep.py with --output-csv first)"

# ── Brief 18 — IPO baseline executor deliverable (docs/ gitignored) ──
mkdir -p docs/executor_briefs
rsync -avz \
    "$SNEL:$REMOTE/docs/executor_briefs/18_ipo_deliverable.md" \
    docs/executor_briefs/18_ipo_deliverable.md 2>/dev/null \
    || echo "  (18_ipo_deliverable.md not on Snellius yet — use the local Mac copy)"

# Phase B figures + summary table (docs/ is fully gitignored, so this is
# the ONLY way these reach local).
mkdir -p docs/figures/phase_b
rsync -avz --include='*.pdf' --include='*.png' --include='*.md' --exclude='*' \
    "$SNEL:$REMOTE/docs/figures/phase_b/" docs/figures/phase_b/

# Phase 2 figures (bathtub from Brief 17 §9.2, etc.) — same docs/-gitignored
# situation as phase_b.
mkdir -p docs/figures/phase2
rsync -avz --include='*.pdf' --include='*.png' --include='*.md' --exclude='*' \
    "$SNEL:$REMOTE/docs/figures/phase2/" docs/figures/phase2/

echo "Refresh complete."
