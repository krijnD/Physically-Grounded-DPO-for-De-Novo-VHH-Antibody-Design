#!/bin/bash
# Snellius SLURM script: three-way evaluation of the seed42_dedup finetune.
#
# Runs Block 2 of docs/finetune_evaluation_handoff.md end-to-end:
#   1. evaluate.py --mode design on the three checkpoints (pretrained,
#      seed42, seed42_dedup) on the test_antigen_disjoint split — the
#      headline three-way comparison.
#   2. evaluate.py on seed42_dedup against val and train splits — for
#      the train/test AAR gap that diagnoses overfit (§diagnostic-1).
#   3. build_comparison_table.py → comparison markdown.
#   4. check_sample_diversity.py × 3 → diversity / mode-collapse reports.
#   5. export_wandb_run.py + summarize_run.py for both seed42 runs.
#   6. Inline AAR-gap summary (train vs val vs test).
#
# Wall budget: ~10h on a single A100. test_antigen_disjoint × 3 ≈ 2.5h,
# val ≈ 45m, train ≈ 6h (181 entries × 3 CDRs × 8 samples), post-processing
# < 5m. 14h slot leaves headroom for the train eval ballooning if entries
# are large.
#
# Submit from the project root:
#     sbatch scripts/diffab_ft/slurm/evaluate_three_way.sbatch
#
# Outputs:
#   logs/slurm/eval_three_way_<jobid>.out         — combined stdout/stderr
#   runs/baseline_pretrained/eval_*.{json,csv}    — pretrained on test split
#   runs/vhh_ft/seed42/eval_*.{json,csv}          — seed42 on test split
#   runs/vhh_ft/seed42_dedup/eval_*.{json,csv}    — seed42_dedup on 3 splits
#   runs/_eval/comparison_test_antigen_disjoint.md
#   runs/_eval/diversity_{pretrained,seed42,seed42_dedup}.md
#   runs/_eval/training_diagnostic.md
#   runs/_eval/aar_gaps.md
#   runs/vhh_ft/{seed42,seed42_dedup}/wandb_export/
#
# Each step short-circuits on failure (`set -e`), so a missing checkpoint
# or a malformed splits JSON stops the whole job rather than producing
# partial reports.

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --job-name=eval_three_way
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

# ── Modules + venv ──
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate

mkdir -p logs/slurm runs/_eval runs/baseline_pretrained

# ── Provenance ──
echo "JobID:    ${SLURM_JOB_ID:-n/a}"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader || true)"
echo "Commit:   $(git rev-parse HEAD 2>/dev/null || echo n/a)"
echo "Branch:   $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
echo "Started:  $(date -Iseconds)"

# ── Paths ──
CONFIG=configs/diffab_ft/vhh_ft.yml
PRETRAINED_CKPT=/projects/0/hpmlprjs/interns/krijn/checkpoints/diffab/codesign_multicdrs.pt
SEED42_CKPT=runs/vhh_ft/seed42/checkpoints/best_ema.pt
DEDUP_CKPT=runs/vhh_ft/seed42_dedup/checkpoints/best_ema.pt
NUM_SAMPLES=8
SPLIT=test_antigen_disjoint

# ── Pre-flight: checkpoints ──
for entry in "pretrained:$PRETRAINED_CKPT" "seed42:$SEED42_CKPT" "seed42_dedup:$DEDUP_CKPT"; do
    name="${entry%%:*}"
    path="${entry#*:}"
    if [ ! -f "$path" ]; then
        echo "ERROR: $name checkpoint not found: $path" >&2
        if [ "$name" = "seed42" ]; then
            echo "       (handoff §open-questions item 1: confirm" >&2
            echo "        runs/vhh_ft/seed42/checkpoints/best_ema.pt was not deleted)" >&2
        fi
        exit 1
    fi
done
echo "All 3 checkpoints found."

# ── Pre-flight: splits use PDB-atom-sequence dedup ──
python -c "
import json, sys
d = json.load(open('data/datasets/clustering/cluster_splits.json'))
dedupe_by = d['params'].get('dedupe_by', None)
n_members = d['params'].get('n_members', 0)
if dedupe_by != 'pdb_atom_sequence':
    print(f'ERROR: splits used --dedupe-by={dedupe_by!r}, not pdb_atom_sequence.', file=sys.stderr)
    print('       The pretrained/seed42/seed42_dedup comparison is invalid against', file=sys.stderr)
    print('       the wrong splits. Re-run run_data_prep.py first.', file=sys.stderr)
    sys.exit(1)
print(f'splits OK: dedupe_by={dedupe_by!r}, n_members={n_members}')
"

# ── GPU evals ──
echo "=== Eval 1/5: pretrained on $SPLIT ==="
python scripts/diffab_ft/evaluate.py \
    --checkpoint "$PRETRAINED_CKPT" --config "$CONFIG" \
    --split "$SPLIT" --mode design --num-samples "$NUM_SAMPLES" \
    --save-pdbs runs/baseline_pretrained/eval_pdbs_${SPLIT} \
    --output runs/baseline_pretrained/eval_${SPLIT}.json

echo "=== Eval 2/5: seed42 on $SPLIT ==="
python scripts/diffab_ft/evaluate.py \
    --checkpoint "$SEED42_CKPT" --config "$CONFIG" \
    --split "$SPLIT" --mode design --num-samples "$NUM_SAMPLES" \
    --save-pdbs runs/vhh_ft/seed42/eval_pdbs_${SPLIT} \
    --output runs/vhh_ft/seed42/eval_${SPLIT}.json

echo "=== Eval 3/5: seed42_dedup on $SPLIT ==="
python scripts/diffab_ft/evaluate.py \
    --checkpoint "$DEDUP_CKPT" --config "$CONFIG" \
    --split "$SPLIT" --mode design --num-samples "$NUM_SAMPLES" \
    --save-pdbs runs/vhh_ft/seed42_dedup/eval_pdbs_${SPLIT} \
    --output runs/vhh_ft/seed42_dedup/eval_${SPLIT}.json

# val + train: no PDBs needed (only used for AAR-gap numbers; saves disk).
echo "=== Eval 4/5: seed42_dedup on val ==="
python scripts/diffab_ft/evaluate.py \
    --checkpoint "$DEDUP_CKPT" --config "$CONFIG" \
    --split val --mode design --num-samples "$NUM_SAMPLES" \
    --output runs/vhh_ft/seed42_dedup/eval_val.json

echo "=== Eval 5/5: seed42_dedup on train (the long one, ~6h) ==="
python scripts/diffab_ft/evaluate.py \
    --checkpoint "$DEDUP_CKPT" --config "$CONFIG" \
    --split train --mode design --num-samples "$NUM_SAMPLES" \
    --output runs/vhh_ft/seed42_dedup/eval_train.json

# ── Post-processing (CPU) ──
echo "=== Post 1/5: comparison table ==="
python scripts/diffab_ft/build_comparison_table.py \
    --evals runs/baseline_pretrained/eval_${SPLIT}.json \
            runs/vhh_ft/seed42/eval_${SPLIT}.json \
            runs/vhh_ft/seed42_dedup/eval_${SPLIT}.json \
    --labels pretrained seed42 seed42_dedup \
    --out runs/_eval/comparison_${SPLIT}.md \
    --title "Three-way evaluation on ${SPLIT}"

echo "=== Post 2/5: diversity reports ==="
python scripts/diffab_ft/check_sample_diversity.py \
    --csv runs/baseline_pretrained/eval_${SPLIT}.csv \
    --out runs/_eval/diversity_pretrained.md --label pretrained
python scripts/diffab_ft/check_sample_diversity.py \
    --csv runs/vhh_ft/seed42/eval_${SPLIT}.csv \
    --out runs/_eval/diversity_seed42.md --label seed42
python scripts/diffab_ft/check_sample_diversity.py \
    --csv runs/vhh_ft/seed42_dedup/eval_${SPLIT}.csv \
    --out runs/_eval/diversity_seed42_dedup.md --label seed42_dedup

echo "=== Post 3/5: W&B history export ==="
# seed42 may have been launched without W&B (early in the project). If
# its .wandb file is missing, fall back to export_log.py against log.txt.
if python scripts/diffab_ft/export_wandb_run.py \
        runs/vhh_ft/seed42 --out-dir runs/vhh_ft/seed42/wandb_export; then
    echo "seed42: W&B export OK."
else
    echo "seed42: W&B export failed; trying export_log.py fallback."
    python scripts/diffab_ft/export_log.py \
        runs/vhh_ft/seed42/log.txt \
        --out-dir runs/vhh_ft/seed42/wandb_export || \
        echo "WARN: seed42 has no log.txt either; training diagnostic will be single-run."
fi
python scripts/diffab_ft/export_wandb_run.py \
    runs/vhh_ft/seed42_dedup --out-dir runs/vhh_ft/seed42_dedup/wandb_export

echo "=== Post 4/5: training diagnostic ==="
if [ -d runs/vhh_ft/seed42/wandb_export ] && \
   [ "$(ls -A runs/vhh_ft/seed42/wandb_export 2>/dev/null)" ]; then
    python scripts/diffab_ft/summarize_run.py \
        runs/vhh_ft/seed42_dedup/wandb_export \
        --compare-to runs/vhh_ft/seed42/wandb_export \
        --out runs/_eval/training_diagnostic.md
else
    python scripts/diffab_ft/summarize_run.py \
        runs/vhh_ft/seed42_dedup/wandb_export \
        --out runs/_eval/training_diagnostic.md
fi

echo "=== Post 5/5: AAR gap summary ==="
python - <<'PYEOF'
from __future__ import annotations
import json
from pathlib import Path

out = Path("runs/_eval/aar_gaps.md")
out.parent.mkdir(parents=True, exist_ok=True)

paths = {
    "train": "runs/vhh_ft/seed42_dedup/eval_train.json",
    "val":   "runs/vhh_ft/seed42_dedup/eval_val.json",
    "test":  "runs/vhh_ft/seed42_dedup/eval_test_antigen_disjoint.json",
}
stats: dict[str, dict] = {}
for split, p in paths.items():
    try:
        stats[split] = json.loads(Path(p).read_text()).get("design", {}) or {}
    except FileNotFoundError:
        stats[split] = {}

def _aar(split: str, cdr: str) -> tuple[str, float | None]:
    s = (stats.get(split) or {}).get(cdr) or {}
    aar = s.get("aar_mean")
    std = s.get("aar_std")
    if aar is None:
        return "—", None
    return f"{aar:.3f} ± {std:.3f}", aar

with out.open("w") as f:
    f.write("# seed42_dedup AAR gap (overfit diagnostic)\n\n")
    f.write("Per `docs/finetune_evaluation_handoff.md` §diagnostic-question-1.\n\n")
    f.write("| CDR | train AAR | val AAR | test AAR | train−test gap |\n")
    f.write("|---|---|---|---|---|\n")
    for cdr in ("H1", "H2", "H3"):
        train_cell, train_v = _aar("train", cdr)
        val_cell,   _       = _aar("val",   cdr)
        test_cell,  test_v  = _aar("test",  cdr)
        if train_v is not None and test_v is not None:
            gap_cell = f"{train_v - test_v:+.3f}"
        else:
            gap_cell = "—"
        f.write(f"| {cdr} | {train_cell} | {val_cell} | {test_cell} | {gap_cell} |\n")
    f.write("\n**Interpretation (per handoff §decision-tree):**\n")
    f.write("- train−test H3 AAR gap < 0.10 → healthy generalization\n")
    f.write("- train−test H3 AAR gap > 0.15 → overfit; consider stronger regularization,\n")
    f.write("  shorter max_iters, or larger ema_decay\n")
print(f"Wrote {out}")
PYEOF

echo "Finished: $(date -Iseconds)"
