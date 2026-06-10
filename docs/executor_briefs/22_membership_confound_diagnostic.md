# Brief 22 — Membership-confound diagnostic (E0 / E1-B / E2 + rigor wins)

**Campaign:** [`docs/expanded_ft_handoff.md`](../expanded_ft_handoff.md)
**Progress log:** [`docs/expanded_ft_progress.md`](../expanded_ft_progress.md)
**Authoritative spec:** [`docs/EXPERIMENT_PLAN_brief22.md`](../EXPERIMENT_PLAN_brief22.md) — v2 reviewer-corrected scoped execution plan. **All §3 conventions, §6 interpretation gates, and §11 cut list defer to that file.**
**Previous brief:** [`19_brief17_brief18_synthesis.md`](19_brief17_brief18_synthesis.md) — sextuple-verified decoupling + data-property Ile collapse
**Concurrent prior art:** Hu, Zhang, Kuang, "D-Fusion: Direct Preference Optimization with Diffusion Consistency", *ICML 2025*, **PMLR 267:24869–24892**, arXiv:**2505.22002**.
**Branch:** `dpo-membership-diagnostic` (3 brief-22 commits already landed: `96f3238`, `1e05bf4`, `ec705d4`)
**Authored:** 2026-06-10 by orchestrator
**Triggering input:** three independent reviewers converged on the v2 corrections — channel-scope confound, circular membership/energy decomposition, terminology, D-Fusion prior art

---

## §1. TL;DR

Membership-confound diagnostic on the floor pipeline. **Three experiments (E0 / E1-B / E2) + AAR bootstrap CI rigor win + D-Fusion citation.** Experimental hard stop **36 h**. **NO reframe on spec** — channel-scope confound is closed, with correct attribution to the rotation channel.

- **E0 already produced on laptop** (single t=50 on 928 floor pairs). Headline: |m_seq| median = **+0.22** (≪ 0.5) → seq channel carries little membership signal → E1 will likely be flat → interpret as confound-closer.
- **E1-B** = matched seq-only GT-vs-decoy on the floor's **928 whitelisted IDs** (no partial-window filter). Pair pool, config, sbatch already on disk; only Snellius launch remains.
- **E2** = matched-manifold sample-vs-sample (winner = lowest-E_Rep π_ref sample, loser = highest-E_Rep π_ref sample). **678 pairs, 173 GTs contributing.** Pair pool, config, sbatch already on disk.
- **AAR bootstrap CI** (rigor-regardless): 1/9 cells excludes zero — **H2 Expanded(OLD) π_ref→π_θ: Δ = −2.04 pp [−3.91, −0.17]** — joins existing H2-designable + H2-CAAR significant cells.

**Decision tree (per `EXPERIMENT_PLAN_brief22.md` §2 hard 36 h gate):**

```
Run E0 (done) + E1-B + E2 + light eval → read against §4.7 interpretation table → DECIDE
  ├─ Gate-pass clean/positive → modest reframe (Days 4–8); promote bathtub; full Phase 4 on winning ckpt only.
  └─ Ambiguous (likely) → NO reframe; fold runs in as null-strengthening ablations; ship polished thesis.
```

---

## §2. Anti-hallucination guarantee

Every load-bearing number below is sourced to a file path or a CSV/parquet row. Match this convention for any downstream deliverable.

| # | Claim | Source |
|---|---|---|
| 1 | E0 m_rot median = +4.42 | [`data/analysis_outputs/e0_per_channel_ref_margin_928.csv`](../../data/analysis_outputs/e0_per_channel_ref_margin_928.csv) col `m_rot`, n=928 |
| 2 | E0 m_pos median = +0.20 | same CSV, col `m_pos` |
| 3 | E0 m_seq median = +0.22 | same CSV, col `m_seq` |
| 4 | E0 m_rot mean ± std = +5.79 ± 5.23 | same CSV; arithmetic on 928 rows |
| 5 | E0 m_pos mean ± std = +0.29 ± 0.63 | same CSV |
| 6 | E0 m_seq mean ± std = +0.24 ± 1.51 | same CSV |
| 7 | Bathtub t=0 means (rot +2.61, pos +0.13, seq −0.23) on the unfiltered 1492 pool | `data/results/decoy_t_sweep.csv` t=0 row; Brief 17 §9.2; Brief 19 §1(a) |
| 8 | E0 figure | [`docs/figures/phase2/per_channel_ref_margin_real_pairs.{png,pdf}`](../figures/phase2/) |
| 9 | E1-B pair pool: 928 rows, 188 unique GTs, 0 missing pair_ids | [`data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_decoy_t1_floor928.parquet`](../../data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_decoy_t1_floor928.parquet); commit `1e05bf4` body |
| 10 | E1 sign-flip diagnostic: 643/928 (69.3%) both>0; 285/928 (30.7%) floor>0 → decoy<0; 0 anti-flips | commit `1e05bf4` body; [`scripts/dpo/build_floor928_pair_pool.py`](../../scripts/dpo/build_floor928_pair_pool.py) |
| 11 | E1 post-substitution per-channel decoy (means / medians): rot +2.42 / +2.02; pos +0.10 / +0.08; seq −0.08 / −0.01 | commit `1e05bf4` body |
| 12 | E1 rot drop on 928 floor → 928 decoy: +5.79 → +2.42 (58% reduction, 3.37 absolute units) | rows 4 + 11 |
| 13 | E2 pair pool: 678 pairs, 173 GTs contributing, `winner_provenance="sample_min_erep"` | [`data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_sample_minErep_brief22.parquet`](../../data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_sample_minErep_brief22.parquet); commit `1e05bf4` body |
| 14 | E2 D0 funnel: 1680 → 1672 (drop 8 NaN-e_rep) → 1504 (restrict to floor's 188 GTs); 2 GTs dropped for range<1; 186 proceed | [`data/analysis_outputs/e2_d0_within_gt_spread.csv`](../../data/analysis_outputs/e2_d0_within_gt_spread.csv) |
| 15 | E2 D0 e_rep_range: mean 168.9, q25 86.5, median 139.5, q75 218.0, max 783.9 REU | same CSV; commit `1e05bf4` body |
| 16 | E2 D0 dropped GTs (range<1.0 REU): `7saj` (range 0.124), `8cyb` (range 0.059) | same CSV |
| 17 | E2 pairing params: THRESHOLD = 50.0 REU, CAP_PER_GT = 4 | [`scripts/dpo/select_by_e_rep_extrema.py`](../../scripts/dpo/select_by_e_rep_extrema.py) |
| 18 | E2 per-GT count: min=0, median=4, max=4, mean=3.65 | commit `1e05bf4` body |
| 19 | E2 13 zero-pair GTs (all gaps ≤ 50 REU): `7nll, 7s2r, 7uby, 8c5h, 8cxn, 8dam, 8en3, 8evd, 8f0k, 8f6u, 8hbi, 8p88, 8q95` | commit `1e05bf4` body |
| 20 | E1-B / E2 fixed-budget overrides: `max_iters=1000`, `val_freq=50`, `early_stop_patience=0`, β=0.05 (explicit), `loss_weights {rot:0,pos:0,seq:1}` | [`configs/dpo/vhh_dpo_seqonly_decoy_t1_floor928_brief22.yml`](../../configs/dpo/vhh_dpo_seqonly_decoy_t1_floor928_brief22.yml) lines 70-89, 119, 135; [`configs/dpo/vhh_dpo_seqonly_sample_minErep_brief22.yml`](../../configs/dpo/vhh_dpo_seqonly_sample_minErep_brief22.yml) lines 76-100, 138, 147 |
| 21 | E1-B W&B group + run name: `brief22_membership_diagnostic` / `brief22_e1_decoyt1_seqonly_beta05` | [`scripts/dpo/slurm/train_dpo_brief22_e1_decoyt1.sbatch`](../../scripts/dpo/slurm/train_dpo_brief22_e1_decoyt1.sbatch) lines 46-51 |
| 22 | E2 W&B group + run name: `brief22_membership_diagnostic` / `brief22_e2_sample_minErep_seqonly_beta05` | [`scripts/dpo/slurm/train_dpo_brief22_e2_sample.sbatch`](../../scripts/dpo/slurm/train_dpo_brief22_e2_sample.sbatch) lines 39-43 |
| 23 | AAR-CI headline: H2 Expanded(OLD) π_ref → π_θ Δ = −2.04 pp, 95% CI [−3.91, −0.17] | `master-thesis/data/analysis_outputs/bootstrap_cis_v2_aar.csv` (in master-thesis sibling, not yet committed); `master-thesis/scripts/regenerate_v2_bootstrap_cis_aar.py` |
| 24 | Brief 17 §11 partial-window filter dropped 928 → 658 (claim 409 in spec is AUDIT-2) | `data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_decoy_t1_filtered.parquet`; resolved Snellius-side via `grep pair_parquet runs/dpo/dpo_allchannel_decoy_t1_*/log_*.txt` |
| 25 | Floor π_θ reference AAR (n=29 OLD): H1 49.3 / H2 29.7 / H3 25.1 | Brief 17 §12 table; Brief 19 §1(d) |

**Bathtub vs E0 reconciliation** (AUDIT-3 below): both numerical views are honest; cite the correct pool when quoting.

| Pool | Source | rot mean | pos mean | seq mean | rot median | pos median | seq median |
|---|---|---|---|---|---|---|---|
| Unfiltered 1492 (bathtub t=0) | Brief 17 §9.2 | +2.61 | +0.13 | −0.23 | — | — | — |
| Ref-margin-filtered 928 (E0) | row 1–6 | +5.79 | +0.29 | +0.24 | +4.42 | +0.20 | +0.22 |
| 928 IDs after GT→decoy-t1 (E1 pair pool) | row 11 | +2.42 | +0.10 | −0.08 | +2.02 | +0.08 | −0.01 |

---

## §3. Hard rules (inherited from spec §3; do not deviate)

- **Pin one commit** (this branch's HEAD after pull) for every run record. Latest brief-22 commit `ec705d4`. No new fine-tune of π_ref.
- **v2 ANARCI-aware Chothia design-side slicing mandatory** for every metric (re-introducing the legacy `(95,102)` window reproduces the KPEDTAVY artifact; Brief 15 v2 fix).
- **Fixed-budget training**: `max_iters=1000`, `val_freq=50`, `early_stop_patience=0`, `seed=42`. Save EMA best-val + final; eval the best-val. E2 may extend to `max_iters=1500` if val still descending at 1000 (spec stretch — manually edit YAML and resubmit).
- **Precision**: **TF32-on, no-AMP** (matches existing trainer regime — NO FP32 flag is set; Krijn Q3). Record this verbatim in the run record's `precision` field.
- **AUDIT-4** — do not trust the `_beta05` suffix in template configs. `vhh_dpo_seqonly_filtered_beta05.yml` actually has `beta_dpo: 0.5`. The Brief 22 configs set `beta_dpo: 0.05` **explicitly** (cite paths in §2 row 20). Confirm by grepping the YAML before launch.
- **One auto-retry per GPU task** on crash/NaN; on second failure write `status:"failed"` + last 50 log lines and release the GPU. Monitor grad-norm vs clip (‖·‖₂ ≤ 1.0).
- **No master-thesis `sections/*.tex` edits today.** Reframe is conditional on the gate (§4.7); writing is Days 4–8 only if a result lands.
- **W&B**: entity `krijnd`, project `vhh-dpo`, group `brief22_membership_diagnostic` (shared by E1-B and E2 for visual comparison).

---

## §4. Runbook (numbered steps with gates)

Each step = (a) what to run, (b) what to check before proceeding, (c) the gate criterion that opens the next step.

### §4.1 — Branch sync on Snellius (~3 min)

```bash
cd /home/krijnds/Physically-Grounded-DPO-for-De-Novo-VHH-Antibody-Design
git fetch origin
git checkout dpo-membership-diagnostic
git pull origin dpo-membership-diagnostic
git log --oneline -5
```

**Check:** the three brief-22 commits land:

```
ec705d4 brief 22 §EX-4/6: E1-B + E2 configs + sbatches (seq-only β=0.05, fixed 1000 iters)
1e05bf4 brief 22 §EX-2/3/5/7: E0 figure + E1/E2 pair pools + AAR bootstrap CI
96f3238 brief 22: scoped execution plan v2 (membership-confound diagnostic)
```

**Confirm artifacts present:**

```bash
ls -la \
  docs/EXPERIMENT_PLAN_brief22.md \
  docs/executor_briefs/22_membership_confound_diagnostic.md \
  configs/dpo/vhh_dpo_seqonly_decoy_t1_floor928_brief22.yml \
  configs/dpo/vhh_dpo_seqonly_sample_minErep_brief22.yml \
  scripts/dpo/slurm/train_dpo_brief22_e1_decoyt1.sbatch \
  scripts/dpo/slurm/train_dpo_brief22_e2_sample.sbatch \
  data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_decoy_t1_floor928.parquet \
  data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_sample_minErep_brief22.parquet \
  data/analysis_outputs/e0_per_channel_ref_margin_928.csv \
  data/analysis_outputs/e2_d0_within_gt_spread.csv \
  docs/figures/phase2/per_channel_ref_margin_real_pairs.{png,pdf}
```

**Gate:** every path exists, non-empty. If any file missing → re-pull or contact orchestrator.

### §4.2 — E0 t=50 result is already in the branch — NO Snellius work

E0 was computed on laptop (`scripts/thesis/e0_per_channel_ref_margin_real_pairs.py`) at single fixed t=50 on the 928 ref-margin-filtered floor pairs.

**Read against §6 of the spec:**

| Quantity | Value | Decision rule trigger |
|---|---|---|
| `median(m_seq)` | **+0.22** | < 0.5 → seq channel has small membership signal → **E1 will likely be flat; interpret as confound-closer** ([SPEC §6](../EXPERIMENT_PLAN_brief22.md) row "Both small / both crystallize"). |
| `median(m_rot)` | **+4.42** | rot dominates the membership signal (as expected; bathtub also showed rot-dominance). |

**Gate:** confirmed; no E0 sub-step blocks E1-B or E2 launch.

### §4.3 — E0 t-sensitivity check (optional Snellius follow-up, ~1 GPU-hr total)

A robustness adjunct, **not a separate brief**. Re-run the per-channel diag at t=25 and t=75 on the 928 IDs and report `median(m_seq)` at each — confirms the t=50 reading isn't a one-point artifact.

```bash
sbatch --partition=gpu_a100 --gpus=1 --time=00:45:00 --wrap='
  set -euo pipefail
  module load 2024 && module load Python/3.12.3-GCCcore-13.3.0
  source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate
  for T in 25 75; do
    python scripts/dpo/diag_per_channel_reward.py \
        --pi-ref-checkpoint runs/vhh_ft/seed42_jfix/checkpoints/best_ema.pt \
        --pairs-parquet data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_filtered_marginGTp0.0.parquet \
        --output data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/lwref_per_channel_floor_t${T}.parquet \
        --num-timesteps 100 \
        --t-eval ${T} \
        --device cuda
  done
'
```

Then locally:

```python
import pandas as pd
T = 100
for t in (25, 50, 75):
    df = pd.read_parquet(f"data/aapr/.../lwref_per_channel_floor_t{t}.parquet")
    margin_seq = -T * (df["L_w_ref_seq"] - df["L_l_ref_seq"])
    print(f"t={t}: median(m_seq)={margin_seq.median():+.3f}")
```

**Pass condition:** all three medians within ±0.3 of each other and ≪ 0.5 → robustness confirmed. Report as a one-line note in the deliverable §"Optional Snellius follow-up". Wallclock: ~20 min A100 per t-value.

### §4.4 — E1-B launch (~30 min wallclock A100)

```bash
sbatch scripts/dpo/slurm/train_dpo_brief22_e1_decoyt1.sbatch
```

**Pre-launch sanity:**

```bash
test -f data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_decoy_t1_floor928.parquet
grep -n "beta_dpo: 0.05" configs/dpo/vhh_dpo_seqonly_decoy_t1_floor928_brief22.yml  # AUDIT-4 guard
grep -n "max_iters: 1000" configs/dpo/vhh_dpo_seqonly_decoy_t1_floor928_brief22.yml
grep -n "early_stop_patience: 0" configs/dpo/vhh_dpo_seqonly_decoy_t1_floor928_brief22.yml
```

**During training, monitor via W&B (`vhh-dpo` / group `brief22_membership_diagnostic` / run `brief22_e1_decoyt1_seqonly_beta05`):**

- val DPO loss curve — fixed-budget 1000 iters, val_freq=50; expect ≥20 val points.
- per-channel grad-norm — seq dominant (rot:0/pos:0 weights); rot/pos grads NOT propagated (loss weights zero by construction).
- iter-0 implicit reward ≡ 0 (since π_θ = π_ref at init); confirms loop is on the new pair pool.
- NaN spike → auto-retry once; second failure → record `status:"failed"` + last 50 log lines.

**Gate (§4.4 → §4.6):** training completes without NaN; best EMA val saved to `runs/dpo/brief22_e1_decoyt1_seqonly_beta05/checkpoints/best_ema.pt`.

### §4.5 — E2 launch (~30 min wallclock A100; can run in parallel with §4.4)

```bash
sbatch scripts/dpo/slurm/train_dpo_brief22_e2_sample.sbatch
```

**Pre-launch sanity** (identical to §4.4, swapping the config + parquet path):

```bash
test -f data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_sample_minErep_brief22.parquet
grep -n "beta_dpo: 0.05" configs/dpo/vhh_dpo_seqonly_sample_minErep_brief22.yml
grep -n "max_iters: 1000" configs/dpo/vhh_dpo_seqonly_sample_minErep_brief22.yml
```

**During training, monitor:** same metrics as §4.4. **If val DPO loss is still descending at iter 1000** (no plateau, no upward drift) → spec §3.3 stretch: manually edit the YAML to `max_iters: 1500`, resubmit. Document the deviation in the run record.

**Gate (§4.5 → §4.6):** identical to §4.4 — clean completion + saved best EMA.

### §4.6 — Gate-eval on both checkpoints (~30 min each on A100; sequence-side only, no folding)

Mirror Brief 17 §12 + Brief 18 §8 eval recipe. Concretely:

**Step A — extend `EVAL_CSV_MAP`** in [`scripts/eval/compute_per_position_modal_picks_v2.py`](../../scripts/eval/compute_per_position_modal_picks_v2.py) to include the two new variants:

```python
EVAL_CSV_MAP = {
    # ... existing entries ...
    ("brief22_e1_decoyt1_seqonly_beta05", "oldtest"): "runs/dpo/brief22_e1_decoyt1_seqonly_beta05/eval_test_design.csv",
    ("brief22_e2_sample_minErep_seqonly_beta05", "oldtest"): "runs/dpo/brief22_e2_sample_minErep_seqonly_beta05/eval_test_design.csv",
}
```

**Step B — design-eval sbatch** (one per run, modeled on `scripts/dpo/slurm/eval_ipo_floor_beta05.sbatch`):

```bash
sbatch --partition=gpu_a100 --gpus=1 --time=01:00:00 --job-name=eval_brief22_e1 --wrap='
  set -euo pipefail
  module load 2024 && module load Python/3.12.3-GCCcore-13.3.0
  source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate
  python scripts/diffab_ft/evaluate.py \
      --checkpoint runs/dpo/brief22_e1_decoyt1_seqonly_beta05/checkpoints/best_ema.pt \
      --config configs/diffab_ft/vhh_ft.yml \
      --split test \
      --mode design \
      --num-samples 4 \
      --output runs/dpo/brief22_e1_decoyt1_seqonly_beta05/eval_test_design.json
'
# same for E2
sbatch --partition=gpu_a100 --gpus=1 --time=01:00:00 --job-name=eval_brief22_e2 --wrap='
  set -euo pipefail
  module load 2024 && module load Python/3.12.3-GCCcore-13.3.0
  source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate
  python scripts/diffab_ft/evaluate.py \
      --checkpoint runs/dpo/brief22_e2_sample_minErep_seqonly_beta05/checkpoints/best_ema.pt \
      --config configs/diffab_ft/vhh_ft.yml \
      --split test \
      --mode design \
      --num-samples 4 \
      --output runs/dpo/brief22_e2_sample_minErep_seqonly_beta05/eval_test_design.json
'
```

**Step C — modal-pick analysis** (login node, after both JSONs land):

```bash
python scripts/eval/compute_per_position_modal_picks_v2.py \
    --variants brief22_e1_decoyt1_seqonly_beta05,brief22_e2_sample_minErep_seqonly_beta05 \
    --test-sets oldtest \
    --output data/eval/per_position_modal_picks_brief22.parquet
```

**Gate-eval axes (the spec's §9 policy):** per-CDR AAR, modal-match, H3 modal motif (L=8), homopolymer rate, per-channel reference margin, grad-norm/θ-drift. Sequence-side only — **no scRMSD / CAAR / EpiF1** at this stage.

**Reference table to fill (your deliverable §4.6):**

| Variant | H1 AAR | H2 AAR | H3 AAR | H3 modal motif | H3 modal-match |
|---|---|---|---|---|---|
| floor π_ref (anchor) | 48.6% | 30.0% | 25.0% | (existing) | (existing) |
| floor π_θ (E1-A; seq-only β=0.05, n=928) | 49.3% | 29.7% | 25.1% | `YCAAAGGGVYDYPYTYDY` | 27.8% |
| **E1-B decoy-t1 seq-only β=0.05 (n=928)** | ? | ? | ? | ? | ? |
| **E2 sample-min-E_Rep seq-only β=0.05 (n=678)** | ? | ? | ? | ? | ? |

### §4.7 — Interpretation gate (read against `EXPERIMENT_PLAN_brief22.md` §6)

**E1 outcomes — verbatim from spec §6:**

| Outcome | Licenses |
|---|---|
| **A learns + B crystallizes** (small E1-B update vs E1-A) | seq channel had residue-identity contrast worth removing — promote, but frame as **residue-identity, NOT noise-signature** membership. |
| **Both small / both crystallize** *(likely, given E0's m_seq median +0.22)* | "shortcut is **structural**, not sequence; seq-only null is **margin-sharpening** (§5.1.5); **channel-scope confound closed**." Defensible refinement, not an overturn. |
| **B learns but sample metrics flat** | shortcut isn't the whole story (unlikely given small m_seq). |

**E2 outcomes — verbatim from spec §6:**

| Outcome | Licenses |
|---|---|
| **AAR / modal-match move beyond noise** | quality CAN transmit once membership removed → **best outcome**; n=29 humility; "promising rescue, NOT a solved design method." |
| **No-op** | localizes bottleneck **below** the membership confound at this data scale → keep as limitation / preliminary diagnostic (ambiguous per spec §5 scope note). |

**Spec caveat threaded through both** ([SPEC §6 closing](../EXPERIMENT_PLAN_brief22.md)): the headline is **no longer** "the DPO update was the membership shortcut." E0 relocates the shortcut to the structural channels; the all-channel decoy crystallization (thesis Table 4.11) already evidences that the structural signal is membership-dominated.

**Write `results/decision.md`** capturing which §6 cell each experiment landed on + reframe go/no-go + evidence.

### §4.8 — Full Phase 4 (ONLY on a gate-passing checkpoint)

scRMSD designability + CAAR + EpiF1 via ABodyBuilder2. **Reserve compute. v2 ANARCI-aware Chothia slicing mandatory.**

Reuse the existing dispatchers:

```bash
python scripts/eval/run_scrmsd_array.py \
    --variants brief22_<winning_variant> \
    --test-sets oldtest
python scripts/eval/run_caar_epif1_array.py \
    --variants brief22_<winning_variant> \
    --test-sets oldtest
```

Record `abb2_reject_count`. If §4.7 verdict is "ambiguous / both crystallize" — **skip §4.8** and treat the gate-eval table as the deliverable.

---

## §5. Deliverable format

Map each spec deliverable (`EXPERIMENT_PLAN_brief22.md` §10) to where it lives:

| Deliverable | Path | Status |
|---|---|---|
| `figs/per_channel_ref_margin_real_pairs.{png,pdf}` | [`docs/figures/phase2/per_channel_ref_margin_real_pairs.{png,pdf}`](../figures/phase2/) | ✓ produced (EX-2) |
| E0 three median margins | rows 1–3 of §2 anti-hallucination table | ✓ produced |
| E1-B checkpoint | `runs/dpo/brief22_e1_decoyt1_seqonly_beta05/checkpoints/best_ema.pt` | pending §4.4 |
| E1-B gate-eval | `runs/dpo/brief22_e1_decoyt1_seqonly_beta05/eval_test_design.{json,csv}` | pending §4.6 |
| E2 checkpoint | `runs/dpo/brief22_e2_sample_minErep_seqonly_beta05/checkpoints/best_ema.pt` | pending §4.5 |
| E2 gate-eval | `runs/dpo/brief22_e2_sample_minErep_seqonly_beta05/eval_test_design.{json,csv}` | pending §4.6 |
| `results/within_gt_spread.csv` | [`data/analysis_outputs/e2_d0_within_gt_spread.csv`](../../data/analysis_outputs/e2_d0_within_gt_spread.csv) | ✓ produced (EX-5) |
| `runs/*.json` (schema-valid, ref_margin fields) | per-run subdir | pending |
| `results/master_results.csv` (merge of EX-2 / EX-7 / both gate-evals) | `data/results/brief22_master_results.csv` | pending §4.6 |
| AAR bootstrap-CI table | `master-thesis/data/analysis_outputs/bootstrap_cis_v2_aar.csv` (sibling repo, not committed) | ✓ produced (EX-7); writer integrates |
| `results/decision.md` | `docs/executor_briefs/22_decision.md` (or similar) | pending §4.7 |
| Modal-pick parquet | `data/eval/per_position_modal_picks_brief22.parquet` | pending §4.6 |

---

## §6. Audit trail

| Artifact | Producing EX | Script + commit | W&B URL (planned / actual) |
|---|---|---|---|
| `data/analysis_outputs/e0_per_channel_ref_margin_928.csv` | EX-2 | [`scripts/thesis/e0_per_channel_ref_margin_real_pairs.py`](../../scripts/thesis/e0_per_channel_ref_margin_real_pairs.py); commit `1e05bf4` | n/a (laptop CPU/CPU+CUDA) |
| `docs/figures/phase2/per_channel_ref_margin_real_pairs.{png,pdf}` | EX-2 | same script; commit `1e05bf4` | n/a |
| `data/aapr/.../pairs_decoy_t1_floor928.parquet` | EX-3 | [`scripts/dpo/build_floor928_pair_pool.py`](../../scripts/dpo/build_floor928_pair_pool.py); commit `1e05bf4` | n/a |
| `data/aapr/.../pairs_sample_minErep_brief22.parquet` | EX-5 | [`scripts/dpo/select_by_e_rep_extrema.py`](../../scripts/dpo/select_by_e_rep_extrema.py); commit `1e05bf4` | n/a |
| `data/analysis_outputs/e2_d0_within_gt_spread.csv` | EX-5 | same script; commit `1e05bf4` | n/a |
| `master-thesis/data/analysis_outputs/bootstrap_cis_v2_aar.csv` | EX-7 | `master-thesis/scripts/regenerate_v2_bootstrap_cis_aar.py`; **not yet committed** | n/a |
| `configs/dpo/vhh_dpo_seqonly_decoy_t1_floor928_brief22.yml` | EX-4 | hand-authored; commit `ec705d4` | n/a |
| `scripts/dpo/slurm/train_dpo_brief22_e1_decoyt1.sbatch` | EX-4 | hand-authored; commit `ec705d4` | n/a |
| `configs/dpo/vhh_dpo_seqonly_sample_minErep_brief22.yml` | EX-6 | hand-authored; commit `ec705d4` | n/a |
| `scripts/dpo/slurm/train_dpo_brief22_e2_sample.sbatch` | EX-6 | hand-authored; commit `ec705d4` | n/a |
| E1-B training run | §4.4 (pending) | `train_dpo_brief22_e1_decoyt1.sbatch` | planned: `vhh-dpo / brief22_membership_diagnostic / brief22_e1_decoyt1_seqonly_beta05` |
| E2 training run | §4.5 (pending) | `train_dpo_brief22_e2_sample.sbatch` | planned: `vhh-dpo / brief22_membership_diagnostic / brief22_e2_sample_minErep_seqonly_beta05` |
| E1-B gate-eval JSON+CSV | §4.6 (pending) | `evaluate.py --mode design --num-samples 4` | n/a (offline eval) |
| E2 gate-eval JSON+CSV | §4.6 (pending) | same | n/a |
| Modal-pick parquet | §4.6 (pending) | `scripts/eval/compute_per_position_modal_picks_v2.py` | n/a |

---

## §7. Deviations / uncertainties

**AUDIT-1** — Q1 / spec §"averaged over 20 diffusion timesteps":
- Thesis §3.3.4 + §3.4.3 say the ref-margin filter is "averaged over 20 diffusion timesteps".
- The actual filter at [`scripts/dpo/diag_lwref_distribution.py:148`](../../scripts/dpo/diag_lwref_distribution.py) uses **single fixed t=50**.
- The 928 filter therefore used single t=50, not 20-step averaging.
- **Krijn Q1 decision:** keep single t=50 throughout (apples-to-apples with bathtub t=0 row); spec's "averaged over 20" is a misstatement.
- **Writer handoff:** fix the thesis text (cheap) — do NOT re-run the filter (expensive; invalidates downstream).
- **Optional Snellius follow-up:** §4.3 t-sensitivity at t ∈ {25, 75} confirms the t=50 reading isn't a one-point artifact.

**AUDIT-2** — Brief 17 §11 pair-count claim:
- Spec text says "Brief 17 §11 kept only 409 pairs".
- The actual parquet `pairs_decoy_t1_filtered.parquet` on disk is **658 rows** (lwref filter).
- Snellius-side resolution: `grep pair_parquet runs/dpo/dpo_allchannel_decoy_t1_*/log_*.txt` will reveal what Brief 17 §11 actually trained on.
- **NOT blocking Brief 22** — E1-B uses the unfiltered-and-whitelisted 928 regardless of what Brief 17 §11 used.

**AUDIT-3** — bathtub vs E0 pool reconciliation:
- Bathtub t=0 (1492-pair unfiltered pool, means): rot +2.61 / pos +0.13 / **seq −0.23**.
- E0 (928 ref-margin-filtered pool, medians: rot +4.42 / pos +0.20 / **seq +0.22**; means: rot +5.79 / pos +0.29 / seq +0.24).
- **Both honest.** The 928's filter-uplift shifts all channels positive (the 928 was preselected for composite-margin > 0, which is rot-dominated); seq's sign **flips** because of this filter uplift.
- **Qualitative reading preserved:** rot dominates; |m_seq| stays small (< 0.5).
- **Brief 22 prose must cite the right pool** when quoting per-channel margins.
- **Writer handoff:** thesis caption around Table A.12 should explicitly state "on the unfiltered 1492-pair Pareto pool" since that's the published convention.

**AUDIT-4** — `_beta05` filename ambiguity:
- [`configs/dpo/vhh_dpo_seqonly_filtered_beta05.yml`](../../configs/dpo/vhh_dpo_seqonly_filtered_beta05.yml) actually has `beta_dpo: 0.5` (NOT 0.05).
- Brief 16 convention: `_beta05` → β = 0.5; `_beta0005` → β = 0.005. The **TRUE β=0.05 floor config** is [`vhh_dpo_seqonly_filtered.yml`](../../configs/dpo/vhh_dpo_seqonly_filtered.yml) (no β suffix).
- **EX-4 / EX-6 set β=0.05 explicitly** in the new Brief 22 configs (the `_brief22.yml` files); no impact on runs.
- **Pre-launch guard in §4.4 / §4.5** greps `beta_dpo: 0.05` from the YAML to confirm.

**Q1–Q4 decisions (baked into the runbook):**
- **Q1 (E0 t-eval):** single t=50 (apples-to-apples with bathtub). Optional §4.3 t-sensitivity at t∈{25, 75} as one-line robustness adjunct.
- **Q2 (E1 pool):** floor's 928 IDs as a hard whitelist on the **unfiltered** `pairs_decoy_t1.parquet`. No lwref filter, no partial-window filter. **0 IDs were missing** from the decoy pool (commit `1e05bf4` body).
- **Q3 (FP32):** no FP32 flag; TF32-on + no-AMP matches existing trainer regime. Run record must carry `precision: "tf32-matmul, no-amp"`.
- **Q4 (E2 scope):** floor's 188 GTs (hard); pairing = E_Rep-gap-thresholded multi-pair-per-GT capped at 4, fenced to those 188.

**Krijn's two riders on E1 (logging discipline):**
- **Log don't filter.** The sign-flip diagnostic (643 both>0 / 285 floor>0 → decoy<0) is recorded as a column in the pair pool — do not drop sign-flip pairs from E1-B training. Both arms see identical 928 IDs.
- **Record precision regime.** Every run record's `precision` field must read `"tf32-matmul, no-amp"` (matches Q3).

**E2 sub-pool limitation (not a confound):**
- 13 of 188 GTs contribute 0 pairs (all within-GT E_Rep gaps ≤ 50 REU): `7nll, 7s2r, 7uby, 8c5h, 8cxn, 8dam, 8en3, 8evd, 8f0k, 8f6u, 8hbi, 8p88, 8q95`.
- 173 GTs contribute ≥1 pair; final n = 678 pairs.
- This is a sub-pool restriction (not a structural bias) — the 13 dropped GTs have low within-GT E_Rep variance, so they cannot supply a winner ≠ loser pair under the E_Rep ranker. Note as a limitation in §"Honest scope" prose; do not over-claim coverage.

---

## §8. Files produced (this branch — 3 brief-22 commits)

Grouped by EX-N.

**EX-1 — Authoritative spec** (commit `96f3238`):
- [`docs/EXPERIMENT_PLAN_brief22.md`](../EXPERIMENT_PLAN_brief22.md) — 198-line v2 reviewer-corrected scoped plan; source of §3 hard rules, §4.7 interpretation gates, §"Honest scope" prose.

**EX-2 — E0 per-channel reference-margin decomposition** (commit `1e05bf4`):
- [`scripts/thesis/e0_per_channel_ref_margin_real_pairs.py`](../../scripts/thesis/e0_per_channel_ref_margin_real_pairs.py) — 181 lines; reads 928 floor pairs + lwref_per_channel parquet, computes m_c per pair, plots 3-panel figure.
- [`data/analysis_outputs/e0_per_channel_ref_margin_928.csv`](../../data/analysis_outputs/e0_per_channel_ref_margin_928.csv) — 929 rows (header + 928 pairs); cols `pair_id, gt_id, m_rot, m_pos, m_seq, m_composite`.
- [`docs/figures/phase2/per_channel_ref_margin_real_pairs.png`](../figures/phase2/per_channel_ref_margin_real_pairs.png) + `.pdf` — 3-panel histograms with median annotations.

**EX-3 — E1-B pair pool** (commit `1e05bf4`):
- [`scripts/dpo/build_floor928_pair_pool.py`](../../scripts/dpo/build_floor928_pair_pool.py) — 185 lines; ID-filters the unfiltered 1492-pair decoy-t1 parquet to the 928 floor IDs; emits sign-flip diagnostic.
- [`data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_decoy_t1_floor928.parquet`](../../data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_decoy_t1_floor928.parquet) — 928 rows × 13 cols; `winner_provenance="decoy_t1"`.

**EX-4 — E1-B config + sbatch** (commit `ec705d4`):
- [`configs/dpo/vhh_dpo_seqonly_decoy_t1_floor928_brief22.yml`](../../configs/dpo/vhh_dpo_seqonly_decoy_t1_floor928_brief22.yml) — 221 lines; overrides vs `vhh_dpo_seqonly_filtered.yml` (the true β=0.05 floor): `max_iters 10000→1000`, `val_freq 100→50`, `early_stop_patience 30→0`, `pair_parquet → pairs_decoy_t1_floor928.parquet`. β=0.05 explicit.
- [`scripts/dpo/slurm/train_dpo_brief22_e1_decoyt1.sbatch`](../../scripts/dpo/slurm/train_dpo_brief22_e1_decoyt1.sbatch) — 53 lines; run-name `brief22_e1_decoyt1_seqonly_beta05`, group `brief22_membership_diagnostic`.

**EX-5 — E2 pair pool + D0 within-GT spread** (commit `1e05bf4`):
- [`scripts/dpo/select_by_e_rep_extrema.py`](../../scripts/dpo/select_by_e_rep_extrema.py) — 332 lines; D0 pre-check + pairing fenced to floor's 188 GTs; `THRESHOLD_E_REP_REU=50.0`, `CAP_PER_GT=4`.
- [`data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_sample_minErep_brief22.parquet`](../../data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo/pairs_sample_minErep_brief22.parquet) — 678 rows; `winner_provenance="sample_min_erep"`.
- [`data/analysis_outputs/e2_d0_within_gt_spread.csv`](../../data/analysis_outputs/e2_d0_within_gt_spread.csv) — 189 rows (header + 188 GTs); cols `gt_id, n_samples, e_rep_min, e_rep_max, e_rep_median, e_rep_range, kept`.

**EX-6 — E2 config + sbatch** (commit `ec705d4`):
- [`configs/dpo/vhh_dpo_seqonly_sample_minErep_brief22.yml`](../../configs/dpo/vhh_dpo_seqonly_sample_minErep_brief22.yml) — 233 lines; same overrides as E1-B + `pair_parquet → pairs_sample_minErep_brief22.parquet`.
- [`scripts/dpo/slurm/train_dpo_brief22_e2_sample.sbatch`](../../scripts/dpo/slurm/train_dpo_brief22_e2_sample.sbatch) — 46 lines; run-name `brief22_e2_sample_minErep_seqonly_beta05`, group `brief22_membership_diagnostic`.

**EX-7 — AAR bootstrap CI (sibling, master-thesis repo, NOT committed)**:
- `master-thesis/scripts/regenerate_v2_bootstrap_cis_aar.py` — 120 lines; sibling to existing `regenerate_v2_bootstrap_cis.py` (designable + CAAR); does NOT touch that file.
- `master-thesis/data/analysis_outputs/bootstrap_cis_v2_aar.csv` — 9 rows; identical column shape to `bootstrap_cis_v2.csv` so forest plots can append.
- **Headline cell:** H2 AAR Expanded(OLD) π_ref → π_θ: Δ = **−2.04 pp, 95% CI [−3.91, −0.17]** (excludes zero). Joins existing H2-designable [−12.07, −0.86] + H2-CAAR [−6.50, −0.17]. All other 8 AAR cells contain zero.
- **Writer integrates** when committing the master-thesis sibling.

**EX-8 — This brief** (pending commit by orchestrator):
- [`docs/executor_briefs/22_membership_confound_diagnostic.md`](22_membership_confound_diagnostic.md) — the run-ready spec a fresh executor will use to launch §4.1 → §4.8 on Snellius.

---

## §9. D-Fusion citation + framing (rigor win 1)

**Verbatim citation:** Hu, Zhang, Kuang, "D-Fusion: Direct Preference Optimization with Diffusion Consistency", *ICML 2025*, **PMLR 267:24869–24892**, arXiv:**2505.22002**.

**Single paragraph for thesis-prose insertion** (NOT applied to thesis until gate opens):

> Concurrent work in image diffusion has published the same diffusion-DPO consistency problem we observe and characterize here. D-Fusion (Hu et al., ICML 2025, PMLR 267:24869–24892, arXiv:2505.22002) identifies that DPO's preference signal in diffusion models can be dominated by a real-vs-synthetic membership shortcut rather than the intended quality axis, and proposes a matched-consistent-sample construction (mask-guided self-attention fusion) as a working remedy. Our work is the **structural-domain analogue** of their visual-inconsistency analysis, on a different modality (3D structure-conditioned VHH antibody design) and using a different decoy mechanism (forward-noise-to-t-then-π_ref-denoise rather than mask-guided self-attention fusion). Our contribution is the **diagnostic + the bathtub instrument** — the per-channel decomposition that shows the rotation channel carries the dominant membership signature in our pipeline (rot +2.61 / pos +0.13 / seq −0.23 on the unfiltered Pareto pool) — **not the remedy**. We distinguish this from DeDPO (arXiv:2602.06195, already cited), which targets a related-but-distinct synthetic-annotator bias.

**Anti-pattern guard:** delete every "novel failure mode" / "discovery" / "first to identify" phrase from the existing thesis text. D-Fusion precludes that framing.

---

## §10. Terminology fix (rigor win 2)

Replace **"iter-0 implicit reward"** with **"reference NLL margin under π_ref"** throughout new analysis.

**Reasoning:** at DPO init π_θ = π_ref, so the implicit reward `m = -β·T·δ` where `δ = (L_w_θ - L_w_ref) - (L_l_θ - L_l_ref)` is **identically 0** at iter 0. The load-bearing quantity is the **reference NLL margin** `m_ref = log π_ref(y_w) - log π_ref(y_l)` (which `ref_nll_margin` already computes). "Iter-0 implicit reward" is imprecise — call it what it is.

**Add one sentence to thesis §3.3.4 (or wherever the terminology first appears):**

> Because π_θ = π_ref at DPO initialization, the implicit reward `m = -β·T·δ` is identically zero at iter 0; the load-bearing diagnostic is the **reference NLL margin** `m_ref = log π_ref(y_w) - log π_ref(y_l)` (computed per pair under π_ref). All references to "iter-0 implicit reward" in earlier briefs should be read as "reference NLL margin under π_ref."

**Writer handoff:** audit existing thesis usage; replace inline.

---

*End of Brief 22. Hand-off to Snellius execution: pull this branch, follow §4.1 → §4.8. The default landing zone is **modest reframe-less null-strengthening**: closed channel-scope confound, correctly-attributed membership signature, sextuple-decoupled loss-quality story intact, three rigor wins (AAR bootstrap CI / D-Fusion cite / terminology fix). Reframe only if §4.7 opens.*
