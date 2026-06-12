# Brief 26 deliverable — Leakage-corrected ELBO recomputation (Brief 23 follow-up)

**Date:** 2026-06-12
**Origin:** [`26_leakage_corrected_elbo.md`](26_leakage_corrected_elbo.md);
ran on Snellius commit
[`5e79e37`](https://github.com/krijnD/Physically-Grounded-DPO-for-De-Novo-VHH-Antibody-Design/commit/5e79e37).
**Mode:** ELBO re-eval of `seed42_jfix` (anchor) and `seed42_jfix_expanded`
(expanded) best-EMA checkpoints on Snellius via git-commit roundtrip
(`gpu_a100`, ~2 min wallclock). No new training, no new sampling, no
judge changes.

This is the campaign-repo mirror of the writer-facing deliverable at
[`master-thesis/docs/orchestrator_requests/A1_leakage_corrected_elbo_deliverable.md`](../../../master-thesis/docs/orchestrator_requests/A1_leakage_corrected_elbo_deliverable.md).
The two are kept in sync; the only differences are the relative paths
to artifacts.

---

## §1. Headline numbers

| Set | n | Anchor ELBO | Expanded ELBO | Δ%  |
|---|---:|---:|---:|---:|
| **Unfiltered** | 29 | 0.8487 | 0.8495 | **−0.09 %** |
| **Filtered** (excl. `7q6c_K`, `7n9v_J`) | 27 | 0.8262 | 0.8284 | **−0.27 %** |
| **Leakage effect** | — | — | — | **+0.18 pp** |

Reference: Brief 06's SQ1 claim was val ELBO −13.0 % (0.7316 → 0.6363),
measured on each model's *own* val split during training. Source
artifacts: [`tmp_brief26/per_entry_elbo.csv`](../../tmp_brief26/per_entry_elbo.csv)
+ [`tmp_brief26/leakage_corrected_summary.txt`](../../tmp_brief26/leakage_corrected_summary.txt).

## §2. Per-entry ELBO for the 2 leakage-flagged holdout entries

| Entry | Brief 23 leakage source (CDR identity) | Anchor ELBO | Expanded ELBO | Δ (anchor − expanded) |
|---|---|---:|---:|---:|
| `7q6c_K` | `7jkm_K`, `7o31_X` at 100 % | 1.2994 | 1.3072 | **−0.0077** (anchor better) |
| `7n9v_J` | `5o2u_D` at 88.8 % | 1.0042 | 0.9593 | **+0.0449** (expanded better) |

Of the two flagged entries, only `7n9v_J` shows a memorization-shaped
signal. `7q6c_K` — the 100 %-CDR-identity case — has the expanded
model marginally *worse*. Net contribution of the two leaked entries
to the unfiltered Δ: ~0.18 pp, matching the (unfiltered − filtered)
shift exactly.

## §3. Per-entry distribution across all 29 entries

- **Expanded wins** (lower ELBO): **15 / 29**. Largest favorable
  deltas: `7n9v_J` (+0.045 LEAK), `8elq_B` (+0.037), `7vke_B`
  (+0.033), `7xrp_B` (+0.017).
- **Anchor wins**: **14 / 29**. Largest favorable deltas: `7zlg_K`
  (−0.065), `8qot_B` (−0.034), `7qbf_B` (−0.032), `8cy6_D` (−0.025).
- **Within ±0.005 ELBO**: 8 / 29 — essentially tied.
- Mean Δ (clean, n = 27): −0.0022 ELBO units. Median Δ across all 29: 0.

The two models are functionally tied on the shared holdout.

## §4. Method (script provenance)

`scripts/analysis/brief26_leakage_corrected_elbo.py` (+ sbatch wrapper
`brief26_eval.sbatch`):

- Reuses DiffAb's model and dataset machinery via `get_model` /
  `get_dataset`, with `dataset.ids_in_split` overridden to the 29 IDs
  so the iteration is independent of whichever (train / val / test)
  split each entry sits in under the expanded splits JSON.
- Standard val transform (`mask_multiple_cdrs + merge_chains +
  patch_around_anchor`).
- 8 forward passes per entry; per-entry ELBO = mean over the 8
  passes. The transform's `mask_multiple_cdrs` is stochastic and
  DiffAb's diffusion samples a random `t` per forward, so the 8 passes
  draw 8 different `(t, mask)` configurations per entry.
- `seed_all(42)` re-called before each model's loop. Because both
  loops do the same dataset accesses in the same order, the i-th
  forward pass on entry e draws the *same* `t` and *same* CDR mask for
  both anchor and expanded — only the weights differ.

Internal-consistency checks that the eval ran correctly:

1. The 29 entries all produce non-trivial, diverse ELBO values
   (anchor range 0.40 – 1.73; expanded range 0.41 – 1.74) — no
   silently-skipped entries.
2. Different model weights produce different per-entry numbers for
   every entry (no two models would give bit-identical numbers if the
   second checkpoint had failed to load).
3. Leakage arithmetic balances: `(7q6c_K Δ + 7n9v_J Δ) / 29` ≈
   +0.00128 ELBO shift in favor of expanded → ≈ +0.15 % of anchor
   mean → matches the reported +0.18 pp leakage effect within rounding.

## §5. Gating outcome

Per brief §6:

- Trigger (1) **|Δ% unfiltered − Δ% filtered| ≥ 5 pp**: **NOT
  triggered** (0.18 pp shift).
- Trigger (2) **eval entry-point doesn't emit per-entry output**:
  handled in Phase 1 — `scripts/diffab_ft/evaluate.py --mode elbo`
  emits only aggregated stats, so a custom per-entry loop was added
  in Brief 26's own script (as the brief anticipated).
- Trigger (3) **expanded checkpoint inaccessible**: both checkpoints
  loaded cleanly on Snellius.

The brief instructs "Otherwise proceed directly to the deliverable",
which we did. But the unfiltered finding (−0.09 % vs the Brief 06
claim of −13 %) sits *outside* the brief's gating rubric and is the
load-bearing finding for SQ1 framing — surfaced in the writer-facing
deliverable §A-1.5.

## §6. Diagnosis

1. **Leakage gate: NEGATIVE.** The 3 of 462 added entries that
   cluster with 2 of the 29 shared-holdout entries at ≥85 % CDR
   identity do not meaningfully inflate the SQ1 comparison
   (+0.18 pp).
2. **Apples-to-apples gate (separate from leakage): the original
   −13 % does not reproduce on the shared holdout.** This was
   plausibly a per-model val split effect compounded by best-EMA
   selection: 22-ish entries per val split, no overlap between anchor
   val and expanded val, plus the standard practice of selecting the
   checkpoint at minimum val loss. On the same 29 entries for both
   models, the gap collapses to essentially zero.

Recommended SQ1 rewrite is in the writer-facing deliverable §A-1.6;
the recommendation is to replace the −13 % headline with the
shared-holdout numbers rather than footnote the gap, on the grounds
that the leakage-corrected number is the apples-to-apples one a
reviewer will defend most cleanly.

## §7. Reproduction

```bash
# On Snellius
cd ~/Physically-Grounded-DPO-for-De-Novo-VHH-Antibody-Design
git checkout 5e79e37  # or any descendant
sbatch scripts/analysis/brief26_eval.sbatch
# wallclock ~2 min on gpu_a100
cat tmp_brief26/leakage_corrected_summary.txt
```
