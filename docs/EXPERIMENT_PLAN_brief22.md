# Scoped One-Day Plan (v2) — Membership-Confound Diagnostic for VHH Diffusion-DPO

**Owner:** Krijn Dignum · **Repo:** `KrijnD/Physically-Grounded-DPO-for-De-Novo-VHH-Antibody-Design` · **Cluster:** Snellius (A100/H100)
**Audience:** orchestrator + executors. Supersedes v1 (`one_day_execution_plan.md`).
**Submission window:** 9 days. **Experimental hard stop:** 36 h. **Reframe is conditional on results — do NOT touch the abstract/intro today.**

---

## 0. What changed from v1, and why

Three independent reviewers converged on the same corrections. They are right; v1 is amended:

| v1 element | Verdict | Correction |
|---|---|---|
| Seq-only decoy-depth sweep as the **headline "linchpin"** | **Wrong premise.** Table A.12 shows the +2.61 shortcut is ~entirely **rotation-channel** (r_rot +2.606, r_pos +0.125, **r_seq −0.226**). The seq channel — the only channel the principal runs and the linchpin train — carries a tiny, slightly-negative separation that stays small across the sweep. The "update → 0 as the shortcut vanishes" prediction is predicated on a seq-channel shortcut that the data says isn't there. | **Demote.** Keep a seq-only decoy run, but only as a **channel-scope confound-closer** (E1), not a smoking gun. Expect an ambiguous/flat outcome and read it as: "the shortcut is structural, not sequence; the seq-only null is margin-sharpening (§5.1.5)." |
| Reward decomposition (membership vs quality) | **Circular.** In your data membership and energy are collinear by construction (real ⇒ low E_Rep, synthetic ⇒ high E_Rep). You can't regress reward on both as separate predictors — they're the same variable. | **Replace** with a per-channel **reference-margin decomposition on the real training pairs** (E0), which is non-circular and decision-relevant. Energy-orthogonal membership detector is **optional stretch** only. |
| "Hold out a clean quality axis" for the rescue test | **Not executable.** PSH is coin-flip (52/50%, §3.4.2); E_cdr is collinear with E_Rep. Energy is your only discriminative ranker. | Run the rescue (E2) ranked by **E_Rep**. State plainly: it removes the real-vs-synthetic gap but **does not** break energy=quality. Don't claim it does. |
| "Iter-0 implicit reward" terminology | **Imprecise.** At the DPO start state π_θ = π_ref, so the implicit reward is identically **zero**. | Call it the **reference NLL margin under π_ref** (the quantity your `ref_nll_margin` already computes). Apply to new work and audit existing usage. |
| "Novel DPO failure mode" framing | **Precluded.** D-Fusion (ICML 2025) published the phenomenon + matched-pair remedy in image-diffusion. | **Cite D-Fusion. Remove all "discovery" language.** Contribution = domain-extension + the bathtub instrument + the energy-axis recontextualization. |
| Cross-antigen hard negatives (WS-E) | **Cut** (all three reviewers). Pair validity under different conditioning is ambiguous; too much loader logic. | Future work. |
| Reframe abstract/intro on spec; promote bathtub to centerpiece on Day 1 | **Premature** (all three). | **Gate it.** Default = polish the honest null with the confound elevated. Reframe only if a result lands. |

**Net:** the scope shrank, which makes the **one-day target more realistic, not less** — there is far less to run. Today = three cheap, causally-clean experiments + two rigor wins + a decision. Writing is deferred and conditional.

---

## 1. Objective (revised, non-overclaiming)

> Characterize the **GT-vs-synthetic membership confound** in this VHH Diffusion-DPO pipeline: show (E0) the confound is **structural** (rotation-channel reference-likelihood gap), not sequence; (E1) close the **channel-scope confound** in the existing all-channel decoy ablation with a matched seq-only GT-vs-decoy control; and (E2) run the one new experiment with an informative null — **matched-manifold sample-vs-sample** preference — asking whether any quality signal transmits once the real-vs-synthetic gap is removed.

Likely landing zone (per E0/channel-mismatch): **a correctly-attributed null with the membership confound elevated from a buried appendix to a prominent diagnostic**, the existing position-conservative / margin-sharpening reading intact and *refined*, and the channel-scope confound closed. Upgrade path (if E2 is positive): "removing the membership gap partially restores transmission." Either is a real improvement; neither bets the thesis.

---

## 2. The decision gate (hard 36 h)

```
Run E0 + E1 + E2 + light eval  ──►  read against §6 interpretation table  ──►  DECIDE
   │
   ├─ A result lands clean/positive  ──►  modest reframe (Days 4–8): subtitle + abstract sentence,
   │                                       promote bathtub to main text, elevate confound section,
   │                                       cite D-Fusion, run full Phase 4 on the winning checkpoint only.
   │
   └─ Ambiguous (the likely case)    ──►  NO reframe. Fold runs in as ablations that strengthen the
                                          null ("shortcut is structural; seq-only signal is margin-
                                          sharpening; channel-scope confound closed"), elevate the
                                          confound to a Discussion/diagnostic section (correctly
                                          attributed), ship the polished current thesis.
```
**If the matched experiment is not clean within 36 h, stop and polish.** The existing thesis is already defensible.

---

## 3. Shared conventions (carried from v1, trimmed)

- **Pin one commit** for the whole day; record SHA in every run record. No new fine-tune of π_ref.
- **v2 ANARCI-aware Chothia design-side slicing is mandatory** for every metric (re-introducing the `(95,102)` window reproduces the KPEDTAVY artifact).
- **Fixed-budget training:** `max_iters=1000`, `early_stop_patience=0`, `val_every=50`, `seed=42`, FP32. Save EMA best-val + final; eval best-val. (E2 may extend to 1500 if val still descending at 1000.)
- **One auto-retry** per GPU task on crash/NaN; on second failure write `status:"failed"` + last 50 log lines and release the GPU. Monitor grad-norm vs clip (‖·‖₂≤1.0).
- **Config override grammar** (⚠ bind to real entrypoint/flags):
```bash
python -m train_dpo --config configs/dpo_floor.yaml \
  dpo.beta_dpo=0.05 dpo.loss_channels=seq dpo.aggregation=residue \
  dpo.winner_source=${gt|decoy|sample} dpo.decoy_depth=${T} \
  dpo.pair_pool=${POOL_PATH} dpo.max_iters=1000 dpo.early_stop_patience=0 \
  dpo.val_every=50 run.seed=42 run.fp32=true
```
- SLURM template + GPU-tier dispatch logic: reuse v1 §3.8 unchanged.

### Run-record schema (delta from v1)
Rename the reward fields to reflect §0's terminology fix and add per-channel:
```json
"ref_margin_rot": null, "ref_margin_pos": null, "ref_margin_seq": null,
"ref_margin_total": null, "ref_margin_pct_negative": null,
"grad_norm_trajectory_path": "", "theta_drift_final_l2": null,
"n_pairs": 0, "homopolymer_rate": null
```
(All other schema fields from v1 §3.6 carry over: AAR/modal-match/H3-motif/scRMSD/CAAR/EpiF1/abb2_reject_count/checkpoints/wandb/commit/wallclock.)

---

## 4. Setup barrier (must emit `GO` first; ~30 min)

| Step | Check | On fail |
|---|---|---|
| S1 | Pin commit; env imports (PyRosetta, ANARCI, ABodyBuilder2, ChimeraBench) | block |
| S2 | **Canonical ID set = the 928 ref-margin-filtered floor pairs** (the set existing floor π_θ `jm11qoch` trained on). Confirm π_θ ↔ 928 IDs. | block E1 |
| S3 | **Gate G3:** raw per-sample **E_Rep** for all K=8/GT candidates is loadable | `GO-E2` else `FALLBACK-E2` (regenerate E_Rep via PyRosetta on K=8/GT, ~1–2 h) |
| S4 | Confirm `ref_nll_margin` code path can emit **per-channel** margins (rot/pos/seq) | needed for E0/E1 logging |
| S5 | GPU inventory + tier dispatch | — |

**Emit `GO`.** Nothing GPU-bound starts before it.

---

## 5. Experiments

### E0 — Per-channel reference-margin decomposition on the REAL training pairs
**Type:** analysis (cheap GPU/CPU) · **Est:** 1–2 h · **First / decision-relevant** · this is reviewer 1's "checkable in an afternoon before you commit."

- **Input:** the 928 filtered floor pairs (winner = GT crystal, loser = DiffAb sample), floor π_ref.
- **Action:** per pair, compute `m_c = β·[log π_ref(y_w) − log π_ref(y_l)]_c`, averaged over the 20 diffusion timesteps used for `ref_nll_margin`, for `c ∈ {rot, pos, seq}`. ⚠ reuse the `ref_nll_margin` path, emitting per-channel instead of all-channel-summed. **Run on all 928** (not the 200-sample bathtub budget).
- **Compare** to bathtub t=0 (r_rot +2.606, r_pos +0.125, r_seq −0.226).
- **Decision rule:**
  - `median(m_seq)` small (≈ the −0.2 ballpark) → seq channel carries little membership signal → **E1 will likely be flat; do not expect collapse; interpret E1 as confound-closer.** (Expected.)
  - `median(m_seq)` surprisingly large → E1 becomes genuinely informative; proceed with the stronger reading.
- **Output:** `figs/per_channel_ref_margin_real_pairs.*`, three median margins + distributions.

### E1 — Matched seq-only GT-vs-decoy control (channel-scope confound-closer)
**Type:** infer + train · **Est:** decoy-gen <1 h + 1 run ~2 h · closes the "principal=seq-only but existing decoy=all-channel" confound (reviewer 3) and the "409-vs-928 pair-count" confound (reviewer 4).

- **Canonical IDs:** the 928 (S2). **Both arms use the identical 928 IDs.**
- **E1-A (GT winners):** = existing floor π_θ `jm11qoch`. **Reuse**; pull existing eval (Table 4.4: AAR 49.3/29.7/25.1; modal-match 85.7/50/27.8; motif `YCAAAGGGVYDYPYTYDY`). No run.
- **E1-B (t=1 decoy winners):** regenerate decoys for **all 928 IDs, NO PARTIAL-window filter** (the §4.11 run kept only 409 — that is the confound). Seq-only, β=0.05, fixed budget. 1 run.
- **E1-B′ (t=4 decoy):** optional, **only if E1-B finishes fast.** Same IDs. **Do not run t=20** (reviewer 4: lower value).
- **Sanity:** E1-B decoys' per-channel margin must trace the bathtub left wall (rot drops toward ~−0.22 at t=1). If not → decoy gen mis-wired → `BLOCKED`.
- **Log:** per-channel ref margin, grad-norm trajectory, `theta_drift_final_l2`, light eval.

### E2 — Matched-manifold sample-vs-sample rescue (the one new experiment with an informative null)
**Type:** train · **Est:** pre-check (CPU) + 1 run ~2 h · **Depends on G3.**

- **Pre-check (gate):** within-GT **E_Rep spread** across the K=8 samples per GT. Report median range → `results/within_gt_spread.csv`. If degenerate (winner≈loser), flag and switch to top-2-vs-bottom-2 pooled across GTs, or note the limitation.
- **Pairing:** per GT, winner = **lowest-E_Rep** π_ref sample, loser = **highest-E_Rep** π_ref sample. Optionally add pairs with E_Rep gap > threshold, capped per GT. **Both winner and loser are π_ref samples** → real-vs-synthetic gap removed. Report `n_pairs`.
- **Run:** seq-only, β=0.05, fixed budget (extend to 1500 if descending), seed 42.
- **Honest scope (write this verbatim into the thesis):** removes the membership gap; **does not** break energy=quality (E_Rep is still the ranker; no orthogonal axis available — PSH coin-flip, E_cdr collinear). A null is **ambiguous** across {no-shortcut, weak-oracle, too-few-pairs, E_Rep-misaligned}. A positive is valuable but **n=29 → frame as a promising rescue, not a solved design method.**
- **FALLBACK-E2 (no raw E_Rep):** regenerate E_Rep on K=8/GT, or fall back to GT-vs-sample re-ranking by E_Rep (weaker; document the compromise).

---

## 6. Interpretation gates (decide against this table; do not extrapolate beyond it)

**E1 (seq-only GT vs decoy):**
| Outcome | Licenses |
|---|---|
| A learns, B crystallizes (small update) | seq-only learnable signal was the GT-residue-vs-synthetic contrast → promote, but frame as residue-identity, **not** noise-signature membership (see caveat). |
| **Both small / both crystallize** *(likely, if E0 shows small m_seq)* | seq channel has no large signal to remove → "shortcut is **structural**, not sequence; seq-only null is margin-sharpening (§5.1.5)." **Closes the channel-scope confound.** Defensible refinement, not an overturn. |
| B learns, sample metrics flat | shortcut isn't the whole story (unlikely given small m_seq). |

**E2 (sample-vs-sample):**
| Outcome | Licenses |
|---|---|
| AAR/modal-match move beyond noise | quality **can** transmit once membership removed → best outcome; **n=29 humility**; "promising rescue." |
| No-op | localizes bottleneck **below** the membership confound at this data scale → keep as limitation/preliminary diagnostic (ambiguous per scope note). |

**Caveat threaded through both:** the headline is **no longer** "the DPO update was the membership shortcut." E0 relocates the shortcut to the structural channels; the all-channel decoy crystallization (Table 4.11) already evidences that the structural signal is membership-dominated — **modulo** the t=1-in-PARTIAL-window ambiguity (winner≈loser trivially vs signal-was-shortcut), which E0 + E2 disambiguate rather than a new all-channel run.

---

## 7. Rigor-regardless tasks (do these whatever the gate says — parallel, independent)

1. **Cite D-Fusion** — Hu, Zhang, Kuang, *ICML 2025*, **PMLR 267:24869–24892**, arXiv:**2505.22002**. Frame your shortcut as the **structural-domain analogue** of their visual-inconsistency problem, and your decoy as **conceptually analogous** to their matched-consistent-sample construction. **Note the mechanism differs:** D-Fusion uses mask-guided self-attention fusion; you use forward-noise-then-π_ref-denoise. They use it as a **working fix**; your contribution is the **diagnostic + the bathtub instrument**, not the remedy. Distinguish **DeDPO** (already cited, arXiv:2602.06195) as targeting synthetic-annotator bias — related but distinct. **Remove every "novel failure mode"/"discovery" phrase.**
2. **Per-CDR AAR bootstrap CIs** — §5.8/A.8 flag these missing. Extend ⚠ `scripts/regenerate_v2_bootstrap_cis.py` to the AAR axis (needs per-CDR GT sequence labels under the v2 pipeline). Cheap rigor win an examiner will notice is absent. Independent of E0–E2.
3. **Terminology fix** — replace "iter-0 implicit reward" with "reference NLL margin under π_ref" throughout new analysis; add one sentence noting π_θ=π_ref ⇒ implicit reward ≡ 0 at start, so the reference margin is the load-bearing quantity. Audit existing thesis usage.

---

## 8. Parallel timeline (one focused day)

```
h0.0–0.5  ┃ Setup barrier (S1–S5) ──────────────────────────────► GO
h0.5      ┃ ┌ E0 decomposition on 928 real pairs (cheap) .......... (done ~h2.0)  [DECISION-RELEVANT]
          ┃ ├ E1-B: decoy-gen t=1 for 928 (no filter) ──► E1-B train (seq-only) ── ~h2.5
          ┃ ├ E2: within-GT spread pre-check ──► pairing ──► E2 train ──────────── ~h2.5
          ┃ ├ Rigor #2: AAR bootstrap CIs (CPU+inference) ........................ (rolling)
          ┃ └ Rigor #1/#3: D-Fusion cite (verified) + terminology (text) ......... (anytime)
h0.5–2.5  ┃ training runs in parallel, fixed 1000 iters
h2.0–3.0  ┃ light gate-eval (AAR + modal-match + H3 motif + homopolymer + margin/θ-drift) per run
h3.0–4.0  ┃ read E0 + gate-evals against §6 ──► DECIDE
          ┃   └ if a checkpoint is clean/positive: queue FULL Phase 4 (scRMSD/CAAR/EpiF1) on THAT one only
          ┃ (E1-B′ t=4 slots in here only if E1-B was fast)
```
Critical path ≈ **3–4 h** at modest parallelism (2–3 GPUs suffice). Reframe writing is **not today** — it is Days 4–8 **and only if the gate opens.**

---

## 9. Eval policy

- **Gate-eval (all runs; sequence-side; fast, no folding):** per-CDR AAR, modal-match, H3 modal motif (L=8), homopolymer rate, per-channel reference margin, grad-norm/θ-drift. This is sufficient for the causal point (reviewer 4).
- **Full Phase 4 (ABodyBuilder2 scRMSD, CAAR, EpiF1):** **only** on a checkpoint that passes the §6 gate clean/positive. Reserve compute; v2 slicing; record `abb2_reject_count`.

---

## 10. Deliverables (end of day)

- [ ] `figs/per_channel_ref_margin_real_pairs.*` + the 3 median margins (E0) — **the decision input**
- [ ] E1-B (and optional t=4) checkpoint + gate-eval; E1-A numbers pulled from existing π_θ
- [ ] `results/within_gt_spread.csv` + E2 checkpoint + gate-eval
- [ ] `runs/*.json` (schema-valid, ref_margin fields)
- [ ] `results/master_results.csv` (all runs × gate-eval axes; + Phase 4 cols only where run)
- [ ] AAR bootstrap-CI table (rigor #2)
- [ ] `results/decision.md` — which §6 cell each experiment landed on + reframe go/no-go, evidence-cited
- [ ] D-Fusion citation drafted; "discovery" language removed; terminology fixed (text-only, can land same day)

---

## 11. Explicit cut list (do NOT run today)

Cross-antigen negatives · the circular membership-vs-energy regression · decoy depths beyond {0,1,(4)} · the elaborate energy-orthogonal membership classifier (optional stretch at most) · any abstract/intro rewrite · OAS-VHH SFT · full retitle around an "open problem."
