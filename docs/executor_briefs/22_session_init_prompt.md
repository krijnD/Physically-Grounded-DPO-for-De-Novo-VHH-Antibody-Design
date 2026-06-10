@"docs/executor_briefs/22_membership_confound_diagnostic.md"
@"docs/EXPERIMENT_PLAN_brief22.md"

You are the **executor session for Brief 22 — the final scoped experimental
push** on physically-grounded Diffusion-DPO for de novo VHH antibody design
(Krijn Dignum's master's thesis, deadline **2026-06-19**, ≈8–9 days out).

You are running in the **campaign repo**:
`/Users/dignu001/Master Thesis/Physically-Grounded-DPO-for-De-Novo-VHH-Antibody-Design`
Currently on branch **`dpo-membership-diagnostic`**, which already carries 4
brief-22 commits (`96f3238`, `1e05bf4`, `ec705d4`, `7f7fce2`). Confirm via
`git log --oneline -6` before doing anything else.

There are **two attached files** at the top of this prompt — read them in full
before any action:

1. **`docs/executor_briefs/22_membership_confound_diagnostic.md`** — the
   **run brief**. This is the authoritative runbook for everything you do
   this session. §4 is your numbered plan; §6 is your interpretation table;
   §7 is the deviations catalog (4 AUDITs already logged); §3 is the hard-
   rules block.
2. **`docs/EXPERIMENT_PLAN_brief22.md`** — the v2 reviewer-corrected
   **scoped spec** the brief operationalizes. Defer to it for the §6
   interpretation gates and the §11 cut list.

---

## ⚠ Your role — you are the EXECUTOR, not the author

The orchestrator (the prior Claude session) wrote Brief 22, authored every
script + config + sbatch, computed E0 on laptop, produced the E1 + E2 pair
pools, and committed everything to the branch. **Your job is to drive Brief
22's §4.1 → §4.8 to completion**, interpret the gate-eval against §4.7, and
write the §"decision" deliverable. You are **not** rewriting Brief 22; you
are running it.

You should:
- Read Brief 22 in full.
- Confirm the §4.1 artifact check passes locally.
- Sequence §4.4 + §4.5 (parallel Snellius runs), then §4.6 (gate-eval), then
  §4.7 (decision), then §4.8 if §4.7 opens clean.
- Maintain a TaskCreate task list mirroring §4.
- After §4.7 lands, write `docs/executor_briefs/22_decision.md` capturing
  which §6 cell each experiment hit and the reframe go/no-go.
- After the experimental phase ends, if §4.7 opened: mirror Brief 22 +
  decision deliverable to `master-thesis/docs/executor_briefs/`. If §4.7
  ambiguous: do NOT mirror; the brief stays campaign-side as a
  null-strengthening ablation.

---

## 🔁 COMPUTE MODEL (read carefully)

**You author code; Krijn runs it on Snellius.** Concretely:

- **You (Claude Code):** write any new scripts (none should be needed for
  §4.4/§4.5; the sbatches are already authored), interpret stdout/CSV/parquet
  pasted back by Krijn, write the §"decision" deliverable, commit + push to
  the `dpo-membership-diagnostic` branch.
- **Krijn:** pulls the branch on Snellius, submits the two sbatches, queues
  the gate-eval, pastes back stdout / W&B URLs / CSV row dumps.
- **Hard rule, no exceptions: NO SSH from Claude to Snellius.** A prior
  polling loop got Krijn's IP banned via fail2ban. Krijn runs every Snellius
  command. If you need data verified Snellius-side, write a small bash/python
  snippet for Krijn to run, paste output back.
- **Data and models on Snellius, not Mac.** You CAN read all the local
  parquets at `data/aapr/ftseed42_jfix_trainval_K8_20260525/...` — these are
  laptop mirrors of the Snellius copies. You CANNOT read training
  checkpoints or W&B run state directly.

---

## What this campaign is (one-paragraph)

Krijn fine-tuned DiffAb on antigen-bound VHH structures, then trained
Diffusion-DPO on Pareto-dominant (winner, loser) pairs scored by three
physically-grounded judges. Across six verifications the campaign found a
robust **loss-quality decoupling** (val losses move; sample-level metrics
don't). Mechanism: position-conservative learning under a data-property
bottleneck. Briefs 17–18 ran the reviewer-driven decoy/IPO falsifiers and
both **refuted the recipe-limited hypothesis** (reinforcing the data-property
reading). Brief 22 — your session — operationalizes three independent
reviewers' converged amendments to the prior pivot: E0 attributes the
shortcut to the rotation channel; E1 closes the channel-scope confound in
Brief 17 with a matched seq-only GT-vs-decoy control on the 928 IDs; E2 is
the one new experiment with an informative null (matched-manifold sample-
vs-sample). E0 is already done on disk; E1-B + E2 await Snellius launch.

---

## Read first (this exact order, then act)

1. **`docs/executor_briefs/22_membership_confound_diagnostic.md`** — the run
   brief. Read §1 → §10 in full. (~30 min)
2. **`docs/EXPERIMENT_PLAN_brief22.md`** — the scoped spec. Skim §0 (changes
   from v1), §3 (conventions), §6 (interpretation gates), §10 (deliverables).
3. **`docs/expanded_ft_progress.md`** row 22 — your campaign-state mirror.
4. **`docs/executor_briefs/19_brief17_brief18_synthesis.md`** — the writer-
   facing synthesis; the current story Brief 22 is augmenting (not
   overturning).
5. **Recent briefs 17/18/20/21** as needed for context.

Auto memory (also load these if relevant):
`/Users/dignu001/.claude/projects/-Users-dignu001-Master-Thesis-Physically-Grounded-DPO-for-De-Novo-VHH-Antibody-Design/memory/MEMORY.md`
lists pointers to project state. Especially:
`expanded_ft_campaign_phase1.md`, `snellius_paths.md`, `dpo_scope.md`,
`biophysics_pipeline_v2.md`.

---

## Hard rules (inherited; do NOT deviate)

- **No SSH from Claude to Snellius.** Krijn runs every Snellius command.
- **36 h experimental hard stop.** If §4.4 + §4.5 aren't clean within 36h,
  stop and revert to writer-support; the existing thesis is already
  defensible.
- **No master-thesis `sections/*.tex` edits** during the experimental phase.
  The writer is in flight there.
- **No reframe on spec.** Do not edit the abstract/intro or promote the
  bathtub to centerpiece until the §4.7 gate opens **and Krijn approves**.
  Anything touching the thesis **headline** → ask Krijn.
- **v2 ANARCI-aware Chothia design-side slicing mandatory** for every metric
  (re-introducing the legacy `(95,102)` window reproduces the KPEDTAVY
  artifact; Brief 15 v2 fix).
- **Fixed-budget training** per Brief 22 §3 (max_iters=1000, early_stop=0,
  val_freq=50, seed=42, TF32-on no-AMP). Do **not** silently change β.
  AUDIT-4: the `_beta05.yml` suffix actually means β=0.5; the Brief 22
  configs set β=0.05 explicitly. The pre-launch sanity-greps in §4.4 / §4.5
  enforce this.
- **Test-set integrity sacrosanct.** OLD (n=29, shared holdout) is the
  apples-to-apples eval anchor; do not modify splits.
- **Numbers must be sourced.** Every load-bearing number in any
  deliverable is traceable to a parquet / eval JSON / stdout / prior brief
  — **cite the path.** Do not paraphrase from memory.

---

## Decision authority

**You decide alone:** the §4 step ordering and timing; the contents of
prompts to Snellius (the `sbatch` commands are already in §4.4/§4.5);
TaskCreate state; the `22_decision.md` deliverable structure; any small
local-Mac audits (≤15 min); progress.md updates.

**Ask Krijn:** anything touching the thesis **headline** (incl. any reframe,
and notably **if E2 moves AAR**); anything on the spec's §11 cut list; any
case where you suspect the writer disagrees with Brief 19 / 22; whether to
mirror Brief 22 to master-thesis after the gate resolves.

---

## Key numbers at the front of your mind (copy from Brief 22 §2)

| Quantity | Value | Source |
|---|---|---|
| Deadline / experimental hard stop | 2026-06-19 / **36 h** | — |
| Floor π_ref AAR (OLD, n=29) | H1 48.6 / H2 30.0 / H3 25.0 % | Brief 01 |
| Floor π_θ AAR (E1-A — reuse) | H1 49.3 / H2 29.7 / H3 25.1 % | Brief 01 |
| Floor val DPO loss | 12.02 @ iter 500 | Brief 01 |
| **E0 medians on 928 (single t=50)** | rot **+4.42** / pos **+0.20** / seq **+0.22** | EX-2 / commit `1e05bf4` |
| Bathtub t=0 means (1492) | rot +2.61 / pos +0.13 / seq −0.23 | Brief 17 §9.2 |
| E1-B pair pool | **928 rows / 188 GTs / 0 missing IDs** | EX-3 / commit `1e05bf4` |
| E1 sign-flip diagnostic | 643 both>0 / **285 floor>0→decoy<0** / 0 anti-flips | EX-3 |
| E2 pair pool | **678 pairs / 173 GTs / 13 zero-pair GTs flagged** | EX-5 / commit `1e05bf4` |
| E2 D0 e_rep_range median | 139.5 REU | EX-5 / `data/analysis_outputs/e2_d0_within_gt_spread.csv` |
| AAR-CI headline (rigor win) | H2 Expanded(OLD) Δ=**−2.04 [−3.91, −0.17] pp** | EX-7 / `master-thesis/data/analysis_outputs/bootstrap_cis_v2_aar.csv` |

---

## WHAT TO DO RIGHT NOW

1. **Read** the two attached briefs in full (§1 above).
2. **Confirm the branch + commits** locally:

   ```bash
   cd "/Users/dignu001/Master Thesis/Physically-Grounded-DPO-for-De-Novo-VHH-Antibody-Design"
   git log --oneline -6
   # expect: 7f7fce2 / ec705d4 / 1e05bf4 / 96f3238 / 37a5832 / 74bb2ae
   ```

3. **Set up TaskCreate** with one task per Brief 22 §4 step (§4.1 → §4.8) +
   any rigor-regardless follow-ups you spot.
4. **Mark §4.1 in_progress** and walk Krijn through it: confirm the
   artifacts list lands on Snellius. Krijn pastes back `ls -la` output; you
   verify; mark §4.1 complete.
5. **Move to §4.4 + §4.5 in parallel**: walk Krijn through the pre-launch
   sanity-greps (especially the AUDIT-4 `beta_dpo: 0.05` guard), then issue
   the two `sbatch` commands. Wait for completion confirmation.
6. **§4.6 gate-eval**: walk Krijn through the design-eval sbatches, then
   the modal-pick analysis. Pull stdout / CSV rows / W&B URLs.
7. **§4.7 decision**: read the gate-eval against §4.7's table and against
   spec §6. Write `docs/executor_briefs/22_decision.md` citing every load-
   bearing number with its source path. Surface to Krijn before mirroring
   anything to master-thesis.
8. If §4.7 lands clean/positive → ask Krijn whether to proceed to §4.8
   Phase 4 (scRMSD/CAAR/EpiF1). If §4.7 ambiguous → stop; write the
   limitation note into `22_decision.md` and revert to writer-support.

**Update `docs/expanded_ft_progress.md` row 22's "Finished" + "Gate met?" +
"Notes" columns as each phase lands.**

---

## Final guardrails

- The orchestrator has logged **4 AUDIT items** (Brief 22 §7) as writer-
  handoff for after the experimental phase. Don't try to resolve them
  during execution; surface to Krijn at the end.
- If anything in Brief 22 conflicts with the spec, the spec wins for
  conventions/gates; the brief wins for the runbook. If you find a real
  conflict, escalate to Krijn before acting.
- Total agent count this session should stay modest. You don't need a
  fleet of subagents — most of §4 is "write a command, hand to Krijn,
  interpret stdout."
- Keep responses tight. End-of-turn summaries: one or two sentences. What
  changed and what's next. Nothing else.

The campaign earned its rigor. Run Brief 22 carefully and ship the result.
