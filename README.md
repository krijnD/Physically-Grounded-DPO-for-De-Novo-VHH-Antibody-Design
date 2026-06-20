# Physically-Grounded DPO for De Novo VHH Antibody Design

> **MSc Artificial Intelligence thesis — University of Amsterdam**
> *Diagnosing the Limits of Preference Optimization for In-Silico VHH Antibody Design*
> Krijn Felipe Dignum · Supervisor: dr. Monica Rotulo · University supervisor: Cong Liu · Examiner: dr. Katja Rogers

---

## TL;DR

Structure-based generative models for antibody design produce candidates that pass model-likelihood
checks but fail **orthogonal physical validation** — the *designability gap*. This thesis tests,
**entirely in silico**, whether Direct Preference Optimization (DPO) on physically-grounded preference
pairs narrows that gap for single-domain camelid antibodies (**VHHs / nanobodies**).

The pipeline fine-tunes **DiffAb** on a curated VHH+antigen manifest, mines preference pairs with
**three orthogonal judges** (biology, biophysics, physics), and trains a **Diffusion-DPO** policy on
strict-Pareto-dominant (winner, loser) pairs.

**Central finding — a four-axis loss–quality decoupling.** Diffusion-DPO lowers its proxy loss
(2.7–3.7 %) but moves **no** sample-level quality metric beyond the test-set noise floor — not
amino-acid recovery (AAR), self-consistency designability (scRMSD), contact-restricted recovery
(CAAR), or epitope F1. Neither does expanding the fine-tuning manifest (a matched shared-holdout tie,
Δ ≈ 0). The policy is **position-conservative**: it reproduces the modal training residue most on the
hypervariable H3 loop. Three sensitivity ablations (a β-sweep, an all-channel decoy variant, and an
IPO objective) rescue nothing and locate the failure **upstream of DPO**, in the preference signal
rather than the objective. The contribution is a **diagnostic methodology** that surfaces this
mechanism while ruling out the obvious methodological explanations.

This repository contains the full pipeline: data curation, reference-policy fine-tuning, the
three-judge AAPR loop, Diffusion-DPO training, and the field-positioning evaluation stack.

---

## The four-phase pipeline

```
┌─ PHASE 1 ─ Reference-policy fine-tuning ────────┐   ┌─ PHASE 2 ─ AAPR hard-negative mining ───────────────┐
│  DiffAb codesign_multicdrs.pt                   │   │  For each GT VHH+antigen complex, sample K=8        │
│  fine-tuned on curated VHH+antigen data,        │   │  candidates from π_ref (mask H1+H2+H3)              │
│  multi-CDR masking objective.                   │   │       │                                             │
│   • floor   manifest  =  465 ANDD entries       │   │       ▼  three orthogonal judges                    │
│   • expanded manifest =  911 ANDD+SAbDab        │   │   Biology · Biophysics (TNP) · Physics (Rosetta)    │
│                  │                              │   │       │  strict Pareto dominance (winner=GT ≻ loser)│
│                  └──────────► π_ref ◄───────────┼───┤       │  + reference-margin filter                  │
└─────────────────────────────────────────────────┘   │       ▼                                             │
                                                       │   (winner = GT, loser = sample) preference pairs    │
                                                       └──────────────────────┬──────────────────────────────┘
                                                                              │
┌─ PHASE 3 ─ Diffusion-DPO ───────────────────────┐   ┌─ PHASE 4 ─ Field-positioning evaluation ────────────┐
│  Train π_θ against frozen π_ref on the           │   │  Re-evaluate every checkpoint on:                   │
│  Pareto pairs.                                   │   │   • AAR + Cα-RMSD          (per-CDR sample metrics)  │
│   • β = 0.05, sequence-only channel              │──►│   • TNP developability    (Green/Amber/Red zones)   │
│   • per-residue AbDPO direct-energy loss         │   │   • scRMSD designability  (vs ABodyBuilder2)         │
│  Ablations: β-sweep · all-channel decoy · IPO    │   │   • CAAR + EpiF1          (ChimeraBench)             │
└─────────────────────────────────────────────────┘   │   • per-position modal-pick reconciliation          │
                                                       └─────────────────────────────────────────────────────┘
```

The four methodological commitments that distinguish this work from prior antibody-DPO (AbDPO,
Zhou et al. 2024) are: **structure-direct generation** (DiffAb emits coordinates, not sequences),
**three orthogonal judges** over one composite reward, **Pareto-dominant** pair selection over a
scalarized energy reward, and **Diffusion-DPO** over supervised fine-tuning.

---

## Phase 1 — Fine-tuning the reference policy on VHH data

The DPO reference policy `π_ref` must encode a usable prior over the design space. The published
AbDPO recipe initializes `π_ref` from DiffAb's `codesign_multicdrs.pt`, pretrained on the
paired-heavy/light-dominated SAbDab corpus. That prior is weakly specialized for the VHH-specific
solvent-exposed Framework Region 2 (the YERL hallmark motif), the longer CDR-H3 conformations, and the
absent inter-domain packing geometry of camelid nanobodies. Phase 1 fine-tunes the upstream checkpoint
on curated, single-domain VHH data to isolate VHH-specific geometry.

**Two production manifests, a deliberate data-scale ablation:**

| Manifest | Entries | Source | Best EMA val neg-ELBO | π_ref role |
|---|---|---|---|---|
| **floor** | 465 | ANDD-only (post-DiffAb-cutoff) | 0.7316 | head-to-head comparison baseline |
| **expanded** | 911 (of 927) | ANDD + SAbDab after cross-source dedup | **0.6363 (−13 %)** | production reference policy |

The expanded manifest reduces the per-model validation negative-ELBO (the diffusion denoising loss;
lower is better) by **13 %**. Crucially, a standardized re-evaluation of both checkpoints on a shared
29-entry holdout under identical RNG and masking **ties the two** (Δ ≈ 0): the 13 % is a per-partition
training statistic, not a generalization win. Both checkpoints feed the downstream head-to-head
comparison so the only varying factor is the fine-tuning manifest.

**Scope of `π_ref` — multi-CDR masking.** Fine-tuning uses a multi-CDR masking objective (a
uniformly-random subset of {H1, H2, H3} masked per example) rather than the CDR-H3-only objective used
in AbDPO. The multi-CDR scope matches the downstream judges: TNP's surface metrics are computed
globally, the Rosetta repulsion term scores whole-structure interactions, and the Biology Judge's
framework checks operate outside CDR-H3. (The CDR-H3-only scope is retained as the `seed42_v2`
ablation.)

The five-run diagnostic arc that selected `π_ref` — `seed42` (diverged at Adam cold-start) →
`seed42_v2` (AbDPO-replication, CDR-H3-only) → `seed42_v3` (multi-CDR) → **`seed42_jfix`** (floor) →
**`seed42_jfix_expanded`** (expanded production) — is reproduced via `scripts/diffab_ft/`.

- **Training:** [`scripts/diffab_ft/train.py`](scripts/diffab_ft/train.py) — thin DiffAb trainer adding EMA shadow weights, early-stop on EMA val ELBO, top-K checkpointing, warmup, W&B logging.
- **Data prep:** [`scripts/diffab_ft/prepare_manifest.py`](scripts/diffab_ft/prepare_manifest.py), [`clean_manifest.py`](scripts/diffab_ft/clean_manifest.py), [`cluster_split.py`](scripts/diffab_ft/cluster_split.py) — MMseqs2 cluster-based train/val/test splits at 70 % CDR-concatenated identity.
- **Diagnostics:** [`export_wandb_run.py`](scripts/diffab_ft/export_wandb_run.py) + [`summarize_run.py`](scripts/diffab_ft/summarize_run.py) — export → summarize a finished run into a markdown diagnostic (see [run-diagnostics](#diffab-fine-tune-run-diagnostics) below).
- **Config:** [`configs/diffab_ft/vhh_ft.yml`](configs/diffab_ft/vhh_ft.yml) (floor), [`vhh_ft_expanded.yml`](configs/diffab_ft/vhh_ft_expanded.yml) (expanded).

---

## Phase 2 — AAPR: adversarial hard-negative mining with three judges

The **AAPR** (Antigen-Aware Pareto Refinement) loop converts `π_ref` into a stream of (winner, loser)
preference pairs. For each ground-truth complex it masks H1+H2+H3 and samples **K = 8** candidate
structures from `π_ref` (210 GT complexes × 8 = 1680 candidates, seed 42). Each candidate is
side-chain-packed and scored by the three judges in parallel; the GT serves as the candidate winner.
A candidate the model was confident in but a judge rejected is a **hard negative**.

- **Sampling:** [`scripts/aapr/sample_candidates.py`](scripts/aapr/sample_candidates.py) — masks via `MaskMultipleCDRs`, runs DiffAb reverse diffusion on flagged CDR residues only (antigen coordinates preserved end-to-end), writes a `candidates.csv` manifest.
- **Masking module:** [`src/masking/`](src/masking/) — four strategies (PARATOPE, CDR_FOCUSED for positive candidates; FR2_REVERSION, UNANCHORED_CLASH for hard negatives) with Kabat mapping and anchor/FR2 protection.

### Judge 1 — Biology (VHH hallmark grammar + localized SAP)

Rejects sequences that violate the VHH-specific hallmark-residue grammar. The absence of a light chain
in VHHs exposes Framework Region 2 (FR2) to solvent; the **YERL motif** (Tyr/Phe-37, Glu-44, Arg-45,
Gly/Leu-47, Kabat) compensates. A deterministic ANARCI pass emits flags, resolved by a **localized
Spatial Aggregation Propensity** check:

- `W47` → **absolute reject** (exposed indole ring drives irreversible aggregation; no 3D analysis).
- `V37`, `G44`, `L45`, `W99` → **conditional flags**, resolved by SAP_loc on the flagged residue's Cα.

SAP_loc weights each neighbor's Shrake-Rupley SASA (within 10 Å) by Black & Mould normalized
hydrophobicity, bounded ∈ [−1, +1]. **SAP_loc > 0.15 → fail**; threshold from the aggregation-propensity
literature (Chennamsetty 2009, Sankar 2018), fixed *before* any AAPR data is generated to avoid
circularity. On the natural ANDD distribution all four flag-position 80th percentiles are negative, so
the threshold is correctly dormant and does not over-reject natural VHHs.

**Code:** [`src/biology_judge/`](src/biology_judge/) — `sequence_filter.py` (Kabat numbering + flags),
`sap_calculator.py` (localized SAP), `judge.py` (orchestrator).

### Judge 2 — Biophysics (Therapeutic Nanobody Profiler)

Evaluates clinical developability via three continuous surface metrics from the
[Therapeutic Nanobody Profiler (TNP)](https://github.com/oxpig/TNP) (Gordon et al. 2026), calibrated on
36 clinical-stage nanobody therapeutics. TAP (the paired-antibody profiler) is **not** used: its
hydrophobicity/charge bands assume a light-chain-shielded FR2 and misclassify viable VHHs.

| Metric | Clinical band | Rejection |
|---|---|---|
| **PSH** — Patch of Surface Hydrophobicity | 79.59 – 126.83 Å² | outside band |
| **PPC** — Positive Patch Charge | < 0.39 | > 0.39 |
| **CDR3 Compactness** — L_CDR3 / R_max | 0.81 – 1.57 | outside band |

TNP's metric functions are called **directly on the side-chain-packed DiffAb structure** — TNP's CLI
re-folds the input with NanoBodyBuilder2, but bypassing that preserves the geometry DiffAb actually
generated (and avoids a 49.6 % NB2 coverage loss observed in a pilot). DiffAb is backbone-only, so a
deterministic PyRosetta side-chain pack (`RestrictToRepacking`, no design/backbone motion) is inserted
at sample time to prevent a +58 Å² PSH false-positive inflation.

**Code:** [`src/biophysics_judge/`](src/biophysics_judge/) — `pdb_utils.py` (VHH chain → clean
monomer), `tnp_direct.py` (score via theraprofnano, no NB2 re-fold), `judge.py` (threshold checks).

### Judge 3 — Physics (Rosetta energy decomposition)

Evaluates thermodynamic viability via Rosetta's `ref2015` all-atom energy, decomposed into a repulsive
term (clash detector) and a non-repulsive binding-quality term — blocking the two AbDPO-documented
reward-hacking modes (low-energy non-binders and atom-overlap pseudo-attraction).

| Metric | Rejection | Meaning |
|---|---|---|
| **E_Rep** (Lennard-Jones 6-12 repulsive) | > **+3.271 REU** | steric clash / "physical hallucination" |
| **E_cdr** (per-CDR-residue total energy) | > **+2.844 REU/residue** | weak binder / "rock" |

Thresholds are **empirically calibrated** on the natural VHH distribution following AbDPO's actual
Appendix E.1 methodology (80th percentile of the Physics scalars on the 465-entry ANDD ground-truth
set), **not** literature-imported values. A key finding of the calibration: the same corpus yields
three different 80th-percentile thresholds under three refinement modes (`none`, `pack_cdrs`, `full`)
spanning ~10 REU/residue — a refinement-regime ambiguity in AbDPO's published thresholds. The operative
regime is **`refinement_mode=none`** (score what DiffAb actually produced). Under that regime raw DiffAb
backbones sit orders of magnitude above the threshold, so the Physics judge passes **0 %** of AAPR
candidates and is non-discriminative — the architecture operationally collapses to two-judge eligibility
plus single-axis E_Rep ranking.

**Code:** [`src/physics_judge/`](src/physics_judge/) — `rosetta_scorer.py` (PyRosetta wrapper:
constrained FastRelax, E_Rep, per-residue CDR energy), `judge.py` (thresholds). Calibration:
[`scripts/calibration/`](scripts/calibration/), thresholds centralized in
[`src/common/config.py`](src/common/config.py).

### Pair selection — strict Pareto dominance + reference-margin filter

Pairs are selected by **strict Pareto dominance** on the three judges' raw metric vectors
(PSH-outside-zone, E_cdr, E_Rep) rather than a weighted sum — a candidate cannot game the objective by
trading off a low-weighted axis. A GT winner `y_w` and AAPR loser `y_l` form a valid pair iff `y_l`
fails ≥ 1 judge **and** `y_w` strictly Pareto-dominates `y_l`. A second **reference-margin filter** then
drops pairs that `π_ref` already orders the wrong way (negative `ref_nll_margin`).

| AAPR funnel | Floor | Expanded |
|---|---|---|
| Total candidates (210 GT × K=8) | 1680 | 1680 |
| Biology pass | 99.5 % | 99.5 % |
| Biophysics pass | 34.8 % | 36.5 % |
| Physics pass | 0.0 % | 0.0 % |
| Pareto-accepted pairs | 1492 | 1377 |
| **After reference-margin filter** | **928** | **809** |

- **Pair selection:** [`scripts/dpo/select_pareto_pairs.py`](scripts/dpo/select_pareto_pairs.py)
- **Reference-margin filter:** [`scripts/dpo/filter_pairs_by_ref_margin.py`](scripts/dpo/filter_pairs_by_ref_margin.py)

---

## Phase 3 — Diffusion-DPO

Phase 3 trains a policy `π_θ` to prefer GT winners over Pareto-dominated AAPR losers, adapting Direct
Preference Optimization (Rafailov 2023) to a diffusion model via the per-timestep Diffusion-DPO
formulation (Wallace 2024) on the energy-decomposed AbDPO loss (Zhou 2024). The implementation departs
from AbDPO in two ways: strict Pareto dominance over three judges replaces the scalarized reward, and
the loss is **restricted to the sequence channel** (residue-type logits) — rotation and centroid-position
channels are held off the DPO gradient.

**Operative configuration:** β = 0.05, sequence-only loss, per-residue (AbDPO Eq. 8) aggregation,
20 ground-truth validation holdouts at seed 42.

| Pipeline | Val DPO loss | vs iter-1 baseline (~12.48) |
|---|---|---|
| floor | 12.02 @ iter 500 | −3.7 % |
| expanded | 12.15 @ iter 300 | −2.7 % |

- **Trainer:** [`scripts/dpo/train_dpo.py`](scripts/dpo/train_dpo.py) — loads π_θ (trainable) + π_ref (frozen), forks the EMA / early-stop / checkpoint scaffolding from the Phase-1 trainer.
- **Loss:** [`src/dpo/loss.py`](src/dpo/loss.py) (AbDPO per-residue direct-energy loss), [`src/dpo/dataset.py`](src/dpo/dataset.py) (pair-aware dataset with aligned masks / shapes across winner & loser).
- **Configs:** [`configs/dpo/vhh_dpo_seqonly_filtered.yml`](configs/dpo/vhh_dpo_seqonly_filtered.yml) and β / channel-scope variants.

### Robustness ablations

| Ablation | Config | Outcome |
|---|---|---|
| **β-sweep** | `vhh_dpo_seqonly_filtered_beta0005.yml`, `_beta05.yml` | β ∈ {0.05, 0.5} preserve position-conservative behaviour; **β = 0.005 → 99.7 % Isoleucine-homopolymer collapse** (reward hacking under unbounded low-β). |
| **All-channel decoy DPO** | [`sample_decoy_winners.py`](scripts/dpo/sample_decoy_winners.py) + `vhh_dpo_allchannel_decoy_t1_*.yml` | Replaces each GT winner with a noise-matched decoy (forward+reverse diffusion to depth t) to dismantle the GT-vs-synthetic shortcut, and activates rot/pos/seq channels. β = 0.005 reaches the **lowest** val loss (10.63) at the **worst** H3 AAR (5.0 %, RMSD ~2600 Å) — an active inverse relationship. |
| **IPO** | [`src/dpo/loss_ipo.py`](src/dpo/loss_ipo.py) + `vhh_ipo_*.yml` | Bounded squared-margin objective (Azar 2023). β = 0.005 collapses to the **same** Ile-homopolymer as DPO → the failure is in the preference signal, not the log-sigmoid. |

The all-channel diagnostic also localizes the GT-vs-synthetic preference signal: the per-channel
reference-NLL margin is carried almost entirely by the **rotation (backbone-frame) channel**
(median +4.4) vs a negligible sequence channel (median +0.2). The sequence-only DPO objective cannot
access most of the signal — the membership cue is a structural likelihood gap, not a sequence-identity
one.

---

## Phase 4 — Field-positioning evaluation stack

A re-evaluation of the existing checkpoints (no new fine-tunes, no new DPO) against 2024–2026 SOTA
evaluation conventions, layering four sample-level axes on top of AAR and Cα-RMSD. For each of the four
model variants (`π_ref^floor`, `π_θ^floor`, `π_ref^exp`, `π_θ^exp`) it persists K = 4 design samples per
CDR per test entry (3384 design PDBs total).

| Axis | Metric | Script |
|---|---|---|
| Sequence recovery | per-CDR **AAR** + **Cα-RMSD** | [`scripts/diffab_ft/evaluate.py`](scripts/diffab_ft/evaluate.py) |
| Clinical developability | **TNP** Green/Amber/Red zone occupancy | [`scripts/eval/compute_interface_dG.py`](scripts/eval/compute_interface_dG.py), `build_master_parquet.py` |
| Self-consistency designability | **scRMSD** < 2 Å vs ABodyBuilder2-predicted backbone | [`scripts/eval/run_scrmsd_array.py`](scripts/eval/run_scrmsd_array.py), `join_scrmsd_into_master.py` |
| Antigen-contact specificity | **CAAR** (contact-restricted AAR) + **EpiF1** (ChimeraBench) | [`scripts/eval/run_caar_epif1_array.py`](scripts/eval/run_caar_epif1_array.py) |
| Position-conservatism | per-position **modal-pick** reconciliation | [`scripts/eval/compute_per_position_modal_picks_v2.py`](scripts/eval/compute_per_position_modal_picks_v2.py) |

**Headline results (the four-axis decoupling):**

- **AAR:** every floor→expanded and π_ref→π_θ delta within ±1.2 pp on every CDR — DPO is a sample-level no-op.
- **TNP developability:** all four variants 37.1–50.1 % Green vs 80.9 % for the natural-VHH calibration; DPO moves zone occupancy by ≤ 1.2 pp.
- **scRMSD designability:** 17.2–34.0 % across variants/splits; DPO movement within per-entry σ.
- **CAAR / EpiF1:** max |ΔCAAR| = 4.67 pp, max |ΔEpiF1| = 0.012 — both inside per-entry σ.
- **Position-conservative learning:** the model matches the GT modal residue at **86 % of H1, 50 % of H2, 28–58 % of H3** positions; the asymmetry tracks training-data density.
- **Statistics:** n = 29 shared holdout; minimum detectable effect ≈ 9 pp on H3 AAR (α = 0.05, power 0.80). Sub-10-pp DPO effects are not resolvable by this design.

Thesis figure-regeneration scripts: [`scripts/thesis/`](scripts/thesis/),
[`scripts/dpo/plot_decoy_t_sweep.py`](scripts/dpo/plot_decoy_t_sweep.py).

---

## Datasets

### 1. ANDD (Antibody and Nanobody Design Dataset)

**Source:** [Zenodo – ANDD_pdb.zip](https://zenodo.org/records/18151718/files/ANDD_pdb.zip?download=1) ·
**Metadata:** `Antibody and Nanobody Design Dataset (ANDD)_v2.xlsx`

Starting from the full ANDD Excel, the filters keep `Ab_or_Nano == 'Nanobody/VHH'` and split by
structure availability:

| File | Sequences | Description |
|---|---|---|
| `ANDD_VHH_only.csv` | 30,119 | All VHH sequences |
| `ANDD_VHH_with_structure.csv` | 3,178 | VHH sequences with a known PDB structure |
| `ANDD_VHH_no_structure.csv` | 26,941 | VHH sequences without a structure |

**`Predicted_or_Not` audit.** ANDD v2's label column has three values: `real`, `predicted`, and `\`
(an unlabelled sentinel). Naïvely filtering `== "real"` drops **571 real PDBs** that are only labelled
`\`. Verified against RCSB: 559 resolve as current entries, 12 are obsoleted, zero are hallucinated. The
contamination check therefore keeps any row whose label is not explicitly `predicted`. Net candidate
pool entering DiffAb: **1,287** verified-real VHH structures (vs. 728 with the naïve filter).

| Directory | PDB files | Description |
|---|---|---|
| `All_structures/` | 8,214 | All structures from the ANDD bulk download |
| `VHH_structures/` | 1,261 | VHH-only structures |
| `VHH_structures_post_iglm/` | — | Subset deposited after the IgLM cutoff (2022-01-01) |
| `VHH_structures_post_diffab/` | — | Subset deposited after the DiffAb cutoff (2021-12-25) |

### 2. SAbDab Nanobody Dataset

**Source:** [SAbDab nanobody summary](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/nanobody/)

Starting from `sabdab_nano_summary.tsv` (2,422 entries), filters: **post-IgLM cutoff** (`date >=
2023-01-01`) and **high resolution** (`resolution <= 2.5 Å`, required for reliable PyRosetta E_Rep) →
**38 PDB files** downloaded from RCSB.

### Data-contamination check & curation (3-step)

When fine-tuning a generative model, structures deposited before the model's training cutoff may have
been seen during training and are unsuitable as ground-truth winners.

1. **[`fetch_deposition_dates.py`](data%20scripts/fetch_deposition_dates.py)** — queries RCSB for each
   PDB's `initial_release_date` (the ANDD `Update_Date` column is unreliable) and flags entries
   deposited after the cutoff. Cutoffs: **IgLM 2022-01-01**, **DiffAb 2021-12-25**.
2. **[`subset_vhh_structures.py`](data%20scripts/subset_vhh_structures.py)** — copies
   contamination-safe PDBs into a clean subset directory and produces a filtered metadata CSV.
3. **[`curate_andd.py`](data%20scripts/curate_andd.py)** — geometry-verifies each entry's VHH chain
   (ANARCI) and antigen chain(s) via contact geometry, overwriting the chain columns with
   structure-verified values. Notable fixes: a relaxed J-motif regex `[WRK]G[x]GT` (recovers complete
   VHH domains with a known-rare W→R/K FR4 substitution), comma-separated homodimer chain parsing, and
   exclusion of *all* VH-type chains from antigen scoring (prevents VHH-VHH / VHH-Fab mislabelling).
   Yield on the 561-row post-DiffAb slice: 465 curated, 36 `no_vhh`, 32 `ambiguous_vhh`, 28 `no_antigen`.

`diagnose_rejections.py` classifies why each rejected row failed; it informed the J-motif and homodimer
fixes above.

---

## Repository structure

```
src/
├── common/              # NanobodyCandidate dataclass, centralized thresholds, PDB utils, SAbDab loader
├── masking/             # AAPR masking: 4 strategies, Kabat mapper, paratope detector, engine
├── biology_judge/       # YERL hallmark grammar + localized SAP
├── biophysics_judge/    # TNP (PSH/PPC/Compactness) scored directly on DiffAb geometry
├── physics_judge/       # Rosetta ref2015 energy decomposition (E_Rep, E_cdr)
├── dpo/                 # Diffusion-DPO loss (AbDPO per-residue), IPO variant, pair-aware dataset
├── diffab_ft/           # Fine-tune support package
└── pipeline.py          # Multi-judge orchestrator: score one geometry, judge many → Parquet

scripts/
├── diffab_ft/           # Phase 1: data prep, cluster splits, trainer, W&B export/summarize
├── aapr/                # Phase 2: candidate sampling from π_ref
├── calibration/         # Phase 2: percentile threshold calibration
├── judges/              # Phase 2: refinement-mode pilots
├── dpo/                 # Phase 3: Pareto selection, ref-margin filter, decoy winners, DPO trainer, diagnostics
├── eval/                # Phase 4: scRMSD, CAAR/EpiF1, modal-pick, master-parquet assembly
├── thesis/              # Figure regeneration
├── analysis/            # Leakage / holdout audits
└── test_sabdab_judges.py  # End-to-end judge sanity test on SAbDab crystals

data scripts/            # ANDD/SAbDab download, filtering, contamination check, curation
configs/                 # YAML run configs (diffab_ft/, dpo/)
docs/                    # Research briefs, handoffs, calibration rationale
third_party/diffab/      # DiffAb submodule (luost26/diffab)
```

---

## Setup (Snellius)

### 1. Clone with submodule

```bash
git clone --recurse-submodules https://github.com/krijnD/Physically-Grounded-DPO-for-De-Novo-VHH-Antibody-Design.git
# or, if already cloned:
git submodule update --init --recursive
```

### 2. Python virtual environment

```bash
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

python -m venv /projects/0/hpmlprjs/interns/krijn/venvs/DPO
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate
pip install --upgrade pip
```

### 3. Install TNP (Therapeutic Nanobody Profiler) — Python package only

The pipeline uses `theraprofnano`'s metric functions (PSH, PPC, PNC, Compactness, CDR lengths)
directly on the DiffAb-generated structure. The TNP CLI and its NanoBodyBuilder2 fold-back are **not**
used.

```bash
cd /projects/0/hpmlprjs/interns/krijn/tools/
git clone https://github.com/oxpig/TNP.git && cd TNP && pip install .

python -c "
from theraprofnano.CDR_Profiler.CDR3_Conf_Assigner import main_compactness
from theraprofnano.Hydrophobicity_and_Charge_Profiler.Hydrophobicity_and_Charge_Assigner import CreateAnnotation
print('OK')"
```

### 4. Install DSSP (required by theraprofnano `CreateAnnotation`)

DSSP must be built from source on Snellius (needs GCC 13+ / C++20 and a recent SQLite, both built into
the venv).

```bash
module purge && module load 2024 && module load GCCcore/13.3.0

# 4a. Build SQLite from source (system version is too old for DSSP)
cd /projects/0/hpmlprjs/interns/krijn/tools/
wget https://www.sqlite.org/2024/sqlite-autoconf-3460000.tar.gz
tar xzf sqlite-autoconf-3460000.tar.gz && cd sqlite-autoconf-3460000
./configure --prefix=$VIRTUAL_ENV && make && make install

# 4b. Clone and build DSSP against the venv's SQLite
cd /projects/0/hpmlprjs/interns/krijn/tools/
git clone https://github.com/PDB-REDO/dssp.git && cd dssp
cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV -DCMAKE_PREFIX_PATH=$VIRTUAL_ENV \
  -DSQLite3_INCLUDE_DIR=$VIRTUAL_ENV/include \
  -DSQLite3_LIBRARY=$VIRTUAL_ENV/lib/libsqlite3.so
cmake --build build && cmake --install build
mkdssp --version
```

### 5. Install PyRosetta (Physics Judge + side-chain packing)

```bash
pip install pyrosetta \
  --find-links https://graylab.jhu.edu/download/PyRosetta4/archive/release-quarterly/release
```

### 6. Download datasets

```bash
wget -O ANDD_pdb.zip "https://zenodo.org/records/18151718/files/ANDD_pdb.zip?download=1"
wget https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/nanobody/ -O sabdab_nano_summary.tsv
```

---

## Reproducing the pipeline

```bash
# ── Phase 1: fine-tune the reference policy ──────────────────────────────
python scripts/diffab_ft/prepare_manifest.py   # build curated manifest
python scripts/diffab_ft/cluster_split.py       # MMseqs2 70% cluster splits
python scripts/diffab_ft/train.py --config configs/diffab_ft/vhh_ft_expanded.yml
python scripts/diffab_ft/export_wandb_run.py  runs/vhh_ft/seed42_jfix_expanded --out-dir .../wandb_export
python scripts/diffab_ft/summarize_run.py     .../wandb_export --out diagnostic.md

# ── Phase 2: AAPR hard-negative mining ───────────────────────────────────
python scripts/aapr/sample_candidates.py        # sample K=8 losers per GT from π_ref
python scripts/test_sabdab_judges.py --csv ANDD_VHH_with_structure.csv \
    --pdb-dir VHH_structures_post_diffab --score-biophysics --output candidates_scored.parquet
python scripts/dpo/select_pareto_pairs.py        # strict Pareto-dominant (y_w, y_l) pairs
python scripts/dpo/filter_pairs_by_ref_margin.py # drop pairs π_ref orders the wrong way

# ── Phase 3: Diffusion-DPO ───────────────────────────────────────────────
python scripts/dpo/train_dpo.py --config configs/dpo/vhh_dpo_seqonly_filtered.yml

# ── Phase 4: field-positioning evaluation ────────────────────────────────
python scripts/eval/build_design_manifest.py
python scripts/eval/run_scrmsd_array.py
python scripts/eval/run_caar_epif1_array.py
python scripts/eval/compute_per_position_modal_picks_v2.py
python scripts/eval/build_master_parquet.py      # assemble all axes into one parquet
```

Most compute steps run on Snellius via `sbatch` wrappers under each `scripts/*/slurm/` directory
(single NVIDIA A100 for sampling / training, CPU `rome` partition for PyRosetta + TNP scoring).

---

### DiffAb fine-tune run diagnostics

Standard workflow after a run finishes: **export → summarize**. Both scripts run wherever the training
venv lives — no web-UI clicking, no laptop round-trip.

```bash
# 1) Read the run's W&B history from disk → per-metric CSVs
python scripts/diffab_ft/export_wandb_run.py runs/vhh_ft/seed42_jfix_expanded \
  --out-dir runs/vhh_ft/seed42_jfix_expanded/wandb_export --include train/ val/

# 2) Reduce to a ~30-line markdown diagnostic (warmup detection, per-phase loss/grad-norm stats,
#    full val trajectory, best-val landmark). Stdlib-only.
python scripts/diffab_ft/summarize_run.py runs/vhh_ft/seed42_jfix_expanded/wandb_export \
  --compare-to runs/vhh_ft/seed42_jfix/wandb_export --out diagnostic.md
```

`export_wandb_run.py` auto-detects local mode (parses the binary `.wandb` log; works for online and
offline runs) vs cloud mode (any `wandb.ai` URL or `entity/project/run_id`).

### Testing the judges (`scripts/test_sabdab_judges.py`)

Runs the full multi-judge pipeline on real crystal structures. Three input modes:

```bash
# Mode 1 — SAbDab (TSV metadata)
python scripts/test_sabdab_judges.py --tsv sabdab_nano_summary.tsv \
  --pdb-dir filtered_vhh_pdbs --output data/results/sabdab_judge_test.parquet --score-biophysics

# Mode 2 — ANDD (CSV metadata; chains read automatically)
python scripts/test_sabdab_judges.py --csv ANDD_VHH_with_structure.csv \
  --pdb-dir VHH_structures_post_iglm --output data/results/andd_judge_test.parquet --score-biophysics

# Mode 3 — plain PDB directory (chains specified manually)
python scripts/test_sabdab_judges.py --pdb-dir /path/to/pdbs \
  --chain A --antigen-chain B --output data/results/custom_judge_test.parquet --score-biophysics
```

| Flag | Default | Description |
|---|---|---|
| `--tsv` / `--csv` | — | SAbDab TSV / ANDD CSV metadata (mutually exclusive with plain mode) |
| `--pdb-dir` | *(required)* | Directory containing PDB files |
| `--output` | `data/results/sabdab_judge_test.parquet` | Output Parquet path |
| `--limit` | — | Process only first N entries (quick sanity check) |
| `--score-biophysics` (alias `--run-tnp`) | off | Score PSH/PPC/PNC/Compactness on each PDB's VHH chain; enables the Biophysics Judge |
| `--chain` / `--antigen-chain` | `A` / — | Chain IDs (plain PDB directory mode only) |

---

## Key references

- **DiffAb** — Luo et al., *Antigen-Specific Antibody Design with Diffusion Models*, NeurIPS 2022 (upstream backbone)
- **AbDPO** — Zhou et al., *Antigen-Specific Antibody Design via Direct Energy-based Preference Optimization*, NeurIPS 2024 (closest prior work)
- **Diffusion-DPO** — Wallace et al., *Diffusion Model Alignment Using DPO*, CVPR 2024
- **DPO** — Rafailov et al., *Direct Preference Optimization*, NeurIPS 2023 · **IPO** — Azar et al., 2023
- **TNP** — Gordon et al., *Characterising nanobody developability*, Commun. Biol. 2026 · **ChimeraBench** — Ahmed et al., 2026
- **VHH hallmark residues** — Uto et al. 2025; Vincke et al. 2009 · **SAP** — Chennamsetty et al. 2009
- **Rosetta ref2015** — Alford et al. 2017 · **PyRosetta** — Chaudhury et al. 2010

> Licensed CC-BY-NC 4.0.
