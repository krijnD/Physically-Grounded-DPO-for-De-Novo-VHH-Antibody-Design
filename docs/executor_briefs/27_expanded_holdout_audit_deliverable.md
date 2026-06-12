# Brief 27 deliverable — Expanded-holdout MMseqs2 audit

**Date:** 2026-06-12
**Origin:** [`27_expanded_holdout_audit.md`](27_expanded_holdout_audit.md); ran
on Snellius commit `c283608`.
**Mode:** read-only MMseqs2 audit on Snellius via git-commit roundtrip;
CPU-only. No new training, sampling, or judge recalibration.

This is the campaign-repo mirror of the writer-facing deliverable at
`master-thesis/docs/orchestrator_requests/A1_expanded_holdout_audit_deliverable.md`.
The two are kept in sync; the only differences are the relative paths to
the artifacts.

Follows up on the open item from Brief 23 §"Open Items" #3 (the 83-entry
expanded holdout was not audited under the EXPANDED re-clustering).

---

## §1. Method

For each of the 463 expanded-train members (from
`data/datasets/clustering/cluster_splits_expanded.json` → `splits.train`),
loaded concat-CDR strings from {floor `concat_cdrs.fasta`} ∪ {Brief 23's
`tmp_brief23/added_concat_cdrs.fasta`}. Ran `mmseqs easy-search` against
the 83-entry expanded-holdout concat-CDRs (pulled from the same union) at
floor-clustering parameters: `--min-seq-id 0.7 -c 0.8 --cov-mode 0`.
Flagged any hit at ≥85 % identity (Brief 23's gate).

Coverage: 462 of 463 train members had CDR records (1 missing); 82 of 83
holdout members had CDR records (missing `7ph2_D`).

---

## §2. Headline result

| Metric | Value |
|---|---:|
| Total flagged train→holdout pairs (≥85 % identity) | **0** |
| Raw MMseqs2 hits (≥70 % identity, floor-clustering minimum) | **0** |
| Brief 23-known training entries appearing in train→holdout hits | **0** |
| NEW leakage cases | **0** |
| Distinct expanded-holdout entries contaminated | **0** |
| Max % identity among any hit | n/a (no hits) |

Full hit table: [`tmp_brief27/expanded_holdout_leakage_summary.csv`](../../tmp_brief27/expanded_holdout_leakage_summary.csv)
(header only, 0 data rows). Audit summary text:
[`tmp_brief27/audit_summary.txt`](../../tmp_brief27/audit_summary.txt).

---

## §3. Why the Brief 23 cases don't appear

Brief 23 identified 3 added training entries (`5o2u_D`, `7jkm_K`,
`7o31_X`) that clustered with shared-holdout members (`7n9v_J`, `7q6c_K`)
at ≥85 % CDR identity (two at 100 %). Under the expanded re-clustering's
pinning step, all five IDs land on the **test** side of the train/test
split:

| Entry | Brief 23 role | `cluster_splits_expanded.json` role |
|---|---|---|
| `5o2u_D` | added training entry | `splits.test` (NOT in train) |
| `7jkm_K` | added training entry | `splits.test` (NOT in train) |
| `7o31_X` | added training entry | `splits.test` (NOT in train) |
| `7n9v_J` | floor holdout entry | `splits.test` ✓ |
| `7q6c_K` | floor holdout entry | `splits.test` ✓ |

`splits.test` contains 86 entries: the 83-entry hardcoded eval holdout
**plus** the 3 extras above. The pinning step explicitly placed the 3
Brief 23 contaminating entries on the test side alongside their
100 %-identical cluster-mates, so the expanded fine-tune never trained
on them. The audit correspondingly finds zero train↔holdout pairs at
any threshold ≥70 %.

Diagnostic verification (Snellius, post-run):

```python
splits.train  ∩  {5o2u_D, 7jkm_K, 7o31_X}     →  ∅
splits.test   ∩  {5o2u_D, 7jkm_K, 7o31_X}     →  {5o2u_D, 7jkm_K, 7o31_X}
splits.test   ∩  {7n9v_J, 7q6c_K}             →  {7n9v_J, 7q6c_K}
```

---

## §4. Gating outcome

Per Brief 27 §6:

- ≥5 new flagged pairs: NO — zero flagged.
- Any single new pair at ≥95 % identity: NO — zero hits at any threshold.
- Expanded splits JSON cannot be located: NO — found and used.
- 83-entry hardcoded list doesn't match `splits.test` subset: NO — 83/83
  match.

No gates triggered; deliverable proceeds directly.

---

## §5. Diagnosis

Per the brief's tree: **0 new cases → "expanded clustering preserves
identity integrity beyond Brief 23's 3 cases. SQ1 framing safe to extend
to the 83-entry expanded holdout."**

Stronger: the audit is also clean for the 3 Brief 23 cases themselves
(they are in `splits.test`, not `splits.train`). Implication: SQ1 (val
ELBO −13 %) is not attributable to memorization of CDR-identical
training entries, and Brief 26's leakage-corrected ELBO recompute
should show a near-zero leakage effect (its measured numbers
supersede this prediction).

---

## §6. Secondary observation

The top-level `data/datasets/clustering/concat_cdrs.fasta` on Snellius
currently contains 242 entries (not the 465-entry floor file Brief 23
referenced). The full 465-entry floor CDR data presumably lives
elsewhere (LMDB, or `expanded_raw/concat_cdrs.fasta`); the top-level
file appears rewritten by a later pipeline step. This did not affect
Brief 27's audit — the union with `tmp_brief23/added_concat_cdrs.fasta`
(462 records) covered 462/463 train members and 82/83 holdout members,
sufficient for the comparison.

---

## §7. Reproduction

```bash
# On Snellius
cd ~/Physically-Grounded-DPO-for-De-Novo-VHH-Antibody-Design
git checkout c283608   # or any descendant
module purge
module load 2025 2024 gompi/2024a HMMER/3.4-gompi-2024a
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate
python scripts/analysis/brief27_expanded_holdout_audit.py \
    --expanded-splits data/datasets/clustering/cluster_splits_expanded.json \
    --out-dir tmp_brief27
cat tmp_brief27/audit_summary.txt
```
