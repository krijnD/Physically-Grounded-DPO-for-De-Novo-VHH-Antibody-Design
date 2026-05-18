"""AAPR (Aggressive-Accept Physical-Reject) loser-pool generation.

The thesis pipeline produces DPO training pairs by:
  1. Taking each natural GT VHH-antigen complex
  2. Masking its CDRs (default: H1+H2+H3 multi-CDR, deterministic)
  3. Sampling K candidate structures from DiffAb's reverse diffusion
  4. Preserving the original antigen on each generated candidate
  5. Scoring candidates via three calibrated judges (Physics/Biophysics/Biology)
  6. Selecting Pareto-dominated losers as hard negatives

This module owns step 2–4 (mask + generate + antigen-preserved PDB output).
Steps 5–6 live in the judge modules and a future pair-selection module.

See ``docs/aapr_generation_context.md`` for the full handoff.
"""
