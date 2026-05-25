"""Diffusion-DPO training over DiffAb's joint sequence-structure outputs.

Composed of:
  * :mod:`src.dpo.loss`     — AbDPO residue-level loss (Zhou et al. 2024 Eq. 8)
                              + ``compute_per_residue_losses`` adapter on top of
                              DiffAb's ``FullDPM``.
  * :mod:`src.dpo.dataset`  — pair dataset + collate that yields aligned
                              winner/loser DiffAb-format batches from a
                              ``select_pareto_pairs.py`` parquet.

The trainer entrypoint lives at ``scripts/dpo/train_dpo.py``; see
``docs/dpo_training_context.md`` (design) and
``docs/post_jfix_dpo_handoff.md`` (operational state) for the full picture.
"""
