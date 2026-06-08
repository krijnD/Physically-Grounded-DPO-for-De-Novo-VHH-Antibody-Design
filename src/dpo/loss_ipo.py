"""IPO loss (Azar et al., arXiv 2310.12036) — bounded-regression DPO variant.

Parallel to :mod:`src.dpo.loss` but replaces the unbounded log-sigmoid
objective with a bounded margin regression toward target τ = 1/(2β).

    L_DPO = -log σ(β · T · Δreward)           # unbounded; reward-hackable
    L_IPO = (T · Δreward - τ)²                # bounded; quadratic both sides

Motivation — Brief 18, post-Brief-16 reviewer follow-up. Brief 16 found
that DPO at β=0.005 reward-hacked into Isoleucine homopolymers (99.7%
of generated sequences = ``IIIIIIIIIIIIIIII``, scRMSD 700-1450 Å). The
hypothesis is that this is a *DPO-objective* failure (unbounded loss
can push δ → -∞ via a trivial degenerate solution), not a data
failure. IPO's bounded objective should structurally prevent that
exploit because the loss penalises *over-shooting* the margin just as
much as under-shooting.

The four-forward shared-noise plumbing
(:func:`~src.dpo.loss.forward_pair_with_shared_noise`,
:func:`~src.dpo.loss.compute_per_residue_losses`, :class:`PairLosses`)
is reused verbatim — only the final aggregation step changes. The
trainer dispatches to :func:`ipo_loss` vs
:func:`~src.dpo.loss.abdpo_loss` based on ``config.dpo.objective``.
"""

from __future__ import annotations

from typing import Dict

import torch

from src.dpo.loss import CHANNELS, PairLosses, _composite


def ipo_loss(
    pair: PairLosses,
    mask: torch.Tensor,
    loss_weights: Dict[str, float],
    beta_dpo: float,
    num_timesteps: int,
    aggregation: str = "residue",
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """The IPO bounded-regression preference loss (Azar et al. 2023, Eq. 17).

    Same composite per-residue ELBO as AbDPO:
        L^t(j) = w_rot · loss_rot(j) + w_pos · loss_pos(j) + w_seq · loss_seq(j)

    Same per-residue DPO delta:
        δ(j) = (L_w_θ(j) - L_w_ref(j)) - (L_l_θ(j) - L_l_ref(j))

    Same implicit reward margin convention (positive when winner preferred):
        m(j) = -T · δ(j)

    IPO regression target:
        τ = 1 / (2β)

    Loss (residue aggregation, default):
        L_IPO = E[ Σ_j  (m(j) - τ)² ]

    Loss (sequence aggregation, ablation):
        L_IPO = E[ (Σ_j m(j) - τ)² ]

    Parameters mirror :func:`~src.dpo.loss.abdpo_loss` so the trainer
    can dispatch with one line. ``beta_dpo`` is interpreted as
    Azar et al.'s β — same symbol, different role: in DPO it scales
    the log-sigmoid; in IPO it sets the regression target via τ=1/(2β).

    Returns
    -------
    loss : torch.Tensor — scalar to ``.backward()`` on. IPO loss values
                          live on a different scale than DPO loss values
                          (quadratic distance vs negative log-sigmoid)
                          and are NOT directly comparable across runs of
                          different objective; use ``margin_distance_from_tau``
                          as the cross-objective convergence proxy.
    diagnostics : dict — detached scalars for W&B / logging.
    """
    if aggregation not in ("residue", "sequence"):
        raise ValueError(
            f"aggregation must be 'residue' or 'sequence', got {aggregation!r}"
        )

    L_w_theta = _composite(pair.w_theta, loss_weights)
    L_l_theta = _composite(pair.l_theta, loss_weights)
    L_w_ref = _composite(pair.w_ref, loss_weights)
    L_l_ref = _composite(pair.l_ref, loss_weights)

    delta = (L_w_theta - L_w_ref) - (L_l_theta - L_l_ref)  # [B, L]
    mask_f = mask.float()
    mask_count = mask_f.sum(dim=-1).clamp_min(1.0)         # [B]

    # Implicit reward residual per residue, scaled by T (same sign
    # convention as DPO: positive when π_θ prefers the winner over π_ref).
    m_per_res = -num_timesteps * delta                     # [B, L]

    # IPO regression target. β is reused as the IPO hyperparameter symbol;
    # at β=0.05 τ=10, at β=0.005 τ=100, at β=0.5 τ=1.
    tau = 1.0 / (2.0 * beta_dpo)

    if aggregation == "residue":
        # Per-residue quadratic penalty around τ, summed per pair.
        # Matches AbDPO's residue-level granularity (Eq. 8 fine-grained
        # credit assignment) but with the bounded IPO inner objective.
        ipo_per_res = (m_per_res - tau).pow(2)              # [B, L]
        per_pair = (ipo_per_res * mask_f).sum(dim=-1)       # [B]
        loss = per_pair.mean()
    else:  # sequence — Wallace-style: sum reward first, then regress
        m_per_pair_raw = (m_per_res * mask_f).sum(dim=-1)   # [B]
        loss = (m_per_pair_raw - tau).pow(2).mean()

    with torch.no_grad():
        # Mean per-residue margin per pair (interpretable in (negative)
        # ELBO units; same metric as DPO's margin_mean).
        margin_per_pair = (m_per_res * mask_f).sum(dim=-1) / mask_count
        accuracy = (margin_per_pair > 0).float().mean()

        # IPO convergence proxy: distance of the per-pair mean margin
        # from τ. At convergence on average this approaches 0; on a run
        # that is collapsing or failing to learn it stays near τ.
        margin_distance_from_tau = (margin_per_pair - tau).abs()
        # Loose convergence indicator: fraction of pairs within ½τ of τ.
        # Brief 18 §5 — useful headline scalar for the deliverable §7.
        converged_pair_fraction = (
            margin_distance_from_tau < 0.5 * tau
        ).float().mean()

        diagnostics = {
            "loss": loss.detach(),
            "margin_mean": margin_per_pair.mean(),
            "margin_per_residue_mean": m_per_res.mean(),
            "tau_target": torch.tensor(tau, device=loss.device),
            "margin_distance_from_tau_mean": margin_distance_from_tau.mean(),
            "converged_pair_fraction": converged_pair_fraction,
            "accuracy": accuracy,
            "L_w_theta": (L_w_theta * mask_f).sum(dim=-1).mean(),
            "L_l_theta": (L_l_theta * mask_f).sum(dim=-1).mean(),
            "L_w_ref":   (L_w_ref   * mask_f).sum(dim=-1).mean(),
            "L_l_ref":   (L_l_ref   * mask_f).sum(dim=-1).mean(),
            "delta_mean": (delta * mask_f).sum(dim=-1).mean(),
            "mask_count_mean": mask_count.mean(),
        }

    return loss, diagnostics
