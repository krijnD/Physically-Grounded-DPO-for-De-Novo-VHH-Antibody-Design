"""AbDPO loss for DiffAb (Zhou et al., NeurIPS 2024 — Eq. 8).

This module provides:

  * :func:`compute_per_residue_losses` — a thin reimplementation of
    DiffAb's ``FullDPM.forward`` that returns per-residue, per-channel
    losses ``[B, L]`` instead of the masked-and-averaged scalars the
    upstream returns. The arithmetic is byte-identical to upstream;
    only the final ``.sum() / mask.sum()`` aggregation is removed so
    DPO can do its own masked log-sigmoid sum.

  * :func:`abdpo_loss` — the AbDPO direct-energy preference loss,
    operating on the per-residue composite ELBO. Two aggregation modes:

      ``residue`` (default, AbDPO Eq. 8): apply ``log σ`` per residue,
                  sum residues per pair, mean over pairs. This is the
                  fine-grained credit assignment that AbDPO's Fig. 4
                  ablation shows beats the sequence-level form.

      ``sequence`` (Wallace et al. 2024 §3): sum residues first into a
                   per-pair scalar, then ``log σ``. Provided as an
                   ablation toggle; not the recommended default for
                   antibody design.

  * Shared-noise plumbing helpers — sampling the diffusion noise once
    and reusing it across the winner/loser/reference/policy passes so
    the four ELBO terms are computed at the *same* point in noise space
    (variance reduction trick from Wallace et al. §3.2).

The whole module is pure-tensor: no I/O, no model construction, no
config dependency. It assumes the caller has already built two
``DiffAb`` instances (``π_θ`` trainable, ``π_ref`` frozen+``eval()``)
and is feeding the same ``batch_w``/``batch_l`` dicts that come out of
:class:`src.dpo.dataset.PairCollate`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

# DiffAb internals — must be importable; the training script puts
# third_party/diffab on sys.path before importing this module.
from diffab.modules.common.so3 import rotation_to_so3vec, so3vec_to_rotation
from diffab.modules.diffusion.dpm_full import rotation_matrix_cosine_loss


# ── Channel names match DiffAb's FullDPM.forward return keys ────────────
CHANNELS = ("rot", "pos", "seq")


# ────────────────────────────────────────────────────────────────────────
# Per-residue loss extraction
# ────────────────────────────────────────────────────────────────────────
def compute_per_residue_losses(
    model,
    batch: dict,
    t: torch.Tensor,
    *,
    rng_state: Optional[tuple] = None,
) -> Dict[str, torch.Tensor]:
    """Mirror of ``FullDPM.forward`` returning per-residue losses ``[B, L]``.

    Parameters
    ----------
    model
        A ``DiffusionAntibodyDesign`` instance (the DiffAb model). We
        bypass ``model.forward`` so we can keep losses un-aggregated.
    batch
        DiffAb-format batch dict (output of ``PaddingCollate``).
    t
        Per-pair diffusion timestep, shape ``[B]``. Sample with
        ``torch.randint(1, num_steps + 1, (B,))`` per AbDPO §3.1.
    rng_state
        Optional ``(cpu_state, cuda_state_or_None)`` tuple. If
        provided, the RNG is reset to this state before noise sampling
        so the same noise is drawn across paired calls. See
        :func:`forward_pair_with_shared_noise` for the wrapper that
        manages this end-to-end.

    Returns
    -------
    dict
        ``{"rot": [B, L], "pos": [B, L], "seq": [B, L]}``. Each tensor
        carries per-residue ELBO terms. ``generate_flag`` masking is
        the caller's responsibility (we don't divide by mask sum here).
    """
    if rng_state is not None:
        _restore_rng(rng_state, device=batch["aa"].device)

    mask_generate = batch["generate_flag"]
    mask_res = batch["mask"]

    # encode() produces residue/pair features from the *context*
    # (~generate_flag) only — see DiffusionAntibodyDesign.encode. Since
    # winner and loser share the antigen and framework, their encoder
    # outputs are nearly identical; the difference comes from the
    # noisy CDR predictions below.
    res_feat, pair_feat, R_0, p_0 = model.encode(
        batch,
        remove_structure=model.cfg.get("train_structure", True),
        remove_sequence=model.cfg.get("train_sequence", True),
    )
    v_0 = rotation_to_so3vec(R_0)
    s_0 = batch["aa"]

    diff = model.diffusion
    p_0 = diff._normalize_position(p_0)
    R_0_mat = so3vec_to_rotation(v_0)

    # Add noise — this is where shared-RNG matters. We always denoise
    # both structure and sequence; restricting to one channel would
    # contradict the multi-CDR composite-loss decision in
    # docs/dpo_training_context.md §"Architectural decisions" #6.
    v_noisy, _ = diff.trans_rot.add_noise(v_0, mask_generate, t)
    p_noisy, eps_p = diff.trans_pos.add_noise(p_0, mask_generate, t)
    _, s_noisy = diff.trans_seq.add_noise(s_0, mask_generate, t)

    beta = diff.trans_pos.var_sched.betas[t]
    _v_pred, R_pred, eps_p_pred, c_denoised = diff.eps_net(
        v_noisy, p_noisy, s_noisy,
        res_feat, pair_feat, beta, mask_generate, mask_res,
    )

    # Per-residue losses — arithmetic matches FullDPM.forward except
    # for the final `.sum() / mask_sum` reduction that we drop.
    loss_rot = rotation_matrix_cosine_loss(R_pred, R_0_mat)          # [B, L]
    loss_pos = F.mse_loss(eps_p_pred, eps_p, reduction="none").sum(dim=-1)  # [B, L]

    post_true = diff.trans_seq.posterior(s_noisy, s_0, t)
    log_post_pred = torch.log(
        diff.trans_seq.posterior(s_noisy, c_denoised, t) + 1e-8
    )
    loss_seq = F.kl_div(
        input=log_post_pred,
        target=post_true,
        reduction="none",
        log_target=False,
    ).sum(dim=-1)                                                    # [B, L]

    return {"rot": loss_rot, "pos": loss_pos, "seq": loss_seq}


# ────────────────────────────────────────────────────────────────────────
# Shared-noise wrapper for the four-forward DPO step
# ────────────────────────────────────────────────────────────────────────
@dataclass
class PairLosses:
    """The four per-residue composite-loss tensors for one DPO step."""
    w_theta: Dict[str, torch.Tensor]
    l_theta: Dict[str, torch.Tensor]
    w_ref: Dict[str, torch.Tensor]
    l_ref: Dict[str, torch.Tensor]


def forward_pair_with_shared_noise(
    model_theta,
    model_ref,
    batch_w: dict,
    batch_l: dict,
    t: torch.Tensor,
    *,
    device: torch.device,
) -> PairLosses:
    """Run all four forward passes sharing diffusion noise across them.

    The variance-reduction trick from Wallace et al. §3.2: same ε for
    (y_w, y_l, π_θ, π_ref) at a given diffusion step. We implement
    sharing by capturing the RNG state once and resetting it before
    each of the four forwards. Both the position/rotation/sequence
    noise samplers in :mod:`diffab.modules.diffusion.transition` and any
    in-model dropout consume the same RNG sequence; this is fine — the
    correlation across the four passes is the intended invariant, not
    a bug.

    π_ref is run under :func:`torch.no_grad` and is expected to be in
    ``.eval()`` mode (caller's responsibility).
    """
    rng_state = _capture_rng(device)

    # π_θ winner — gradient flows here
    losses_w_theta = compute_per_residue_losses(
        model_theta, batch_w, t, rng_state=rng_state,
    )

    # π_θ loser — gradient flows here too
    losses_l_theta = compute_per_residue_losses(
        model_theta, batch_l, t, rng_state=rng_state,
    )

    # π_ref winner — frozen, no grad
    with torch.no_grad():
        losses_w_ref = compute_per_residue_losses(
            model_ref, batch_w, t, rng_state=rng_state,
        )
        losses_l_ref = compute_per_residue_losses(
            model_ref, batch_l, t, rng_state=rng_state,
        )

    return PairLosses(
        w_theta=losses_w_theta,
        l_theta=losses_l_theta,
        w_ref=losses_w_ref,
        l_ref=losses_l_ref,
    )


# ────────────────────────────────────────────────────────────────────────
# AbDPO loss
# ────────────────────────────────────────────────────────────────────────
def abdpo_loss(
    pair: PairLosses,
    mask: torch.Tensor,
    loss_weights: Dict[str, float],
    beta_dpo: float,
    num_timesteps: int,
    aggregation: str = "residue",
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """The AbDPO direct-energy preference loss (Zhou et al. 2024 Eq. 8).

    Composite per-residue ELBO term:
        L^t(j) = w_rot · loss_rot(j) + w_pos · loss_pos(j) + w_seq · loss_seq(j)

    Per-residue DPO delta (positive when π_θ is *worse* on the winner
    and *better* on the loser than π_ref — i.e. exactly what DPO
    should push down):
        δ(j) = (L_w_θ(j) - L_w_ref(j)) - (L_l_θ(j) - L_l_ref(j))

    Loss:
        L_AbDPO = - E[ Σ_j  log σ(-β · T · δ(j)) ]   (residue, default)
        L_DPO   = - E[      log σ(-β · T · Σ_j δ(j)) ]  (sequence, Wallace)

    Parameters
    ----------
    pair
        Output of :func:`forward_pair_with_shared_noise`.
    mask
        ``[B, L]`` boolean / 0-1 tensor — residues to include in the
        DPO loss. Use the winner's ``generate_flag``; we assume the
        pair dataset has enforced winner/loser mask alignment.
    loss_weights
        ``{"rot": w_r, "pos": w_p, "seq": w_s}`` — typically inherited
        from the fine-tune config (1.0 / 1.0 / 1.0).
    beta_dpo
        The DPO temperature β (AbDPO defaults to 0.1).
    num_timesteps
        DiffAb's diffusion horizon ``T`` (default 100). Multiplies δ
        because the diffusion ELBO is an *upper bound* on the negative
        log-likelihood divided by T — see Wallace et al. §3.1.
    aggregation
        ``"residue"`` (AbDPO Eq. 8, recommended) or ``"sequence"``
        (Wallace-style scalar DPO). Exposed as a config switch so we
        can run the ablation if needed.

    Returns
    -------
    loss : torch.Tensor  — scalar to ``.backward()`` on.
    diagnostics : dict  — detached scalars for W&B / logging.
    """
    if aggregation not in ("residue", "sequence"):
        raise ValueError(
            f"aggregation must be 'residue' or 'sequence', got {aggregation!r}"
        )

    L_w_theta = _composite(pair.w_theta, loss_weights)   # [B, L]
    L_l_theta = _composite(pair.l_theta, loss_weights)
    L_w_ref = _composite(pair.w_ref, loss_weights)
    L_l_ref = _composite(pair.l_ref, loss_weights)

    delta = (L_w_theta - L_w_ref) - (L_l_theta - L_l_ref)  # [B, L]
    mask_f = mask.float()
    # Per-pair residue count, ≥ 1 so we never divide by zero. With our
    # PairDataset alignment check, mask_count > 0 for every emitted pair.
    mask_count = mask_f.sum(dim=-1).clamp_min(1.0)         # [B]

    if aggregation == "residue":
        # AbDPO Eq. 8: per-residue log σ, summed over residues per pair.
        # The sign convention: δ > 0 means π_θ predicts the *winner*
        # worse than π_ref relative to the loser; -β·T·δ < 0; log σ(·)
        # is negative; -mean is positive — i.e. loss is high when π_θ
        # is mis-preferring, which is what we want to push down.
        log_sig = F.logsigmoid(-beta_dpo * num_timesteps * delta)   # [B, L]
        per_pair = (log_sig * mask_f).sum(dim=-1)                   # [B]
        loss = -per_pair.mean()
    else:  # sequence
        masked_delta_sum = (delta * mask_f).sum(dim=-1)             # [B]
        log_sig = F.logsigmoid(-beta_dpo * num_timesteps * masked_delta_sum)  # [B]
        loss = -log_sig.mean()

    with torch.no_grad():
        # Implicit reward margin per AbDPO §3.3 / Wallace §3.1:
        # m_j := -T · δ(j). Positive m means π_θ "prefers" the winner
        # relative to π_ref. Average over residues per pair, then over
        # pairs — interpretable in units of (negative) ELBO.
        margin_per_res = -num_timesteps * delta                     # [B, L]
        margin_per_pair = (margin_per_res * mask_f).sum(dim=-1) / mask_count
        # Pairwise accuracy: fraction of pairs where the average
        # margin is positive (π_θ correctly prefers winner over loser).
        accuracy = (margin_per_pair > 0).float().mean()

        diagnostics = {
            "loss": loss.detach(),
            "margin_mean": margin_per_pair.mean(),
            "margin_per_residue_mean": margin_per_res.mean(),
            "accuracy": accuracy,
            "L_w_theta": (L_w_theta * mask_f).sum(dim=-1).mean(),
            "L_l_theta": (L_l_theta * mask_f).sum(dim=-1).mean(),
            "L_w_ref":   (L_w_ref   * mask_f).sum(dim=-1).mean(),
            "L_l_ref":   (L_l_ref   * mask_f).sum(dim=-1).mean(),
            "delta_mean": (delta * mask_f).sum(dim=-1).mean(),
            "mask_count_mean": mask_count.mean(),
        }

    return loss, diagnostics


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────
def _composite(
    losses: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> torch.Tensor:
    """Weighted sum of per-residue channel losses → [B, L] composite."""
    out = None
    for k in CHANNELS:
        w = float(weights.get(k, 1.0))
        contrib = w * losses[k]
        out = contrib if out is None else out + contrib
    return out


def _capture_rng(device: torch.device) -> tuple:
    """Snapshot CPU + (optionally) CUDA RNG state for later restoration."""
    cpu = torch.get_rng_state()
    cuda = (
        torch.cuda.get_rng_state(device)
        if device.type == "cuda"
        else None
    )
    return cpu, cuda


def _restore_rng(state: tuple, device: torch.device) -> None:
    cpu, cuda = state
    torch.set_rng_state(cpu)
    if cuda is not None and device.type == "cuda":
        torch.cuda.set_rng_state(cuda, device)


def check_pair_alignment(batch_w: dict, batch_l: dict) -> bool:
    """Verify that winner and loser share the same generate_flag.

    Returns True if the masks are identical. Pre-condition for the
    AbDPO Eq. 8 sum to be well-defined (per-residue δ assumes residue
    j on the winner corresponds to residue j on the loser).
    """
    fw = batch_w["generate_flag"]
    fl = batch_l["generate_flag"]
    if fw.shape != fl.shape:
        return False
    return bool(torch.equal(fw, fl))
