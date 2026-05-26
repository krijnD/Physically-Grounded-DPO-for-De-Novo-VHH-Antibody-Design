#!/usr/bin/env python3
"""Reproduce the trainer's iter-1 forward path exactly and compare.

The previous diagnostic (diag_forward_determinism.py) showed 0 diff
for direct calls to compute_per_residue_losses with manual RNG
restoration. But the trainer reports L_w_θ - L_w_ref = 0.128 at iter 1.
That means the discrepancy is somewhere in the path the trainer takes
that the simple diagnostic doesn't replicate. This script narrows it
down by running the EXACT trainer flow — forward_pair_with_shared_noise
+ abdpo_loss — and printing the same diagnostic dict the trainer logs.

Tests:
  A. forward_pair_with_shared_noise once, inspect per-residue losses
     for L_w_θ vs L_w_ref (and L_l_θ vs L_l_ref). These should match
     bit-exactly at iter-0 with π_θ == π_ref.
  B. Same as A but with model_theta requires_grad=False (matches the
     no_grad block on ref side). Isolates whether autograd-graph
     construction is changing numerical output.
  C. Repeat A with model_theta in train mode (not eval). Confirms
     whether mode actually changes anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "diffab"))

import torch
from torch.utils.data import DataLoader

from diffab.datasets import get_dataset
from diffab.models import get_model
from diffab.utils.misc import load_config, seed_all
from diffab.utils.train import recursive_to
from diffab.utils.transforms import get_transform

import src.diffab_ft.datasets  # noqa: F401
from src.dpo.dataset import PairCollate, PairDataset
from src.dpo.loss import abdpo_loss, forward_pair_with_shared_noise


def report_diag(label: str, diag: dict, beta_dpo: float, T: int):
    print(f"  [{label}]")
    print(f"    L_w_θ = {diag['L_w_theta']:.6f}  L_w_ref = {diag['L_w_ref']:.6f}  "
          f"diff = {diag['L_w_theta'] - diag['L_w_ref']:+.6e}")
    print(f"    L_l_θ = {diag['L_l_theta']:.6f}  L_l_ref = {diag['L_l_ref']:.6f}  "
          f"diff = {diag['L_l_theta'] - diag['L_l_ref']:+.6e}")
    print(f"    margin_mean = {diag['margin_mean']:+.6f}   accuracy = {diag['accuracy']:.4f}")


def main() -> int:
    config, _ = load_config(str(PROJECT_ROOT / "configs/dpo/vhh_dpo.yml"))
    seed_all(42)
    device = "cuda"
    T = int(config.model.diffusion.num_steps)
    beta_dpo = float(config.dpo.beta_dpo)
    loss_weights = dict(config.train.loss_weights)

    print(f"β = {beta_dpo}, T = {T}, loss_weights = {loss_weights}")
    print("Building models (mimicking trainer order)...")
    model_theta = get_model(config.model).to(device)
    model_ref = get_model(config.model).to(device)

    ckpt_path = config.dpo.pi_ref_checkpoint
    if not Path(ckpt_path).is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = ck["model"]
    model_ref.load_state_dict(sd, strict=False)
    model_theta.load_state_dict(sd, strict=False)

    model_ref.eval()
    for p in model_ref.parameters():
        p.requires_grad_(False)

    base_dataset = get_dataset(config.dataset.train)
    transform = get_transform(config.dataset.train.transform)
    ds = PairDataset(
        pairs_parquet=str(PROJECT_ROOT / config.dpo.pair_parquet),
        base_dataset=base_dataset,
        transform=transform,
        split="train",
    )
    # Use shuffle=True to match the trainer's DataLoader behaviour.
    loader = DataLoader(
        ds, batch_size=4, collate_fn=PairCollate(eight=False), shuffle=True,
    )
    batch = next(iter(loader))
    batch_w = recursive_to(batch["winner"], device)
    batch_l = recursive_to(batch["loser"], device)
    t = torch.randint(1, T + 1, (4,), dtype=torch.long, device=device)

    # ── Test A: trainer's exact path, π_θ in eval mode ───────────────
    print("\n[Test A] forward_pair_with_shared_noise, π_θ.eval()")
    model_theta.eval()
    pair_losses = forward_pair_with_shared_noise(
        model_theta, model_ref, batch_w, batch_l, t, device=torch.device(device),
    )
    # Per-channel byte-equality check
    for k in ("rot", "pos", "seq"):
        diff_w = (pair_losses.w_theta[k] - pair_losses.w_ref[k]).abs()
        diff_l = (pair_losses.l_theta[k] - pair_losses.l_ref[k]).abs()
        print(f"    {k}  W max={diff_w.max().item():.3e}  L max={diff_l.max().item():.3e}")
    loss, diag = abdpo_loss(
        pair_losses, mask=batch_w["generate_flag"],
        loss_weights=loss_weights, beta_dpo=beta_dpo, num_timesteps=T,
        aggregation="residue",
    )
    diag_floats = {k: float(v.item()) for k, v in diag.items()}
    report_diag("Test A diagnostics", diag_floats, beta_dpo, T)

    # ── Test B: π_θ in train mode (rules out a mode effect we missed) ─
    print("\n[Test B] forward_pair_with_shared_noise, π_θ.train()")
    model_theta.train()
    pair_losses_b = forward_pair_with_shared_noise(
        model_theta, model_ref, batch_w, batch_l, t, device=torch.device(device),
    )
    for k in ("rot", "pos", "seq"):
        diff_w = (pair_losses_b.w_theta[k] - pair_losses_b.w_ref[k]).abs()
        diff_l = (pair_losses_b.l_theta[k] - pair_losses_b.l_ref[k]).abs()
        print(f"    {k}  W max={diff_w.max().item():.3e}  L max={diff_l.max().item():.3e}")
    loss_b, diag_b = abdpo_loss(
        pair_losses_b, mask=batch_w["generate_flag"],
        loss_weights=loss_weights, beta_dpo=beta_dpo, num_timesteps=T,
        aggregation="residue",
    )
    diag_b_floats = {k: float(v.item()) for k, v in diag_b.items()}
    report_diag("Test B diagnostics", diag_b_floats, beta_dpo, T)

    # ── Test C: π_θ eval, but freeze its params (mimics ref side) ─────
    # If this matches Test A, autograd graph construction has no effect.
    print("\n[Test C] forward_pair_with_shared_noise, π_θ.eval() + frozen params")
    model_theta.eval()
    for p in model_theta.parameters():
        p.requires_grad_(False)
    pair_losses_c = forward_pair_with_shared_noise(
        model_theta, model_ref, batch_w, batch_l, t, device=torch.device(device),
    )
    for k in ("rot", "pos", "seq"):
        diff_w = (pair_losses_c.w_theta[k] - pair_losses_c.w_ref[k]).abs()
        diff_l = (pair_losses_c.l_theta[k] - pair_losses_c.l_ref[k]).abs()
        print(f"    {k}  W max={diff_w.max().item():.3e}  L max={diff_l.max().item():.3e}")
    loss_c, diag_c = abdpo_loss(
        pair_losses_c, mask=batch_w["generate_flag"],
        loss_weights=loss_weights, beta_dpo=beta_dpo, num_timesteps=T,
        aggregation="residue",
    )
    diag_c_floats = {k: float(v.item()) for k, v in diag_c.items()}
    report_diag("Test C diagnostics", diag_c_floats, beta_dpo, T)

    print("\nInterpretation:")
    print("  If Test A shows L_w_θ - L_w_ref ≠ 0 (e.g. ~0.13)        → bug is in")
    print("                  forward_pair_with_shared_noise itself (probably the")
    print("                  no_grad block interacts with autograd-on θ calls).")
    print("  If Test A == 0 but trainer iter-1 ≠ 0                   → something")
    print("                  in the trainer setup we haven't replicated.")
    print("  If Test C matches Test A                                → autograd")
    print("                  is not the cause.")
    print("  If Test B differs from Test A                           → mode effect")
    print("                  exists somewhere despite earlier evidence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
