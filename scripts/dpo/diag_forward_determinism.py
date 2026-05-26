#!/usr/bin/env python3
"""Diagnose whether DiffAb's forward path is deterministic on CUDA.

Three tests, each on a real batch from the PairDataset:

  1. SAME-MODEL × 2  — call model_theta(batch_w) twice with RNG
     restored between calls. If outputs differ, CUDA has
     non-deterministic ops somewhere in the forward path
     (prime suspect: torch.multinomial in
     AminoacidCategoricalTransition._sample).

  2. WINNER θ vs ref — same input batch_w, both models loaded from
     the same checkpoint, RNG restored to the same state. If outputs
     differ but test 1 passes, there's a non-state-dict difference
     between the model instances.

  3. LOSER  θ vs ref — same as 2 but on batch_l. Confirms whether
     the gap is larger on OOD inputs (would explain the training
     log's asymmetric L_w gap=0.13 vs L_l gap=2.28).

Run from the project root:
    python scripts/dpo/diag_forward_determinism.py
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

import src.diffab_ft.datasets  # noqa: F401 — registry side effect
from src.dpo.dataset import PairCollate, PairDataset
from src.dpo.loss import compute_per_residue_losses


def main() -> int:
    config, _ = load_config(str(PROJECT_ROOT / "configs/dpo/vhh_dpo.yml"))
    seed_all(42)
    device = "cuda"

    print("Building models...")
    model_theta = get_model(config.model).to(device).eval()
    model_ref = get_model(config.model).to(device).eval()

    ckpt_path = config.dpo.pi_ref_checkpoint
    if not Path(ckpt_path).is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    print(f"Loading both from: {ckpt_path}")
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = ck["model"]
    model_ref.load_state_dict(sd, strict=False)
    model_theta.load_state_dict(sd, strict=False)

    print("Building one batch via PairDataset...")
    base_dataset = get_dataset(config.dataset.train)
    transform = get_transform(config.dataset.train.transform)
    ds = PairDataset(
        pairs_parquet=str(PROJECT_ROOT / config.dpo.pair_parquet),
        base_dataset=base_dataset,
        transform=transform,
        split="train",
    )
    loader = DataLoader(
        ds, batch_size=4, collate_fn=PairCollate(eight=False), shuffle=False,
    )
    batch = next(iter(loader))
    batch_w = recursive_to(batch["winner"], device)
    batch_l = recursive_to(batch["loser"], device)
    t = torch.randint(1, 101, (4,), dtype=torch.long, device=device)

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device)

    def restore():
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, device)

    def report(label: str, a: dict, b: dict):
        for k in ("rot", "pos", "seq"):
            diff = (a[k] - b[k]).abs()
            print(
                f"  {label:30s} {k:3s}  "
                f"max={diff.max().item():.6e}  "
                f"mean={diff.mean().item():.6e}  "
                f"sum={diff.sum().item():.6e}"
            )

    # Test 1: same model, same input, RNG restored — twice
    print("\n[Test 1] SAME MODEL (θ), SAME INPUT (winner), RNG restored:")
    restore()
    a = compute_per_residue_losses(model_theta, batch_w, t)
    restore()
    b = compute_per_residue_losses(model_theta, batch_w, t)
    report("θ(w) call 1 vs call 2", a, b)

    # Test 2: same input (winner), different model instances (both
    # loaded from the same checkpoint)
    print("\n[Test 2] WINNER input, θ-instance vs ref-instance:")
    restore()
    losses_theta_w = compute_per_residue_losses(model_theta, batch_w, t)
    restore()
    losses_ref_w = compute_per_residue_losses(model_ref, batch_w, t)
    report("θ(w) vs ref(w)", losses_theta_w, losses_ref_w)

    # Test 3: same input (loser, OOD), different model instances
    print("\n[Test 3] LOSER input (OOD), θ-instance vs ref-instance:")
    restore()
    losses_theta_l = compute_per_residue_losses(model_theta, batch_l, t)
    restore()
    losses_ref_l = compute_per_residue_losses(model_ref, batch_l, t)
    report("θ(l) vs ref(l)", losses_theta_l, losses_ref_l)

    print("\nInterpretation:")
    print("  Test 1 nonzero  → forward is non-deterministic on CUDA")
    print("                    (multinomial is the prime suspect)")
    print("                    Fix: torch.use_deterministic_algorithms(True)")
    print("  Tests 2/3 nonzero but Test 1 zero → non-state-dict difference")
    print("                    between model instances (closures, untracked state)")
    print("  All zero        → bug is elsewhere in the trainer pipeline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
