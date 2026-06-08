#!/usr/bin/env python3
"""Brief 17 §8 — per-channel iter-0 reference loss + implicit reward.

This is the per-channel extension of ``diag_lwref_distribution.py``. The
original collapses (rot, pos, seq) into a single composite L_ref scalar.
The brief needs the breakdown so we can check whether the rot+pos
"structural shortcut" (+2.28 of margin on the floor pool) has been
killed by the decoy intervention.

Procedure: load π_ref, iterate over every (winner, loser) pair in the
provided parquet at a fixed timestep (t=t_eval, defaults to 50), compute
``compute_per_residue_losses`` once for the winner and once for the
loser with shared RNG (matches the DPO training-time noise sharing).
Sum each per-channel loss per pair (masked by generate_flag) and write
one row per pair.

Output schema
-------------
Each row covers one pair.

    split, pair_id, gt_id, mask_count,
    L_w_ref_rot, L_w_ref_pos, L_w_ref_seq, L_w_ref_composite, L_w_ref,
    L_l_ref_rot, L_l_ref_pos, L_l_ref_seq, L_l_ref_composite, L_l_ref,
    ref_margin              # composite L_l_ref - L_w_ref (no T)

The ``L_w_ref`` / ``L_l_ref`` / ``ref_margin`` aliases keep this parquet
drop-in compatible with ``filter_pairs_by_ref_margin.py`` (Brief 17 §10).

Computing the gate from this parquet
------------------------------------
At the login node, after this script lands its parquet::

    import pandas as pd
    df = pd.read_parquet('lwref_per_channel_decoy_t10.parquet')
    T = 100   # diffusion horizon, must match config.model.diffusion.num_steps
    for ch in ('rot', 'pos', 'seq'):
        m = -T * (df[f'L_w_ref_{ch}'] - df[f'L_l_ref_{ch}'])    # implicit reward
        print(f'reward_{ch}: mean={m.mean():+.3f}  median={m.median():+.3f}  '
              f'q10={m.quantile(0.1):+.3f}  q90={m.quantile(0.9):+.3f}')

CLI
---
::

    python scripts/dpo/diag_per_channel_reward.py \\
        --pi-ref-checkpoint runs/vhh_ft/seed42_jfix/checkpoints/best_ema.pt \\
        --pairs-parquet     data/aapr/.../dpo/pairs_decoy_t10.parquet \\
        --output            data/aapr/.../dpo/lwref_per_channel_decoy_t10.parquet \\
        --t-eval            50 \\
        --device            cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "diffab"))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from diffab.datasets import get_dataset
from diffab.models import get_model
from diffab.utils.misc import load_config, seed_all
from diffab.utils.train import recursive_to
from diffab.utils.transforms import get_transform

import src.diffab_ft.datasets  # noqa: F401 — registry side effect
from src.dpo.dataset import PairCollate, PairDataset
from src.dpo.loss import compute_per_residue_losses, _capture_rng

CHANNELS = ("rot", "pos", "seq")


def _composite(losses_dict, weights):
    """Weighted sum of per-residue channel losses → [B, L] composite."""
    out = None
    for k in CHANNELS:
        w = float(weights.get(k, 1.0))
        c = w * losses_dict[k]
        out = c if out is None else out + c
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=Path,
                    default=PROJECT_ROOT / "configs/dpo/vhh_dpo.yml",
                    help="DPO config YAML. Provides model arch, "
                         "loss_weights, val_gt_holdout, val_split_seed, "
                         "and the dataset block used to build the LMDB-"
                         "backed base dataset. Default: configs/dpo/vhh_dpo.yml.")
    ap.add_argument("--pi-ref-checkpoint", "--pi-ref", dest="pi_ref",
                    type=Path, default=None,
                    help="Override config.dpo.pi_ref_checkpoint.")
    ap.add_argument("--pairs-parquet", "--pairs", dest="pairs",
                    type=Path, default=None,
                    help="Override config.dpo.pair_parquet.")
    ap.add_argument("--output", "--out", dest="out",
                    type=Path, required=True,
                    help="Output parquet path. REQUIRED.")
    ap.add_argument("--t-eval", type=int, default=50,
                    help="Fixed diffusion timestep for L_ref evaluation. "
                         "Default 50 (matches diag_lwref_distribution.py).")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    config, _ = load_config(str(args.config))
    seed_all(args.seed)
    device = args.device
    T = int(config.model.diffusion.num_steps)
    loss_weights = dict(config.train.loss_weights)
    print(f"[config]  {args.config}")
    print(f"[config]  loss_weights = {loss_weights}")
    print(f"[config]  num_steps T  = {T}")
    print(f"[config]  t_eval       = {args.t_eval}")
    print(f"[config]  val_gt_holdout = {config.dpo.get('val_gt_holdout', 20)}")
    print(f"[config]  val_split_seed = {config.dpo.get('val_split_seed', 42)}")

    pi_ref_path = args.pi_ref if args.pi_ref else Path(config.dpo.pi_ref_checkpoint)
    if not pi_ref_path.is_absolute():
        pi_ref_path = PROJECT_ROOT / pi_ref_path
    print(f"[π_ref]   {pi_ref_path}")

    pair_parquet_path = args.pairs if args.pairs else Path(config.dpo.pair_parquet)
    if not pair_parquet_path.is_absolute():
        pair_parquet_path = PROJECT_ROOT / pair_parquet_path
    print(f"[pairs]   {pair_parquet_path}")

    print(f"[out]     {args.out}")
    print(f"[seed]    {args.seed}")

    print("\nLoading π_ref...")
    model = get_model(config.model).to(device).eval()
    ck = torch.load(str(pi_ref_path), map_location=device, weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    result = model.load_state_dict(sd, strict=False)
    print(f"  load_state_dict: missing={len(result.missing_keys)} "
          f"unexpected={len(result.unexpected_keys)}")
    if result.missing_keys:
        print(f"    missing sample: {result.missing_keys[:5]}")
    if result.unexpected_keys:
        print(f"    unexpected sample: {result.unexpected_keys[:5]}")
    for p in model.parameters():
        p.requires_grad_(False)

    print("Building base VHH dataset...")
    base_dataset = get_dataset(config.dataset.train)
    transform = get_transform(config.dataset.train.transform)

    print("Building pair datasets (both PairDataset splits, val_gt_holdout from config)...")
    rows = []
    for split in ("train", "val"):
        ds = PairDataset(
            pairs_parquet=str(pair_parquet_path),
            base_dataset=base_dataset,
            transform=transform,
            split=split,
            val_split_seed=int(config.dpo.get("val_split_seed", 42)),
            val_gt_holdout=int(config.dpo.get("val_gt_holdout", 20)),
        )
        loader = DataLoader(
            ds, batch_size=args.batch_size, collate_fn=PairCollate(eight=False),
            shuffle=False, num_workers=0,
        )
        t_fixed = torch.full((args.batch_size,), args.t_eval,
                             dtype=torch.long, device=device)
        n = len(ds)
        print(f"\n[{split}] {n} pairs")
        t_start = time.time()
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                batch_w = recursive_to(batch["winner"], device)
                batch_l = recursive_to(batch["loser"], device)
                B = batch_w["aa"].size(0)
                t_use = t_fixed[:B]
                if B < args.batch_size:
                    t_use = torch.full((B,), args.t_eval,
                                       dtype=torch.long, device=device)

                # Shared RNG: capture once, both forwards restore from it.
                # This mirrors forward_pair_with_shared_noise (src/dpo/loss.py
                # §159-208) so the per-channel decomposition we emit here is
                # the same noise-correlated quantity DPO sees at training time.
                rng_state = _capture_rng(torch.device(device))
                losses_w = compute_per_residue_losses(
                    model, batch_w, t_use, rng_state=rng_state,
                )
                losses_l = compute_per_residue_losses(
                    model, batch_l, t_use, rng_state=rng_state,
                )

                mask = batch_w["generate_flag"].float()
                mask_count = mask.sum(dim=-1).cpu().numpy()

                # Per-pair masked sums per channel.
                per_pair = {}
                for ch in CHANNELS:
                    per_pair[f"L_w_ref_{ch}"] = (
                        (losses_w[ch] * mask).sum(dim=-1).cpu().numpy()
                    )
                    per_pair[f"L_l_ref_{ch}"] = (
                        (losses_l[ch] * mask).sum(dim=-1).cpu().numpy()
                    )

                # Composite — sum of weighted per-channel losses, masked.
                L_w_comp = (_composite(losses_w, loss_weights) * mask).sum(dim=-1).cpu().numpy()
                L_l_comp = (_composite(losses_l, loss_weights) * mask).sum(dim=-1).cpu().numpy()

                for i in range(B):
                    row = {
                        "split":              split,
                        "pair_id":            batch["pair_id"][i],
                        "gt_id":              batch["gt_id"][i],
                        "mask_count":         float(mask_count[i]),
                        # Composite (also exposed as L_w_ref/L_l_ref for
                        # filter_pairs_by_ref_margin.py compatibility).
                        "L_w_ref_composite":  float(L_w_comp[i]),
                        "L_l_ref_composite":  float(L_l_comp[i]),
                        "L_w_ref":            float(L_w_comp[i]),
                        "L_l_ref":            float(L_l_comp[i]),
                        "ref_margin":         float(L_l_comp[i] - L_w_comp[i]),
                    }
                    for ch in CHANNELS:
                        row[f"L_w_ref_{ch}"] = float(per_pair[f"L_w_ref_{ch}"][i])
                        row[f"L_l_ref_{ch}"] = float(per_pair[f"L_l_ref_{ch}"][i])
                    rows.append(row)
                if (batch_idx + 1) % 25 == 0:
                    elapsed = time.time() - t_start
                    print(f"  batch {batch_idx+1} | elapsed {elapsed:.1f}s")
        print(f"  done. n_rows so far: {len(rows)}, elapsed {time.time() - t_start:.1f}s")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nWrote {len(df)} rows → {args.out}")

    # Per-channel implicit reward summary (Brief 17 §8 gate inputs).
    print("\n=== Iter-0 implicit reward per channel  (margin = -T·(L_w − L_l), T=%d) ===" % T)
    for ch in CHANNELS:
        m = -T * (df[f"L_w_ref_{ch}"] - df[f"L_l_ref_{ch}"])
        print(
            f"reward_{ch}: "
            f"n={len(m)} "
            f"mean={m.mean():+.3f}  "
            f"median={m.median():+.3f}  "
            f"q10={m.quantile(0.1):+.3f}  "
            f"q90={m.quantile(0.9):+.3f}"
        )
    print("\n=== ref_margin composite (L_l - L_w, no T — filter_pairs_by_ref_margin units) ===")
    print(
        f"ref_margin: "
        f"n={len(df)} "
        f"mean={df['ref_margin'].mean():+.3f}  "
        f"median={df['ref_margin'].median():+.3f}  "
        f"pct_pos={(df['ref_margin'] > 0).mean() * 100:.1f}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
