#!/usr/bin/env python3
"""Dump per-pair (L_w_ref, L_l_ref) for the entire pair pool.

Background (unchanged from v1, 2026-05-26)
------------------------------------------
Question this answers: is the train pool bimodal? Specifically, is
L_w_ref close to 0 for pairs whose GT was in pi_ref's fine-tune train
split (memorized) and at baseline for pairs whose GT was in the val
split? If so, the trainval pair pool mixes two regimes and DPO is
trying to satisfy contradictory objectives.

Procedure: load pi_ref, iterate over all pairs (no DPO update), compute
compute_per_residue_losses for winner and loser at a fixed timestep
(t=50, mid-diffusion) with deterministic noise. Sum across residues
(masked). Save to parquet.

v2 changes (2026-06-01) — added by Brief 06.5
---------------------------------------------
- CLI args (--config / --pi-ref / --pairs / --out / --seed) so the
  same script can score arbitrary (pi_ref, pair) combinations without
  editing YAMLs or paths in the source. Defaults preserve the
  original v1 behaviour exactly (vhh_dpo.yml config, all-channels
  weights, config.dpo.pi_ref_checkpoint, config.dpo.pair_parquet).
- --out is REQUIRED to avoid accidentally overwriting the existing
  lwref_distribution.parquet.
- Everything else (fixed t=50, batch_size=4, num_workers=0,
  val_gt_holdout from config, PairDataset train+val iteration,
  composite() with config.train.loss_weights, output schema
  {split, pair_id, gt_id, L_w_ref, L_l_ref, mask_count, ref_margin})
  is byte-identical to v1.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'third_party' / 'diffab'))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from diffab.datasets import get_dataset
from diffab.models import get_model
from diffab.utils.misc import load_config, seed_all
from diffab.utils.train import recursive_to
from diffab.utils.transforms import get_transform

import src.diffab_ft.datasets  # noqa: F401  — registry side effect
from src.dpo.dataset import PairCollate, PairDataset
from src.dpo.loss import compute_per_residue_losses, _capture_rng

CHANNELS = ('rot', 'pos', 'seq')


def composite(losses_dict, weights):
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
    ap.add_argument('--config', type=Path,
                    default=PROJECT_ROOT / 'configs/dpo/vhh_dpo.yml',
                    help='DPO config YAML. Provides model arch, '
                         'loss_weights, val_gt_holdout, and defaults '
                         'for pi_ref / pair_parquet. Default: '
                         'configs/dpo/vhh_dpo.yml (the original convention).')
    ap.add_argument('--pi-ref', type=Path, default=None,
                    help='Override config.dpo.pi_ref_checkpoint.')
    ap.add_argument('--pairs', type=Path, default=None,
                    help='Override config.dpo.pair_parquet.')
    ap.add_argument('--out', type=Path, required=True,
                    help='Output parquet path. REQUIRED to avoid '
                         'overwriting the existing lwref_distribution.parquet.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    config, _ = load_config(str(args.config))
    seed_all(args.seed)
    device = args.device
    T = int(config.model.diffusion.num_steps)
    loss_weights = dict(config.train.loss_weights)
    print(f'[config]  {args.config}')
    print(f'[config]  loss_weights = {loss_weights}')
    print(f'[config]  num_steps T = {T}')
    print(f'[config]  val_gt_holdout = {config.dpo.get("val_gt_holdout", 20)}')
    print(f'[config]  val_split_seed = {config.dpo.get("val_split_seed", 42)}')

    pi_ref_path = args.pi_ref if args.pi_ref else Path(config.dpo.pi_ref_checkpoint)
    if not pi_ref_path.is_absolute():
        pi_ref_path = PROJECT_ROOT / pi_ref_path
    print(f'[π_ref]   {pi_ref_path}')

    pair_parquet_path = args.pairs if args.pairs else Path(config.dpo.pair_parquet)
    if not pair_parquet_path.is_absolute():
        pair_parquet_path = PROJECT_ROOT / pair_parquet_path
    print(f'[pairs]   {pair_parquet_path}')

    print(f'[out]     {args.out}')
    print(f'[seed]    {args.seed}')

    print('\nLoading π_ref...')
    model = get_model(config.model).to(device).eval()
    ck = torch.load(str(pi_ref_path), map_location=device, weights_only=False)
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    result = model.load_state_dict(sd, strict=False)
    print(f'  load_state_dict: missing={len(result.missing_keys)} '
          f'unexpected={len(result.unexpected_keys)}')
    if result.missing_keys:
        print(f'    missing sample: {result.missing_keys[:5]}')
    if result.unexpected_keys:
        print(f'    unexpected sample: {result.unexpected_keys[:5]}')
    for p in model.parameters():
        p.requires_grad_(False)

    print('Building base VHH dataset...')
    base_dataset = get_dataset(config.dataset.train)
    transform = get_transform(config.dataset.train.transform)

    print('Building pair datasets (both PairDataset splits, val_gt_holdout from config)...')
    rows = []
    for split in ('train', 'val'):
        ds = PairDataset(
            pairs_parquet=str(pair_parquet_path),
            base_dataset=base_dataset,
            transform=transform,
            split=split,
            val_split_seed=int(config.dpo.get('val_split_seed', 42)),
            val_gt_holdout=int(config.dpo.get('val_gt_holdout', 20)),
        )
        loader = DataLoader(
            ds, batch_size=4, collate_fn=PairCollate(eight=False),
            shuffle=False, num_workers=0,
        )
        t_fixed = torch.full((4,), 50, dtype=torch.long, device=device)
        n = len(ds)
        print(f'\n[{split}] {n} pairs')
        t_start = time.time()
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                batch_w = recursive_to(batch['winner'], device)
                batch_l = recursive_to(batch['loser'], device)
                B = batch_w['aa'].size(0)
                t_use = t_fixed[:B]
                if B < 4:
                    t_use = torch.full((B,), 50, dtype=torch.long, device=device)
                # Capture RNG once, restore for both winner+loser forwards
                rng_state = _capture_rng(torch.device(device))
                losses_w = compute_per_residue_losses(model, batch_w, t_use, rng_state=rng_state)
                losses_l = compute_per_residue_losses(model, batch_l, t_use, rng_state=rng_state)
                L_w = composite(losses_w, loss_weights)  # [B, L]
                L_l = composite(losses_l, loss_weights)
                mask = batch_w['generate_flag'].float()
                # Per-pair sums
                L_w_per_pair = (L_w * mask).sum(dim=-1).cpu().numpy()
                L_l_per_pair = (L_l * mask).sum(dim=-1).cpu().numpy()
                mask_count = mask.sum(dim=-1).cpu().numpy()
                for i in range(B):
                    rows.append({
                        'split': split,
                        'pair_id': batch['pair_id'][i],
                        'gt_id': batch['gt_id'][i],
                        'L_w_ref': float(L_w_per_pair[i]),
                        'L_l_ref': float(L_l_per_pair[i]),
                        'mask_count': float(mask_count[i]),
                        'ref_margin': float(L_l_per_pair[i] - L_w_per_pair[i]),
                    })
                if (batch_idx + 1) % 25 == 0:
                    elapsed = time.time() - t_start
                    print(f'  batch {batch_idx+1} | elapsed {elapsed:.1f}s')
        print(f'  done. n_rows so far: {len(rows)}, elapsed {time.time() - t_start:.1f}s')

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f'\nWrote {len(df)} rows → {args.out}')
    print('\n=== Summary ===')
    for split in ('train', 'val'):
        sub = df[df.split == split]
        print(f'\n[{split}] n={len(sub)}')
        for col in ('L_w_ref', 'L_l_ref', 'ref_margin', 'mask_count'):
            s = sub[col]
            print(f'  {col}: min={s.min():.3f} q10={s.quantile(.1):.3f} '
                  f'median={s.median():.3f} mean={s.mean():.3f} '
                  f'q90={s.quantile(.9):.3f} max={s.max():.3f}')
    print(f'\npct_neg_margin overall: {100 * (df["ref_margin"] < 0).mean():.1f}%')
    print(f'pct_neg_margin train:   {100 * (df[df.split == "train"]["ref_margin"] < 0).mean():.1f}%')
    print(f'pct_neg_margin val:     {100 * (df[df.split == "val"]["ref_margin"] < 0).mean():.1f}%')
    return 0


if __name__ == '__main__':
    sys.exit(main())
