#!/usr/bin/env python3
"""Pull W&B histories for the 4 DPO runs and dump CSVs locally.

Saves to /tmp/dpo_wandb/ — one CSV per run plus a summary.txt.
"""
from __future__ import annotations

import json
from pathlib import Path

import wandb

OUT = Path("/tmp/dpo_wandb")
OUT.mkdir(parents=True, exist_ok=True)

PROJECT = "krijnd/vhh-dpo"
RUN_NAMES = [
    "dpo_seqonly",
    "dpo_seqagg",
    "dpo_abdpo_match",
    "dpo_seed42_jfix_trainval",
]
KEYS = [
    "train/loss",
    "train/grad_norm",
    "train/margin_mean",
    "train/accuracy",
    "train/L_w_theta",
    "train/L_l_theta",
    "train/L_w_ref",
    "train/L_l_ref",
    "train/delta_mean",
    "val/loss",
    "val/margin_mean",
    "val/accuracy",
    "val/best",
]

api = wandb.Api()
summary_lines = []
for name in RUN_NAMES:
    runs = list(api.runs(PROJECT, filters={"display_name": name}))
    if not runs:
        print(f"[skip] no run with name {name!r}")
        continue
    # Most recent first
    runs.sort(key=lambda r: r.created_at, reverse=True)
    r = runs[0]
    print(f"[{name}] id={r.id}  state={r.state}  created={r.created_at}")

    # Full history (samples=10_000 to avoid downsampling)
    hist = r.history(samples=10000, keys=KEYS, pandas=True)
    out_csv = OUT / f"{name}.csv"
    hist.to_csv(out_csv, index=False)
    print(f"   → {out_csv} ({len(hist)} rows)")

    summary_lines.append(f"=== {name} (id={r.id}, state={r.state}) ===")
    for k, v in r.summary.items():
        # Filter out W&B internal keys
        if k.startswith("_") or k in ("graph", "model", "artifacts"):
            continue
        summary_lines.append(f"  {k}: {v}")
    summary_lines.append("")

(OUT / "summary.txt").write_text("\n".join(summary_lines))
print(f"\nWrote {OUT/'summary.txt'}")
print(f"All files in {OUT}:")
for p in sorted(OUT.iterdir()):
    print(f"   {p.name}  ({p.stat().st_size} bytes)")
