#!/usr/bin/env python3
"""Export a DiffAb fine-tune ``log.txt`` to panel-format CSVs.

Companion to ``export_wandb_run.py``. Reads the file logger output
written by ``scripts/diffab_ft/train.py`` and emits the same panel-CSV
format that ``export_wandb_run.py`` produces — so the existing
``summarize_run.py`` pipeline works unchanged on top of it.

Why this exists
---------------
``export_wandb_run.py`` is the primary path: it reads the ``.wandb``
binary directly via ``wandb.sdk.internal.datastore``. But that API is
private and has changed across wandb versions; if a venv upgrade
breaks it, you still have ``log.txt`` on disk with everything we
actually need to diagnose a run:

* ``train/loss``, ``train/loss_{rot,pos,seq}``, ``train/grad_norm``,
  ``train/lr``, ``train/ms`` — sampled every 25 iters by
  [train.py](scripts/diffab_ft/train.py)'s logger.info call.
* ``val/loss`` (overall + best) and per-component ``val/loss_rot``,
  ``val/loss_pos``, ``val/loss_seq`` — written every ``val_freq``
  iters via upstream ``log_losses`` and ``train.py``'s val block.

The 25-iter sampling is plenty for percentile and slope estimates;
val is logged at the same frequency W&B sees it, so val statistics
are exact.

Output is one CSV per metric, format matching ``export_wandb_run.py``::

    train_loss.csv:
      Step,seed42_v2 - train/loss
      1,0.9123
      26,0.8901
      ...

Usage
-----
::

    python scripts/diffab_ft/export_log.py \\
        runs/vhh_ft/seed42_v2/log.txt \\
        --out-dir runs/vhh_ft/seed42_v2/log_export

    python scripts/diffab_ft/summarize_run.py \\
        runs/vhh_ft/seed42_v2/log_export \\
        --compare-to runs/vhh_ft/seed42/log_export \\
        --out runs/vhh_ft/seed42_v2/diagnostic.md
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Regexes for the log lines emitted by scripts/diffab_ft/train.py ────
# Train (every 25 iters):
#   "iter   500 | loss 0.4500 (rot 0.300 pos 0.100 seq 0.050)
#                 | grad 5.20 | lr 1.00e-05 | 120ms"
TRAIN_RE = re.compile(
    r"iter\s+(\d+)\s*\|\s*loss\s+([\d.]+)\s*"
    r"\(rot\s+([\d.]+)\s+pos\s+([\d.]+)\s+seq\s+([\d.]+)\)\s*"
    r"\|\s*grad\s+([\d.]+)\s*\|\s*lr\s+([\d.eE+-]+)\s*\|\s*(\d+)ms"
)
# Val (overall + best, written by train.py at line 512):
#   "iter 200 | EMA val loss 0.7500 (best 0.7500)"
VAL_OVERALL_RE = re.compile(
    r"iter\s+(\d+)\s*\|\s*EMA val loss\s+([\d.]+)\s*\(best\s+([\d.]+)\)"
)
# Val (per-component, written by upstream log_losses via tape.log).
# Example:
#   "[val] Iter 00200 | loss 0.7500 | loss(rot) 0.6000 | loss(pos) 0.1500 | loss(seq) 0.0500"
VAL_HEADER_RE = re.compile(
    r"\[val\]\s+Iter\s+(\d+)\s*\|\s*loss\s+([\d.]+)(.*)$"
)
VAL_COMP_RE = re.compile(r"loss\(([A-Za-z_]+)\)\s+([\d.]+)")


def _label_from_log_path(p: Path) -> str:
    """Pick the user-meaningful run label from a path like
    ``runs/vhh_ft/seed42_v2/log.txt`` → ``seed42_v2``.
    """
    parent = p.parent
    if parent.name and parent.name not in ("", "/", "."):
        return parent.name
    return p.stem


def parse_log(path: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Return ``{metric_name: [(step, value), ...]}`` parsed from log.txt."""
    by_metric: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            m = TRAIN_RE.search(line)
            if m:
                step = int(m.group(1))
                by_metric["train/loss"].append((step, float(m.group(2))))
                by_metric["train/loss_rot"].append((step, float(m.group(3))))
                by_metric["train/loss_pos"].append((step, float(m.group(4))))
                by_metric["train/loss_seq"].append((step, float(m.group(5))))
                by_metric["train/grad_norm"].append((step, float(m.group(6))))
                by_metric["train/lr"].append((step, float(m.group(7))))
                by_metric["train/ms"].append((step, float(m.group(8))))
                continue
            m = VAL_HEADER_RE.search(line)
            if m:
                step = int(m.group(1))
                by_metric["val/loss"].append((step, float(m.group(2))))
                rest = m.group(3)
                for comp_name, comp_val in VAL_COMP_RE.findall(rest):
                    by_metric[f"val/loss_{comp_name}"].append(
                        (step, float(comp_val))
                    )
                continue
            m = VAL_OVERALL_RE.search(line)
            if m:
                step = int(m.group(1))
                # Use as fallback if no per-component val line was seen
                # at this step. Don't overwrite an existing val/loss.
                if not any(s == step for s, _ in by_metric.get("val/loss", [])):
                    by_metric["val/loss"].append((step, float(m.group(2))))
                # `best so far` series — useful for sanity-checking the
                # early-stop logic against the diagnostic.
                by_metric["val/best"].append((step, float(m.group(3))))
                continue
    return by_metric


def write_csvs(
    by_metric: Dict[str, List[Tuple[int, float]]],
    out_dir: Path,
    label: str,
) -> int:
    """Write one panel-format CSV per metric. Returns total row count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_total = 0
    for metric in sorted(by_metric):
        safe_name = metric.replace("/", "_") + ".csv"
        path = out_dir / safe_name
        col_header = f"{label} - {metric}"
        # Dedupe by step (keep latest value).
        latest: Dict[int, float] = {}
        for step, val in by_metric[metric]:
            latest[step] = val
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", col_header])
            for step in sorted(latest):
                writer.writerow([step, latest[step]])
        n_total += len(latest)
        print(f"  {safe_name:32s} {len(latest):>6d} rows")
    return n_total


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "log", type=Path,
        help="Path to runs/<run>/log.txt produced by scripts/diffab_ft/train.py",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Directory to write per-metric panel-format CSVs into.",
    )
    parser.add_argument(
        "--run-label", type=str, default=None,
        help="String used in CSV column headers ('<label> - <metric>'). "
             "Defaults to the run output dir name (e.g. 'seed42_v2').",
    )
    args = parser.parse_args()

    if not args.log.exists():
        sys.exit(f"log file not found: {args.log}")
    if args.log.is_dir():
        sys.exit(f"expected a log.txt file, got a directory: {args.log}")

    label = args.run_label or _label_from_log_path(args.log)
    by_metric = parse_log(args.log)
    if not by_metric:
        sys.exit(
            f"No metric lines matched in {args.log}. "
            "Is this the log.txt produced by scripts/diffab_ft/train.py?"
        )
    print(f"Source: log file: {args.log}")
    print(f"Label:  {label}")
    n_rows = write_csvs(by_metric, args.out_dir, label)
    print(
        f"\nExported {len(by_metric)} metrics ({n_rows} rows) "
        f"→ {args.out_dir}",
    )
    print(
        "Next: python scripts/diffab_ft/summarize_run.py "
        f"{args.out_dir}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
