#!/usr/bin/env python3
"""Export a W&B run's full history as panel-format CSVs.

Companion to ``summarize_run.py``. Goes from a finished W&B run on
wandb.ai to one CSV per metric in the exact format
``summarize_run.py`` expects, without the manual "Export panel data"
clicking. Designed to run on Snellius (or wherever you train) so the
diagnostic loop never needs to leave the remote machine:

    python scripts/diffab_ft/export_wandb_run.py <run> \\
        --out-dir runs/vhh_ft/seed42_v2/wandb_export
    python scripts/diffab_ft/summarize_run.py \\
        runs/vhh_ft/seed42_v2/wandb_export \\
        --out runs/vhh_ft/seed42_v2/diagnostic.md

The ``<run>`` argument accepts either of:

    entity/project/run_id
    https://wandb.ai/entity/project/runs/run_id   (paste from address bar)

Authentication is whatever ``wandb`` already uses (``~/.netrc`` /
``WANDB_API_KEY`` env / ``wandb login`` on the login node). Pulls the
full step-by-step history via ``run.scan_history()`` (no downsampling).

Output format
-------------
One CSV per metric, named after the metric with ``/`` → ``_``::

    train_loss.csv:
      Step,seed42_v2 - train/loss
      1,0.9123
      2,0.8901
      ...

Two-column CSVs in this exact shape are what ``summarize_run.py``'s
header parser expects — it splits on `` - `` to recover ``(run, metric)``.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional

# ── Run-path parsing ────────────────────────────────────────────────────
_WANDB_URL_RE = re.compile(
    r"https?://(?:wandb\.ai|api\.wandb\.ai)/"
    r"(?P<entity>[^/]+)/(?P<project>[^/]+)/runs/(?P<run_id>[^/?#]+)"
)


def _parse_run_path(s: str) -> str:
    """Accept either ``entity/project/run_id`` or a wandb.ai URL.

    Returns the canonical ``entity/project/run_id`` form expected by
    ``wandb.Api().run(path)``.
    """
    s = s.strip()
    m = _WANDB_URL_RE.match(s)
    if m:
        return f"{m['entity']}/{m['project']}/{m['run_id']}"
    if s.count("/") != 2:
        sys.exit(
            f"Could not parse '{s}' as a W&B run. Expected "
            "'entity/project/run_id' or a wandb.ai URL."
        )
    return s


# ── Filtering helpers ──────────────────────────────────────────────────
def _matches_any_prefix(key: str, prefixes: Optional[List[str]]) -> bool:
    if prefixes is None:
        return True
    return any(key.startswith(p) for p in prefixes)


def _is_skipped_internal(key: str) -> bool:
    """Skip W&B's bookkeeping fields (_step, _runtime, _timestamp, ...)."""
    return key.startswith("_")


# ── Main ────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run",
        help="W&B run identifier: 'entity/project/run_id' or a wandb.ai URL.",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Directory to write per-metric CSVs into (created if needed).",
    )
    parser.add_argument(
        "--run-label", type=str, default=None,
        help="Override the run-name string used in CSV column headers. "
             "Defaults to the W&B run.name. Useful when run.name is auto-"
             "generated and you want a stable label across exports.",
    )
    parser.add_argument(
        "--include", nargs="+", default=None, metavar="PREFIX",
        help="Only export metrics whose key starts with any of these "
             "prefixes. E.g. --include train/ val/ . Default: all.",
    )
    parser.add_argument(
        "--exclude", nargs="+", default=None, metavar="PREFIX",
        help="Exclude metrics whose key starts with any of these prefixes. "
             "Applied after --include.",
    )
    args = parser.parse_args()

    try:
        import wandb  # noqa: WPS433  — runtime dep
    except ImportError:
        sys.exit(
            "wandb not installed in this environment. "
            "Activate the project venv or `pip install wandb`."
        )

    run_path = _parse_run_path(args.run)
    api = wandb.Api()
    try:
        run = api.run(run_path)
    except Exception as exc:  # broad: wandb wraps many auth/network errors
        sys.exit(f"Failed to fetch run '{run_path}': {exc}")

    label = args.run_label or run.name
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect step-indexed values per metric ─────────────────────────
    # scan_history returns the full unsampled history; rows are dicts
    # where keys are metric names and missing metrics are simply absent
    # (W&B doesn't pad). Most rows only carry train/* keys; rows at
    # validation steps additionally carry val/*.
    by_metric: dict[str, List[tuple[int, float]]] = defaultdict(list)
    n_rows_seen = 0
    for row in run.scan_history():
        n_rows_seen += 1
        step = row.get("_step")
        if step is None:
            continue
        for key, val in row.items():
            if _is_skipped_internal(key):
                continue
            if not _matches_any_prefix(key, args.include):
                continue
            if _matches_any_prefix(key, args.exclude):
                continue
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                # Skip non-numeric (e.g. logged tables, images, strings).
                continue
            by_metric[key].append((int(step), fval))

    if not by_metric:
        sys.exit(
            f"Fetched {n_rows_seen} history rows but no numeric metrics "
            "matched the include/exclude filters."
        )

    # ── Write one CSV per metric ───────────────────────────────────────
    n_total = 0
    for metric in sorted(by_metric):
        # Build a filesystem-safe filename. Slashes in keys (e.g.
        # 'train/loss') are the canonical W&B convention; replace them
        # with underscores for portability across platforms.
        safe_name = metric.replace("/", "_").replace(" ", "_") + ".csv"
        path = args.out_dir / safe_name
        col_header = f"{label} - {metric}"
        # Dedupe by step in case the run has duplicate logs at the same
        # step (rare but happens with resumes); keep the first.
        seen_steps: set[int] = set()
        unique_rows: List[tuple[int, float]] = []
        for step, val in sorted(by_metric[metric]):
            if step in seen_steps:
                continue
            seen_steps.add(step)
            unique_rows.append((step, val))
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", col_header])
            for step, val in unique_rows:
                writer.writerow([step, val])
        n_total += len(unique_rows)
        print(f"  {safe_name:30s} {len(unique_rows):>6d} rows")

    print(
        f"\nExported {len(by_metric)} metrics ({n_total} rows) "
        f"from run '{label}' → {args.out_dir}",
    )
    print(
        "Next: python scripts/diffab_ft/summarize_run.py "
        f"{args.out_dir}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
