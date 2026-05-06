#!/usr/bin/env python3
"""Compact summarizer for W&B-exported DiffAb fine-tune run CSVs.

Why this exists
---------------
A single fine-tune run produces thousands of per-step ``train/*`` data
points and ~100 ``val/*`` data points across multiple metrics. Pasting
those into a chat for diagnosis is impractical; eyeballing screenshots
loses precision and doesn't aggregate across runs. This script reduces a
W&B export directory to a markdown report (~30 lines per run) that
captures the load-bearing diagnostics:

  * **Warmup ramp detection** — finds the iter at which ``train/lr``
    plateaued, so post-warmup statistics are reported separately from
    warmup. Critical because the first ~1000 iters of a fine-tune have
    very different gradient/loss dynamics than steady state.
  * **Per-phase percentiles + slopes** for ``train/loss`` and
    ``train/grad_norm``. Slope/1k captures whether train loss is
    drifting up (bad) or down (good); percentiles flag spikes that
    would otherwise be hidden by means.
  * **Full val trajectory table** (small, paste-friendly) including
    per-component breakdowns (``val/loss_rot``, ``loss_pos``,
    ``loss_seq``) — these are the load-bearing signal for whether
    fine-tuning helps and *which loss component* drove the change.
  * **Best-val landmark + termination iter**, so "killed early vs
    converged vs diverged" is unambiguous.

Input format
------------
W&B "Export panel data" produces one CSV per panel/metric with columns:

    Step, "<run> - <metric>", "<run> - <metric>__MIN", ...

Multiple runs become extra column groups. Multiple panels become
separate files. Drop them all in one directory and point this script at
it; metric names are parsed from headers, filenames are ignored.

Stdlib-only on purpose — runs in any Python ≥ 3.9 with no install step.

Usage
-----
::

    python scripts/diffab_ft/summarize_run.py path/to/wandb_export_dir
    python scripts/diffab_ft/summarize_run.py path/to/v2 --compare-to path/to/v1
    python scripts/diffab_ft/summarize_run.py path/to/v2 --out report.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Type aliases ────────────────────────────────────────────────────────
Series = List[Tuple[int, float]]                    # [(step, value), ...]
RunDict = Dict[str, Series]                         # {metric: series}
Runs = Dict[str, RunDict]                           # {run_name: RunDict}

# Order val/* columns in this canonical order (others append at end).
VAL_METRIC_ORDER = ("val/loss", "val/loss_rot", "val/loss_pos",
                    "val/loss_seq", "val/best")


# ── CSV loading ─────────────────────────────────────────────────────────
def _parse_header(col: str) -> Optional[Tuple[str, str]]:
    """Parse a W&B column header ``'<run> - <metric>'``.

    Returns ``None`` for the ``Step`` column or for ``__MIN`` / ``__MAX``
    band columns we don't summarize.
    """
    if " - " not in col:
        return None
    run, metric = col.split(" - ", 1)
    metric = metric.strip()
    if metric.endswith("__MIN") or metric.endswith("__MAX"):
        return None
    return run.strip(), metric


def load_csv(path: Path) -> Runs:
    """Return ``{run: {metric: [(step, val), ...]}}`` from one CSV file."""
    runs: Runs = {}
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return runs
        try:
            step_idx = header.index("Step")
        except ValueError:
            return runs  # not a W&B panel CSV
        col_to_key: Dict[int, Tuple[str, str]] = {}
        for i, col in enumerate(header):
            if i == step_idx:
                continue
            key = _parse_header(col)
            if key is not None:
                col_to_key[i] = key
                runs.setdefault(key[0], {}).setdefault(key[1], [])
        for row in reader:
            if not row or step_idx >= len(row) or not row[step_idx]:
                continue
            try:
                step = int(float(row[step_idx]))
            except ValueError:
                continue
            for i, (run, metric) in col_to_key.items():
                if i >= len(row) or row[i] == "":
                    continue
                try:
                    runs[run][metric].append((step, float(row[i])))
                except ValueError:
                    pass
    return runs


def load_dir(d: Path) -> Runs:
    """Merge every ``*.csv`` in ``d`` into a single ``{run: {metric: ...}}``."""
    out: Runs = {}
    csv_paths = sorted(d.glob("*.csv"))
    if not csv_paths:
        sys.exit(f"No CSV files found in {d}")
    for path in csv_paths:
        for run, metrics in load_csv(path).items():
            dst = out.setdefault(run, {})
            for metric, series in metrics.items():
                # Concatenate; we sort + dedupe at the end.
                dst.setdefault(metric, []).extend(series)
    # Sort by step and dedupe (a metric can appear in two CSVs on
    # overlapping step ranges — keep latest value).
    for metrics in out.values():
        for metric, series in metrics.items():
            seen: Dict[int, float] = {}
            for step, val in series:
                seen[step] = val
            metrics[metric] = sorted(seen.items())
    return out


# ── Analysis primitives ─────────────────────────────────────────────────
def percentiles(vals: List[float], ps: List[float]) -> Dict[float, float]:
    """Nearest-rank percentiles. Returns NaN for empty input."""
    if not vals:
        return {p: float("nan") for p in ps}
    s = sorted(vals)
    n = len(s)
    return {p: s[max(0, min(n - 1, int(p / 100 * (n - 1))))] for p in ps}


def linear_slope(series: Series) -> float:
    """Ordinary-least-squares slope of value vs step. NaN if undefined."""
    n = len(series)
    if n < 2:
        return float("nan")
    sx = sum(s for s, _ in series)
    sy = sum(v for _, v in series)
    sxy = sum(s * v for s, v in series)
    sxx = sum(s * s for s, _ in series)
    denom = n * sxx - sx * sx
    return float("nan") if denom == 0 else (n * sxy - sx * sy) / denom


def detect_warmup(lr_series: Series) -> Optional[int]:
    """Return the iter at which LR plateaus (warmup ends), or None.

    Heuristic: find the first step where lr ≥ 99% of its run-max and
    stays within 50% of run-max for at least 50 subsequent points. If
    LR is flat (max == min), no warmup was applied.
    """
    if len(lr_series) < 50:
        return None
    vals = [v for _, v in lr_series]
    max_lr = max(vals)
    if max_lr == 0 or min(vals) >= max_lr * 0.99:
        return None  # flat lr, no warmup detectable
    threshold = max_lr * 0.99
    floor = max_lr * 0.5
    for i, (step, v) in enumerate(lr_series):
        if v >= threshold:
            tail = lr_series[i:i + 50]
            if all(t_v >= floor for _, t_v in tail):
                return step
    return None


def split_by_phase(series: Series, boundary: int) -> Tuple[Series, Series]:
    pre = [(s, v) for s, v in series if s <= boundary]
    post = [(s, v) for s, v in series if s > boundary]
    return pre, post


# ── Reporting ───────────────────────────────────────────────────────────
def _fmt_phase(label: str, sub: Series) -> str:
    vals = [v for _, v in sub]
    if not vals:
        return f"- {label}: (empty)"
    pcts = percentiles(vals, [50, 90, 99])
    slope_per_1k = linear_slope(sub) * 1000
    return (
        f"- {label} (n={len(sub)}): "
        f"p50={pcts[50]:.3f}, p90={pcts[90]:.3f}, p99={pcts[99]:.3f}, "
        f"max={max(vals):.3f} | slope/1k={slope_per_1k:+.4f}"
    )


def _fmt_train_metric(name: str, series: Series, warmup_end: Optional[int]) -> List[str]:
    out = [f"\n### `{name}`"]
    if warmup_end is not None:
        pre, post = split_by_phase(series, warmup_end)
        out.append(_fmt_phase("warmup    ", pre))
        out.append(_fmt_phase("post-warmup", post))
    else:
        out.append(_fmt_phase("all", series))
    return out


def _val_metric_order(metrics: List[str]) -> List[str]:
    canonical = [m for m in VAL_METRIC_ORDER if m in metrics]
    extras = sorted(m for m in metrics if m.startswith("val/")
                    and m not in VAL_METRIC_ORDER)
    return canonical + extras


def _fmt_val_table(metrics_dict: RunDict) -> List[str]:
    val_metrics = _val_metric_order(list(metrics_dict.keys()))
    if not val_metrics:
        return []
    out = ["\n### Val trajectory\n"]
    short_names = [m.replace("val/", "") for m in val_metrics]
    out.append("| step | " + " | ".join(short_names) + " |")
    out.append("|" + "|".join(["---"] * (len(val_metrics) + 1)) + "|")
    steps = sorted({s for m in val_metrics for s, _ in metrics_dict[m]})
    for step in steps:
        row = [str(step)]
        for m in val_metrics:
            val = next((v for s, v in metrics_dict[m] if s == step), None)
            row.append(f"{val:.4f}" if val is not None else "—")
        out.append("| " + " | ".join(row) + " |")
    if "val/loss" in metrics_dict and metrics_dict["val/loss"]:
        best_step, best_val = min(metrics_dict["val/loss"], key=lambda x: x[1])
        n_after = sum(1 for s, _ in metrics_dict["val/loss"] if s > best_step)
        out.append(
            f"\n- **Best `val/loss`:** {best_val:.4f} at iter {best_step} "
            f"({n_after} validations after best)"
        )
    return out


def report_run(run: str, metrics: RunDict) -> str:
    out = [f"## Run: `{run}`\n"]
    last_iter = max(
        (max((s for s, _ in series), default=-1) for series in metrics.values()),
        default=-1,
    )
    out.append(f"- **Last logged iter:** {last_iter}")
    lr = metrics.get("train/lr", [])
    warmup_end = detect_warmup(lr)
    if warmup_end is not None:
        out.append(
            f"- **Warmup detected:** ramp ended at iter {warmup_end} "
            f"(LR plateau at {max(v for _, v in lr):.2e})"
        )
    elif lr:
        out.append(f"- **Warmup detected:** none (LR flat at {lr[0][1]:.2e})")
    else:
        out.append("- **Warmup detected:** unknown (no `train/lr` series)")

    for name in ("train/loss", "train/grad_norm"):
        if name in metrics and metrics[name]:
            out.extend(_fmt_train_metric(name, metrics[name], warmup_end))
    out.extend(_fmt_val_table(metrics))
    return "\n".join(out)


def report_all(runs: Runs, title: str) -> str:
    parts = [f"# {title}\n"]
    for run in sorted(runs):
        parts.append(report_run(run, runs[run]))
    return "\n\n".join(parts)


# ── CLI ─────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_dir", type=Path,
        help="Directory of W&B-exported panel CSVs to summarize.",
    )
    parser.add_argument(
        "--compare-to", type=Path, default=None,
        help="Optional second directory; its summary is appended below.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Write the markdown report to this file instead of stdout.",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"Not a directory: {args.input_dir}")
    primary = report_all(load_dir(args.input_dir), "W&B run summary")
    sections = [primary]
    if args.compare_to is not None:
        if not args.compare_to.is_dir():
            sys.exit(f"Not a directory: {args.compare_to}")
        sections.append(report_all(load_dir(args.compare_to),
                                   f"Comparison: `{args.compare_to.name}`"))
    output = "\n\n".join(sections) + "\n"
    if args.out is not None:
        args.out.write_text(output)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
