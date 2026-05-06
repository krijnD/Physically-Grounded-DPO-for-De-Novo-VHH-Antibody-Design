#!/usr/bin/env python3
"""Export a W&B run's full history as panel-format CSVs.

Companion to ``summarize_run.py``. Designed to run on the same machine
that did the training — no laptop round-trip, no manual "Export panel
data" clicking. Two modes, auto-detected from the argument:

* **Local mode** (default for any path that exists on disk): reads the
  binary ``.wandb`` log file that ``wandb`` writes alongside every run.
  No network needed. Works for both online and offline runs — the
  binary is present either way.
* **Cloud mode** (any non-path: ``entity/project/run_id`` or wandb.ai
  URL): uses ``wandb.Api()`` to fetch history from wandb.ai. Useful
  for older runs whose local dir is no longer available.

Output is one CSV per metric, named after the metric with ``/`` →
``_``. The format matches what ``summarize_run.py`` expects::

    train_loss.csv:
      Step,seed42_v2 - train/loss
      1,0.9123
      2,0.8901
      ...

End-to-end remote workflow::

    python scripts/diffab_ft/export_wandb_run.py \\
        runs/vhh_ft/seed42_v2 \\
        --out-dir runs/vhh_ft/seed42_v2/wandb_export

    python scripts/diffab_ft/summarize_run.py \\
        runs/vhh_ft/seed42_v2/wandb_export \\
        --out runs/vhh_ft/seed42_v2/diagnostic.md

The local-mode argument can point at any of:

  * the run output dir (``runs/vhh_ft/seed42_v2``),
  * the run's ``wandb/`` subdir,
  * a specific ``run-<id>/`` dir or ``latest-run`` symlink,
  * the ``.wandb`` file directly.

The script walks the path to find the ``.wandb`` file automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ── Run-path parsing for cloud mode ────────────────────────────────────
_WANDB_URL_RE = re.compile(
    r"https?://(?:wandb\.ai|api\.wandb\.ai)/"
    r"(?P<entity>[^/]+)/(?P<project>[^/]+)/runs/(?P<run_id>[^/?#]+)"
)


def _parse_cloud_run_path(s: str) -> str:
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
            f"Could not parse '{s}' as a W&B run identifier. Expected "
            "either a local path that exists, a wandb.ai URL, or "
            "'entity/project/run_id'."
        )
    return s


# ── Local-mode path resolution ──────────────────────────────────────────
def _resolve_local_run(path: Path) -> Tuple[Path, str]:
    """Return ``(wandb_file, default_label)`` for a local run path.

    The ``default_label`` is the deepest non-wandb-internal directory
    name in the path — for ``runs/vhh_ft/seed42_v2/...`` it's
    ``seed42_v2``. The user can override via ``--run-label``.
    """
    p = path.resolve()
    if not p.exists():
        sys.exit(f"Path does not exist: {path}")

    # Direct .wandb file.
    if p.is_file():
        if p.suffix != ".wandb":
            sys.exit(f"File is not a .wandb log: {path}")
        return p, _label_from_path(p.parent)

    # Directory: search for *.wandb in priority order.
    candidates: List[Path] = []
    candidates.extend(p.glob("*.wandb"))                      # run-XX/ itself
    candidates.extend(p.glob("latest-run/*.wandb"))           # wandb/latest-run
    candidates.extend(p.glob("run-*/*.wandb"))                # wandb/run-XX
    candidates.extend(p.glob("wandb/latest-run/*.wandb"))     # run-output/wandb/latest-run
    candidates.extend(p.glob("wandb/run-*/*.wandb"))          # run-output/wandb/run-XX

    # Deduplicate while preserving order; some patterns can match the same file
    # (e.g. latest-run is often a symlink to a real run-* dir).
    seen: set = set()
    unique: List[Path] = []
    for c in candidates:
        rc = c.resolve()
        if rc in seen:
            continue
        seen.add(rc)
        unique.append(c)

    if not unique:
        sys.exit(
            f"No .wandb file found under {path}. Looked for "
            "'*.wandb', 'run-*/*.wandb', 'latest-run/*.wandb', "
            "and the 'wandb/' subdir of those."
        )

    # Most-recently-modified wins. For a run-output dir like
    # runs/vhh_ft/seed42_v2 with a single run inside wandb/, this picks
    # that run; for a parent dir with multiple runs it picks the most
    # recent, which is almost always what the user wants.
    wandb_file = max(unique, key=lambda f: f.stat().st_mtime)
    return wandb_file, _label_from_path(p)


def _label_from_path(p: Path) -> str:
    """Pick the user-meaningful label segment of ``p``.

    Skips ``wandb`` and ``run-*`` / ``latest-run`` segments which are
    W&B internal naming, walking up to the first directory that looks
    like the user's own run name (e.g. ``seed42_v2``).
    """
    for part in reversed(p.parts):
        if part in ("wandb", "latest-run") or part.startswith("run-"):
            continue
        if part in ("", "/", "."):
            continue
        return part
    return p.name


# ── Local-mode history reader ──────────────────────────────────────────
def _iter_local_history(
    wandb_file: Path,
) -> Iterable[Tuple[int, str, float]]:
    """Yield ``(step, key, value)`` tuples by scanning ``wandb_file``.

    Uses ``wandb.sdk.internal.datastore.DataStore`` to read W&B's
    binary log format directly. This API has been stable across
    recent wandb versions; if it breaks on a future upgrade, fall
    back to cloud mode (``--source <wandb_url>``).
    """
    try:
        from wandb.proto import wandb_internal_pb2 as wandb_pb2
        from wandb.sdk.internal.datastore import DataStore
    except ImportError as exc:
        sys.exit(
            f"Cannot import wandb internals for local mode ({exc}). "
            "Either install/upgrade wandb in this environment, or "
            "pass the run as a wandb.ai URL / 'entity/project/run_id' "
            "to use cloud mode instead."
        )

    ds = DataStore()
    ds.open_for_scan(str(wandb_file))
    try:
        while True:
            try:
                raw = ds.scan_data()
            except Exception:
                # Malformed tail (run was interrupted) — stop cleanly.
                break
            if raw is None:
                break
            record = wandb_pb2.Record()
            try:
                record.ParseFromString(raw)
            except Exception:
                continue
            if record.WhichOneof("record_type") != "history":
                continue
            history = record.history

            # Step is an item with key '_step'. Pull it first; skip the
            # row if absent (defensive — every history record should
            # have it, but the format isn't formally guaranteed).
            step: Optional[int] = None
            for item in history.item:
                if item.key == "_step":
                    try:
                        step = int(json.loads(item.value_json))
                    except Exception:
                        pass
                    break
            if step is None:
                continue

            for item in history.item:
                if item.key.startswith("_"):
                    continue  # _step, _runtime, _timestamp, ...
                try:
                    val = json.loads(item.value_json)
                    fval = float(val)
                except (json.JSONDecodeError, TypeError, ValueError):
                    # Skip non-numeric (tables, images, NaN as string, ...).
                    continue
                yield step, item.key, fval
    finally:
        # DataStore exposes no explicit close; rely on GC. Wrapped in
        # try/finally so a parser exception doesn't strand handles.
        pass


# ── Cloud-mode history reader ──────────────────────────────────────────
def _iter_cloud_history(
    run_path: str,
) -> Tuple[Iterable[Tuple[int, str, float]], str]:
    """Return ``(history_iter, run_name)`` for a cloud-hosted run."""
    try:
        import wandb  # noqa: WPS433  — runtime dep
    except ImportError:
        sys.exit(
            "wandb not installed in this environment. "
            "Activate the project venv or `pip install wandb`."
        )

    api = wandb.Api()
    try:
        run = api.run(run_path)
    except Exception as exc:
        sys.exit(f"Failed to fetch run '{run_path}': {exc}")

    def _gen() -> Iterable[Tuple[int, str, float]]:
        for row in run.scan_history():
            step = row.get("_step")
            if step is None:
                continue
            for key, val in row.items():
                if key.startswith("_"):
                    continue
                if val is None:
                    continue
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                yield int(step), key, fval

    return _gen(), run.name


# ── Filtering helpers ──────────────────────────────────────────────────
def _matches_any_prefix(key: str, prefixes: Optional[List[str]]) -> bool:
    if prefixes is None:
        return True
    return any(key.startswith(p) for p in prefixes)


# ── CSV writing ─────────────────────────────────────────────────────────
def _write_csvs(
    by_metric: Dict[str, List[Tuple[int, float]]],
    out_dir: Path,
    label: str,
) -> int:
    """Write one panel-format CSV per metric. Returns total row count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_total = 0
    for metric in sorted(by_metric):
        safe_name = metric.replace("/", "_").replace(" ", "_") + ".csv"
        path = out_dir / safe_name
        col_header = f"{label} - {metric}"

        # Dedupe by step (resumes can produce duplicates); keep the
        # latest value at each step.
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


# ── CLI ─────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "source",
        help="Either a local path (run output dir, wandb/ subdir, "
             "run-<id>/ dir, or .wandb file) — or a wandb.ai URL / "
             "'entity/project/run_id' for cloud mode. Auto-detected: "
             "if the argument is an existing path, local mode is used.",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Directory to write per-metric CSVs into (created if needed).",
    )
    parser.add_argument(
        "--run-label", type=str, default=None,
        help="String used in CSV column headers ('<label> - <metric>'). "
             "Defaults to the run output dir name (local mode) or the "
             "W&B run.name (cloud mode).",
    )
    parser.add_argument(
        "--include", nargs="+", default=None, metavar="PREFIX",
        help="Only export metrics whose key starts with any of these "
             "prefixes (e.g. --include train/ val/). Default: all.",
    )
    parser.add_argument(
        "--exclude", nargs="+", default=None, metavar="PREFIX",
        help="Exclude metrics whose key starts with any of these prefixes. "
             "Applied after --include.",
    )
    args = parser.parse_args()

    # Mode dispatch: existing path → local; otherwise → cloud.
    src = args.source
    is_local = Path(src).exists()

    if is_local:
        wandb_file, default_label = _resolve_local_run(Path(src))
        history_iter = _iter_local_history(wandb_file)
        label = args.run_label or default_label
        source_desc = f"local file: {wandb_file}"
    else:
        run_path = _parse_cloud_run_path(src)
        history_iter, default_label = _iter_cloud_history(run_path)
        label = args.run_label or default_label
        source_desc = f"cloud run: {run_path}"

    # Collect, applying include/exclude filters.
    by_metric: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    n_records = 0
    for step, key, val in history_iter:
        n_records += 1
        if not _matches_any_prefix(key, args.include):
            continue
        if _matches_any_prefix(key, args.exclude):
            continue
        by_metric[key].append((step, val))

    if not by_metric:
        sys.exit(
            f"Read {n_records} records from {source_desc} but no "
            "numeric metrics matched the include/exclude filters."
        )

    print(f"Source: {source_desc}")
    print(f"Label:  {label}")
    n_rows = _write_csvs(by_metric, args.out_dir, label)
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
