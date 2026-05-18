#!/usr/bin/env python3
"""End-to-end orchestrator for the ANDD → DiffAb data prep pipeline.

Chains the six data-prep scripts in order:

    1. filter_andd_vhh.py            (Excel → ANDD_VHH_with_structure.csv)
    2. fetch_deposition_dates.py     (add RCSB deposition dates)
    3. subset_vhh_structures.py      (filter to post-cutoff PDBs)
    4. curate_andd.py                (ANARCI-verify VHH + antigen chains)
    5. prepare_manifest.py           (DiffAb-compatible TSV)
    6. cluster_split.py              (PDB-ATOM dedup + cluster splits)

Then runs the post-Step-6 audits (PDB↔CSV consistency, PDB-ATOM diversity)
and prints a verdict table summarising the state of the dataset for the
upcoming finetune.

Idempotent by default: each step is skipped if its output already exists,
unless --force-step <name> or --force-all is set. Use --skip-step <name>
to bypass a step entirely (e.g. resuming after a partial run).

A side-by-side comparison against the previous manifest (if present) is
written at the end, so you can see at a glance whether re-running the
pipeline changed the row count (it usually won't — see the docs file
docs/data_pipeline_investigation.md for why).

See docs/data_pipeline_investigation.md for the full motivation, the
investigation that led to this script, and the verification gates.

Run on Snellius (all defaults assume the standard ANDD layout):
    python scripts/diffab_ft/run_data_prep.py

Dry-run on the laptop (prints commands without executing):
    python scripts/diffab_ft/run_data_prep.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable


# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────


class TeeFormatter(logging.Formatter):
    """Compact one-line log format with relative timestamps."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-5s %(message)s",
                         datefmt="%H:%M:%S")


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = TeeFormatter()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    root.addHandler(fh)


log = logging.getLogger("data_prep")


# ─────────────────────────────────────────────────────────────────────
# Step abstraction
# ─────────────────────────────────────────────────────────────────────


@dataclass
class Step:
    """One step in the pipeline."""

    name: str
    description: str
    cmd: list[str]
    output_paths: list[Path]   # all must exist to count as "already done"
    audit: Callable[[], None] | None = None
    elapsed_seconds: float = 0.0
    skipped: bool = False

    def outputs_exist(self) -> bool:
        return all(p.exists() for p in self.output_paths)


# ─────────────────────────────────────────────────────────────────────
# Step builders
# ─────────────────────────────────────────────────────────────────────


def build_steps(args: argparse.Namespace) -> list[Step]:
    """Construct the 6-step pipeline using args-resolved paths."""
    base = args.base_dir
    label = args.label
    cutoff = args.cutoff_date
    repo_data = args.repo_data_dir

    # Snellius-canonical paths (filter_andd_vhh.py hardcodes these)
    with_structure_csv = base / "ANDD_VHH_with_structure.csv"
    no_structure_csv = base / "ANDD_VHH_no_structure.csv"
    vhh_structures_dir = base / "VHH_structures"

    # Date / subset paths (parameterised by --label)
    dates_csv = base / f"andd_real_deposition_dates_{label}.csv"
    subset_dir = base / f"VHH_structures_{label}"
    subset_csv = base / f"ANDD_VHH_with_structure_{label}.csv"

    # Curation paths
    curated_csv = base / f"ANDD_VHH_curated_{label}.csv"
    rejected_csv = base / f"ANDD_VHH_rejected_{label}.csv"

    # Repo-side paths (manifest + splits)
    manifest_tsv = repo_data / "diffab_manifest.tsv"
    splits_dir = repo_data / "clustering"
    splits_json = splits_dir / "cluster_splits.json"

    # Stash for post-run audits / summary
    args._curated_csv = curated_csv
    args._rejected_csv = rejected_csv
    args._manifest_tsv = manifest_tsv
    args._splits_json = splits_json
    args._subset_dir = subset_dir
    args._with_structure_csv = with_structure_csv

    steps: list[Step] = [
        # ── Step 1: Excel → VHH-only CSV (hardcoded I/O paths inside script) ─
        Step(
            name="filter",
            description="Excel → VHH-only CSV (with-structure / no-structure split)",
            cmd=[PYTHON, str(PROJECT_ROOT / "data scripts/filter_andd_vhh.py")],
            output_paths=[with_structure_csv, no_structure_csv, vhh_structures_dir],
            audit=lambda: _audit_filter(with_structure_csv, no_structure_csv,
                                        vhh_structures_dir),
        ),
        # ── Step 2: Fetch RCSB deposition dates ────────────────────────────
        Step(
            name="dates",
            description=f"Fetch RCSB deposition dates (cutoff {cutoff})",
            cmd=[
                PYTHON, str(PROJECT_ROOT / "data scripts/fetch_deposition_dates.py"),
                "--input", str(with_structure_csv),
                "--cutoff", cutoff,
                "--label", label,
                "--output", str(dates_csv),
            ],
            output_paths=[dates_csv],
            audit=lambda: _audit_dates(dates_csv, label),
        ),
        # ── Step 3: Subset PDB files by date ───────────────────────────────
        Step(
            name="subset",
            description=f"Subset PDBs and metadata to {label} cohort",
            cmd=[
                PYTHON, str(PROJECT_ROOT / "data scripts/subset_vhh_structures.py"),
                "--dates-csv", str(dates_csv),
                "--structures-dir", str(vhh_structures_dir),
                "--output-dir", str(subset_dir),
                "--metadata-csv", str(with_structure_csv),
                "--output-csv", str(subset_csv),
                "--label", label,
            ],
            output_paths=[subset_dir, subset_csv],
            audit=lambda: _audit_subset(subset_dir, subset_csv),
        ),
        # ── Step 4: Curate (ANARCI verify VHH + antigen chains) ────────────
        Step(
            name="curate",
            description="ANARCI-verify VHH + antigen chains",
            cmd=[
                PYTHON, str(PROJECT_ROOT / "data scripts/curate_andd.py"),
                "--input-csv", str(subset_csv),
                "--pdb-dir", str(subset_dir),
                "--output-csv", str(curated_csv),
                "--rejected-csv", str(rejected_csv),
                "--overwrite-output",
            ],
            output_paths=[curated_csv, rejected_csv],
            audit=lambda: _audit_curate(curated_csv, rejected_csv),
        ),
        # ── Step 5: Build DiffAb manifest TSV ──────────────────────────────
        Step(
            name="manifest",
            description="Build DiffAb manifest TSV from curated CSV",
            cmd=[
                PYTHON, str(PROJECT_ROOT / "scripts/diffab_ft/prepare_manifest.py"),
                "--curated-csv", str(curated_csv),
                "--pdb-dir", str(subset_dir),
                "--output-tsv", str(manifest_tsv),
                "--overwrite",
            ],
            output_paths=[manifest_tsv],
            audit=lambda: _audit_manifest(manifest_tsv),
        ),
        # ── Step 6: Cluster + dedup + splits ───────────────────────────────
        Step(
            name="cluster",
            description="Cluster CDRs, dedup by PDB-ATOM sequence, build splits",
            cmd=[
                PYTHON, str(PROJECT_ROOT / "scripts/diffab_ft/cluster_split.py"),
                "--curated-csv", str(curated_csv),
                "--manifest-tsv", str(manifest_tsv),
                "--pdb-dir", str(subset_dir),
                "--output-dir", str(splits_dir),
                "--dedupe-by", args.dedupe_by,
                "--ratios", str(args.ratios[0]), str(args.ratios[1]), str(args.ratios[2]),
                "--seed", str(args.seed),
                "--audit-antigens",
                "--overwrite",
            ],
            output_paths=[splits_json],
            audit=lambda: _audit_cluster(splits_json),
        ),
    ]
    return steps


# ─────────────────────────────────────────────────────────────────────
# Audit helpers (per-step lightweight summaries)
# ─────────────────────────────────────────────────────────────────────


def _row_count(path: Path, sep: str = ",") -> int:
    """Cheap line count minus header. Used when pandas would be overkill."""
    if not path.exists():
        return 0
    with open(path) as f:
        return max(0, sum(1 for _ in f) - 1)


def _audit_filter(with_csv: Path, no_csv: Path, pdb_dir: Path) -> None:
    n_with = _row_count(with_csv)
    n_no = _row_count(no_csv)
    n_pdbs = len(list(pdb_dir.glob("*.pdb"))) if pdb_dir.exists() else 0
    log.info("  filter audit: %d rows with structure, %d without, %d PDBs copied",
             n_with, n_no, n_pdbs)


def _audit_dates(dates_csv: Path, label: str) -> None:
    try:
        import csv
        n_total = 0
        n_safe = 0
        with open(dates_csv) as f:
            reader = csv.DictReader(f)
            if label not in (reader.fieldnames or []):
                log.warning("  dates audit: label column %r not in %s — header is %s",
                            label, dates_csv.name, reader.fieldnames)
                return
            for row in reader:
                n_total += 1
                if row[label].strip().lower() == "true":
                    n_safe += 1
        log.info("  dates audit: %d PDBs with deposition dates, %d marked safe "
                 "(%s=True)", n_total, n_safe, label)
    except Exception as exc:  # noqa: BLE001
        log.warning("  dates audit failed: %s", exc)


def _audit_subset(subset_dir: Path, subset_csv: Path) -> None:
    n_pdbs = len(list(subset_dir.glob("*.pdb"))) if subset_dir.exists() else 0
    n_rows = _row_count(subset_csv)
    log.info("  subset audit: %d PDBs copied, %d metadata rows", n_pdbs, n_rows)


def _audit_curate(curated_csv: Path, rejected_csv: Path) -> None:
    try:
        import csv
        from collections import Counter
        statuses: Counter[str] = Counter()
        with open(curated_csv) as f:
            for row in csv.DictReader(f):
                statuses[row.get("curation_status", "?")] += 1
        with open(rejected_csv) as f:
            for row in csv.DictReader(f):
                statuses[row.get("curation_status", "rejected")] += 1
        log.info("  curate audit: %d total entries; breakdown:", sum(statuses.values()))
        for status, n in statuses.most_common():
            log.info("    %-25s %d", status, n)
    except Exception as exc:  # noqa: BLE001
        log.warning("  curate audit failed: %s", exc)


def _audit_manifest(manifest_tsv: Path) -> None:
    n = _row_count(manifest_tsv, sep="\t")
    log.info("  manifest audit: %d rows in %s", n, manifest_tsv.name)


def _audit_cluster(splits_json: Path) -> None:
    try:
        d = json.loads(splits_json.read_text())
        params = d.get("params", {})
        n_members = params.get("n_members", "?")
        n_clusters = params.get("n_clusters", "?")
        dedupe_by = params.get("dedupe_by", "?")
        n_before = params.get("n_entries_before_dedup", "?")
        n_after = params.get("n_entries_after_dedup", "?")
        log.info("  cluster audit: %d clusters from %d members "
                 "(dedupe_by=%s, %s→%s)",
                 n_clusters, n_members, dedupe_by, n_before, n_after)
        for split_name, ids in d.get("splits", {}).items():
            log.info("    split %-6s %d entries", split_name, len(ids))
    except Exception as exc:  # noqa: BLE001
        log.warning("  cluster audit failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────
# Post-Step-6 audits (the critical ones)
# ─────────────────────────────────────────────────────────────────────


def run_consistency_audit(args: argparse.Namespace) -> dict:
    """Run audit_pdb_csv_consistency.py and parse its stdout for verdict.

    Returns a dict with keys: 'total', 'consistent', 'consistent_pct',
    'len_mismatch', 'seq_mismatch', 'returncode', 'healthy' (bool).
    """
    audit_script = PROJECT_ROOT / "scripts/diffab_ft/audit_pdb_csv_consistency.py"
    cmd = [
        PYTHON, str(audit_script),
        "--curated-csv", str(args._curated_csv),
        "--manifest-tsv", str(args._manifest_tsv),
        "--pdb-dir", str(args._subset_dir),
    ]
    log.info("running consistency audit...")
    log.info("  cmd: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    # Parse the lines we care about. Use regex to robustly handle the
    # "N ( XX.X%)" formatting where whitespace varies inside parens.
    import re
    result = {"returncode": proc.returncode, "raw_stdout": proc.stdout}
    re_count_pct = re.compile(r":\s*(\d+)\s*\(\s*([\d.]+)%\)")
    re_count = re.compile(r":\s*(\d+)\s*\(")
    for line in proc.stdout.splitlines():
        ls = line.strip()
        if ls.startswith("manifest rows:"):
            m = re.search(r":\s*(\d+)", ls)
            if m:
                result["total"] = int(m.group(1))
        elif "fully consistent:" in ls:
            m = re_count_pct.search(ls)
            if m:
                result["consistent"] = int(m.group(1))
                result["consistent_pct"] = float(m.group(2))
        elif "length mismatch" in ls:
            m = re_count.search(ls)
            if m:
                result["len_mismatch"] = int(m.group(1))
        elif "seq mismatch" in ls:
            m = re_count.search(ls)
            if m:
                result["seq_mismatch"] = int(m.group(1))
    result["healthy"] = result.get("consistent_pct", 0.0) >= 30.0
    return result


def run_diversity_audit(args: argparse.Namespace) -> dict:
    """Run audit_pdb_atom_diversity.py and parse its stdout for verdict."""
    audit_script = PROJECT_ROOT / "scripts/diffab_ft/audit_pdb_atom_diversity.py"
    cmd = [
        PYTHON, str(audit_script),
        "--manifest-tsv", str(args._manifest_tsv),
        "--pdb-dir", str(args._subset_dir),
        "--curated-csv", str(args._curated_csv),
    ]
    log.info("running diversity audit...")
    log.info("  cmd: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    result = {"returncode": proc.returncode, "raw_stdout": proc.stdout}
    for line in proc.stdout.splitlines():
        ls = line.strip()
        if ls.startswith("unique PDB-ATOM sequences:"):
            result["unique_pdb"] = int(ls.split(":")[1].strip())
        elif ls.startswith("unique CSV sequences"):
            # "unique CSV sequences (for comparison): 221"
            result["unique_csv"] = int(ls.split(":")[-1].strip())
        elif ls.startswith("manifest rows processed:"):
            result["total"] = int(ls.split(":")[1].strip())
    upd = result.get("unique_pdb", 0)
    result["healthy"] = 200 <= upd <= 280
    return result


# ─────────────────────────────────────────────────────────────────────
# "More data" diff vs previous manifest
# ─────────────────────────────────────────────────────────────────────


def snapshot_previous_manifest(manifest_tsv: Path) -> Path | None:
    """Copy the existing manifest aside before this run touches it.
    Returns the snapshot path, or None if there's nothing to snapshot."""
    if not manifest_tsv.exists():
        return None
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    snap = manifest_tsv.with_suffix(f".tsv.previous.{ts}")
    shutil.copy2(manifest_tsv, snap)
    log.info("snapshotted previous manifest → %s", snap.name)
    return snap


def manifest_diff(prev_path: Path | None, new_path: Path) -> dict:
    """Compare row IDs between two manifest TSVs."""
    if prev_path is None or not prev_path.exists():
        return {"status": "no_previous", "previous_count": 0,
                "new_count": _row_count(new_path, sep="\t")}
    try:
        import csv
        def _ids(path: Path) -> set[str]:
            with open(path) as f:
                reader = csv.DictReader(f, delimiter="\t")
                return {f"{r['pdb']}_{r['Hchain']}" for r in reader if r.get("pdb")}
        prev_ids = _ids(prev_path)
        new_ids = _ids(new_path)
        added = sorted(new_ids - prev_ids)
        removed = sorted(prev_ids - new_ids)
        return {
            "status": "diffed",
            "previous_count": len(prev_ids),
            "new_count": len(new_ids),
            "added": added,
            "removed": removed,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "diff_failed", "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────
# Step execution
# ─────────────────────────────────────────────────────────────────────


def run_step(step: Step, args: argparse.Namespace) -> None:
    log.info("=" * 60)
    log.info("STEP %s: %s", step.name, step.description)
    log.info("=" * 60)

    # Skip rules
    if step.name in args.skip_step:
        log.info("  --skip-step %s set; skipping.", step.name)
        step.skipped = True
        return

    force = (step.name in args.force_step) or args.force_all
    if step.outputs_exist() and not force:
        log.info("  outputs already exist:")
        for p in step.output_paths:
            log.info("    ✓ %s", p)
        log.info("  skipping (--force-step %s or --force-all to re-run).", step.name)
        step.skipped = True
        if step.audit:
            step.audit()
        return

    log.info("  cmd: %s", " ".join(step.cmd))
    if args.dry_run:
        log.info("  [DRY-RUN] not executing.")
        return

    t0 = datetime.datetime.now()
    proc = subprocess.run(step.cmd, check=False)
    step.elapsed_seconds = (datetime.datetime.now() - t0).total_seconds()

    if proc.returncode != 0:
        log.error("  step %s failed with exit code %d. Aborting pipeline.",
                  step.name, proc.returncode)
        sys.exit(proc.returncode)
    log.info("  step %s completed in %.1f s.", step.name, step.elapsed_seconds)

    if step.audit:
        step.audit()


# ─────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────


def print_final_summary(
    args: argparse.Namespace,
    steps: list[Step],
    consistency: dict,
    diversity: dict,
    diff: dict,
) -> int:
    log.info("=" * 60)
    log.info("FINAL SUMMARY")
    log.info("=" * 60)

    log.info("Steps run:")
    for s in steps:
        tag = "SKIP" if s.skipped else f"{s.elapsed_seconds:5.1f}s"
        log.info("  [%s] %s — %s", tag, s.name, s.description)

    log.info("")
    log.info("PDB ↔ CSV consistency audit:")
    if consistency:
        log.info("  manifest rows:        %s", consistency.get("total", "?"))
        log.info("  fully consistent:     %s (%.1f%%)",
                 consistency.get("consistent", "?"),
                 consistency.get("consistent_pct", 0.0))
        log.info("  length mismatch:      %s", consistency.get("len_mismatch", "?"))
        log.info("  same-len seq mismatch:%s", consistency.get("seq_mismatch", "?"))
        log.info("  → %s", "HEALTHY" if consistency.get("healthy") else
                 "WARN: <30% fully consistent")

    log.info("")
    log.info("PDB-ATOM diversity audit:")
    if diversity:
        log.info("  total entries:        %s", diversity.get("total", "?"))
        log.info("  unique PDB-ATOM seqs: %s (expected 200–280)",
                 diversity.get("unique_pdb", "?"))
        log.info("  unique CSV seqs:      %s (for comparison)",
                 diversity.get("unique_csv", "?"))
        log.info("  → %s", "HEALTHY" if diversity.get("healthy") else
                 "WARN: outside expected band (regression?)")

    log.info("")
    log.info("Manifest diff vs previous run:")
    status = diff.get("status", "?")
    if status == "no_previous":
        log.info("  no previous manifest snapshot; new manifest has %d rows",
                 diff.get("new_count", "?"))
    elif status == "diffed":
        prev_n, new_n = diff.get("previous_count", 0), diff.get("new_count", 0)
        delta = new_n - prev_n
        if delta == 0:
            log.info("  manifest unchanged: %d rows", new_n)
        elif delta > 0:
            log.info("  manifest GREW by %d rows: %d → %d", delta, prev_n, new_n)
        else:
            log.info("  manifest SHRANK by %d rows: %d → %d", -delta, prev_n, new_n)
        added, removed = diff.get("added", []), diff.get("removed", [])
        if added:
            log.info("  added (%d): %s%s", len(added), added[:10],
                     " ..." if len(added) > 10 else "")
        if removed:
            log.info("  removed (%d): %s%s", len(removed), removed[:10],
                     " ..." if len(removed) > 10 else "")
    else:
        log.info("  diff failed: %s", diff.get("error", "?"))

    log.info("")
    healthy = (consistency.get("healthy", False) and diversity.get("healthy", False))
    if healthy:
        log.info("VERDICT: HEALTHY — proceed to sbatch scripts/diffab_ft/slurm/"
                 "train_seed42_dedup.sbatch")
        return 0
    else:
        log.info("VERDICT: WARNINGS RAISED — review the audit output before "
                 "training. See docs/data_pipeline_investigation.md §verification "
                 "for guidance.")
        return 2


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--cutoff-date", default="2021-12-25",
        help="Training-data cutoff date (YYYY-MM-DD). Default 2021-12-25 = DiffAb.",
    )
    p.add_argument(
        "--label", default="post_diffab",
        help="Boolean-flag column name + file suffix. Default 'post_diffab'.",
    )
    p.add_argument(
        "--base-dir",
        default=Path("/projects/0/hpmlprjs/interns/krijn/ANDD_nano_dataset_IgLM"),
        type=Path,
        help="Base dir containing the ANDD Excel and all derived CSVs.",
    )
    p.add_argument(
        "--repo-data-dir",
        default=PROJECT_ROOT / "data/datasets",
        type=Path,
        help="Repo data dir (manifest + clustering outputs land here).",
    )
    p.add_argument(
        "--dedupe-by", default="pdb_atom_sequence",
        choices=["pdb_atom_sequence", "raw_sequence", "concat_cdrs", "none"],
        help="Dedup key passed to cluster_split.py.",
    )
    p.add_argument(
        "--ratios", nargs=3, type=float, default=[0.75, 0.15, 0.10],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Cluster-level split ratios.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Seed for cluster-level shuffle.",
    )
    p.add_argument(
        "--force-step", action="append", default=[], metavar="NAME",
        help="Force re-run of this step even if outputs exist. Repeatable.",
    )
    p.add_argument(
        "--force-all", action="store_true",
        help="Force re-run of every step.",
    )
    p.add_argument(
        "--skip-step", action="append", default=[], metavar="NAME",
        help="Skip this step entirely (no output check). Repeatable.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print all subprocess commands without executing.",
    )
    p.add_argument(
        "--log-dir", default=PROJECT_ROOT / "logs",
        type=Path,
        help="Where to write the run log.",
    )
    args = p.parse_args()

    # Logging
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = args.log_dir / f"data_prep_{ts}.log"
    setup_logging(log_path)
    log.info("orchestrator started; logging to %s", log_path)
    log.info("project root: %s", PROJECT_ROOT)
    log.info("args: %s", vars(args))

    # Validate base_dir exists (laptop dry-runs may use a fake path)
    if not args.dry_run and not args.base_dir.exists():
        log.error("base dir not found: %s (use --dry-run to skip validation)",
                  args.base_dir)
        return 1

    # Build pipeline
    steps = build_steps(args)
    log.info("pipeline: %d steps", len(steps))
    for s in steps:
        log.info("  %s — %s", s.name, s.description)

    # Snapshot existing manifest for later diff
    prev_manifest = None
    if not args.dry_run:
        prev_manifest = snapshot_previous_manifest(args._manifest_tsv)

    # Execute pipeline
    for step in steps:
        run_step(step, args)

    # Post-Step-6 critical audits
    consistency: dict = {}
    diversity: dict = {}
    diff: dict = {"status": "no_previous"}

    if args.dry_run:
        log.info("[DRY-RUN] skipping post-step audits and final summary.")
        return 0

    try:
        consistency = run_consistency_audit(args)
    except Exception as exc:  # noqa: BLE001
        log.error("consistency audit raised: %s", exc)

    try:
        diversity = run_diversity_audit(args)
    except Exception as exc:  # noqa: BLE001
        log.error("diversity audit raised: %s", exc)

    diff = manifest_diff(prev_manifest, args._manifest_tsv)

    rc = print_final_summary(args, steps, consistency, diversity, diff)
    log.info("log written to %s", log_path)
    return rc


if __name__ == "__main__":
    sys.exit(main())
