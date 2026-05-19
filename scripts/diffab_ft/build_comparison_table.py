#!/usr/bin/env python3
"""Build the §core-evaluation comparison table from multiple eval JSONs.

Reads one or more ``evaluate.py --mode design`` JSON outputs and produces a
single markdown table comparing them on H1/H2/H3 AAR and RMSD.

The table format matches ``docs/finetune_evaluation_handoff.md``
§core-evaluation so the Block-2 deliverable can be pasted straight in.

Usage
-----
::

    python scripts/diffab_ft/build_comparison_table.py \\
        --evals  runs/baseline_pretrained/eval_test_antigen_disjoint.json \\
                 runs/vhh_ft/seed42/eval_test_antigen_disjoint.json \\
                 runs/vhh_ft/seed42_dedup/eval_test_antigen_disjoint.json \\
        --labels pretrained seed42 seed42_dedup \\
        --out    runs/_eval/comparison_test_antigen_disjoint.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CDRS = ("H1", "H2", "H3")


def _format_aar(stats: dict | None) -> str:
    if not stats or stats.get("n", 0) == 0:
        return "—"
    return f"{stats['aar_mean']:.3f} ± {stats['aar_std']:.3f}"


def _format_rmsd(stats: dict | None) -> str:
    if not stats or stats.get("n", 0) == 0:
        return "—"
    return f"{stats['rmsd_mean']:.2f} ± {stats['rmsd_std']:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--evals", nargs="+", required=True, type=Path,
                        help="Eval JSON paths produced by evaluate.py --mode design.")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Row labels (one per --evals entry).")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output markdown path.")
    parser.add_argument("--title", default="Three-way evaluation comparison",
                        help="Title line for the markdown report.")
    args = parser.parse_args()

    if len(args.evals) != len(args.labels):
        print(f"ERROR: {len(args.evals)} eval files but {len(args.labels)} labels",
              file=sys.stderr)
        return 2

    rows = []
    splits_seen: set[str] = set()
    n_entries_seen: set[int] = set()
    for path, label in zip(args.evals, args.labels):
        if not path.exists():
            print(f"ERROR: eval JSON not found: {path}", file=sys.stderr)
            return 2
        data = json.loads(path.read_text())
        if data.get("mode") != "design":
            print(f"ERROR: {path} has mode={data.get('mode')!r}; need 'design'",
                  file=sys.stderr)
            return 2
        design = data.get("design") or {}
        for cdr in CDRS:
            if cdr not in design:
                print(f"WARN: {path} missing design.{cdr}; row will show '—'.",
                      file=sys.stderr)
        meta = data.get("ckpt_meta") or {}
        rows.append({
            "label": label,
            "iter": meta.get("iteration", "n/a"),
            "val_loss": meta.get("val_loss"),
            "split": data.get("split", "?"),
            "n_entries": data.get("n_entries", "?"),
            "design": design,
            "source": path,
        })
        splits_seen.add(rows[-1]["split"])
        if isinstance(rows[-1]["n_entries"], int):
            n_entries_seen.add(rows[-1]["n_entries"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write(f"# {args.title}\n\n")
        if len(splits_seen) == 1:
            split = next(iter(splits_seen))
            entries_str = f"{next(iter(n_entries_seen))} entries" if len(n_entries_seen) == 1 else "varying entries"
            f.write(f"**Split:** `{split}` ({entries_str})\n\n")
        else:
            f.write(f"**Splits (mixed!):** {sorted(splits_seen)}\n\n")

        f.write("| Checkpoint | iter | H1 AAR | H1 RMSD (Å) | "
                "H2 AAR | H2 RMSD (Å) | H3 AAR | H3 RMSD (Å) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for row in rows:
            d = row["design"]
            f.write(
                f"| `{row['label']}` | {row['iter']} | "
                f"{_format_aar(d.get('H1'))} | {_format_rmsd(d.get('H1'))} | "
                f"{_format_aar(d.get('H2'))} | {_format_rmsd(d.get('H2'))} | "
                f"{_format_aar(d.get('H3'))} | {_format_rmsd(d.get('H3'))} |\n"
            )

        f.write("\n## Source eval JSONs\n\n")
        for row in rows:
            vl = f"{row['val_loss']:.4f}" if isinstance(row["val_loss"], float) else row["val_loss"]
            f.write(f"- `{row['label']}` (val_loss={vl}): `{row['source']}`\n")

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
