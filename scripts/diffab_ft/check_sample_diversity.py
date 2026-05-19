#!/usr/bin/env python3
"""Sample-diversity check on ``evaluate.py --mode design`` CSV outputs.

Detects mode collapse (the model emitting identical sequences across
all N samples for a given entry+CDR) — a key Block-2 diagnostic per
``docs/finetune_evaluation_handoff.md`` §diagnostic-question-1.

Reads a per-sample CSV with columns
``entry_id,cdr,sample,native_seq,gen_seq,aar,rmsd``, groups by
``(entry_id, cdr)``, and reports per-CDR stats plus an overall
unique-count histogram.

Usage
-----
::

    python scripts/diffab_ft/check_sample_diversity.py \\
        --csv  runs/vhh_ft/seed42_dedup/eval_test_antigen_disjoint.csv \\
        --out  runs/_eval/diversity_seed42_dedup.md \\
        --label seed42_dedup
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


CDRS = ("H1", "H2", "H3")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", required=True, type=Path,
                        help="Per-sample CSV produced by evaluate.py --mode design.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output markdown path.")
    parser.add_argument("--label", default=None,
                        help="Run label (defaults to CSV stem).")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        return 2

    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    with args.csv.open() as f:
        reader = csv.DictReader(f)
        required = {"entry_id", "cdr", "gen_seq"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            print(f"ERROR: CSV missing columns: {sorted(missing)}", file=sys.stderr)
            return 2
        for row in reader:
            groups[(row["entry_id"], row["cdr"])].append(row["gen_seq"])

    if not groups:
        print(f"ERROR: no rows in {args.csv}", file=sys.stderr)
        return 2

    label = args.label or args.csv.stem
    samples_per_group = max(len(seqs) for seqs in groups.values())

    per_cdr: dict[str, dict] = {
        cdr: {"unique_counts": [], "collapsed": 0, "n_groups": 0}
        for cdr in CDRS
    }
    overall_hist: Counter[int] = Counter()
    collapsed_entries: list[tuple[str, str]] = []

    for (entry, cdr), seqs in groups.items():
        n_unique = len(set(seqs))
        overall_hist[n_unique] += 1
        if cdr not in per_cdr:
            continue
        per_cdr[cdr]["unique_counts"].append(n_unique)
        per_cdr[cdr]["n_groups"] += 1
        if n_unique == 1:
            per_cdr[cdr]["collapsed"] += 1
            collapsed_entries.append((entry, cdr))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write(f"# Sample-diversity report: `{label}`\n\n")
        f.write(f"**Source CSV:** `{args.csv}`\n\n")
        f.write(f"**Samples per (entry, CDR):** {samples_per_group}\n\n")
        f.write(f"**Total (entry, CDR) groups:** {len(groups)}\n\n")

        f.write("## Per-CDR diversity\n\n")
        f.write("| CDR | groups | mean unique | median | min | max | collapsed (1/N) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for cdr in CDRS:
            d = per_cdr[cdr]
            if not d["unique_counts"]:
                f.write(f"| {cdr} | 0 | — | — | — | — | — |\n")
                continue
            uc = d["unique_counts"]
            mean_u = statistics.mean(uc)
            median_u = statistics.median(uc)
            pct = d["collapsed"] / d["n_groups"] * 100
            f.write(
                f"| {cdr} | {d['n_groups']} | {mean_u:.2f} | {median_u:.1f} | "
                f"{min(uc)} | {max(uc)} | "
                f"{d['collapsed']} ({pct:.0f}%) |\n"
            )

        f.write("\n## Overall histogram (unique samples → # of (entry, CDR) groups)\n\n")
        f.write("| n_unique | count |\n|---|---|\n")
        for n in range(1, samples_per_group + 1):
            f.write(f"| {n} | {overall_hist.get(n, 0)} |\n")

        f.write("\n## Verdict (per handoff §diagnostic-question-1)\n\n")
        total = len(groups)
        any_collapsed = sum(per_cdr[c]["collapsed"] for c in CDRS)
        if any_collapsed == 0:
            f.write(f"- **No full mode collapse** detected across {total} (entry, CDR) groups.\n")
        else:
            pct = any_collapsed / total * 100
            f.write(f"- **{any_collapsed}/{total} ({pct:.0f}%) groups are fully collapsed** "
                    "(all N samples identical).\n")
            f.write("- A model that emits identical sequences for every sample cannot "
                    "produce winner/loser variance for DPO. If collapse exceeds ~20%, "
                    "rerun with stronger noise schedule or revisit early-stop iter.\n")

        if collapsed_entries:
            f.write(f"\n## Fully collapsed (1/N) groups ({len(collapsed_entries)})\n\n")
            for entry, cdr in sorted(collapsed_entries):
                f.write(f"- `{entry}` {cdr}\n")

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
