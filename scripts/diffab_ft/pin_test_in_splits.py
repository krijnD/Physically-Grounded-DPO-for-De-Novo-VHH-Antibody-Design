#!/usr/bin/env python3
"""Post-process cluster_split.py output to pin the current test set.

Brief 05 §4.6 procedure:
  1. Read raw cluster_assignments produced by cluster_split.py.
  2. Read the current cluster_splits.json :: splits.test PDB IDs.
  3. Identify every cluster that contains at least one current-test PDB.
  4. Assign ALL members of those clusters to the new splits.test
     (cluster-level integrity — prevents CDR-side leakage from old-test
     into new-train).
  5. For the remaining clusters: seeded 80/10/10 train/val/test.
  6. Verify every current-test PDB is in the new splits.test. Hard fail
     if any are missing.

PDB-vs-Hchain matching: pinning matches on PDB ID only (not the full
entry_id ``<pdb>_<Hchain>``). Rationale: the new env's curate may pick a
different Hchain letter than the old run did (ANARCI evolution between
abnumber versions), so the new entry's chain might differ even though
biologically it's the same VHH. PDB-level pinning preserves the biology;
Hchain-level matching would silently drop entries on chain renames.
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("pin_test")


def _entry_pdb(entry_id: str) -> str:
    return entry_id.rsplit("_", 1)[0].lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-splits-json", required=True, type=Path,
                        help="cluster_splits.json output of cluster_split.py "
                             "on the expanded pool.")
    parser.add_argument("--current-splits-json", required=True, type=Path,
                        help="Current data/datasets/clustering/cluster_splits.json.")
    parser.add_argument("--output-json", required=True, type=Path,
                        help="Path for the pinned splits JSON.")
    parser.add_argument("--ratios", type=float, nargs=3,
                        default=[0.8, 0.1, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Cluster-level split ratios for non-pinned "
                             "clusters (default 0.8 0.1 0.1).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for p in (args.raw_splits_json, args.current_splits_json):
        if not p.exists():
            sys.exit(f"Input not found: {p}")
    if args.output_json.exists() and not args.overwrite:
        sys.exit(f"Output exists: {args.output_json} (use --overwrite)")

    # Load raw cluster assignments
    raw = json.loads(args.raw_splits_json.read_text())
    cluster_assignments: dict[str, str] = raw["cluster_assignments"]
    logger.info("Raw cluster assignments: %d entries across %d clusters",
                len(cluster_assignments),
                len(set(cluster_assignments.values())))

    # Load current test PDB IDs
    cur = json.loads(args.current_splits_json.read_text())
    old_test_entries: list[str] = cur["splits"]["test"]
    old_test_pdbs = {_entry_pdb(e) for e in old_test_entries}
    logger.info("Current test: %d entries (unique PDBs: %d)",
                len(old_test_entries), len(old_test_pdbs))

    # Group entries by cluster rep
    rep_to_members: dict[str, list[str]] = defaultdict(list)
    for entry, rep in cluster_assignments.items():
        rep_to_members[rep].append(entry)
    for k in rep_to_members:
        rep_to_members[k].sort()
    all_reps = sorted(rep_to_members.keys())

    # Identify pinned clusters (those touching any current-test PDB)
    pinned_reps: set[str] = set()
    for rep, members in rep_to_members.items():
        if any(_entry_pdb(m) in old_test_pdbs for m in members):
            pinned_reps.add(rep)

    pinned_member_count = sum(len(rep_to_members[r]) for r in pinned_reps)
    logger.info("Pinned clusters: %d (containing %d entries)",
                len(pinned_reps), pinned_member_count)

    # Verify every old-test PDB is reachable via clustering
    clustered_pdbs = {_entry_pdb(e) for e in cluster_assignments}
    missing_from_clustering = old_test_pdbs - clustered_pdbs
    if missing_from_clustering:
        logger.error(
            "Current-test PDBs NOT FOUND in cluster_assignments: %s "
            "(probably failed CDR extraction in cluster_split.py)",
            sorted(missing_from_clustering),
        )
        sys.exit(2)

    # Split the remaining (non-pinned) clusters by ratios
    remaining_reps = [r for r in all_reps if r not in pinned_reps]
    rng = random.Random(args.seed)
    rng.shuffle(remaining_reps)
    n = len(remaining_reps)
    train_r, val_r, test_r = args.ratios
    n_train = int(round(n * train_r))
    n_val   = int(round(n * val_r))
    train_reps = set(remaining_reps[:n_train])
    val_reps   = set(remaining_reps[n_train : n_train + n_val])
    extra_test_reps = set(remaining_reps[n_train + n_val:])

    logger.info("Non-pinned cluster split: train=%d  val=%d  test=%d",
                len(train_reps), len(val_reps), len(extra_test_reps))

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for rep in all_reps:
        bucket = (
            "test"  if rep in pinned_reps      else
            "train" if rep in train_reps        else
            "val"   if rep in val_reps          else
            "test"  if rep in extra_test_reps   else
            None
        )
        if bucket is None:
            sys.exit(f"Bug: cluster {rep!r} not assigned to any split.")
        splits[bucket].extend(rep_to_members[rep])
    for k in splits:
        splits[k].sort()

    logger.info("Final splits (members): train=%d  val=%d  test=%d  "
                "(total=%d)",
                len(splits["train"]), len(splits["val"]), len(splits["test"]),
                sum(len(v) for v in splits.values()))

    # ── Test-preservation hard verification (PDB-level) ──────────────
    new_test_pdbs = {_entry_pdb(e) for e in splits["test"]}
    missing = old_test_pdbs - new_test_pdbs
    if missing:
        logger.error(
            "PRESERVATION FAILED — %d old-test PDBs missing from new test "
            "(should be impossible after pinning): %s",
            len(missing), sorted(missing),
        )
        sys.exit(2)
    logger.info("Test preservation OK: all %d old-test PDBs present in "
                "new test split (PDB-level).", len(old_test_pdbs))

    # Audit: how many current-test entries (PDB+Hchain) match exactly?
    old_test_entry_set = set(old_test_entries)
    new_test_entry_set = set(splits["test"])
    exact_match = len(old_test_entry_set & new_test_entry_set)
    logger.info("Test preservation entry-level (PDB+Hchain exact match): "
                "%d / %d old-test entries (%d PDBs had chain renames).",
                exact_match, len(old_test_entries),
                len(old_test_pdbs) - exact_match)

    # Write
    output = {
        "splits": splits,
        "cluster_assignments": cluster_assignments,
        "params": {
            **raw.get("params", {}),
            "pinned_clusters": len(pinned_reps),
            "pinned_members":  pinned_member_count,
            "remaining_ratios": list(args.ratios),
            "remaining_seed": args.seed,
            "n_clusters": len(all_reps),
            "test_preservation": "pdb_level",
        },
        "antigen_audit": raw.get("antigen_audit"),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2))
    logger.info("Wrote pinned splits to %s", args.output_json)


if __name__ == "__main__":
    main()
