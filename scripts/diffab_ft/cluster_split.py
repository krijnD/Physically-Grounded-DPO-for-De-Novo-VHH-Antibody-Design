#!/usr/bin/env python3
"""Cluster ANDD VHH manifest entries by concatenated CDR identity and
emit train/val/test splits at the cluster level.

Pipeline:
  1. Read the manifest TSV (output of prepare_manifest.py) — defines the
     set of (pdb, Hchain) entries to cluster.
  2. Look up each entry's full VHH sequence in the curated ANDD CSV
     (column ``Ab/Nano H_Chain AA``).
  3. Run abnumber (Chothia scheme) to extract CDR-H1, CDR-H2, CDR-H3.
  4. Write the concatenated CDR sequences to a FASTA file and run
     MMseqs2 ``easy-cluster`` at --min-seq-id 0.7 -c 0.8 --cov-mode 0.
  5. Assign each cluster to train / val / test (default 0.8 / 0.1 / 0.1)
     using a seeded shuffle. All members of a cluster go to the same
     split — this is the only way to prevent CDR-side leakage when
     several PDBs in the dataset are the same nanobody crystallized
     against different antigen states.
  6. Optionally audit antigen-side overlap: extract antigen sequences
     from the PDBs, cluster them at 50%% identity, and report what
     fraction of test-set antigens also appear in train. If overlap is
     high, the user should construct a secondary held-out test set
     where antigen clusters are also disjoint from train.

Outputs (under --output-dir):
  - concat_cdrs.fasta            — input to MMseqs2 (CDR clustering)
  - cluster_result_cluster.tsv   — raw MMseqs2 output (representative,
                                   member) pairs
  - cluster_splits.json          — primary outputs:
        {
          "splits": {"train": [...], "val": [...], "test": [...]},
          "cluster_assignments": {"<pdb_id>": "<rep_pdb_id>", ...},
          "params": {"identity": 0.7, "coverage": 0.8, ...},
          "antigen_audit": {... if enabled ...}
        }
  - antigen_seqs.fasta           — only if --audit-antigens is set
  - antigen_cluster_*.tsv        — only if --audit-antigens is set

The JSON's ``splits`` use **PDB-id-with-Hchain composite keys** of the
form ``"<pdb>_<Hchain>"`` to match DiffAb's entry ID convention (see
sabdab.py line 248). This way the split file is directly consumable by
the DiffAb dataset subclass we'll build later.

Usage:
    python scripts/diffab_ft/cluster_split.py \\
        --curated-csv  /path/to/ANDD_VHH_curated_diffab.csv \\
        --manifest-tsv data/datasets/diffab_manifest.tsv \\
        --pdb-dir      /path/to/VHH_structures_post_diffab \\
        --output-dir   data/datasets/clustering \\
        --audit-antigens
"""

import argparse
import json
import logging
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from abnumber import Chain as AbnumberChain
from abnumber.exceptions import ChainParseError

# Project root is two levels up from this script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.sabdab_loader import extract_chain_sequence  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("cluster_split")


# ── CDR extraction ────────────────────────────────────────────────────────
def extract_concat_cdrs(seq: str, scheme: str = "chothia") -> str | None:
    """Run abnumber on a VHH sequence and return CDR-H1+H2+H3 concatenated.

    Returns None if the sequence fails to parse as a heavy chain. Edge
    cases (truncated CDR3, ANARCI errors) are logged by the caller; we
    just propagate None.
    """
    try:
        chain = AbnumberChain(seq, scheme=scheme)
    except (ChainParseError, ValueError):
        return None
    if chain.chain_type != "H":
        return None
    cdr1 = chain.cdr1_seq or ""
    cdr2 = chain.cdr2_seq or ""
    cdr3 = chain.cdr3_seq or ""
    if not (cdr1 and cdr2 and cdr3):
        return None
    return cdr1 + cdr2 + cdr3


# ── MMseqs2 wrapper ───────────────────────────────────────────────────────
def run_mmseqs_easy_cluster(
    fasta_path: Path,
    output_prefix: Path,
    tmp_dir: Path,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
) -> Path:
    """Run ``mmseqs easy-cluster`` and return the cluster TSV path.

    The TSV output has two columns (representative, member) per line.
    """
    if shutil.which("mmseqs") is None:
        raise RuntimeError(
            "mmseqs not found in PATH. On Snellius, load it via "
            "'module load 2024 MMseqs2/...' or install in your venv."
        )
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mmseqs", "easy-cluster",
        str(fasta_path.resolve()),
        str(output_prefix.resolve()),
        str(tmp_dir.resolve()),
        "--min-seq-id", f"{min_seq_id}",
        "-c", f"{coverage}",
        "--cov-mode", f"{cov_mode}",
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    cluster_tsv = output_prefix.parent / f"{output_prefix.name}_cluster.tsv"
    if not cluster_tsv.exists():
        raise FileNotFoundError(
            f"MMseqs2 finished but cluster TSV not found at {cluster_tsv}"
        )
    return cluster_tsv


def parse_cluster_tsv(tsv_path: Path) -> dict[str, str]:
    """Map each member ID to its cluster representative ID."""
    member_to_rep: dict[str, str] = {}
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            rep, member = parts
            member_to_rep[member] = rep
    return member_to_rep


# ── Cluster-level split ───────────────────────────────────────────────────
def cluster_level_split(
    member_to_rep: dict[str, str],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[str]]:
    """Split clusters (not members) into train/val/test by the ratios.

    All members of a given cluster end up in the same split. Splits at
    the cluster boundary, not the member boundary, so smaller-cluster
    splits may be slightly over-/under-represented in member counts —
    this is expected and correct behavior.
    """
    train_r, val_r, test_r = ratios
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, \
        f"ratios must sum to 1.0, got {ratios}"

    reps = sorted(set(member_to_rep.values()))
    rng = random.Random(seed)
    rng.shuffle(reps)

    n = len(reps)
    n_train = int(round(n * train_r))
    n_val = int(round(n * val_r))
    train_reps = set(reps[:n_train])
    val_reps = set(reps[n_train : n_train + n_val])
    test_reps = set(reps[n_train + n_val:])

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for member, rep in member_to_rep.items():
        if rep in train_reps:
            splits["train"].append(member)
        elif rep in val_reps:
            splits["val"].append(member)
        else:
            splits["test"].append(member)

    for k in splits:
        splits[k].sort()
    return splits


# ── Antigen audit ─────────────────────────────────────────────────────────
def audit_antigen_overlap(
    manifest: pd.DataFrame,
    pdb_dir: Path,
    splits: dict[str, list[str]],
    output_dir: Path,
    min_seq_id: float = 0.5,
) -> dict:
    """Cluster antigen sequences and report train/test overlap.

    For each manifest entry, extract the sequences of all antigen chains
    from the PDB and concatenate them with '|' separators. Cluster the
    resulting strings at --min-seq-id 0.5 (50%% identity, the standard
    threshold for protein homology). Then for each split, compute the
    set of antigen clusters present, and report:
      - |train ∩ test| / |test|  → fraction of test antigens leaked
      - |train ∩ val|  / |val|   → same for val

    Returns a dict serializable to JSON.
    """
    fasta_path = output_dir / "antigen_seqs.fasta"
    entry_to_antigen_seq: dict[str, str] = {}
    skipped = 0

    logger.info("Extracting antigen sequences from %d PDBs ...", len(manifest))
    for _, row in manifest.iterrows():
        pdb_id = row["pdb"]
        # antigen_chain is pipe-delimited, e.g. "A | E"
        ag_chains = [c.strip() for c in str(row["antigen_chain"]).split("|") if c.strip()]
        pdb_path = pdb_dir / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            pdb_path = pdb_dir / f"{pdb_id.upper()}.pdb"
        if not pdb_path.exists():
            skipped += 1
            continue
        seqs = []
        for chain_id in ag_chains:
            s = extract_chain_sequence(str(pdb_path), chain_id)
            if s:
                seqs.append(s)
        if not seqs:
            skipped += 1
            continue
        entry_id = f"{pdb_id}_{row['Hchain']}"
        # Join multi-chain antigens with a stretch of X (untyped residue)
        # so MMseqs2 sees one logical sequence per entry but never aligns
        # across the chain boundary.
        entry_to_antigen_seq[entry_id] = ("X" * 10).join(seqs)

    if skipped:
        logger.warning("Antigen audit: skipped %d entries (chain extraction failed).", skipped)

    # Write FASTA
    with open(fasta_path, "w") as f:
        for entry_id, seq in entry_to_antigen_seq.items():
            f.write(f">{entry_id}\n{seq}\n")
    logger.info("Wrote %d antigen sequences to %s",
                len(entry_to_antigen_seq), fasta_path)

    # Cluster
    out_prefix = output_dir / "antigen_cluster"
    tmp_dir = output_dir / "antigen_tmp"
    cluster_tsv = run_mmseqs_easy_cluster(
        fasta_path=fasta_path,
        output_prefix=out_prefix,
        tmp_dir=tmp_dir,
        min_seq_id=min_seq_id,
        coverage=0.5,    # antigens often have variable lengths; lenient coverage
        cov_mode=1,
    )
    member_to_rep = parse_cluster_tsv(cluster_tsv)

    # Compute overlaps
    split_to_clusters: dict[str, set[str]] = {}
    for split_name, members in splits.items():
        clusters = {member_to_rep[m] for m in members if m in member_to_rep}
        split_to_clusters[split_name] = clusters

    train_set = split_to_clusters["train"]
    val_set = split_to_clusters["val"]
    test_set = split_to_clusters["test"]

    def _frac(num: int, den: int) -> float:
        return (num / den) if den else 0.0

    overlap = {
        "n_antigen_clusters_total": len(set(member_to_rep.values())),
        "n_antigen_clusters_train": len(train_set),
        "n_antigen_clusters_val": len(val_set),
        "n_antigen_clusters_test": len(test_set),
        "test_antigens_leaked_to_train":
            len(test_set & train_set),
        "test_antigens_leaked_to_train_frac":
            _frac(len(test_set & train_set), len(test_set)),
        "val_antigens_leaked_to_train":
            len(val_set & train_set),
        "val_antigens_leaked_to_train_frac":
            _frac(len(val_set & train_set), len(val_set)),
        "skipped_entries": skipped,
        "min_seq_id": min_seq_id,
    }
    return overlap


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--curated-csv", required=True, type=Path,
                        help="ANDD curated CSV (provides Ab/Nano H_Chain AA).")
    parser.add_argument("--manifest-tsv", required=True, type=Path,
                        help="DiffAb manifest TSV (output of prepare_manifest.py).")
    parser.add_argument("--pdb-dir", required=True, type=Path,
                        help="PDB directory (only required if --audit-antigens).")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Where to write FASTA, MMseqs2 outputs, and splits JSON.")
    parser.add_argument("--identity", type=float, default=0.7,
                        help="MMseqs2 --min-seq-id for CDR clustering (default: 0.7).")
    parser.add_argument("--coverage", type=float, default=0.8,
                        help="MMseqs2 -c (coverage) for CDR clustering (default: 0.8).")
    parser.add_argument("--cov-mode", type=int, default=0,
                        help="MMseqs2 --cov-mode (default: 0, bidirectional).")
    parser.add_argument("--scheme", default="chothia",
                        choices=["chothia", "kabat", "imgt"],
                        help="Numbering scheme for CDR extraction (default: chothia).")
    parser.add_argument("--ratios", type=float, nargs=3,
                        default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"),
                        help="Cluster-level split ratios (default: 0.8 0.1 0.1).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for cluster-level shuffle (default: 42).")
    parser.add_argument("--audit-antigens", action="store_true",
                        help="Run the antigen-overlap audit (slower; reads all PDBs).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing output files.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    # Validate inputs
    for p in (args.curated_csv, args.manifest_tsv):
        if not p.exists():
            logger.error("File not found: %s", p)
            sys.exit(1)
    if args.audit_antigens and (not args.pdb_dir.exists() or not args.pdb_dir.is_dir()):
        logger.error("PDB dir required for --audit-antigens but not found: %s", args.pdb_dir)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits_path = args.output_dir / "cluster_splits.json"
    if splits_path.exists() and not args.overwrite:
        logger.error("Output exists: %s (use --overwrite).", splits_path)
        sys.exit(1)

    # Load manifest and curated CSV
    manifest = pd.read_csv(args.manifest_tsv, sep="\t")
    curated = pd.read_csv(args.curated_csv)
    logger.info("Manifest: %d entries; curated CSV: %d rows.",
                len(manifest), len(curated))

    # Build a (pdb_lower, Hchain) → sequence lookup from curated CSV.
    seq_lookup: dict[tuple[str, str], str] = {}
    for _, row in curated.iterrows():
        pdb_id = str(row["PDB_ID"]).strip().lower()
        h_chain = str(row["H_Chain Auth Asym ID"]).strip()
        seq = str(row["Ab/Nano H_Chain AA"]).strip()
        if not seq or seq.lower() == "nan":
            continue
        seq_lookup[(pdb_id, h_chain)] = seq

    # Extract CDRs for each manifest entry, write FASTA
    fasta_path = args.output_dir / "concat_cdrs.fasta"
    n_written = 0
    n_no_seq = 0
    n_no_cdr = 0
    entry_to_cdr: dict[str, str] = {}

    with open(fasta_path, "w") as f:
        for _, row in manifest.iterrows():
            pdb_id = str(row["pdb"]).strip().lower()
            h_chain = str(row["Hchain"]).strip()
            entry_id = f"{pdb_id}_{h_chain}"

            seq = seq_lookup.get((pdb_id, h_chain))
            if seq is None:
                logger.warning("No sequence for %s in curated CSV; skipping.", entry_id)
                n_no_seq += 1
                continue
            cdrs = extract_concat_cdrs(seq, scheme=args.scheme)
            if cdrs is None:
                logger.warning("CDR extraction failed for %s; skipping.", entry_id)
                n_no_cdr += 1
                continue

            f.write(f">{entry_id}\n{cdrs}\n")
            entry_to_cdr[entry_id] = cdrs
            n_written += 1

    logger.info("Wrote %d concatenated CDR sequences to %s",
                n_written, fasta_path)
    if n_no_seq:
        logger.warning("Skipped %d entries: missing sequence in curated CSV.", n_no_seq)
    if n_no_cdr:
        logger.warning("Skipped %d entries: CDR extraction failed (abnumber).", n_no_cdr)
    if n_written == 0:
        logger.error("No CDR sequences extracted; aborting.")
        sys.exit(1)

    # Run MMseqs2 clustering
    out_prefix = args.output_dir / "cluster_result"
    tmp_dir = args.output_dir / "cluster_tmp"
    cluster_tsv = run_mmseqs_easy_cluster(
        fasta_path=fasta_path,
        output_prefix=out_prefix,
        tmp_dir=tmp_dir,
        min_seq_id=args.identity,
        coverage=args.coverage,
        cov_mode=args.cov_mode,
    )
    member_to_rep = parse_cluster_tsv(cluster_tsv)
    n_clusters = len(set(member_to_rep.values()))
    logger.info("MMseqs2: %d members → %d clusters at %.2f identity, %.2f coverage.",
                len(member_to_rep), n_clusters, args.identity, args.coverage)

    # Cluster-level split
    splits = cluster_level_split(
        member_to_rep=member_to_rep,
        ratios=tuple(args.ratios),
        seed=args.seed,
    )
    logger.info("Splits: train=%d val=%d test=%d (members)",
                len(splits["train"]), len(splits["val"]), len(splits["test"]))

    # Optional antigen audit
    antigen_audit = None
    if args.audit_antigens:
        antigen_audit = audit_antigen_overlap(
            manifest=manifest,
            pdb_dir=args.pdb_dir,
            splits=splits,
            output_dir=args.output_dir,
            min_seq_id=0.5,
        )
        logger.info("=" * 60)
        logger.info("Antigen overlap audit:")
        for k, v in antigen_audit.items():
            if isinstance(v, float):
                logger.info("  %-40s %.3f", k, v)
            else:
                logger.info("  %-40s %s", k, v)

    # Write splits JSON
    output = {
        "splits": splits,
        "cluster_assignments": member_to_rep,
        "params": {
            "identity": args.identity,
            "coverage": args.coverage,
            "cov_mode": args.cov_mode,
            "scheme": args.scheme,
            "ratios": list(args.ratios),
            "seed": args.seed,
            "n_members": len(member_to_rep),
            "n_clusters": n_clusters,
        },
        "antigen_audit": antigen_audit,
    }
    with open(splits_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("=" * 60)
    logger.info("Wrote splits to %s", splits_path)
    logger.info("Effective dataset size: %d clusters (raw entries: %d).",
                n_clusters, len(member_to_rep))


if __name__ == "__main__":
    main()
