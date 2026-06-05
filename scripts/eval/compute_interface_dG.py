#!/usr/bin/env python3
"""Compute Rosetta InterfaceAnalyzerMover ΔG on a chunked design manifest.

Brief 11 §3 Step 4 (stretch axis): the existing physics judge populates
``e_rep`` (steric clash) and ``cdr_energy_per_res`` (CDR-side interface
residue energy, AbDPO §3.2). The catalog §3 "Rosetta Binding Free
Energy" recommends ``InterfaceAnalyzerMover`` for the full ΔG of the
paratope-epitope interface (bound minus separated) — a different
quantity from ``cdr_energy_per_res`` and the primary thermodynamic axis
in AbDPO + POEA.

This script reads a chunk CSV from the same chunk directory the main
judge array consumes, runs ``InterfaceAnalyzerMover`` per row with
``set_pack_separated(True)`` (repack interface after separation, the
field-standard setting), and writes a parquet with one row per input
candidate: ``candidate_id, dG_separated, dG_cross, dSASA, error``.

The master parquet assembly (Brief 11 §3 Step 5) joins this output to
the main scored parquet on ``candidate_id``. Keeps ``src/`` read-only —
no patch to ``src/physics_judge`` needed.

Usage::

    python scripts/eval/compute_interface_dG.py \\
        --input-csv      data/eval/chunks/all_variants/chunk_00.csv \\
        --output-parquet data/eval/dG_chunks/all_variants/chunk_00.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("compute_interface_dG")


def _normalize_chain(s: str) -> str:
    """Strip whitespace + pipe separators from a chain-id string."""
    if not s:
        return ""
    return s.replace(" ", "").replace("|", "")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input-csv", required=True, type=Path,
                    help="Chunk CSV (AAPR-format manifest from build_design_manifest.py).")
    ap.add_argument("--output-parquet", required=True, type=Path,
                    help="Output parquet with dG metrics keyed by candidate_id.")
    args = ap.parse_args()

    if not args.input_csv.exists():
        logger.error("Input CSV not found: %s", args.input_csv)
        return 2

    # Lazy PyRosetta init — keeps imports cheap when the script is
    # introspected.
    logger.info("Initializing PyRosetta")
    import pyrosetta
    pyrosetta.init(
        "-mute all -ignore_unrecognized_res 1 -ignore_zero_occupancy 0 "
        "-detect_disulf 0 -no_fconfig",
        silent=True,
    )
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    df = pd.read_csv(args.input_csv)
    logger.info("Processing %d candidates from %s", len(df), args.input_csv)

    rows: list[dict] = []
    t_start = time.time()
    n_ok = 0
    n_err = 0
    n_skipped = 0

    for idx, row in df.iterrows():
        cid = row["candidate_id"]
        pdb_path = row["complex_pdb_path"]
        nb_chain = str(row.get("nanobody_chain_id", "") or "")
        ag_chain = _normalize_chain(str(row.get("antigen_chain_ids", "") or ""))

        if not nb_chain or not ag_chain or not Path(pdb_path).exists():
            n_skipped += 1
            rows.append({
                "candidate_id": cid,
                "dG_separated": None,
                "dG_cross":     None,
                "dSASA":        None,
                "error":        "missing chain ids or PDB",
            })
            continue
        # PyRosetta interface string is "<chains_left>_<chains_right>".
        # Pass nanobody on one side and antigen on the other.
        if nb_chain.islower() or any(c.islower() for c in ag_chain):
            n_skipped += 1
            rows.append({
                "candidate_id": cid,
                "dG_separated": None,
                "dG_cross":     None,
                "dSASA":        None,
                "error":        f"lowercase chain id in {nb_chain}_{ag_chain}",
            })
            continue
        interface_str = f"{nb_chain}_{ag_chain}"

        try:
            pose = pyrosetta.pose_from_pdb(pdb_path)
            iam = InterfaceAnalyzerMover(interface_str)
            iam.set_pack_separated(True)
            iam.set_compute_packstat(False)
            iam.set_compute_interface_sc(False)
            iam.set_compute_interface_energy(True)
            iam.apply(pose)
            dG_sep = float(iam.get_interface_dG())
            try:
                dG_cross = float(iam.get_crossterm_interface_energy())
            except Exception:
                dG_cross = None
            try:
                dSASA = float(iam.get_interface_delta_sasa())
            except Exception:
                dSASA = None
            rows.append({
                "candidate_id": cid,
                "dG_separated": dG_sep,
                "dG_cross":     dG_cross,
                "dSASA":        dSASA,
                "error":        None,
            })
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            n_err += 1
            rows.append({
                "candidate_id": cid,
                "dG_separated": None,
                "dG_cross":     None,
                "dSASA":        None,
                "error":        f"{exc.__class__.__name__}: {exc}",
            })

        if (idx + 1) % 25 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / max(elapsed, 1e-6)
            logger.info(
                "  [%d/%d] ok=%d err=%d skip=%d  %.1fs elapsed  %.2f rows/s",
                idx + 1, len(df), n_ok, n_err, n_skipped, elapsed, rate,
            )

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output_parquet, index=False)
    elapsed = time.time() - t_start
    logger.info(
        "Wrote %d rows to %s in %.1fs (ok=%d err=%d skip=%d).",
        len(rows), args.output_parquet, elapsed, n_ok, n_err, n_skipped,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
