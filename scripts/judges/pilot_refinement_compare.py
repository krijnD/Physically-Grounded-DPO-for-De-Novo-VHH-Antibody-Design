#!/usr/bin/env python3
"""Pilot: how much does pack_cdrs actually shift physics scores on GTs?

Scores N GT complexes from the curated manifest under two modes:
  * "none"       — load pose, score directly (no refinement)
  * "pack_cdrs"  — load pose, side-chain repack CDRs + ±2 shell, score

For each entry, prints E_Rep and CDR-energy under both modes, plus the
deltas. This informs the threshold-recalibration choice without
committing to either mode — see the post-fix discussion in
docs/aapr_masking_research_context.md.

Run on Snellius:
    cd ~/Physically-Grounded-DPO-for-De-Novo-VHH-Antibody-Design
    source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate
    python scripts/judges/pilot_refinement_compare.py --n 10

Interpretation:
  * Small deltas (E_Rep |Δ| < 0.5 REU, CDR-energy |Δ| < 0.05 REU/res)
    → pack_cdrs is mostly a no-op on these GTs; safe to use everywhere.
  * Large deltas → pack_cdrs is materially shifting the GT distribution;
    decide based on whether DiffAb outputs need similar treatment to be
    fairly comparable.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import Config  # noqa: E402
from src.physics_judge.rosetta_scorer import (  # noqa: E402
    _ensure_init,
    compute_cdr_energy_per_res,
    compute_e_rep,
    load_complex_pose,
    pack_cdr_shell,
)

DEFAULT_PDB_DIR = (
    "/projects/0/hpmlprjs/interns/krijn/ANDD_nano_dataset_IgLM/"
    "VHH_structures_post_diffab"
)
DEFAULT_MANIFEST = "data/datasets/diffab_manifest.tsv"


def _score(pose, h_chain: str, ag_chain: str) -> tuple[float, float | None]:
    """E_Rep + CDR-energy on the pose AS-IS. No refinement here."""
    interface = f"{h_chain}_{ag_chain}"
    e_rep = compute_e_rep(pose, interface)
    try:
        cdr_e = compute_cdr_energy_per_res(
            pose,
            nanobody_chain_id=h_chain,
            cdr_ranges=Config.VHH_CDR_RANGES,
        )
    except Exception:  # noqa: BLE001
        cdr_e = None
    return e_rep, cdr_e


def _fmt(v: float | None, width: int = 10, prec: int = 3) -> str:
    if v is None:
        return f"{'n/a':>{width}}"
    return f"{v:>{width}.{prec}f}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=10,
                   help="Number of GTs to score (default: 10).")
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--pdb-dir", default=DEFAULT_PDB_DIR)
    p.add_argument("--max-resolution", type=float, default=3.0,
                   help="Skip structures with worse resolution than this "
                        "(default: 3.0 Å — focus on well-resolved GTs).")
    args = p.parse_args()

    _ensure_init()

    rows: list[dict] = []
    with open(args.manifest) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            ag = r["antigen_chain"].split("|")[0].strip()
            if not ag:
                continue
            try:
                res = float(r["resolution"])
                if res > args.max_resolution:
                    continue
            except (ValueError, TypeError):
                continue  # skip 'NOT' / cryo-EM with weird resolution
            rows.append({
                "pdb": r["pdb"].strip().lower(),
                "h": r["Hchain"].strip(),
                "ag": ag,
                "res": res,
            })
            if len(rows) >= args.n:
                break

    print(
        f"Comparing 'none' vs 'pack_cdrs' refinement on {len(rows)} GTs "
        f"(resolution ≤ {args.max_resolution} Å).\n"
    )
    print(f"{'pdb_id':<8} {'res':>5} {'mode':<10}  "
          f"{'E_Rep':>10}  {'CDR_E/res':>10}  {'ms':>6}")
    print("-" * 60)

    e_rep_deltas: list[float] = []
    cdr_e_deltas: list[float] = []

    for row in rows:
        pdb_path = Path(args.pdb_dir) / f"{row['pdb']}.pdb"
        if not pdb_path.exists():
            print(f"{row['pdb']:<8}  -- missing PDB at {pdb_path}")
            continue

        # ── Mode 1: no refinement ──────────────────────────────────
        t0 = time.time()
        pose = load_complex_pose(str(pdb_path))
        try:
            e_none, cdr_none = _score(pose, row["h"], row["ag"])
        except Exception as exc:  # noqa: BLE001
            print(f"{row['pdb']:<8}  -- score failed (none): {exc}")
            continue
        t_none = (time.time() - t0) * 1000

        print(f"{row['pdb']:<8} {row['res']:>5.2f} {'none':<10}  "
              f"{_fmt(e_none)}  {_fmt(cdr_none)}  {t_none:>6.0f}")

        # ── Mode 2: pack_cdrs ──────────────────────────────────────
        t0 = time.time()
        pose = load_complex_pose(str(pdb_path))  # fresh load
        try:
            pack_cdr_shell(pose, row["h"], cdr_ranges=Config.VHH_CDR_RANGES)
            e_pack, cdr_pack = _score(pose, row["h"], row["ag"])
        except Exception as exc:  # noqa: BLE001
            print(f"{row['pdb']:<8}  -- pack/score failed: {exc}")
            continue
        t_pack = (time.time() - t0) * 1000

        print(f"{row['pdb']:<8} {'':>5} {'pack_cdrs':<10}  "
              f"{_fmt(e_pack)}  {_fmt(cdr_pack)}  {t_pack:>6.0f}")

        # ── Delta ──────────────────────────────────────────────────
        d_e = e_pack - e_none
        d_cdr = (cdr_pack - cdr_none) if (cdr_none is not None and cdr_pack is not None) else None
        e_rep_deltas.append(d_e)
        if d_cdr is not None:
            cdr_e_deltas.append(d_cdr)

        d_e_str = f"{d_e:>+10.3f}"
        d_cdr_str = f"{d_cdr:>+10.3f}" if d_cdr is not None else f"{'n/a':>10}"
        print(f"{'':<8} {'':>5} {'Δ(pack-none)':<10}  {d_e_str}  {d_cdr_str}")
        print()

    # ── Summary ────────────────────────────────────────────────────
    if e_rep_deltas:
        n = len(e_rep_deltas)
        mean_e = sum(e_rep_deltas) / n
        max_e = max(e_rep_deltas, key=abs)
        print("=" * 60)
        print(f"Summary across {n} entries:")
        print(f"  E_Rep    Δ(pack-none): mean={mean_e:+.3f} REU, "
              f"max-|Δ|={max_e:+.3f} REU")
        if cdr_e_deltas:
            n2 = len(cdr_e_deltas)
            mean_c = sum(cdr_e_deltas) / n2
            max_c = max(cdr_e_deltas, key=abs)
            print(f"  CDR_E    Δ(pack-none): mean={mean_c:+.3f} REU/res, "
                  f"max-|Δ|={max_c:+.3f} REU/res")
        print()
        print("Interpretation:")
        if abs(mean_e) < 0.5 and (not cdr_e_deltas or abs(sum(cdr_e_deltas)/len(cdr_e_deltas)) < 0.05):
            print("  Small effect — pack_cdrs is mostly a no-op on these GTs.")
            print("  Recommend pack_cdrs everywhere (matches AAPR pipeline).")
        else:
            print("  Material effect — pack_cdrs shifts the GT distribution.")
            print("  Discuss before committing to a recalibration mode.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
