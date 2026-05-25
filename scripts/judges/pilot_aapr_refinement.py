#!/usr/bin/env python3
"""Pilot: does pack_cdrs rescue the AAPR E_Rep distribution?

The 2026-05-25 AAPR canary on the J-anchor-fixed FT (seed42_jfix) shows
97% Physics-judge rejection with E_Rep median 57 REU vs GT p80 of 3.27 REU.
Diagnosis: DiffAb produces diffusion-noisy side-chain rotamers that inflate
Rosetta fa_rep when scored as-is. This pilot tests whether pack_cdrs (CDR +
±2 shell side-chain repack, backbone fixed) collapses the AAPR distribution
back to a sensible range.

Picks N candidates spanning the AAPR scored distribution (worst, median,
best by E_Rep) and scores each twice — once "none", once "pack_cdrs".

Usage:
    python scripts/judges/pilot_aapr_refinement.py \\
        --scored data/aapr/ftseed42_jfix_K8_20260525/scored.parquet \\
        --n 5

Interpretation:
  * If pack_cdrs E_Rep collapses to 1-15 REU range → the issue is purely
    rotamer noise. Switch AAPR scoring to pack_cdrs (asymmetric vs GT
    calibration in "none"; the asymmetry has a biological justification —
    DiffAb side chains aren't experimentally optimized).
  * If pack_cdrs E_Rep stays high (>20 REU) → the issue is deeper than
    rotamers (likely backbone-level clashes in DiffAb's output). Need
    further investigation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

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


def _score(pose, h_chain: str, ag_chain: str) -> tuple[float, float | None]:
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
    p.add_argument("--scored", type=Path, required=True,
                   help="Path to the AAPR scored parquet (output of judges).")
    p.add_argument("--n", type=int, default=5,
                   help="Number of candidates to pilot (default: 5; will be "
                        "split across worst/median/best E_Rep buckets).")
    args = p.parse_args()

    _ensure_init()

    df = pd.read_parquet(args.scored)
    df = df.dropna(subset=["e_rep"])
    df = df.sort_values("e_rep").reset_index(drop=True)
    n_total = len(df)
    if n_total == 0:
        print("ERROR: no rows with e_rep in scored parquet")
        return 1

    # Sample: best, mid, worst per bucket, evenly spread.
    pick_idx = [int(round(i * (n_total - 1) / (args.n - 1))) for i in range(args.n)]
    rows = df.iloc[pick_idx].to_dict("records")

    print(f"Scored AAPR distribution: n={n_total}, "
          f"e_rep p10={df['e_rep'].quantile(0.1):.2f}, "
          f"p50={df['e_rep'].quantile(0.5):.2f}, "
          f"p90={df['e_rep'].quantile(0.9):.2f}")
    print()
    print(f"{'candidate':<18} {'mode':<10}  {'E_Rep':>10}  {'CDR_E/res':>10}  {'ms':>6}")
    print("-" * 65)

    e_rep_deltas: list[float] = []
    for row in rows:
        cid = row["candidate_id"]
        pdb_path = row.get("complex_pdb_path") or row.get("pdb_filepath")
        h_chain = row.get("nanobody_chain_id", "H")
        ag_chain = row.get("antigen_chain_id") or row.get("ag_chain", "A")
        if not pdb_path or not Path(pdb_path).exists():
            print(f"{cid:<18} -- missing PDB: {pdb_path}")
            continue

        # Original (none) — taken from the parquet
        e_orig = row["e_rep"]
        cdr_orig = row.get("cdr_energy_per_res")
        print(f"{cid:<18} {'none':<10}  {_fmt(e_orig)}  {_fmt(cdr_orig)}  {'(cached)':>6}")

        # pack_cdrs — re-score
        t0 = time.time()
        try:
            pose = load_complex_pose(str(pdb_path))
            pack_cdr_shell(pose, h_chain, cdr_ranges=Config.VHH_CDR_RANGES)
            e_pack, cdr_pack = _score(pose, h_chain, ag_chain)
        except Exception as exc:  # noqa: BLE001
            print(f"{cid:<18} -- pack_cdrs failed: {exc}")
            continue
        t_pack = (time.time() - t0) * 1000

        print(f"{cid:<18} {'pack_cdrs':<10}  {_fmt(e_pack)}  {_fmt(cdr_pack)}  {t_pack:>6.0f}")

        # Delta
        d_e = e_pack - e_orig
        d_cdr = (cdr_pack - cdr_orig) if (cdr_orig is not None and cdr_pack is not None) else None
        e_rep_deltas.append(d_e)
        d_e_str = f"{d_e:>+10.3f}"
        d_cdr_str = f"{d_cdr:>+10.3f}" if d_cdr is not None else f"{'n/a':>10}"
        print(f"{'':<18} {'Δ(pack-none)':<10}  {d_e_str}  {d_cdr_str}")
        print()

    if e_rep_deltas:
        n = len(e_rep_deltas)
        mean_e = sum(e_rep_deltas) / n
        max_d = max(e_rep_deltas, key=abs)
        print("=" * 65)
        print(f"Summary across {n} candidates:")
        print(f"  E_Rep   Δ(pack-none): mean={mean_e:+.3f} REU, max-|Δ|={max_d:+.3f}")
        print()
        if mean_e < -10:
            print(">>> pack_cdrs SUBSTANTIALLY LOWERS E_Rep. Rotamer-noise diagnosis")
            print(">>> confirmed. Switch AAPR scoring to pack_cdrs.")
        elif mean_e < -2:
            print(">>> pack_cdrs moderately lowers E_Rep. Helpful but not dramatic.")
        else:
            print(">>> pack_cdrs does NOT help E_Rep significantly.")
            print(">>> The issue is not rotamer noise — investigate backbone-level clashes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
