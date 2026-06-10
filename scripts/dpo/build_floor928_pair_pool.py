#!/usr/bin/env python3
"""Build the E1-B pair pool: whitelist the floor's 928 pair_ids on the
UNFILTERED decoy-t1 pool, and emit a sign-flip diagnostic.

Background
----------
Brief E1 (membership diagnostic) compares two pools at fixed pair
identity:

  * Floor (E1-A): π_θ trained on the 928 lwref-filtered, GT-winner
    pairs (``pairs_filtered_marginGTp0.0.parquet``).
  * Decoy-t1 (E1-B): π_θ trained on the SAME 928 pair_ids but with the
    winner side swapped to a t=1 decoy of the GT crystal. We do NOT
    re-filter on lwref — Krijn's Q2 Rider 3 (2026-06-10) says "log,
    don't filter" the sign-flip column so the experimental contrast is
    purely the winner-side substitution and the pair set is held fixed.

This script:

  1. Inner-joins ``pairs_decoy_t1.parquet`` (1492 rows, decoy winners)
     against the floor's 928 ``pair_id`` whitelist.
  2. Writes ``pairs_decoy_t1_floor928.parquet`` with the decoy-pool row
     schema preserved verbatim (``winner_provenance="decoy_t1"`` is
     what triggers ``_resolve_winner_source`` to disk-parse the decoy
     PDB at train time).
  3. Computes a uniform-channel composite ref_margin
     ``m = (L_l_ref_rot - L_w_ref_rot) + (L_l_ref_pos - L_w_ref_pos)
           + (L_l_ref_seq - L_w_ref_seq)``
     for both the floor lwref pool and the decoy-t1 lwref pool,
     restricts to the 928 whitelist, and prints a 4-cell sign-flip
     contingency table (both>0, both<0, floor>0 & decoy<0, floor<0 &
     decoy>0) plus the fraction the decoy pool would have lost to a
     lwref-filter (the 658-pool overlap).
  4. Reports mean/median of ``m_c = L_l_ref_c − L_w_ref_c`` per channel
     on the 928 decoy pool (rot should drop to ≈ −0.22 at t=1, per the
     bathtub left-wall expectation).

The output parquet feeds the E1-B training run. No filtering happens
on the lwref axis; sign-flips are logged only.

Run
---
::

    python scripts/dpo/build_floor928_pair_pool.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DPO_DIR = REPO_ROOT / "data/aapr/ftseed42_jfix_trainval_K8_20260525/dpo"

DECOY_PAIRS_PATH = DPO_DIR / "pairs_decoy_t1.parquet"
FLOOR_FILTERED_PATH = DPO_DIR / "pairs_filtered_marginGTp0.0.parquet"
FLOOR_LWREF_PATH = DPO_DIR / "lwref_per_channel_floor.parquet"
DECOY_LWREF_PATH = DPO_DIR / "lwref_per_channel_decoy_t1.parquet"

OUT_PATH = DPO_DIR / "pairs_decoy_t1_floor928.parquet"

CHANNELS = ("rot", "pos", "seq")


def composite_margin(df: pd.DataFrame) -> pd.Series:
    """Uniform-channel composite ref_margin m = sum_c (L_l_ref_c - L_w_ref_c).

    Returned as a Series indexed by pair_id.
    """
    m = sum(df[f"L_l_ref_{c}"] - df[f"L_w_ref_{c}"] for c in CHANNELS)
    return m.set_axis(df["pair_id"].values)


def main() -> int:
    for p in (
        DECOY_PAIRS_PATH,
        FLOOR_FILTERED_PATH,
        FLOOR_LWREF_PATH,
        DECOY_LWREF_PATH,
    ):
        if not p.exists():
            print(f"ERROR: missing input parquet: {p}", file=sys.stderr)
            return 2

    decoy_pairs = pd.read_parquet(DECOY_PAIRS_PATH)
    floor_filtered = pd.read_parquet(FLOOR_FILTERED_PATH)
    floor_lwref = pd.read_parquet(FLOOR_LWREF_PATH)
    decoy_lwref = pd.read_parquet(DECOY_LWREF_PATH)

    print(f"Loaded:")
    print(f"  pairs_decoy_t1           : {len(decoy_pairs):5d} rows   ({DECOY_PAIRS_PATH.name})")
    print(f"  pairs_filtered_marginGTp0: {len(floor_filtered):5d} rows   ({FLOOR_FILTERED_PATH.name})")
    print(f"  lwref_per_channel_floor  : {len(floor_lwref):5d} rows   ({FLOOR_LWREF_PATH.name})")
    print(f"  lwref_per_channel_decoy_t1: {len(decoy_lwref):5d} rows  ({DECOY_LWREF_PATH.name})")
    print()

    # ---- Step 2-4: whitelist join ----------------------------------------
    floor_ids = set(floor_filtered["pair_id"].tolist())
    decoy_ids = set(decoy_pairs["pair_id"].tolist())

    keep_ids = floor_ids & decoy_ids
    missing_ids = sorted(floor_ids - decoy_ids)

    out = decoy_pairs[decoy_pairs["pair_id"].isin(keep_ids)].copy()
    out = out.reset_index(drop=True)

    n_out = len(out)
    n_gt = out["gt_complex_id"].nunique()
    print(f"Whitelist join: floor∩decoy = {n_out} rows, "
          f"{n_gt} unique gt_complex_id")
    if missing_ids:
        print(f"  MISSING from decoy pool (in floor 928 but absent from decoy_t1): "
              f"{len(missing_ids)} pair_ids")
        for pid in missing_ids:
            print(f"    {pid}")
    else:
        print("  (no missing pair_ids; floor 928 ⊆ decoy_t1 1492)")
    print()

    assert n_out >= 900, f"Expected ≥900 rows after whitelist join; got {n_out}"

    # Schema-preserving write (we keep every column the decoy-pool row had,
    # including winner_provenance='decoy_t1', original_winner_pdb_path, etc.)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"Wrote: {OUT_PATH}")
    print(f"  rows={len(out)}  cols={list(out.columns)}")
    print()

    # ---- Step 5: sign-flip diagnostic ------------------------------------
    m_floor = composite_margin(floor_lwref)
    m_decoy = composite_margin(decoy_lwref)

    # Restrict to the whitelist ids
    m_floor = m_floor.loc[m_floor.index.isin(keep_ids)]
    m_decoy = m_decoy.loc[m_decoy.index.isin(keep_ids)]

    # Align (inner join on pair_id)
    common = m_floor.index.intersection(m_decoy.index)
    m_floor = m_floor.loc[common]
    m_decoy = m_decoy.loc[common]
    n = len(common)
    if n != n_out:
        print(f"NOTE: lwref tables cover {n}/{n_out} whitelisted pair_ids "
              f"(lwref intersection).")

    both_pos = ((m_floor > 0) & (m_decoy > 0)).sum()
    both_neg = ((m_floor < 0) & (m_decoy < 0)).sum()
    flip_pn = ((m_floor > 0) & (m_decoy < 0)).sum()
    flip_np = ((m_floor < 0) & (m_decoy > 0)).sum()
    edge_zero = ((m_floor == 0) | (m_decoy == 0)).sum()
    drop_lwref = (m_decoy <= 0).sum()  # what an lwref-filter on decoy would discard

    def fmt_pct(k: int) -> str:
        return f"{k:4d}/{n:<4d} = {100 * k / n:5.1f}%"

    print("Sign-flip table (m = sum_c (L_l_ref_c − L_w_ref_c); uniform weights):")
    print(f"  both > 0        (preserved positive margin)        : {fmt_pct(both_pos)}")
    print(f"  both < 0        (always-loser; m<0 in both pools)  : {fmt_pct(both_neg)}")
    print(f"  floor>0, decoy<0 (shortcut removed; intended flip) : {fmt_pct(flip_pn)}")
    print(f"  floor<0, decoy>0 (anti-flip; should be small)      : {fmt_pct(flip_np)}")
    if edge_zero:
        print(f"  edge: m==0 on at least one side                   : {fmt_pct(edge_zero)}")
    print(f"  decoy m ≤ 0      (would-be lwref-filter drop)      : {fmt_pct(drop_lwref)}")
    print()

    # ---- Step 6: per-channel sub-diagnostic on the 928 decoy pool --------
    decoy_w = decoy_lwref[decoy_lwref["pair_id"].isin(keep_ids)]
    print(f"Per-channel m_c = L_l_ref_c − L_w_ref_c on whitelist (decoy_t1, "
          f"N={len(decoy_w)}):")
    print(f"  {'channel':<6} {'mean':>10} {'median':>10}")
    for c in CHANNELS:
        m_c = decoy_w[f"L_l_ref_{c}"] - decoy_w[f"L_w_ref_{c}"]
        print(f"  {c:<6} {m_c.mean():10.4f} {m_c.median():10.4f}")
    print()
    print("Sanity: m_rot ≈ −0.22 at t=1 (bathtub left-wall expectation).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
