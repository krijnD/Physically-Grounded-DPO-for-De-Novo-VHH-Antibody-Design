#!/usr/bin/env python3
"""Unit test for PairDataset's winner-source routing.

Verifies that ``PairDataset._resolve_winner_source`` honors the
``winner_provenance`` sentinel introduced in brief 17 §7. Without this
routing, the decoy-winner swap is a silent no-op — exactly the bug
diagnosed on 2026-06-08 (D1 and D2 emitted byte-identical
implicit-reward summaries despite reading different pair parquets).

Pure-pandas; no LMDB, no PDB I/O, no GPU.

Usage::

    python scripts/test_pair_dataset_winner_routing.py
    # or under pytest:
    pytest -q scripts/test_pair_dataset_winner_routing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "diffab"))

import pandas as pd

try:
    from src.dpo.dataset import PairDataset
except ModuleNotFoundError as _exc:
    # The Mac dev venv may lack torch; the DPO venv on Snellius has it.
    # Print a single-line skip and exit clean rather than failing CI for
    # an environment issue. Real regressions still surface on Snellius.
    print(f"SKIP: src.dpo.dataset import failed ({_exc}); "
          f"run this test on Snellius with the DPO venv active.")
    sys.exit(0)


# ── Routing helper tests ────────────────────────────────────────────

def test_floor_pair_routes_to_lmdb() -> None:
    """No winner_provenance column → LMDB lookup (existing floor behaviour)."""
    row = pd.Series({
        "gt_complex_id":    "7b2m",
        "winner_pdb_path":  "/should/not/be/read.pdb",
    })
    src, path = PairDataset._resolve_winner_source(row)
    assert src == "lmdb", f"expected 'lmdb' for floor pair, got {src!r}"
    assert path is None, f"expected None path for floor pair, got {path!r}"


def test_blank_provenance_routes_to_lmdb() -> None:
    """Empty / whitespace winner_provenance → LMDB lookup (backwards-compat)."""
    for blank in ("", "   ", None, float("nan")):
        row = pd.Series({
            "gt_complex_id":     "7b2m",
            "winner_pdb_path":   "/x/y.pdb",
            "winner_provenance": blank,
        })
        src, path = PairDataset._resolve_winner_source(row)
        assert src == "lmdb", (
            f"expected 'lmdb' for blank provenance {blank!r}, got {src!r}"
        )
        assert path is None


def test_decoy_provenance_routes_to_disk() -> None:
    """winner_provenance set → disk parse using winner_pdb_path."""
    row = pd.Series({
        "gt_complex_id":     "7b2m",
        "winner_pdb_path":   "/data/aapr/decoys_t10/pdbs/7b2m__decoy_t10.pdb",
        "winner_provenance": "decoy_t10",
    })
    src, path = PairDataset._resolve_winner_source(row)
    assert src == "disk", f"expected 'disk' for decoy pair, got {src!r}"
    assert path == row["winner_pdb_path"], (
        f"expected disk path {row['winner_pdb_path']!r}, got {path!r}"
    )


def test_arbitrary_provenance_label_routes_to_disk() -> None:
    """Any non-blank provenance string flips routing to disk."""
    for label in ("decoy_t5", "decoy_t20", "manual_curated", "ablation_v3"):
        row = pd.Series({
            "gt_complex_id":     "7b2m",
            "winner_pdb_path":   "/x.pdb",
            "winner_provenance": label,
        })
        src, path = PairDataset._resolve_winner_source(row)
        assert src == "disk", f"label {label!r} → expected 'disk', got {src!r}"
        assert path == "/x.pdb"


def test_dataframe_iloc_row() -> None:
    """The helper accepts a Series sliced from a DataFrame (the real call site)."""
    df = pd.DataFrame({
        "gt_complex_id":     ["7b2m", "7b2p"],
        "winner_pdb_path":   ["/gt.pdb", "/decoy.pdb"],
        "winner_provenance": ["", "decoy_t10"],
    })
    src0, path0 = PairDataset._resolve_winner_source(df.iloc[0])
    src1, path1 = PairDataset._resolve_winner_source(df.iloc[1])
    assert (src0, path0) == ("lmdb", None)
    assert (src1, path1) == ("disk", "/decoy.pdb")


# ── _align_to_gt_scaffold contract test ────────────────────────────

def test_align_to_gt_scaffold_handles_missing_keys() -> None:
    """_align_to_gt_scaffold must be a no-op when either side lacks heavy/antigen."""
    PairDataset._align_to_gt_scaffold({}, {})
    PairDataset._align_to_gt_scaffold({"heavy": None}, {"heavy": None})
    PairDataset._align_to_gt_scaffold({"antigen": None}, {"antigen": None})


# ── _parse_pdb rename guard ─────────────────────────────────────────

def test_parse_pdb_method_exists() -> None:
    """Defends against an accidental rename undoing the routing fix."""
    assert hasattr(PairDataset, "_parse_pdb"), (
        "PairDataset._parse_pdb is missing — was it renamed back to "
        "_parse_loser? The winner-side routing depends on this method."
    )
    assert not hasattr(PairDataset, "_parse_loser"), (
        "PairDataset._parse_loser should no longer exist after brief 17 "
        "fix. If you re-introduce it, also update __getitem__."
    )


# ── Driver ──────────────────────────────────────────────────────────

def _run_all() -> int:
    tests = [
        test_floor_pair_routes_to_lmdb,
        test_blank_provenance_routes_to_lmdb,
        test_decoy_provenance_routes_to_disk,
        test_arbitrary_provenance_label_routes_to_disk,
        test_dataframe_iloc_row,
        test_align_to_gt_scaffold_handles_missing_keys,
        test_parse_pdb_method_exists,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {fn.__name__}: {e.__class__.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
