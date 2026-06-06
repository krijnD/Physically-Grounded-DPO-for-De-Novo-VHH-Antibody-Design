"""
Step 6 of Brief 13: per-position modal-pick reconciliation.

For each integer CDR position in {H1, H2, H3} on each test_set, compute:
  - GT modal AA + frequency (across the unique GT entries in that test set)
  - Model modal AA + frequency (across all samples of the chosen variant)
  - modals_match flag and modal_gap_pp (signed pp shift)

By default operates on the campaign's terminal model `expanded_pi_theta`
(`--variant`); pass another variant to re-run for a comparator.

Output: a tidy parquet at `data/eval/per_position_modal_picks.parquet`
with columns:
    variant, test_set, cdr, position,
    n_gt, n_gen,
    gt_modal_aa, gt_modal_freq,
    gen_modal_aa, gen_modal_freq,
    modals_match, modal_gap_pp.

Sequences are extracted directly from the design + GT PDBs (the master
parquet doesn't carry per-position CDR sequence strings), reusing the
chain + window helpers from run_caar_epif1_array.py.

Usage:
    python scripts/eval/compute_per_position_modal_picks.py
    python scripts/eval/compute_per_position_modal_picks.py \\
        --variant expanded_pi_ref --output /tmp/mp_piref.parquet
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser

# Reuse the dispatcher's CDR window + amino-acid map by direct import.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_caar_epif1_array import (  # noqa: E402
    AA3, CDR_WINDOWS, _classify_chains, _cdr_residues,
)


def _cdr_seq_dict(structure, vhh_chain_id, cdr):
    """{resseq: aa1} over the CDR window for the named VHH chain."""
    cdr_residues = _cdr_residues(structure, vhh_chain_id, cdr)
    return {r.id[1]: AA3.get(r.get_resname(), "X") for r in cdr_residues}


def _extract(pdb_path, vhh_chain_hint, cdr, parser, cache=None):
    """Return {resseq: aa1} for the CDR; uses cache by pdb_path."""
    if cache is not None and pdb_path in cache:
        struct = cache[pdb_path]
    else:
        struct = parser.get_structure("x", pdb_path)
        if cache is not None:
            cache[pdb_path] = struct
    vhh, _ag = _classify_chains(struct, vhh_chain_hint=vhh_chain_hint)
    if vhh is None:
        return {}
    return _cdr_seq_dict(struct, vhh, cdr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="data/eval/design_samples_master.parquet")
    ap.add_argument("--gt-pdb-map", default="data/eval/gt_pdb_map.json")
    ap.add_argument("--variant", default="expanded_pi_theta")
    ap.add_argument("--output", default="data/eval/per_position_modal_picks.parquet")
    args = ap.parse_args()

    master = pd.read_parquet(args.master)
    print(f"[load] master: {len(master)} rows × {len(master.columns)} cols")

    sub = master[master["variant"] == args.variant].copy()
    print(f"[filter] variant='{args.variant}': {len(sub)} rows")
    if not len(sub):
        sys.exit(f"FATAL: no rows for variant '{args.variant}'")

    if "pdb_filepath" not in sub.columns:
        sys.exit("FATAL: master parquet has no 'pdb_filepath' column; "
                 "Brief 13 §3 Step 6 needs the design-PDB paths.")
    gt_map = json.loads(Path(args.gt_pdb_map).read_text())
    print(f"[load] gt_pdb_map: {len(gt_map)} entries")

    parser = PDBParser(QUIET=True)
    gt_cache = {}   # {gt_pdb: parsed structure}
    n_missing_gt = 0
    n_missing_design = 0
    n_processed = 0

    # Per (test_set, cdr, position) accumulator: list of (source, aa1)
    # where source ∈ {"gt", "gen"}; GTs counted once per entry, gen
    # counted once per sample.
    accum = {}

    gt_seen = set()   # {(test_set, cdr, entry_id)} so we count each GT once

    for i, row in enumerate(sub.itertuples(index=False)):
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(sub)}] processed")
        cdr = row.cdr
        if cdr not in CDR_WINDOWS:
            continue
        entry = row.entry_id
        gt_pdb = gt_map.get(entry)
        if not gt_pdb or not Path(gt_pdb).exists():
            n_missing_gt += 1
            continue
        design_pdb = row.pdb_filepath
        if not design_pdb or not Path(design_pdb).exists():
            n_missing_design += 1
            continue
        vhh_hint = entry.rsplit("_", 1)[-1] if "_" in entry else None

        # GT residues (cached + count once per entry per test_set per cdr)
        key_gt = (row.test_set, cdr, entry)
        if key_gt not in gt_seen:
            gt_seen.add(key_gt)
            gt_aas = _extract(gt_pdb, vhh_hint, cdr, parser, cache=gt_cache)
            for pos, aa in gt_aas.items():
                accum.setdefault((row.test_set, cdr, pos), []).append(("gt", aa))

        # Design residues — every sample, no cache (each PDB unique)
        dsg_aas = _extract(design_pdb, vhh_hint, cdr, parser)
        for pos, aa in dsg_aas.items():
            accum.setdefault((row.test_set, cdr, pos), []).append(("gen", aa))

        n_processed += 1

    print(f"\n[summary] processed {n_processed} rows; "
          f"missing_gt={n_missing_gt}, missing_design={n_missing_design}")
    print(f"          unique GT (test_set, cdr, entry) keys: {len(gt_seen)}")

    rows = []
    for (test_set, cdr, position), entries in accum.items():
        gt_aas = [aa for src, aa in entries if src == "gt"]
        gen_aas = [aa for src, aa in entries if src == "gen"]
        if not gt_aas and not gen_aas:
            continue
        gt_top = pd.Series(gt_aas).value_counts(normalize=True) if gt_aas else None
        gen_top = pd.Series(gen_aas).value_counts(normalize=True) if gen_aas else None
        gt_modal_aa = gt_top.index[0] if gt_top is not None and len(gt_top) else None
        gt_modal_freq = float(gt_top.iloc[0]) if gt_top is not None and len(gt_top) else None
        gen_modal_aa = gen_top.index[0] if gen_top is not None and len(gen_top) else None
        gen_modal_freq = float(gen_top.iloc[0]) if gen_top is not None and len(gen_top) else None
        rows.append({
            "variant": args.variant,
            "test_set": test_set,
            "cdr": cdr,
            "position": position,
            "n_gt": len(gt_aas),
            "n_gen": len(gen_aas),
            "gt_modal_aa": gt_modal_aa,
            "gt_modal_freq": gt_modal_freq,
            "gen_modal_aa": gen_modal_aa,
            "gen_modal_freq": gen_modal_freq,
            "modals_match": (
                gt_modal_aa is not None
                and gen_modal_aa is not None
                and gt_modal_aa == gen_modal_aa
            ),
            "modal_gap_pp": (
                None if gt_modal_freq is None or gen_modal_freq is None
                else 100.0 * (gen_modal_freq - gt_modal_freq)
            ),
        })
    out = pd.DataFrame(rows).sort_values(
        ["test_set", "cdr", "position"]
    ).reset_index(drop=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output)
    print(f"\n[write] {len(out)} per-position rows → {args.output}")

    # Summary
    print("\n=== Modal-match rate per (test_set × cdr) ===")
    for (test_set, cdr), g in out.groupby(["test_set", "cdr"]):
        n = len(g)
        n_match = int(g["modals_match"].sum())
        print(f"  {test_set}/{cdr} (n positions = {n}): "
              f"model-modal == GT-modal at {n_match}/{n} positions "
              f"({100 * n_match / n:.0f}%)")

    print("\n=== Full per-position table ===")
    cols = ["test_set", "cdr", "position", "n_gt", "n_gen",
            "gt_modal_aa", "gt_modal_freq",
            "gen_modal_aa", "gen_modal_freq",
            "modals_match", "modal_gap_pp"]
    fmt = out[cols].copy()
    fmt["gt_modal_freq"] = fmt["gt_modal_freq"].round(3)
    fmt["gen_modal_freq"] = fmt["gen_modal_freq"].round(3)
    fmt["modal_gap_pp"] = fmt["modal_gap_pp"].round(1)
    print(fmt.to_string(index=False))


if __name__ == "__main__":
    main()
