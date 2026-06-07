"""
Brief 16 §4.9 — consolidated β-sweep comparison parquet.

Reads:
  data/eval/scrmsd_beta0005.parquet                  (β=0.005 design shards, merged)
  data/eval/scrmsd_beta05.parquet                    (β=0.5   design shards, merged)
  data/eval/caar_epif1_beta0005.parquet              (β=0.005 CAAR/EpiF1 shards, merged)
  data/eval/caar_epif1_beta05.parquet                (β=0.5   CAAR/EpiF1 shards, merged)
  data/eval/design_samples_master.parquet            (baselines: π_ref + floor π_θ)
  data/eval/per_position_modal_picks_all.parquet     (modal-match, modal motif, 6 variants×test_set)
  runs/vhh_ft/seed42_jfix/eval_test_design.json      (π_ref AAR)
  runs/dpo/dpo_seqonly_filtered/eval_test_design.json (floor π_θ AAR)
  runs/dpo/floor_dpo_beta0005/eval_test_design.json   (β=0.005 AAR)
  runs/dpo/floor_dpo_beta05/eval_test_design.json     (β=0.5 AAR)

Writes:
  data/eval/beta_sweep_comparison.parquet — 12 rows (4 variants × 3 CDRs).
                                            Backed up to .backup_pre_brief16.parquet
                                            if a previous version exists.

Schema (per orchestrator request):
  beta                    float64 (NaN for π_ref baseline; 0.005 / 0.05 / 0.5 otherwise)
  variant                 string  (modal-pick variant key)
  test_set                string  ("oldtest")
  cdr                     string  ("H1" / "H2" / "H3")
  modal_match_pct         float64 (0-100; mean of per-position modals_match × 100)
  aar_mean_pp             float64 (0-100; from eval_test_design.json)
  scrmsd_designable_pct   float64 (0-100; fraction of samples with scrmsd_<cdr> < 2 Å)
  scrmsd_median_A         float64 (Å; median over the 116 samples for that CDR)
  scrmsd_nan_count        int64   (number of ABB2-failed samples — included for transparency)
  caar_mean_pp            float64 (0-100; mean of caar column, already in percent)
  epif1_mean              float64 (0-1)
  h3_modal_motif          string  (concat of gen_modal_aa over positions 0..L-1; NULL for H1/H2 rows)
  h3_modal_motif_match    bool    (True iff h3_modal_motif starts with 'YCAAAGGG'; NULL for H1/H2)

Run from campaign root:
  python scripts/eval/build_beta_sweep_comparison.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

SCRMSD_DESIGNABLE_THRESHOLD_A = 2.0
H3_REFERENCE_MOTIF            = "YCAAAGGG"
OUTPUT_PATH = Path("data/eval/beta_sweep_comparison.parquet")

# (beta, variant_key, scrmsd_path_or_None, caar_path_or_None, design_json_path)
# beta=None → baseline pulled from design_samples_master.parquet.
VARIANTS = [
    (None,  "seed42_jfix",
        None, None,
        "runs/vhh_ft/seed42_jfix/eval_test_design.json"),
    (0.05,  "floor_pi_theta",
        None, None,
        "runs/dpo/dpo_seqonly_filtered/eval_test_design.json"),
    (0.005, "floor_pi_theta_b0005",
        "data/eval/scrmsd_beta0005.parquet",
        "data/eval/caar_epif1_beta0005.parquet",
        "runs/dpo/floor_dpo_beta0005/eval_test_design.json"),
    (0.5,   "floor_pi_theta_b05",
        "data/eval/scrmsd_beta05.parquet",
        "data/eval/caar_epif1_beta05.parquet",
        "runs/dpo/floor_dpo_beta05/eval_test_design.json"),
]


def _row_for(beta, variant, scrmsd_path, caar_path, json_path,
             master, modal):
    """Build one (variant × cdr) row triple — returns a list of 3 dicts."""
    if not Path(json_path).exists():
        sys.exit(f"FATAL: design JSON not found: {json_path}")
    design = json.load(open(json_path))["design"]

    if scrmsd_path:
        s_df = pd.read_parquet(scrmsd_path)
        c_df = pd.read_parquet(caar_path)
    else:
        s_df = master[(master["variant"] == variant) & (master["test_set"] == "oldtest")]
        c_df = s_df

    rows = []
    for cdr in ("H1", "H2", "H3"):
        scrmsd_col = f"scrmsd_{cdr}"
        ss = s_df[s_df["cdr"] == cdr] if "cdr" in s_df.columns else s_df
        cc = c_df[c_df["cdr"] == cdr] if "cdr" in c_df.columns else c_df

        designable = float((ss[scrmsd_col] < SCRMSD_DESIGNABLE_THRESHOLD_A).mean() * 100) \
                     if scrmsd_col in ss.columns and len(ss) else float("nan")
        scrmsd_med = float(ss[scrmsd_col].median()) if scrmsd_col in ss.columns and len(ss) else float("nan")
        scrmsd_nan = int(ss[scrmsd_col].isna().sum()) if scrmsd_col in ss.columns else 0

        caar_mean  = float(cc["caar"].mean())  if "caar"  in cc.columns and len(cc) else float("nan")
        epif1_mean = float(cc["epif1"].mean()) if "epif1" in cc.columns and len(cc) else float("nan")

        mm = modal[
            (modal["variant"] == variant)
            & (modal["test_set"] == "oldtest")
            & (modal["cdr"] == cdr)
        ]
        modal_match_pct = float(mm["modals_match"].astype(float).mean() * 100) if len(mm) else float("nan")

        h3_motif = None
        h3_motif_match = None
        if cdr == "H3" and len(mm):
            motif = "".join(mm.sort_values("position")["gen_modal_aa"].fillna("-").astype(str).tolist())
            h3_motif = motif
            h3_motif_match = bool(motif.startswith(H3_REFERENCE_MOTIF)) if len(motif) >= len(H3_REFERENCE_MOTIF) else False

        rows.append(dict(
            beta=beta,
            variant=variant,
            test_set="oldtest",
            cdr=cdr,
            modal_match_pct=modal_match_pct,
            aar_mean_pp=float(design[cdr]["aar_mean"] * 100),
            scrmsd_designable_pct=designable,
            scrmsd_median_A=scrmsd_med,
            scrmsd_nan_count=scrmsd_nan,
            caar_mean_pp=caar_mean,
            epif1_mean=epif1_mean,
            h3_modal_motif=h3_motif,
            h3_modal_motif_match=h3_motif_match,
        ))
    return rows


def main():
    master = pd.read_parquet("data/eval/design_samples_master.parquet")
    modal  = pd.read_parquet("data/eval/per_position_modal_picks_all.parquet")

    all_rows = []
    for spec in VARIANTS:
        all_rows.extend(_row_for(*spec, master=master, modal=modal))

    df = pd.DataFrame(all_rows)

    if OUTPUT_PATH.exists():
        bak = OUTPUT_PATH.with_suffix(".backup_pre_brief16.parquet")
        if not bak.exists():
            shutil.copy2(OUTPUT_PATH, bak)
            print(f"Backed up existing {OUTPUT_PATH} → {bak}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")
    print()

    # Default 2-decimal format rounds 0.005 → 0.01 on stdout; show beta at 3 decimals
    # so 0.005 / 0.05 / 0.5 are unambiguous. Underlying parquet is unaffected.
    pd.options.display.float_format = "{:.2f}".format
    print(df.to_string(
        index=False,
        formatters={"beta": lambda x: "  NaN" if pd.isna(x) else f"{x:.3f}"},
    ))


if __name__ == "__main__":
    main()
