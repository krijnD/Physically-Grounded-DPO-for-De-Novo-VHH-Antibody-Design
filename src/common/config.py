"""Centralized thresholds and constants for all judges and masking.

Sources:
- Biology (SAP): Chennamsetty et al. (2009) PNAS, Black & Mould (1991) Proteins,
                 Tien et al. (2013) PLOS ONE (max sidechain SASA)
- Biophysics: Gordon et al. (2026), Therapeutic Nanobody Profiler
- Physics: Zhou et al. (NeurIPS 2024), AbDPO — residue-level CDR energy
- Masking: Paratope definition (Leem et al.), anchor protection (GeoGAD, AbFlex),
           FR2 hallmarks (Mitchell & Colwell, 2018)
"""


class Config:
    # ── Biology Judge (normalized localized SAP) ──
    # Per-neighbor contribution = (SASA_residue / SASA_max_residue) × hydropathy_BM,
    # averaged over neighbors within SAP_RADIUS. Both factors are bounded ∈ [-1,+1],
    # so the average is also ∈ [-1,+1]. Positive = exposed-hydrophobic neighborhood
    # (aggregation risk); negative = polar-shielded (compensated by CDR loops).
    # Threshold of +0.15 corresponds to Chennamsetty/Sankar "aggregation-prone"
    # surface — never derived from this project's data, taken from literature.
    #
    # Calibration confirms +0.15 is correct as an ABSOLUTE-FAIL CATCH, not a
    # population-percentile gate: empirical p80 across all four FR2 hallmarks
    # on natural ANDD (post-dedup, full arm) is in [-0.06, -0.001] — all
    # natural VHHs pass by ~10 SDs. The judge is calibrated to detect
    # AAPR-generated pathologies (exposed unshielded indole on solvent-facing
    # W47, etc.), not to reject natural VHHs.
    # See docs/calibration/threshold_decisions.md §5.6 for full rationale.
    SAP_SAFETY_THRESHOLD: float = 0.15
    SAP_RADIUS: float = 10.0  # Angstroms

    # Black & Mould (1991) normalized hydrophobicity scale (3-letter keys),
    # ranging F=+1.00 (most hydrophobic) to R=-1.00 (most hydrophilic).
    # This is the scale Chennamsetty (2009) PNAS used for the original SAP.
    BLACK_MOULD_HYDROPHOBICITY: dict[str, float] = {
        "PHE":  1.00, "MET":  0.83, "ILE":  0.81, "LEU":  0.73,
        "VAL":  0.51, "CYS":  0.40, "TRP":  0.25, "ALA":  0.25,
        "THR":  0.10, "GLY": -0.06, "SER": -0.30, "PRO": -0.40,
        "TYR": -0.42, "HIS": -0.46, "GLN": -0.52, "ASN": -0.54,
        "GLU": -0.62, "ASP": -0.78, "LYS": -0.92, "ARG": -1.00,
    }

    # Tien et al. (2013) "theoretical" maximum total SASA per residue type (Å²),
    # used to normalize Shrake-Rupley-computed residue SASA into a [0,1] fraction.
    # Whole-residue (sidechain + backbone) values; fractions clamped at 1.0.
    MAX_RESIDUE_SASA: dict[str, float] = {
        "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0,
        "CYS": 167.0, "GLU": 223.0, "GLN": 225.0, "GLY": 104.0,
        "HIS": 224.0, "ILE": 197.0, "LEU": 201.0, "LYS": 236.0,
        "MET": 224.0, "PHE": 240.0, "PRO": 159.0, "SER": 155.0,
        "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
    }

    # Residues that flag CDR3 hydrophobic override risk
    CDR3_HYDROPHOBIC_RESIDUES: set[str] = {"W", "F"}

    # ── Biophysics Judge (TNP) ──
    # All three TNP thresholds (PSH, PPC, Compactness) are kept at Gordon et al.
    # 2026 clinical-36-nanobody calibration. ANDD natural-VHH empirical p80
    # (post sequence-dedup, full arm, n=205) validates two of three:
    #   - PSH:        empirical p80 = 113.2 [110.4, 115.2] — inside green zone
    #   - Compactness: empirical p80 =  1.44 [+1.42, +1.47] — inside [0.81, 1.57]
    #   - PPC:        empirical p80 = 0.498 [+0.37, +1.11] — ABOVE Gordon's 0.39
    # The PPC mismatch is an intentional clinical-vs-natural distribution shift:
    # Gordon's < 0.39 is a pharmacokinetic constraint (in-vivo clearance),
    # not a folding/aggregation metric. Natural PDB VHHs were not selected
    # for half-life; clinical-stage VHHs were. Keeping Gordon's 0.39 means
    # the Biophysics Judge rejects ~20% of natural ANDD as "clinical-grade
    # PK-disqualifying" — the desired behavior for a developability gate
    # in a DPO pipeline aimed at clinical-grade outputs.
    # See docs/calibration/threshold_decisions.md §5.5 for full rationale.
    PSH_GREEN_LOW: float = 79.59
    PSH_GREEN_HIGH: float = 126.83
    # PSH: extended amber/red boundaries (for logging/reporting)
    PSH_RED_LOW: float = 73.4
    PSH_RED_HIGH: float = 155.47
    PPC_MAX: float = 0.39   # Clinical PK constraint, not natural-VHH prevalence
    COMPACTNESS_LOW: float = 0.81
    COMPACTNESS_HIGH: float = 1.57

    # ── Physics Judge (Rosetta, AbDPO-style residue-level CDR energy) ──
    # Mean Rosetta total energy across CDR residues (REU/residue), per
    # Zhou et al. NeurIPS 2024 §3.2: ε(R⁰) = Σⱼ ε(R⁰[j]) summed over CDR
    # residues, here additionally divided by N_CDR_residues for scope-
    # invariance (works under CDR-H3-only or multi-CDR π_ref scope).
    #
    # ── EMPIRICAL CALIBRATION (AbDPO Appendix E.1 methodology) ──
    # Both thresholds are the 80th-percentile of the natural ANDD VHH
    # GT distribution under `--refinement-mode full` (full-complex
    # side-chain repack + FastRelax on CDR loops), n=458 after dropping
    # 7 PyRosetta-crash rows. Bootstrap 95% CIs from 1000 resamples,
    # seed=42. See scripts/calibration/percentile_analysis.py and
    # docs/calibration/pack_vs_full_summary.md.
    #
    # Refinement-regime choice: the `pack_cdrs` arm yields p80 = +9.72
    # REU/residue for cdr_energy_per_res because it inherits unresolved
    # GT clashes that aren't physics — they're crystallographic noise
    # that FastRelax dissolves. Pack/full CIs are disjoint on 4 of 5
    # physics scalars; this is a regime change, not measurement noise.
    # Thus AAPR generation MUST also use `--refinement-mode full` so the
    # scored candidates and the GT distribution live in the same regime.
    #
    # Note the per-residue convention: AbDPO Table 4 reports values
    # SUMMED over CDR-H3 residues; thesis project divides by
    # N_CDR_residues (mean 33.15 in ANDD full arm) for scope-invariance
    # across CDR-H3-only ablation and multi-CDR main run. Multiply by
    # ~33 to recover AbDPO's summed scale for direct paper comparison.
    #
    # Previous (literature-derived, superseded) values:
    #   CDR_ENERGY_PER_RES_REJECT = -0.2  — misattributed to AbDPO; the
    #     actual figure was never per-residue in the source, and even at
    #     -0.2 it rejected ~40% of natural ANDD when applied under
    #     pack_cdrs. Tellingly, the empirical full p80 of -0.183 is
    #     within rounding distance of -0.2 — so the literature number
    #     was correct in spirit, but the calibration regime that
    #     produced it was full-relax, not pack-only.
    #   E_REP_REJECT = 5.0  — also dropped ~80% of natural ANDD at p80
    #     (both pack and full arms exceed it at p70+), suggesting the
    #     5.0 cap was tuned for a different scoring/interface scope.
    CDR_ENERGY_PER_RES_REJECT: float = -0.183  # REU/residue. Full p80, CI [-0.320, -0.059]. AbDPO §E.1
    E_REP_REJECT: float = 5.746                # REU. Full p80, CI [+5.162, +6.338]. AbDPO §E.1
    # Any |E_cdr| beyond this is non-physical (Rosetta scoring blowup
    # from unresolved clashes in the bound state) — distinguished from
    # weak-binder rejects so downstream DPO pair selection isn't polluted.
    CDR_ENERGY_PATHOLOGICAL: float = 100.0   # REU/residue
    ROSETTA_INTERFACE: str = "H_A"  # Chain interface for E_Rep selector
    CCD_OUTER_CYCLES: int = 1      # AbDPO-specified LoopMover_Refine_CCD param
    CCD_MAX_INNER_CYCLES: int = 10  # AbDPO-specified LoopMover_Refine_CCD param
    PYROSETTA_FLAGS: str = "-mute all -ignore_unrecognized_res"

    # VHH CDR loop boundaries (Kabat numbering) for CCD refinement
    VHH_CDR_RANGES: list[tuple[int, int]] = [
        (26, 32),   # CDR H1
        (52, 56),   # CDR H2
        (95, 102),  # CDR H3
    ]

    # ── Masking Module ──
    # Paratope detection: heavy-atom distance cutoff (Angstroms)
    # Ref: structural paratope = VHH heavy atoms within 5.0 Å of antigen
    PARATOPE_DISTANCE_CUTOFF: float = 5.0

    # Number of anchor residues flanking each CDR boundary to protect
    # Ref: GeoGAD (Tan et al., 2024), AbFlex (Ruffolo et al., 2024)
    ANCHOR_FLANK_SIZE: int = 3

    # FR2 hallmark positions (Kabat) that distinguish VHH from VH
    # Ref: Mitchell & Colwell (2018), Desmyter et al. (2015)
    FR2_HALLMARK_POSITIONS: list[str] = ["37", "44", "45", "47"]
