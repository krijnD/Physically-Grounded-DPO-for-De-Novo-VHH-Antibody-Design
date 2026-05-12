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
    # PSH: strict green zone (Gordon et al., 36 clinical nanobodies)
    PSH_GREEN_LOW: float = 79.59
    PSH_GREEN_HIGH: float = 126.83
    # PSH: extended amber/red boundaries (for logging/reporting)
    PSH_RED_LOW: float = 73.4
    PSH_RED_HIGH: float = 155.47
    PPC_MAX: float = 0.39
    COMPACTNESS_LOW: float = 0.81
    COMPACTNESS_HIGH: float = 1.57
    # TNP runtime
    TNP_NCORES: int = 1

    # ── Physics Judge (Rosetta, AbDPO-style residue-level CDR energy) ──
    # Mean Rosetta total energy across CDR residues (REU/residue), per
    # Zhou et al. NeurIPS 2024 §3.2: ε(R⁰) = Σⱼ ε(R⁰[j]) summed over CDR
    # residues, here additionally divided by N_CDR_residues for scope-
    # invariance (works under CDR-H3-only or multi-CDR π_ref scope).
    #
    # ── CALIBRATION PASS (current) ──
    # Both thresholds are set to a large sentinel value so the Physics
    # Judge fast-fail short-circuit in `rosetta_scorer.score_complex()` is
    # disabled and every non-crashing GT row is fully scored (residue-
    # level + sub-residue side-chain energies populated for all rows).
    # This lets the AAPR/calibration parquet carry the complete empirical
    # distribution of all Physics scalars over the curated ANDD GT set,
    # so percentile-based thresholds can be derived post-hoc following
    # AbDPO Appendix E.1 (Zhou et al. NeurIPS 2024, Table 4) — they
    # report success rates at the 50/55/.../95th percentiles of the real
    # antibody training-set distribution and pick the 80th as the
    # headline cutoff. Same convention applies here on natural VHHs.
    #
    # Replace with empirically-derived percentile values after the
    # `andd_calibration_full.parquet` and `andd_calibration_pack.parquet`
    # arms are merged and analysed. Previous (literature-derived,
    # superseded) values: CDR_ENERGY_PER_RES_REJECT = -0.2,
    # E_REP_REJECT = 5.0. Those rejected ~40% of the natural ANDD
    # distribution because they were imported from AbDPO's paired-
    # antibody CDR-H3-only paper without re-calibration for VHH scope
    # and a different scoring regime.
    CDR_ENERGY_PER_RES_REJECT: float = 1.0e9  # REU/residue — calibration sentinel
    E_REP_REJECT: float = 1.0e9               # REU — calibration sentinel
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
