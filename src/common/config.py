"""Centralized thresholds and constants for all judges.

Sources:
- Biology: Pipeline design document (Dignum, 2026)
- Biophysics: Gordon et al. (2026), Therapeutic Nanobody Profiler
- Physics: Zhou et al. (NeurIPS 2024), AbDPO
"""


class Config:
    # ── Biology Judge ──
    SAP_SAFETY_THRESHOLD: float = 150.0
    SAP_RADIUS: float = 10.0  # Angstroms

    # Kyte-Doolittle-derived hydrophobicity scale for SAP calculation
    HYDROPHOBICITY_SCALE: dict[str, float] = {
        "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8,
        "CYS": 2.5, "MET": 1.9, "ALA": 1.8,
        "GLY": -0.4, "THR": -0.7, "SER": -0.8, "TRP": -0.9,
        "TYR": -1.3, "PRO": -1.6,
        "HIS": -3.2, "GLN": -3.5, "GLU": -3.5, "ASP": -3.5,
        "ASN": -3.5, "LYS": -3.9, "ARG": -4.5,
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

    # ── Physics Judge (Rosetta) ──
    DELTA_G_REJECT: float = -2.0   # > -2.0 REU → non-binder "Rock"
    E_REP_REJECT: float = 5.0      # > 5.0 REU → steric clash
    ROSETTA_INTERFACE: str = "H_A"  # Chain interface for InterfaceAnalyzerMover
    CCD_OUTER_CYCLES: int = 1      # AbDPO-specified LoopMover_Refine_CCD param
    CCD_MAX_INNER_CYCLES: int = 10  # AbDPO-specified LoopMover_Refine_CCD param
    PYROSETTA_FLAGS: str = "-mute all -ignore_unrecognized_res"
