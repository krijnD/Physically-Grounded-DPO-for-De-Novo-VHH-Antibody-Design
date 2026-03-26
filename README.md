# Physically-Grounded DPO for De Novo VHH Antibody Design

## Project Overview

This project builds a DPO (Direct Preference Optimization) training pipeline for de novo VHH (nanobody) antibody design. The core idea: generative models (IgLM) produce candidate sequences that *look* plausible but may be physically unviable. We construct a **winner/loser preference dataset** by running candidates through a multi-judge pipeline grounded in structural biology, biophysics, and physics. Sequences that fail the judges become **hard negatives** (losers), paired with ground-truth sequences (winners) for DPO training.

---

## Multi-Judge Pipeline

The pipeline follows a **"Fold Once, Judge Many"** architecture. A candidate sequence enters, gets numbered, filtered, folded into a 3D structure, and then evaluated by three independent judges. Each judge produces a verdict (`pass` / `fail`). Any failure makes the sequence a hard negative.

```
                    ┌─────────────────────────────┐
                    │     Raw VHH Sequence         │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Phase 1: Kabat Numbering    │
                    │  (ANARCI / abnumber)         │
                    └──────────────┬──────────────┘
                                   │
                          ┌────────▼────────┐
                          │  Absolute rules  │
                          │  (e.g. W47)      │
                          └───┬─────────┬───┘
                     fail     │         │  pass / conditional flag
                              │         │
                  ┌───────────▼┐   ┌────▼──────────────────┐
                  │ Hard        │   │ Phase 2: Fold          │
                  │ Negative    │   │ (NanoBodyBuilder2)     │
                  └─────────────┘   └────┬─────────────────┘
                                         │ PDB structure
                         ┌───────────────┼───────────────┐
                         │               │               │
                  ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
                  │  Biology    │ │  Biophysics │ │  Physics    │
                  │  Judge      │ │  Judge      │ │  Judge      │
                  │  (SAP)      │ │  (TNP)      │ │  (Rosetta)  │
                  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
                         │               │               │
                         └───────────────┼───────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  Final Verdict       │
                              │  → Parquet output    │
                              └─────────────────────┘
```

### Phase 1 — Sequence Pre-filter

Runs entirely on the 1D amino acid sequence. No folding required.

1. **Kabat numbering** — The raw sequence is aligned to the Kabat scheme using ANARCI (via `abnumber`). This identifies hallmark framework positions.
2. **Absolute rejection** — If position 47 is Tryptophan (`W47`), the sequence is immediately rejected. The exposed indole ring drives irreversible aggregation.
3. **Conditional flags** — Non-fatal liabilities that require 3D context to resolve:
   - `L45` — Loss of gatekeeper electrostatic repulsion (Arg→Leu)
   - `V37` — Small aliphatic leaves a structural cavity on the former VL interface
   - `G44` — Loss of Glu solvation shell exposes adjacent hydrophobic atoms
   - `CDR3 W/F` — Bulky hydrophobes in CDR3 that may nucleate aggregation

Sequences with **no flags** pass the Biology Judge immediately (no folding needed). Sequences with **conditional flags** proceed to Phase 2 for structural resolution.

### Phase 2 — 3D Structure Generation (via TNP)

Non-rejected sequences are folded using **NanoBodyBuilder2** through the **TNP** (Therapeutic Nanobody Profiler) pipeline. TNP serves as both the folder and the biophysics analyzer — it folds the sequence and computes all surface metrics (PSH, PPC, PNC, Compactness, CDR lengths) in a single pass. The resulting PDB is shared across all judges, enforcing the "Fold Once, Judge Many" principle.

### Phase 3 — Multi-Judge Evaluation

Each judge independently evaluates the folded structure and writes its verdict to the candidate record.

---

## Biology Judge

**Status: Implemented**

**Purpose:** Determines whether conditional flags from Phase 1 represent real aggregation liabilities by examining the 3D structural context around the flagged residue.

**Method:** Localized Spatial Aggregation Propensity (SAP)
- Computes Shrake-Rupley SASA (solvent-accessible surface area) over the full structure
- Uses a K-D tree (`NeighborSearch`) to find all residues within a 10 Å radius of the flagged position
- Weights each neighbor's SASA by its Kyte-Doolittle hydrophobicity
- Sums the weighted contributions → **SAP score**

**Interpretation:**
- High positive SAP → exposed hydrophobic surface → **aggregation risk** → `fail_conditional`
- Low/negative SAP → the region is shielded by polar/charged residues (CDR loop rescue) → `pass`

**Threshold:** SAP > 150.0 → fail

**Decision flow:**
| Condition | Verdict | Needs folding? |
|-----------|---------|----------------|
| No flags from Phase 1 | `pass` | No |
| W47 detected | `fail_absolute` | No |
| Conditional flag + SAP ≤ 150.0 | `pass` | Yes |
| Conditional flag + SAP > 150.0 | `fail_conditional` | Yes |

**Code:** `src/biology_judge/`
| File | Role |
|------|------|
| `sequence_filter.py` | Phase 1: Kabat numbering, absolute rules, conditional flags |
| `sap_calculator.py` | Localized SAP computation (SASA + NeighborSearch + hydrophobicity) |
| `judge.py` | Orchestrator: routes flags to SAP, writes verdict |

---

## Biophysics Judge (TNP)

**Status: Implemented**

**Purpose:** Evaluates clinical developability via global surface properties using the [Therapeutic Nanobody Profiler](https://github.com/oxpig/TNP) (Gordon et al.), calibrated against 36 clinical-stage nanobody therapeutics.

**Method:** TNP computes 6 metrics from the folded structure. Three are used for rejection (strict green zone), three are stored for analysis:

| Metric | Safe range | Rejection | Meaning |
|--------|-----------|-----------|---------|
| PSH (Patches of Surface Hydrophobicity) | 79.59 – 126.83 | Yes | Too high → aggregation; too low → unfolded |
| PPC (Positive Patch Charge) | < 0.39 | Yes | Too high → non-specific binding, rapid clearance |
| Compactness (CDR3 loop geometry) | 0.81 – 1.57 | Yes | Too low → flailing loop; too high → steric strain |
| PNC (Patches of Negative Charge) | — | No (stored) | Informational |
| Total CDR Length | — | No (stored) | Informational |
| CDR3 Length | — | No (stored) | Informational |

**Decision flow:**
| Condition | Verdict |
|-----------|---------|
| PSH outside [79.59, 126.83] | `fail_psh` |
| PPC > 0.39 | `fail_ppc` |
| Compactness outside [0.81, 1.57] | `fail_compactness` |
| All metrics in range | `pass` |

**Code:** `src/biophysics_judge/`
| File | Role |
|------|------|
| `tnp_runner.py` | TNP CLI subprocess execution + JSON/PDB output parsing |
| `judge.py` | BiophysicsJudge: threshold checks on TNP metrics |

---

## Physics Judge (Rosetta)

**Status: Not yet implemented**

**Purpose:** Evaluates thermodynamic viability using PyRosetta energy calculations.

**Planned metrics and thresholds:**
| Metric | Rejection rule | Meaning |
|--------|---------------|---------|
| ΔG_bind | > -2.0 REU | Non-binder ("Rock") — thermodynamically inert |
| E_Rep | > 5.0 REU | Steric clash — atoms overlap in the predicted structure |

---

## Datasets

### 1. ANDD (Antibody and Nanobody Design Dataset)

**Source:** [Zenodo – ANDD_pdb.zip](https://zenodo.org/records/18151718/files/ANDD_pdb.zip?download=1)
**Metadata:** `Antibody and Nanobody Design Dataset (ANDD)_v2.xlsx`
**Location:** `/projects/0/hpmlprjs/interns/krijn/ANDD_nano_dataset_IgLM/`

#### Filtering (`data scripts/filter_andd_vhh.py`)

Starting from the full ANDD Excel file, the following filters were applied:

1. **VHH/Nanobody only** — rows where `Ab_or_Nano == 'Nanobody/VHH'`
2. **Split by structure availability** — separated on whether `PDB_ID` is present (non-null)

#### Resulting files

| File | Sequences | Description |
|---|---|---|
| `ANDD_VHH_only.csv` | 30,119 | All VHH sequences (full filtered set) |
| `ANDD_VHH_with_structure.csv` | 3,178 | VHH sequences with a known PDB structure |
| `ANDD_VHH_no_structure.csv` | 26,941 | VHH sequences without a structure (require ESMfold) |

#### PDB structures

| Directory | PDB files | Description |
|---|---|---|
| `All_structures/` | 8,214 | All structures from the ANDD bulk download |
| `VHH_structures/` | 1,261 | VHH-only structures copied from `All_structures/` |

---

### 2. SAbDab Nanobody Dataset

**Source:** [SAbDab nanobody summary](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/nanobody/)
**Location:** `/projects/0/hpmlprjs/interns/krijn/sabdab_nano_dataset_IgLM/`

#### Filtering (`data scripts/fetch_nano.py`)

Starting from `sabdab_nano_summary.tsv` (2,422 entries), the following filters were applied:

1. **Post-IgLM training cutoff** — `date >= 2023-01-01` (ensures no data leakage into IgLM training data)
2. **High resolution** — `resolution <= 2.5 Å` (required for reliable PyRosetta `E_Rep` energy evaluation)

#### Resulting files

| Resource | Count | Description |
|---|---|---|
| `sabdab_nano_summary.tsv` | 2,422 entries | Full SAbDab nanobody summary (raw download) |
| `filtered_vhh_pdbs/` | 38 PDB files | Structures passing both filters, downloaded from RCSB |

---

## Project Structure

```
src/
├── common/
│   ├── candidate.py        # NanobodyCandidate dataclass — shared across all judges
│   ├── config.py            # Centralized thresholds for all judges
│   └── pdb_utils.py         # PDB loading via Biopython
├── biology_judge/
│   ├── sequence_filter.py   # Phase 1: Kabat numbering + flag assignment
│   ├── sap_calculator.py    # Localized SAP computation
│   └── judge.py             # Biology Judge orchestrator
├── biophysics_judge/
│   ├── tnp_runner.py        # TNP CLI subprocess + JSON/PDB output parsing
│   └── judge.py             # BiophysicsJudge: threshold checks on TNP metrics
├── physics_judge/           # Placeholder
└── pipeline.py              # Top-level orchestrator: filter → TNP fold → judge → Parquet
data/
├── structures/              # PDB files keyed by candidate_id
├── datasets/                # Input CSVs (ANDD, SAbDab)
└── results/                 # judge_verdicts.parquet (output)
```

---

## Setup (Snellius)

### 1. Create Python virtual environment

```bash
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

python -m venv /projects/0/hpmlprjs/interns/krijn/venvs/DPO
source /projects/0/hpmlprjs/interns/krijn/venvs/DPO/bin/activate
pip install --upgrade pip
```

### 2. Install TNP (Therapeutic Nanobody Profiler)

TNP handles both NanoBodyBuilder2 folding and biophysics metric computation.

```bash
cd /projects/0/hpmlprjs/interns/krijn/tools/
git clone https://github.com/oxpig/TNP.git
cd TNP
pip install .

# Verify
which TNP
TNP --help
```

This installs TNP and its dependencies (ImmuneBuilder/NanoBodyBuilder2, ANARCI, BioPython).

### 3. Install DSSP (required by TNP)

DSSP must be built from source on Snellius since there is no pip package. Requires GCC 13+ for C++20 support.

```bash
# Load the compiler (GCC 13.3.0 — must match the Python toolchain)
module purge
module load 2024
module load GCCcore/13.3.0

# Clone and build
cd /projects/0/hpmlprjs/interns/krijn/tools/
git clone https://github.com/PDB-REDO/dssp.git
cd dssp
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV
cmake --build build
cmake --install build

# Verify
which mkdssp
mkdssp --version
```

### 4. Install PyRosetta (for Physics Judge — future)

```bash
pip install pyrosetta \
  --find-links https://graylab.jhu.edu/download/PyRosetta4/archive/release-quarterly/release
```

### 5. Download datasets

```bash
# ANDD dataset
wget -O ANDD_pdb.zip "https://zenodo.org/records/18151718/files/ANDD_pdb.zip?download=1"

# SAbDab nanobody summary
wget https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/nanobody/ -O sabdab_nano_summary.tsv
```

---

## Scripts

| Script | Description |
|---|---|
| `data scripts/fetch_nano.py` | Downloads post-2023, high-resolution (≤2.5 Å) VHH structures from SAbDab/RCSB |
| `data scripts/filter_andd_vhh.py` | Filters ANDD Excel for VHH sequences and splits by structure availability |
