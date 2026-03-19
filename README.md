# Physically-Grounded DPO for De Novo VHH Antibody Design

## Project Overview

This project builds a DPO (Direct Preference Optimization) training pipeline for de novo VHH (nanobody) antibody design, grounded in physical energy scores computed via PyRosetta.

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

## Setup

### Download ANDD dataset
```bash
wget -O ANDD_pdb.zip "https://zenodo.org/records/18151718/files/ANDD_pdb.zip?download=1"
```

### Download SAbDab nanobody summary
```bash
wget https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/nanobody/ -O sabdab_nano_summary.tsv
```

### Install PyRosetta
```bash
python -m venv /projects/0/hpmlprjs/interns/krijn/venvs/rosetta

pip install pyrosetta \
  --find-links https://graylab.jhu.edu/download/PyRosetta4/archive/release-quarterly/release
```

---

## Scripts

| Script | Description |
|---|---|
| `data scripts/fetch_nano.py` | Downloads post-2023, high-resolution (≤2.5 Å) VHH structures from SAbDab/RCSB |
| `data scripts/filter_andd_vhh.py` | Filters ANDD Excel for VHH sequences and splits by structure availability |
