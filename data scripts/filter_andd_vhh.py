import os
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

# ==========================================
# 1. Define Paths based on your environment
# ==========================================
base_dir = Path("/projects/0/hpmlprjs/interns/krijn")
data_dir = base_dir / "ANDD_nano_dataset_IgLM"

# Path to where the unzipped PDBs live
all_structures_dir = data_dir / "All_structures"
output_pdb_dir = data_dir / "VHH_structures"

# Pointing exactly to the Excel file
metadata_file = base_dir / "ANDD_nano_dataset_IgLM/Antibody and Nanobody Design Dataset (ANDD)_v2.xlsx" 

# Output paths for the two separated datasets
vhh_struct_metadata = data_dir / "ANDD_VHH_with_structure.csv"
vhh_nostruct_metadata = data_dir / "ANDD_VHH_no_structure.csv"

# ==========================================
# 2. Setup & Filter Metadata
# ==========================================
print(f"Loading metadata from {metadata_file.name}...")
try:
    df = pd.read_excel(metadata_file)
except FileNotFoundError:
    print(f"Error: Could not find metadata at {metadata_file}.")
    print("Please ensure the Excel file is in the correct folder.")
    exit(1)

print("Filtering for 'Nanobody/VHH'...")
# Use .copy() to avoid SettingWithCopyWarning later
vhh_df = df[df['Ab_or_Nano'] == 'Nanobody/VHH'].copy()
print(f"Found {len(vhh_df)} total VHH sequences.")

# Clean up PDB_ID column (sometimes empty values are written as strings like 'N/A' or 'NA')
vhh_df['PDB_ID'] = vhh_df['PDB_ID'].replace({'N/A': np.nan, 'NA': np.nan, 'NaN': np.nan, 'None': np.nan, '': np.nan})

# Separate into two DataFrames based on structure availability
vhh_with_struct = vhh_df[vhh_df['PDB_ID'].notna()]
vhh_no_struct = vhh_df[vhh_df['PDB_ID'].isna()]

print(f" -> {len(vhh_with_struct)} sequences HAVE a known structure.")
print(f" -> {len(vhh_no_struct)} sequences DO NOT have a structure (will need ESMfold).")

# Save the separated datasets locally
vhh_with_struct.to_csv(vhh_struct_metadata, index=False)
vhh_no_struct.to_csv(vhh_nostruct_metadata, index=False)

print(f"\nSaved structured metadata to: {vhh_struct_metadata.name}")
print(f"Saved sequence-only metadata to: {vhh_nostruct_metadata.name}")

# ==========================================
# 3. Isolate the VHH PDB files (Only for those with structures)
# ==========================================
output_pdb_dir.mkdir(parents=True, exist_ok=True)

pdb_ids = vhh_with_struct['PDB_ID'].astype(str).unique()
print(f"\nIdentifying {len(pdb_ids)} unique VHH PDB structures to copy...")

copied_count = 0
missing_files = []

for pdb_id in pdb_ids:
    possible_names = [
        f"{pdb_id}.pdb", 
        f"{pdb_id.lower()}.pdb", 
        f"{pdb_id.upper()}.pdb",
        f"pdb{pdb_id.lower()}.ent"
    ]
    
    file_found = False
    for name in possible_names:
        src_file = all_structures_dir / name
        if src_file.exists():
            shutil.copy2(src_file, output_pdb_dir / name)
            copied_count += 1
            file_found = True
            break
            
    if not file_found:
        missing_files.append(pdb_id)

print(f"\n✅ Successfully copied {copied_count} PDB files to: {output_pdb_dir}")

if missing_files:
    print(f"⚠️ Could not find PDB files for {len(missing_files)} IDs.")
    print("This is normal, some files might be missing from the bulk download.")

print("\nDone! Your data is now properly separated for both ground-truth scoring and ESMfold prediction.")