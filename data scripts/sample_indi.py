import os
import pandas as pd
from pathlib import Path

# ==========================================
# 1. Define Paths
# ==========================================
dataset_dir = Path("/projects/0/hpmlprjs/interns/krijn/INDI_dataset/")
output_file = dataset_dir / "INDI_VHH_sampled_50k.csv"

# Find all parquet files (ignoring the _delta_log folder)
parquet_files = list(dataset_dir.rglob("*.parquet"))
print(f"Found {len(parquet_files)} parquet files. Processing...")

# ==========================================
# 2. Extract and Filter Data Chunk by Chunk
# ==========================================
all_vhh_data = []
total_rows_scanned = 0

for i, file_path in enumerate(parquet_files):
    # Read the chunk
    df = pd.read_parquet(file_path)
    total_rows_scanned += len(df)
    
    # Filter 1: Must be a nanobody
    vhh_df = df[df['is_nanobody'] == True].copy()
    
    if not vhh_df.empty:
        # Extract the Amino Acid sequence from the AIRR dictionary
        # AIRR format stores the protein sequence under 'sequence_aa'
        vhh_df['sequence_aa'] = vhh_df['variable_domain_airr'].apply(
            lambda x: x.get('sequence_aa') if isinstance(x, dict) else None
        )
        
        # Extract the organism (optional, but good for metadata)
        vhh_df['organism'] = vhh_df['genbank_metadata_organism'].apply(
            lambda x: x.get('organism') if isinstance(x, dict) else 'Unknown'
        )
        
        # Filter 2: Drop any rows that failed to extract a sequence
        vhh_df = vhh_df.dropna(subset=['sequence_aa'])
        
        # Keep only the columns we actually care about
        clean_df = vhh_df[['protein_id', 'organism', 'sequence_aa']]
        all_vhh_data.append(clean_df)
    
    # Print progress every 10 files
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(parquet_files)} files...")

# ==========================================
# 3. Combine and Sample
# ==========================================
print("\nCombining all extracted VHHs...")
if len(all_vhh_data) == 0:
    print("Error: No valid VHH sequences found!")
    exit()

# Concatenate all the filtered chunks into one massive DataFrame
master_vhh_df = pd.concat(all_vhh_data, ignore_index=True)
print(f"Total rows scanned: {total_rows_scanned:,}")
print(f"Total valid VHHs found: {len(master_vhh_df):,}")

# Sample 50,000 sequences (or all of them if there are fewer than 50k)
sample_size = min(50000, len(master_vhh_df))
sampled_df = master_vhh_df.sample(n=sample_size, random_state=42) # random_state ensures reproducibility

# Save to CSV for ESMfold
sampled_df.to_csv(output_file, index=False)
print(f"\n✅ Successfully saved {sample_size:,} real VHH sequences to:")
print(f"   {output_file}")