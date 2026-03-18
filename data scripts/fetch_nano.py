import pandas as pd
import os
import requests
import time

# 1. Load the renamed SAbDab-nano summary
df = pd.read_csv('sabdab_nano_summary.tsv', sep='\t')

# 2. Convert release date with explicit format to silence the warning
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y', errors='coerce')

# 3. Force the resolution column to be numeric (turns "NA" strings into NaN)
df['resolution'] = pd.to_numeric(df['resolution'], errors='coerce')

# 4. Apply Filters for DPO Ground Truth Winners
# - Date: >= 2023-01-01 (post-IgLM training data)
# - Resolution: <= 2.5 Å (required for stable Rosetta E_Rep evaluation)
filtered_df = df[
    (df['date'] >= '2023-01-01') & 
    (df['resolution'] <= 2.5) 
]

# Get unique PDB IDs to avoid downloading the same file multiple times
target_pdbs = filtered_df['pdb'].dropna().unique()
print(f"Found {len(target_pdbs)} high-resolution, post-IgLM nanobody structures.")

# 5. Download directly from RCSB PDB
target_dir = 'filtered_vhh_pdbs'
os.makedirs(target_dir, exist_ok=True)

success_count = 0
for pdb_id in target_pdbs:
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    file_path = os.path.join(target_dir, f"{pdb_id}.pdb")
    
    # Only download if we don't already have it
    if not os.path.exists(file_path):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                success_count += 1
                print(f"Downloaded {pdb_id} ({success_count}/{len(target_pdbs)})")
            else:
                print(f"Failed to fetch {pdb_id} (Status: {response.status_code})")
        except Exception as e:
            print(f"Error downloading {pdb_id}: {e}")
        
        # Polite delay to avoid hammering the RCSB servers
        time.sleep(0.2)

print(f"\nSuccessfully downloaded {success_count} VHH structures to ./{target_dir}/")
