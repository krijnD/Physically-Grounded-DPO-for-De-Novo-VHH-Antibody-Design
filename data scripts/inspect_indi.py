import os
import pandas as pd
from pathlib import Path

# Set the path to the directory where gdown saved the files
# (Update this if gdown created a specific subfolder like 'AbGenbank')
dataset_dir = Path("/projects/0/hpmlprjs/interns/krijn/INDI_dataset/")

# Find the first .parquet file in the directory (or subdirectories)
parquet_files = list(dataset_dir.rglob("*.parquet"))

if not parquet_files:
    print("Could not find any .parquet files. Check the directory path!")
else:
    first_file = parquet_files[0]
    print(f"Reading file: {first_file.name}\n")
    
    # Read just the first few rows to save memory
    df = pd.read_parquet(first_file)
    
    print("=== COLUMN NAMES ===")
    print(df.columns.tolist())
    
    print("\n=== FIRST ROW SAMPLE ===")
    # Print the first row, transposing it so it's easy to read in the terminal
    print(df.head(1).T)