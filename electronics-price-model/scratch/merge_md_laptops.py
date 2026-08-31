import pandas as pd
import os

scraped_md = 'scratch/scraped_md_laptops.csv'
large_db_path = 'data/raw/laptops_large_dataset.csv'
detailed_db_path = 'data/raw/laptops_detailed.csv'

if not os.path.exists(scraped_md):
    print(f"Error: MD Laptops CSV not found at {scraped_md}")
    exit(1)

df_md = pd.read_csv(scraped_md)
print(f"Loaded {len(df_md)} scraped MD Computers laptop records.")

# Load existing database to inspect columns
if os.path.exists(large_db_path):
    df_existing = pd.read_csv(large_db_path)
    print(f"Loaded {len(df_existing)} existing laptop records from {large_db_path}.")
    cols = df_existing.columns.tolist()
    print("Schema columns:", cols)
    
    # Clean MD Laptops to match the schema
    # The schema might contain Title, Price, Location_Category, Link, Brand, Model, Condition, RAM, Storage, Description
    # Let's align fields
    for col in cols:
        if col not in df_md.columns:
            df_md[col] = "Unknown"
            
    df_md_aligned = df_md[cols]
    
    # Filter out existing MD Computers laptops to prevent duplicates
    df_existing = df_existing[df_existing['Location_Category'] != 'MD Computers']
    
    df_merged = pd.concat([df_existing, df_md_aligned], ignore_index=True)
else:
    df_merged = df_md

# Overwrite large reference database
df_merged.to_csv(large_db_path, index=False)
print(f"Saved merged database to {large_db_path}. Total rows: {len(df_merged)}")

# Overwrite detailed reference database if it exists
if os.path.exists(detailed_db_path):
    # Align for detailed db
    df_existing_det = pd.read_csv(detailed_db_path)
    df_existing_det = df_existing_det[df_existing_det['Location_Category'] != 'MD Computers']
    cols_det = df_existing_det.columns.tolist()
    
    for col in cols_det:
        if col not in df_md.columns:
            df_md[col] = "Unknown"
            
    df_md_aligned_det = df_md[cols_det]
    df_merged_det = pd.concat([df_existing_det, df_md_aligned_det], ignore_index=True)
    df_merged_det.to_csv(detailed_db_path, index=False)
    print(f"Saved merged detailed database to {detailed_db_path}. Total rows: {len(df_merged_det)}")

print("Laptops merging complete!")
