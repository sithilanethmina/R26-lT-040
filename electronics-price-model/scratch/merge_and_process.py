import pandas as pd
import os

scraped_csv = 'scratch/scraped_samsung_tablets.csv'
large_db_path = 'data/raw/tablets_large_dataset.csv'
small_db_path = 'data/raw/tablets_dataset.csv'

if not os.path.exists(scraped_csv):
    print(f"Error: Scraped CSV not found at {scraped_csv}")
    exit(1)

# Load scraped Samsung tablets
df_scraped = pd.read_csv(scraped_csv)
print(f"Loaded {len(df_scraped)} scraped Samsung tablet records.")

# Update tablets_large_dataset.csv
if os.path.exists(large_db_path):
    df_large = pd.read_csv(large_db_path)
    print(f"Original large dataset size: {len(df_large)}")
    
    # Remove old Samsung tablets
    df_large_no_samsung = df_large[df_large['Brand'].astype(str).str.upper() != 'SAMSUNG']
    print(f"Large dataset size after removing Samsung tablets: {len(df_large_no_samsung)}")
    
    # Combine
    df_large_updated = pd.concat([df_large_no_samsung, df_scraped], ignore_index=True)
    print(f"New large dataset size: {len(df_large_updated)}")
    
    # Save back
    df_large_updated.to_csv(large_db_path, index=False)
    print(f"Saved updated large database to {large_db_path}")

# Update tablets_dataset.csv
if os.path.exists(small_db_path):
    df_small = pd.read_csv(small_db_path)
    print(f"Original small dataset size: {len(df_small)}")
    
    # Remove old Samsung tablets
    df_small_no_samsung = df_small[df_small['Brand'].astype(str).str.upper() != 'SAMSUNG']
    print(f"Small dataset size after removing Samsung tablets: {len(df_small_no_samsung)}")
    
    # Combine
    df_small_updated = pd.concat([df_small_no_samsung, df_scraped], ignore_index=True)
    print(f"New small dataset size: {len(df_small_updated)}")
    
    # Save back
    df_small_updated.to_csv(small_db_path, index=False)
    print(f"Saved updated small database to {small_db_path}")

print("Merge completed successfully!")
