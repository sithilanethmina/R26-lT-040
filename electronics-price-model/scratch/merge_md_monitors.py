import pandas as pd
import re
import os

scraped_md = 'scratch/scraped_md_monitors.csv'
large_db_path = 'data/raw/monitors_large_dataset.csv'
small_db_path = 'data/raw/monitors_dataset.csv'

if not os.path.exists(scraped_md):
    print(f"Error: MD Scraped CSV not found at {scraped_md}")
    exit(1)

df_md = pd.read_csv(scraped_md)
print(f"Loaded {len(df_md)} scraped MD Computers monitor records.")

# Clean functions
def clean_brand(title):
    title_upper = str(title).upper()
    brands = ["DELL", "HP", "SAMSUNG", "LG", "ASUS", "ACER", "BENQ", "VIEWSONIC", "PHILIPS", "MSI", "AOC", "LENOVO", "EIZO", "HKC"]
    for b in brands:
        if b in title_upper:
            return b
    return "Other"

def clean_size(title, desc):
    text = (str(title) + " " + str(desc)).upper()
    match = re.search(r'(\d{2})\s*(?:INCH|")', text)
    if match:
        return float(match.group(1))
    return 24.0

def clean_hz(title, desc):
    text = (str(title) + " " + str(desc)).upper()
    match = re.search(r'(\d{2,3})\s*HZ', text)
    if match:
        return float(match.group(1))
    return 60.0

def clean_resolution(title, desc):
    text = (str(title) + " " + str(desc)).upper()
    if "4K" in text or "UHD" in text or "2160P" in text or "3840" in text:
        return "4K"
    if "2K" in text or "QHD" in text or "1440P" in text or "2560" in text:
        return "2K"
    return "FHD"

# Populate columns
df_md['Brand'] = df_md['Title'].apply(clean_brand)
df_md['Size'] = df_md.apply(lambda r: clean_size(r['Title'], r['Description']), axis=1)
df_md['Refresh_Rate'] = df_md.apply(lambda r: clean_hz(r['Title'], r['Description']), axis=1)
df_md['Resolution'] = df_md.apply(lambda r: clean_resolution(r['Title'], r['Description']), axis=1)

# Format target columns schema
cols = ['Title', 'Price', 'Location_Category', 'Link', 'Brand', 'Condition', 'Size', 'Refresh_Rate', 'Resolution', 'Description']
df_md_formatted = df_md[cols]

# Load existing database
if os.path.exists(large_db_path):
    df_existing = pd.read_csv(large_db_path)
    print(f"Loaded {len(df_existing)} existing monitor records.")
    
    # Filter out any MD Computers records that might already be in there (to avoid duplicates)
    df_existing = df_existing[df_existing['Location_Category'] != 'MD Computers']
    
    df_merged = pd.concat([df_existing, df_md_formatted], ignore_index=True)
else:
    df_merged = df_md_formatted

# Overwrite large reference database
df_merged.to_csv(large_db_path, index=False)
print(f"Saved merged database to {large_db_path}. Total rows: {len(df_merged)}")

# Overwrite small reference database
df_merged.to_csv(small_db_path, index=False)
print(f"Saved merged database to {small_db_path}. Total rows: {len(df_merged)}")

print("Monitors merging complete!")
