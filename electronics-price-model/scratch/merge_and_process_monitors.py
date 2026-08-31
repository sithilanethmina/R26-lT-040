import pandas as pd
import re
import os

scraped_csv = 'scratch/scraped_monitors.csv'
large_db_path = 'data/raw/monitors_large_dataset.csv'
small_db_path = 'data/raw/monitors_dataset.csv'

if not os.path.exists(scraped_csv):
    print(f"Error: Scraped CSV not found at {scraped_csv}")
    exit(1)

df_scraped = pd.read_csv(scraped_csv)
print(f"Loaded {len(df_scraped)} scraped monitor records.")

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
df_scraped['Brand'] = df_scraped['Title'].apply(clean_brand)
df_scraped['Size'] = df_scraped.apply(lambda r: clean_size(r['Title'], r['Description']), axis=1)
df_scraped['Refresh_Rate'] = df_scraped.apply(lambda r: clean_hz(r['Title'], r['Description']), axis=1)
df_scraped['Resolution'] = df_scraped.apply(lambda r: clean_resolution(r['Title'], r['Description']), axis=1)

# Format to exact target columns schema
cols = ['Title', 'Price', 'Location_Category', 'Link', 'Brand', 'Condition', 'Size', 'Refresh_Rate', 'Resolution', 'Description']
df_scraped_formatted = df_scraped[cols]

# Overwrite large reference database
df_scraped_formatted.to_csv(large_db_path, index=False)
print(f"Overwrote large monitor database at {large_db_path} with {len(df_scraped_formatted)} cleaned real records.")

# Overwrite small reference database
df_scraped_formatted.to_csv(small_db_path, index=False)
print(f"Overwrote small monitor database at {small_db_path} with {len(df_scraped_formatted)} cleaned real records.")

print("Monitors database overhaul complete!")
