import pandas as pd
import numpy as np
import os
import re

def clean_price(price_str):
    if pd.isna(price_str) or price_str == 'Unknown' or 'Negotiable' in str(price_str):
        return np.nan
    try:
        clean_str = str(price_str).replace('Rs', '').replace(',', '').strip()
        return float(clean_str)
    except ValueError:
        return np.nan

def extract_ram(row):
    text = f"{str(row.get('Model', ''))} {str(row.get('Title', ''))} {str(row.get('Description', ''))}".upper()
    # Looking for patterns like 4GB, 8GB RAM, etc.
    match = re.search(r'(\d+)\s*GB\s*(?:RAM|MEMORY)?', text)
    if match:
        val = int(match.group(1))
        if val in [2, 4, 6, 8, 12, 16, 32]:
            return float(val)
    # Default fallback
    return 4.0

def extract_storage(row):
    text = f"{str(row.get('Model', ''))} {str(row.get('Title', ''))} {str(row.get('Description', ''))}".upper()
    # Look for TB first
    tb_match = re.search(r'(\d+)\s*TB', text)
    if tb_match:
        val = int(tb_match.group(1))
        if val in [1, 2]:
            return float(val * 1024)
            
    # Then GB
    gb_match = re.findall(r'(\d+)\s*GB', text)
    # Get the max GB value that is typical for storage
    valid_storages = [32, 64, 128, 256, 512, 1024]
    found_storages = []
    for m in gb_match:
        val = int(m)
        if val in valid_storages:
            found_storages.append(val)
    if found_storages:
        return float(max(found_storages))
    
    # Default fallback
    return 64.0

def clean_model(brand, title):
    title = str(title).upper()
    brand = str(brand).upper()
    
    models = {
        'APPLE': ['IPAD PRO', 'IPAD AIR', 'IPAD MINI', 'IPAD'],
        'SAMSUNG': ['GALAXY TAB S', 'GALAXY TAB A', 'GALAXY TAB E', 'TAB S', 'TAB A'],
        'HUAWEI': ['MATEPAD', 'MEDIAPAD'],
        'LENOVO': ['TAB P11', 'TAB M10', 'YOGA TAB', 'TAB M8'],
        'XIAOMI': ['MI PAD', 'PAD 5', 'PAD 6']
    }
    
    if brand in models:
        for m in models[brand]:
            if m in title:
                return m
                
    return "Other"

def main():
    input_file = 'data/raw/tablets_large_dataset.csv'
    output_file = 'data/processed/tablets_cleaned.csv'
    
    print(f"Loading tablet data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        return
        
    print(f"Raw data shape: {df.shape}")
    
    # 1. Clean Price
    df['Price_Cleaned'] = df['Price'].apply(clean_price)
    df = df.dropna(subset=['Price_Cleaned'])
    
    # Filter out outliers (misclassified items)
    df = df[(df['Price_Cleaned'] >= 5000) & (df['Price_Cleaned'] <= 500000)]
    print(f"Data shape after price filtering: {df.shape}")
    
    # 2. Extract RAM & Storage
    df['RAM_GB'] = df.apply(extract_ram, axis=1)
    df['Storage_GB'] = df.apply(extract_storage, axis=1)
    
    # 3. Clean Brand
    df['Brand_Cleaned'] = df['Brand'].astype(str).str.upper()
    valid_brands = ['APPLE', 'SAMSUNG', 'HUAWEI', 'XIAOMI', 'ASUS', 'AMAZON']
    df.loc[~df['Brand_Cleaned'].isin(valid_brands), 'Brand_Cleaned'] = 'OTHER'
    
    # 4. Clean Condition
    df['Condition_Cleaned'] = df['Condition'].apply(lambda x: 'New' if 'New' in str(x) else 'Used')
    
    # 5. Clean Model
    df['Model_Cleaned'] = df.apply(lambda row: clean_model(row['Brand_Cleaned'], row['Title']), axis=1)
    
    features_df = df[['Title', 'Brand_Cleaned', 'Model_Cleaned', 'Condition_Cleaned', 'RAM_GB', 'Storage_GB', 'Price_Cleaned']]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    features_df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}. Shape: {features_df.shape}")
    print("\nSample of cleaned data:")
    print(features_df.head())

if __name__ == "__main__":
    main()
