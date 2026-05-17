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

def clean_size(size_str):
    if pd.isna(size_str) or size_str == 'Unknown':
        return 24.0 # Default to 24 inch
    try:
        match = re.search(r'\d+', str(size_str))
        return float(match.group()) if match else 24.0
    except:
        return 24.0

def clean_hz(hz_str):
    if pd.isna(hz_str) or hz_str == 'Unknown':
        return 60.0
    try:
        match = re.search(r'\d+', str(hz_str))
        return float(match.group()) if match else 60.0
    except:
        return 60.0

def main():
    input_file = 'data/raw/monitors_large_dataset.csv'
    output_file = 'data/processed/monitors_cleaned.csv'
    
    print(f"Loading monitor data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        return
        
    print(f"Raw data shape: {df.shape}")
    
    # 1. Clean Price
    df['Price_Cleaned'] = df['Price'].apply(clean_price)
    df = df.dropna(subset=['Price_Cleaned'])
    
    # Filter out extreme outliers (e.g., Mitsubishi Motor Graders misclassified as monitors)
    df = df[(df['Price_Cleaned'] >= 5000) & (df['Price_Cleaned'] <= 400000)]
    print(f"Data shape after price filtering: {df.shape}")
    
    # 2. Clean Size & Hz
    df['Size_Inch'] = df['Size'].apply(clean_size)
    df['Refresh_Rate_Hz'] = df['Refresh_Rate'].apply(clean_hz)
    
    # 3. Clean Brand
    df['Brand_Cleaned'] = df['Brand'].astype(str).str.upper()
    valid_brands = ['DELL', 'SAMSUNG', 'ASUS', 'ACER', 'LG', 'HP', 'BENQ']
    df.loc[~df['Brand_Cleaned'].isin(valid_brands), 'Brand_Cleaned'] = 'OTHER'
    
    # 4. Clean Condition
    df['Condition_Cleaned'] = df['Condition'].apply(lambda x: 'New' if 'New' in str(x) else 'Used')
    
    # 5. Clean Resolution
    df['Resolution_Cleaned'] = df['Resolution'].astype(str).apply(lambda x: x if x in ['FHD', '2K', '4K'] else 'FHD')
    
    features_df = df[['Title', 'Brand_Cleaned', 'Condition_Cleaned', 'Size_Inch', 'Refresh_Rate_Hz', 'Resolution_Cleaned', 'Price_Cleaned']]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    features_df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}. Shape: {features_df.shape}")
    print("\nSample of cleaned data:")
    print(features_df.head())

if __name__ == "__main__":
    main()
