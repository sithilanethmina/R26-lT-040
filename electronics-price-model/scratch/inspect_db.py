import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import re

# Load cleaned DB helpers
from app import clean_brand, clean_model, clean_cpu, clean_generation, clean_price, clean_ram, clean_storage

df = pd.read_csv('data/raw/laptops_large_dataset.csv')
df['Price_Cleaned'] = df['Price'].apply(clean_price)
df = df.dropna(subset=['Price_Cleaned'])

df['Brand_Cleaned'] = df.apply(lambda row: clean_brand(row['Brand'], row['Title']), axis=1)
df['Model_Cleaned'] = df.apply(lambda row: clean_model(row['Brand_Cleaned'], row['Title']), axis=1)
df['CPU_Cleaned'] = df['Title'].apply(clean_cpu)
df['Generation_Cleaned'] = df['Title'].apply(clean_generation)

df['RAM_GB'] = df['RAM'].apply(clean_ram).fillna(8.0)
storage_data = df['Storage'].apply(clean_storage)
df['Storage_Capacity_GB'] = storage_data.apply(lambda x: x[0]).fillna(256.0)
df['Storage_Type'] = storage_data.apply(lambda x: x[1])

# Filter for Acer, I3, Gen 10
matched = df[
    (df['Brand_Cleaned'] == 'ACER') & 
    (df['CPU_Cleaned'] == 'I3') & 
    (df['Generation_Cleaned'] == 10)
]

print(f"Total matched Acer I3 10th Gen laptops: {len(matched)}")
print(matched[['Title', 'Price_Cleaned', 'RAM_GB', 'Storage_Capacity_GB', 'Storage_Type']].head(20).to_string())
