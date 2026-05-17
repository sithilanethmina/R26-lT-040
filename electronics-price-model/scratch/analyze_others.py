import pandas as pd
import os

def analyze(path, name):
    if not os.path.exists(path):
        print(f"{name} dataset not found at {path}")
        return
    df = pd.read_csv(path)
    print(f"\n--- {name} Dataset ---")
    print(f"Total rows: {len(df)}")
    if 'Brand' in df.columns:
        print("Unique Brands:")
        print(df['Brand'].unique())
    if 'Model' in df.columns:
        print("Model value counts:")
        print(df['Model'].value_counts().head(10))

analyze('data/raw/monitors_large_dataset.csv', 'Monitors')
analyze('data/raw/tablets_large_dataset.csv', 'Tablets')
