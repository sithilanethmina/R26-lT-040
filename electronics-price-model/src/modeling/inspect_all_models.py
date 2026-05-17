import joblib
import pandas as pd
import os
import glob

def inspect_project():
    print("="*60)
    print(" PROJECT MODEL & DATA INSPECTION SUMMARY ")
    print("="*60)

    # 1. Inspect Datasets
    print("\n[ DATASET STATISTICS ]")
    datasets = {
        'Laptops': 'data/processed/laptops_cleaned.csv',
        'Monitors': 'data/processed/monitors_cleaned.csv',
        'Tablets': 'data/processed/tablets_cleaned.csv'
    }

    for category, path in datasets.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"- {category:10}: {len(df):>6} rows (items)")
        else:
            print(f"- {category:10}: NOT FOUND (Run preprocessing first)")

    # 2. Inspect Models
    print("\n[ TRAINED MODEL DETAILS ]")
    model_files = glob.glob('models/*.pkl')
    
    if not model_files:
        print("No models found in the models/ directory.")
        return

    # Sort files to group by category
    model_files.sort()

    current_cat = ""
    for file_path in model_files:
        filename = os.path.basename(file_path)
        
        # Categorize by filename prefix
        if 'laptop' in filename: cat = "LAPTOP"
        elif 'monitor' in filename: cat = "MONITOR"
        elif 'tablet' in filename: cat = "TABLET"
        else: cat = "OTHER"

        if cat != current_cat:
            print(f"\n--- {cat} MODELS ---")
            current_cat = cat

        try:
            data = joblib.load(file_path)
            name = data.get('model_name', 'Unknown')
            r2 = data.get('accuracy', 0)
            features = len(data.get('features', []))
            
            print(f"  > {name:20} | R2 Score: {r2:.4f} | Features: {features:3}")
        except Exception as e:
            print(f"  > Error loading {filename}: {e}")

    print("\n" + "="*60)

if __name__ == "__main__":
    inspect_project()
