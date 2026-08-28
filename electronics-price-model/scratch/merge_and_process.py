import pandas as pd
import os
import subprocess

def merge_and_clean_laptops():
    large_path = 'data/raw/laptops_large_dataset.csv'
    new_detailed_path = 'data/raw/laptops_detailed.csv'
    md_path = 'data/raw/md_laptops_data.csv'
    
    if not os.path.exists(large_path):
        print(f"Error: {large_path} not found.")
        return
        
    df_large = pd.read_csv(large_path)
    dfs_to_merge = [df_large]
    
    if os.path.exists(new_detailed_path):
        df_new = pd.read_csv(new_detailed_path)
        dfs_to_merge.append(df_new)
        print(f"New scraped laptop details rows: {len(df_new)}")
        
    if os.path.exists(md_path):
        df_md = pd.read_csv(md_path)
        dfs_to_merge.append(df_md)
        print(f"MD Computers laptop rows: {len(df_md)}")
        
    print(f"Current large laptop dataset rows: {len(df_large)}")
    
    df_combined = pd.concat(dfs_to_merge, ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=['Link'], keep='last')
    df_combined.to_csv(large_path, index=False)
    print(f"Laptops merged successfully. Total rows in large dataset: {len(df_combined)}")
    
    print("Running laptop data cleaning...")
    subprocess.run(['venv\\Scripts\\python.exe', 'src\\preprocessing\\data_cleaning.py'], check=True)
    
    print("Retraining laptop models...")
    subprocess.run(['venv\\Scripts\\python.exe', 'src\\modeling\\train_model.py'], check=True)

def merge_and_clean_tablets():
    large_path = 'data/raw/tablets_large_dataset.csv'
    new_scraped_path = 'data/raw/tablets_dataset.csv'
    
    if not os.path.exists(large_path):
        print(f"Error: {large_path} not found.")
        return
        
    df_large = pd.read_csv(large_path)
    dfs_to_merge = [df_large]
    
    if os.path.exists(new_scraped_path):
        df_new = pd.read_csv(new_scraped_path)
        dfs_to_merge.append(df_new)
        print(f"New scraped tablet rows: {len(df_new)}")
        
    print(f"Current large tablet dataset rows: {len(df_large)}")
    
    df_combined = pd.concat(dfs_to_merge, ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=['Link'], keep='last')
    df_combined.to_csv(large_path, index=False)
    print(f"Tablets merged successfully. Total rows in large dataset: {len(df_combined)}")
    
    print("Running tablet data cleaning...")
    subprocess.run(['venv\\Scripts\\python.exe', 'src\\preprocessing\\tablets_data_cleaning.py'], check=True)
    
    print("Retraining tablet models...")
    subprocess.run(['venv\\Scripts\\python.exe', 'src\\modeling\\train_tablets_model.py'], check=True)

if __name__ == '__main__':
    print("--- PROCESS TABLETS ---")
    merge_and_clean_tablets()
    print("\n--- PROCESS LAPTOPS ---")
    merge_and_clean_laptops()
    print("\nDone!")
