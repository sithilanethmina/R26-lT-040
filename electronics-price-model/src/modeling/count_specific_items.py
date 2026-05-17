import pandas as pd
import os

def check_specific_counts():
    path = 'data/processed/laptops_cleaned.csv'
    if not os.path.exists(path):
        print("Error: Processed data not found.")
        return

    df = pd.read_csv(path)

    # The user's specific example
    print("="*60)
    print(" SEARCHING FOR SPECIFIC CONFIGURATION: ")
    print(" HP Victus, Core i5, 12th Gen, 16GB, 512GB SSD ")
    print("="*60)

    # Filtering logic (Case-insensitive where applicable)
    match = df[
        (df['Brand_Cleaned'].str.upper() == 'HP') &
        (df['Model_Cleaned'].str.upper() == 'VICTUS') &
        (df['CPU_Cleaned'].str.upper() == 'I5') &
        (df['Generation_Cleaned'] == 12) &
        (df['RAM_GB'] == 16) &
        (df['Storage_Capacity_GB'] == 512) &
        (df['Storage_Type'].str.upper() == 'SSD')
    ]

    print(f"\nFOUND: {len(match)} exact matches in your training data.")
    
    if len(match) > 0:
        print(f"Average Price for this spec: Rs {match['Price_Cleaned'].mean():,.2f}")
    
    # Broader Search
    print("\n" + "-"*40)
    print(" BROADER CATEGORY COUNTS: ")
    print("-"*40)
    
    victus_total = df[df['Model_Cleaned'].str.upper() == 'VICTUS']
    print(f"- Total HP Victus Laptops:      {len(victus_total)}")
    
    hp_total = df[df['Brand_Cleaned'].str.upper() == 'HP']
    print(f"- Total HP Laptops:             {len(hp_total)}")
    
    i5_12th_total = df[(df['CPU_Cleaned'].str.upper() == 'I5') & (df['Generation_Cleaned'] == 12)]
    print(f"- Total Core i5 12th Gen:       {len(i5_12th_total)}")

    # Most Common Models
    print("\n" + "-"*40)
    print(" TOP 5 MOST COMMON MODELS IN DATASET: ")
    print("-"*40)
    top_models = df['Model_Cleaned'].value_counts().head(5)
    for model, count in top_models.items():
        print(f"- {model:15}: {count} items")

    print("\n" + "="*60)

if __name__ == "__main__":
    check_specific_counts()
