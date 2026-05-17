import pandas as pd

df = pd.read_csv('data/processed/laptops_cleaned.csv')
print("Missing value counts in processed data:")
print(df.isnull().sum())

print("\nValue counts for CPU:")
print(df['CPU_Cleaned'].value_counts())

print("\nValue counts for Generation:")
print(df['Generation_Cleaned'].value_counts())

print("\nCorrelation between numeric features and Price:")
print(df[['RAM_GB', 'Storage_Capacity_GB', 'Generation_Cleaned', 'Price_Cleaned']].corr()['Price_Cleaned'])
