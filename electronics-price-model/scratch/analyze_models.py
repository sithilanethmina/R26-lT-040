import pandas as pd

df = pd.read_csv('data/raw/laptops_large_dataset.csv')
print("Unique Brands in raw data:")
print(df['Brand'].unique())

print("\nUnique Models in raw data (first 50):")
print(df['Model'].unique()[:50])

print("\nModel counts (top 20):")
print(df['Model'].value_counts().head(20))

# Try to extract model from title if Model is Unknown
def guess_model(row):
    title = str(row['Title']).upper()
    brand = str(row['Brand']).upper()
    
    models = {
        'DELL': ['LATITUDE', 'INSPIRON', 'VOSTRO', 'PRECISION', 'XPS', 'ALIENWARE', 'G15', 'G3', 'G5', 'G7'],
        'HP': ['ELITEBOOK', 'PROBOOK', 'PAVILION', 'SPECTRE', 'ENVY', 'OMEN', 'VICTUS', 'ZBOOK', 'NOTEBOOK', 'ELITE'],
        'LENOVO': ['THINKPAD', 'IDEAPAD', 'LEGION', 'YOGA', 'V15', 'THINKBOOK', 'T470', 'T480', 'T490', 'X1'],
        'ASUS': ['VIVOBOOK', 'ZENBOOK', 'ROG', 'TUF', 'EXPERTBOOK'],
        'ACER': ['ASPIRE', 'SWIFT', 'NITRO', 'PREDATOR', 'TRAVELMATE', 'SPIN'],
        'APPLE': ['MACBOOK PRO', 'MACBOOK AIR', 'MACBOOK'],
        'MSI': ['MODERN', 'STEALTH', 'KATANA', 'SWORD', 'CYBORG', 'GAMING']
    }
    
    if brand in models:
        for m in models[brand]:
            if m in title:
                return m
                
    return "Other"

df['Guessed_Model'] = df.apply(guess_model, axis=1)
print("\nGuessed Model counts (top 20):")
print(df['Guessed_Model'].value_counts().head(20))
