import pandas as pd
import numpy as np
import random
import os

def generate_laptop_data(n=2000):
    brands = {
        'DELL': ['LATITUDE', 'INSPIRON', 'VOSTRO', 'PRECISION', 'XPS', 'ALIENWARE', 'G15'],
        'HP': ['ELITEBOOK', 'PROBOOK', 'PAVILION', 'SPECTRE', 'ENVY', 'OMEN', 'VICTUS'],
        'LENOVO': ['THINKPAD', 'IDEAPAD', 'LEGION', 'YOGA', 'V15'],
        'ASUS': ['VIVOBOOK', 'ZENBOOK', 'ROG', 'TUF'],
        'ACER': ['ASPIRE', 'SWIFT', 'NITRO', 'PREDATOR'],
        'APPLE': ['MACBOOK PRO', 'MACBOOK AIR'],
        'MSI': ['MODERN', 'STEALTH', 'KATANA', 'GAMING']
    }
    
    cpus = {
        'I3': 15000, 'I5': 35000, 'I7': 65000, 'I9': 120000,
        'RYZEN 3': 12000, 'RYZEN 5': 30000, 'RYZEN 7': 55000, 'RYZEN 9': 100000,
        'M1': 90000, 'M2': 130000, 'M3': 180000,
        'CELERON': 5000, 'PENTIUM': 8000, 'Other': 10000
    }
    
    data = []
    for _ in range(n):
        brand = random.choice(list(brands.keys()))
        model = random.choice(brands[brand] + ['Other'])
        
        if brand == 'APPLE':
            cpu = random.choice(['M1', 'M2', 'M3', 'I5', 'I7'])
        else:
            cpu = random.choice(['I3', 'I5', 'I7', 'RYZEN 3', 'RYZEN 5', 'RYZEN 7', 'Other'])
            
        gen = random.randint(4, 13)
        ram = random.choice([4, 8, 16, 32])
        storage = random.choice([128, 256, 512, 1024])
        stype = random.choice(['SSD', 'HDD'])
        
        # Base price
        price = 30000
        
        # Add CPU value
        price += cpus.get(cpu, 10000)
        
        # Add Gen value
        price += (gen - 4) * 8000
        
        # Add RAM value
        price += (ram / 4) * 6000
        
        # Add Storage value
        price += (storage / 128) * 4000
        if stype == 'SSD': price += 10000
        
        # Brand/Model multipliers
        if brand == 'APPLE': price *= 1.4
        if model in ['XPS', 'ALIENWARE', 'ROG', 'PREDATOR', 'SPECTRE', 'LEGION']: price *= 1.3
        
        # Random noise
        price *= random.uniform(0.9, 1.1)
        
        data.append({
            'Title': f"{brand} {model} {cpu} {gen}th Gen {ram}GB {storage}GB {stype}",
            'Brand_Cleaned': brand,
            'Model_Cleaned': model,
            'CPU_Cleaned': cpu,
            'Generation_Cleaned': gen,
            'Condition_Cleaned': 'Used',
            'RAM_GB': ram,
            'Storage_Capacity_GB': storage,
            'Storage_Type': stype,
            'Price_Cleaned': round(price, -2)
        })
        
    return pd.DataFrame(data)

def generate_tablet_data(n=1500):
    brands = {
        'APPLE': ['IPAD PRO', 'IPAD AIR', 'IPAD MINI', 'IPAD'],
        'SAMSUNG': ['GALAXY TAB S', 'GALAXY TAB A', 'GALAXY TAB E'],
        'HUAWEI': ['MATEPAD', 'MEDIAPAD'],
        'LENOVO': ['TAB P11', 'TAB M10', 'YOGA TAB'],
        'XIAOMI': ['MI PAD', 'PAD 5', 'PAD 6']
    }
    
    data = []
    for _ in range(n):
        brand = random.choice(list(brands.keys()))
        model = random.choice(brands[brand] + ['Other'])
        ram = random.choice([2, 4, 8, 16])
        storage = random.choice([32, 64, 128, 256, 512])
        
        price = 20000
        if brand == 'APPLE': price += 40000
        if brand == 'SAMSUNG': price += 20000
        
        price += (ram / 2) * 5000
        price += (storage / 32) * 4000
        
        if 'PRO' in model or 'S' in model or '6' in model: price *= 1.5
        
        price *= random.uniform(0.9, 1.1)
        
        data.append({
            'Title': f"{brand} {model} {ram}GB {storage}GB",
            'Brand_Cleaned': brand,
            'Model_Cleaned': model,
            'Condition_Cleaned': 'Used',
            'RAM_GB': ram,
            'Storage_GB': storage,
            'Price_Cleaned': round(price, -2)
        })
    return pd.DataFrame(data)

# Generate and Save
laptop_syn = generate_laptop_data(3000)
tablet_syn = generate_tablet_data(2000)

# Merge with existing processed data if exists
if os.path.exists('data/processed/laptops_cleaned.csv'):
    orig = pd.read_csv('data/processed/laptops_cleaned.csv')
    laptop_final = pd.concat([orig, laptop_syn], ignore_index=True)
    laptop_final.to_csv('data/processed/laptops_cleaned.csv', index=False)
    print(f"Laptops: Merged {len(orig)} original with {len(laptop_syn)} synthetic. Total: {len(laptop_final)}")
else:
    laptop_syn.to_csv('data/processed/laptops_cleaned.csv', index=False)

if os.path.exists('data/processed/tablets_cleaned.csv'):
    orig = pd.read_csv('data/processed/tablets_cleaned.csv')
    tablet_final = pd.concat([orig, tablet_syn], ignore_index=True)
    tablet_final.to_csv('data/processed/tablets_cleaned.csv', index=False)
    print(f"Tablets: Merged {len(orig)} original with {len(tablet_syn)} synthetic. Total: {len(tablet_final)}")
else:
    tablet_syn.to_csv('data/processed/tablets_cleaned.csv', index=False)
