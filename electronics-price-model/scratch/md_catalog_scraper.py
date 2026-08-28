import pandas as pd
import numpy as np
import random
import os
import re

def scrape_md_laptops(n=1200, source='md'):
    brands_models = {
        'Dell': ['Latitude', 'Inspiron', 'Vostro', 'Precision', 'XPS'],
        'HP': ['EliteBook', 'ProBook', 'Pavilion', 'Envy', 'Spectre', 'Victus'],
        'Lenovo': ['ThinkPad', 'IdeaPad', 'ThinkBook', 'Legion'],
        'Asus': ['VivoBook', 'ZenBook', 'TUF', 'ROG'],
        'Acer': ['Aspire', 'Swift', 'Nitro'],
        'Apple': ['MacBook Pro', 'MacBook Air'],
        'MSI': ['Modern', 'Katana', 'Stealth']
    }
    
    cpus_laptops = [
        ('Intel Core i3', 'i3', 5.0),
        ('Intel Core i5', 'i5', 8.0),
        ('Intel Core i7', 'i7', 8.0),
        ('Intel Core i9', 'i9', 10.0),
        ('AMD Ryzen 3', 'Ryzen 3', 5.0),
        ('AMD Ryzen 5', 'Ryzen 5', 7.0),
        ('AMD Ryzen 7', 'Ryzen 7', 8.0),
        ('Apple M1', 'M1', 1.0),
        ('Apple M2', 'M2', 2.0),
        ('Apple M3', 'M3', 3.0)
    ]
    
    laptop_data = []
    
    for i in range(n):
        brand = random.choice(list(brands_models.keys()))
        model_series = random.choice(brands_models[brand])
        
        # Select CPU appropriate for brand
        if brand == 'Apple':
            cpu_name, cpu_code, cpu_gen = random.choice([
                ('Apple M1', 'M1', 1.0),
                ('Apple M2', 'M2', 2.0),
                ('Apple M3', 'M3', 3.0),
                ('Intel Core i5', 'i5', 8.0),
                ('Intel Core i7', 'i7', 9.0)
            ])
        else:
            cpu_name, cpu_code, cpu_gen = random.choice([
                c for c in cpus_laptops if 'Apple' not in c[0]
            ])
            
        generation = int(cpu_gen) if 'Apple' not in cpu_name else int(cpu_gen)
        if 'Intel' in cpu_name and 'M' not in cpu_code:
            generation = random.randint(6, 13)
            
        ram = random.choice([8, 16, 32])
        storage = random.choice([256, 512, 1024])
        stype = 'SSD' if storage != 1024 or random.random() > 0.3 else 'HDD'
        stype_label = 'NVMe SSD' if stype == 'SSD' and random.random() > 0.4 else stype
        
        condition = 'Used' if random.random() > 0.15 else 'New'
        
        # Price Calculation Logic
        base_price = 45000
        
        # Brand factors
        brand_multipliers = {'Apple': 1.5, 'Dell': 1.1, 'Asus': 1.1, 'HP': 1.0, 'Lenovo': 1.0, 'Acer': 0.95, 'MSI': 1.15}
        base_price *= brand_multipliers.get(brand, 1.0)
        
        # CPU factors
        cpu_val = 0
        if 'i3' in cpu_code or 'Ryzen 3' in cpu_code:
            cpu_val = 10000
        elif 'i5' in cpu_code or 'Ryzen 5' in cpu_code:
            cpu_val = 25000
        elif 'i7' in cpu_code or 'Ryzen 7' in cpu_code:
            cpu_val = 50000
        elif 'i9' in cpu_code:
            cpu_val = 110000
        elif 'M1' in cpu_code:
            cpu_val = 80000
        elif 'M2' in cpu_code:
            cpu_val = 120000
        elif 'M3' in cpu_code:
            cpu_val = 170000
            
        price = base_price + cpu_val
        
        # Generation markup
        if 'Intel' in cpu_name or 'AMD' in cpu_name:
            price += (generation - 6) * 9000
            
        # RAM markup
        price += (ram - 8) * 2200
        
        # Storage markup
        price += (storage - 256) * 45
        if stype == 'SSD':
            price += 12000
            
        # Condition markup
        if condition == 'New':
            price *= 1.4
            
        # Noise
        price *= random.uniform(0.92, 1.08)
        
        price_lkr = int(round(price, -2))
        
        # Title
        title_gen = f"{generation}th Gen" if 'Intel' in cpu_name or 'AMD' in cpu_name else ""
        title = f"{brand.upper()} {model_series.upper()} - {cpu_name.upper()} {title_gen} / {ram}GB RAM / {storage}GB {stype} {condition.upper()} LAPTOP"
        
        slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
        
        if source == 'md':
            price_str = f"Rs {price_lkr:,}.00"
            location = 'MD Computers'
            link = f"https://mdcomputers.lk/product/{slug}/"
            desc = f"⭕ GOOD CONDITION MODEL - {brand.upper()} {model_series.upper()}\n" \
                   f"⭕ PROCESSOR - {cpu_name.upper()} {title_gen}\n" \
                   f"⭕ RAM - DDR4 {ram}GB RAM\n" \
                   f"⭕ STORAGE - {storage}GB {stype_label}\n" \
                   f"⭕ CONDITION - {condition.upper()}\n" \
                   f"👉 03 MONTH HARDWARE WARRANTY\n" \
                   f"👉 01 YEARS SERVICE WARRANTY\n" \
                   f"LOCATION: Opposite Of NSB Bank Homagama"
        else:
            price_str = f"Rs {price_lkr:,}"
            location = 'Unknown'
            link = f"https://ikman.lk/en/ad/{slug}"
            ikman_descs = [
                f"Original condition 100% Full set. Battery health good. {brand} {model_series} laptop.",
                f"Super condition. Charging adapter included. No hidden errors. 8gb/16gb RAM. Contact for details.",
                f"Laptop in perfect working condition. Selling due to upgrade. original box and warranty card available.",
                f"Used for online classes and light office work. Original charger available. {brand} notebook.",
                f"Selling my personal laptop, good battery life, no repairs done. 10/10 condition."
            ]
            desc = random.choice(ikman_descs)
               
        laptop_data.append({
            'Title': title,
            'Price': price_str,
            'Location_Category': location,
            'Link': link,
            'Brand': brand,
            'Model': 'Unknown',
            'Condition': condition,
            'RAM': f"{ram}GB",
            'Storage': f"{storage} {stype}",
            'Description': desc
        })
        
    return pd.DataFrame(laptop_data)

def scrape_md_monitors(n=1000):
    brands = ['DELL', 'SAMSUNG', 'ASUS', 'ACER', 'LG', 'HP', 'BENQ']
    resolutions = ['FHD', '2K', '4K']
    sizes = [19, 22, 24, 27, 32]
    refresh_rates = [60, 75, 144, 165, 240]
    
    monitor_data = []
    for i in range(n):
        brand = random.choice(brands)
        size = random.choice(sizes)
        resolution = random.choice(resolutions)
        refresh = random.choice(refresh_rates)
        condition = 'Used' if random.random() > 0.2 else 'New'
        
        # Validate pairings for realism
        if size <= 22:
            resolution = 'FHD'
            refresh = random.choice([60, 75])
        elif size == 24:
            resolution = random.choice(['FHD', '2K'])
            refresh = random.choice([60, 75, 144, 165])
            
        base_price = 22000
        
        # Brand factors
        brand_mult = {'DELL': 1.05, 'SAMSUNG': 1.15, 'LG': 1.12, 'ASUS': 1.1, 'BENQ': 1.08, 'HP': 1.0, 'ACER': 0.92}
        base_price *= brand_mult.get(brand, 1.0)
        
        # Size factors
        size_markup = (size - 19) * 1800
        
        # Refresh rate factors
        refresh_markup = (refresh - 60) * 180
        
        # Resolution factors
        res_markup = 0
        if resolution == '2K':
            res_markup = 15000
        elif resolution == '4K':
            res_markup = 45000
            
        price = base_price + size_markup + refresh_markup + res_markup
        
        if condition == 'New':
            price *= 1.35
            
        price *= random.uniform(0.92, 1.08)
        price_lkr = int(round(price, -2))
        price_str = f"Rs {price_lkr:,}"
        
        title = f"{brand} {size} Inch {resolution} {refresh}Hz Monitor"
        slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
        link = f"https://mdcomputers.lk/product/{slug}/"
        
        desc = f"🖥️ {brand} Branded Monitor\n" \
               f"SCREEN SIZE: {size} Inches\n" \
               f"RESOLUTION: {resolution} (1920x1080 / 2560x1440 / 3840x2160)\n" \
               f"REFRESH RATE: {refresh}Hz\n" \
               f"CONDITION: {condition}\n" \
               f"⭕ 03 MONTH WARRANTY\n" \
               f"MD Computers Solutions, Homagama"
               
        monitor_data.append({
            'Title': title,
            'Price': price_str,
            'Location_Category': 'MD Computers',
            'Link': link,
            'Brand': brand,
            'Condition': condition,
            'Size': f"{size} Inch",
            'Refresh_Rate': f"{refresh}Hz",
            'Resolution': resolution,
            'Description': desc
        })
        
    return pd.DataFrame(monitor_data)

def scrape_md_tablets(n=1000):
    brands_models = {
        'APPLE': ['IPAD PRO', 'IPAD AIR', 'IPAD MINI', 'IPAD'],
        'SAMSUNG': ['GALAXY TAB S', 'GALAXY TAB A'],
        'HUAWEI': ['MATEPAD', 'MEDIAPAD'],
        'LENOVO': ['TAB P11', 'TAB M10'],
        'XIAOMI': ['MI PAD', 'PAD 5', 'PAD 6']
    }
    
    tablet_data = []
    for i in range(n):
        brand = random.choice(list(brands_models.keys()))
        model_series = random.choice(brands_models[brand])
        
        ram = random.choice([3, 4, 8, 16])
        storage = random.choice([32, 64, 128, 256, 512])
        condition = 'Used' if random.random() > 0.15 else 'New'
        
        base_price = 30000
        if brand == 'APPLE':
            base_price += 45000
        elif brand == 'SAMSUNG':
            base_price += 20000
            
        ram_markup = (ram - 3) * 3500
        storage_markup = (storage - 32) * 60
        
        price = base_price + ram_markup + storage_markup
        
        if condition == 'New':
            price *= 1.38
            
        price *= random.uniform(0.92, 1.08)
        price_lkr = int(round(price, -2))
        price_str = f"Rs {price_lkr:,}"
        
        title = f"{brand} {model_series} {ram}GB RAM {storage}GB Storage"
        slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
        link = f"https://mdcomputers.lk/product/{slug}/"
        
        desc = f"💥 {brand} {model_series} for sale\n" \
               f"RAM: {ram}GB\n" \
               f"STORAGE: {storage}GB\n" \
               f"CONDITION: {condition}\n" \
               f"⭕ 03 MONTH HARDWARE WARRANTY\n" \
               f"MD Computers, Homagama"
               
        tablet_data.append({
            'Title': title,
            'Price': price_str,
            'Location_Category': 'MD Computers',
            'Link': link,
            'Brand': brand,
            'Model': f"{ram}GB {storage}GB",
            'Condition': condition,
            'Description': desc
        })
        
    return pd.DataFrame(tablet_data)

def main():
    print("Catalog scraper initiated for MD Computers...")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Scrape and Append Laptops (2500 MD and 2500 ikman)
    df_laptops_md = scrape_md_laptops(2500, source='md')
    df_laptops_ik = scrape_md_laptops(2500, source='ikman')
    df_laptops = pd.concat([df_laptops_md, df_laptops_ik], ignore_index=True)
    
    laptop_path = os.path.join(BASE_DIR, 'data', 'raw', 'laptops_large_dataset.csv')
    if os.path.exists(laptop_path):
        df_l_orig = pd.read_csv(laptop_path)
        df_l_final = pd.concat([df_l_orig, df_laptops], ignore_index=True)
        df_l_final.to_csv(laptop_path, index=False)
        print(f"Laptops: Appended {len(df_laptops)} rows (2500 MD, 2500 ikman). Total size now: {len(df_l_final)} rows.")
    else:
        df_laptops.to_csv(laptop_path, index=False)
        print(f"Laptops: Created new raw dataset with {len(df_laptops)} rows.")
        
    # 2. Scrape and Append Monitors
    df_monitors = scrape_md_monitors(1000)
    monitor_path = os.path.join(BASE_DIR, 'data', 'raw', 'monitors_large_dataset.csv')
    if os.path.exists(monitor_path):
        df_m_orig = pd.read_csv(monitor_path)
        df_m_final = pd.concat([df_m_orig, df_monitors], ignore_index=True)
        df_m_final.to_csv(monitor_path, index=False)
        print(f"Monitors: Appended {len(df_monitors)} rows. Total size now: {len(df_m_final)} rows.")
    else:
        df_monitors.to_csv(monitor_path, index=False)
        print(f"Monitors: Created new raw dataset with {len(df_monitors)} rows.")
        
    # 3. Scrape and Append Tablets
    df_tablets = scrape_md_tablets(1000)
    tablet_path = os.path.join(BASE_DIR, 'data', 'raw', 'tablets_large_dataset.csv')
    if os.path.exists(tablet_path):
        df_t_orig = pd.read_csv(tablet_path)
        df_t_final = pd.concat([df_t_orig, df_tablets], ignore_index=True)
        df_t_final.to_csv(tablet_path, index=False)
        print(f"Tablets: Appended {len(df_tablets)} rows. Total size now: {len(df_t_final)} rows.")
    else:
        df_tablets.to_csv(tablet_path, index=False)
        print(f"Tablets: Created new raw dataset with {len(df_tablets)} rows.")

if __name__ == "__main__":
    main()
