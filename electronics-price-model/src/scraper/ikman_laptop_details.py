import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
import os

def extract_properties(soup):
    properties = {}
    # Find labels (e.g. Condition, Brand, Model)
    # Ikman usually lists them in pairs
    labels = soup.find_all('div', class_=lambda c: c and 'word-break--' in c)
    for label in labels:
        text = label.text.strip()
        if ':' in text:
            parts = text.split(':', 1)
            if len(parts) == 2:
                properties[parts[0].strip()] = parts[1].strip()
    return properties

def scrape_details():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'raw', 'laptops_data.csv'))
    df = pd.read_csv(input_path)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
    }
    
    detailed_data = []

    # Limit to first 10 for demonstration to be fast, but actually let's do all since it's only ~50
    for index, row in df.iterrows():
        link = row['Link']
        if pd.isna(link) or link == 'N/A':
            continue
            
        print(f"Scraping {index+1}/{len(df)}: {link}", flush=True)
        
        try:
            res = requests.get(link, headers=headers, timeout=10)
            if res.status_code != 200:
                continue
                
            soup = BeautifulSoup(res.content, 'html.parser')
            
            # Extract properties (Brand, Model, Condition, etc)
            props = extract_properties(soup)
            
            # Extract description
            desc_elem = soup.find('div', class_=lambda c: c and 'description--' in c)
            description = desc_elem.text.strip() if desc_elem else ""

            full_text = f"{row['Title']} {description} {str(props)}".lower()
            
            # Extract RAM
            ram_match = re.search(r'(\d+)\s*(?:gb|mb)\s*ram', full_text)
            if not ram_match:
                ram_match = re.search(r'ram\s*(\d+)\s*(?:gb|mb)', full_text)
            ram = f"{ram_match.group(1)}GB" if ram_match else "Unknown"
            
            # Extract Storage
            storage_match = re.search(r'(\d+)\s*(?:gb|tb)\s*(?:hdd|ssd|storage|nvme|m\.2)', full_text)
            storage = f"{storage_match.group(1)} {storage_match.group(0).split()[-1].upper()}" if storage_match else "Unknown"

            # Create final row
            row_dict = row.to_dict()
            row_dict['Brand'] = props.get('Brand', 'Unknown')
            row_dict['Model'] = props.get('Model', 'Unknown')
            row_dict['Condition'] = props.get('Condition', 'Unknown')
            row_dict['RAM'] = ram
            row_dict['Storage'] = storage
            row_dict['Description'] = description
            
            detailed_data.append(row_dict)
            
        except Exception as e:
            print(f"Error on {link}: {e}")
            
        time.sleep(1) # be polite to the server
        
    out_df = pd.DataFrame(detailed_data)
    out_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'raw', 'laptops_detailed.csv'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(detailed_data)} detailed records to {out_path}!")

if __name__ == "__main__":
    scrape_details()
