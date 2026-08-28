import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os

def scrape_md_laptops():
    base_url = "https://mdcomputers.lk/product-category/laptop/used-laptop/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # STEP 1: Get all product links from all 7 pages
    product_links = []
    print("STEP 1: Getting product links from category pages...")
    
    for page in range(1, 8):
        if page == 1:
            url = base_url
        else:
            url = f"{base_url}page/{page}/"
            
        print(f"Reading category page {page}...")
        try:
            res = requests.get(url, headers=headers, timeout=15)
            if res.status_code != 200:
                print(f"Failed to load page {page}. Status: {res.status_code}")
                continue
                
            soup = BeautifulSoup(res.content, 'html.parser')
            # Actual products have 'type-product' class
            products = soup.find_all('li', class_=lambda c: c and 'type-product' in c)
            
            for p in products:
                link_elem = p.find('a', href=True)
                if link_elem:
                    link = link_elem['href']
                    if link not in product_links:
                        product_links.append(link)
                        
            time.sleep(2) # polite delay
        except Exception as e:
            print(f"Error reading page {page}: {e}")
            
    print(f"Found {len(product_links)} unique product links.")
    
    # STEP 2: Scrape details of each product page
    detailed_data = []
    print("\nSTEP 2: Scraping product details...")
    
    for idx, link in enumerate(product_links):
        print(f"Scraping {idx+1}/{len(product_links)}: {link}")
        try:
            res = requests.get(link, headers=headers, timeout=15)
            if res.status_code != 200:
                print(f"Failed to fetch details for {link}. Status: {res.status_code}")
                continue
                
            soup = BeautifulSoup(res.content, 'html.parser')
            
            # Title
            title_elem = soup.find('h1', class_='product_title')
            title = title_elem.text.strip() if title_elem else "Unknown"
            
            # Price
            price_elem = soup.find('p', class_='price') or soup.find('span', class_='price')
            price = "Unknown"
            if price_elem:
                # Find ins tag (sale price)
                ins_tag = price_elem.find('ins')
                if ins_tag:
                    price_text = ins_tag.text.strip()
                else:
                    del_tag = price_elem.find('del')
                    if del_tag:
                        price_text = price_elem.text.replace(del_tag.text, '').strip()
                    else:
                        price_text = price_elem.text.strip()
                
                # Extract actual price number (e.g., Rs:59,000.00 or Rs. 59,000)
                # Keep only Rs:X,XXX part
                price_match = re.search(r'(Rs[:\.]\s*[\d,]+(?:\.\d{2})?)', price_text, re.IGNORECASE)
                if price_match:
                    price = price_match.group(1).replace('Rs:', 'Rs ').replace('Rs.', 'Rs ').strip()
                else:
                    # Fallback to any numeric string
                    num_match = re.search(r'([\d,]+(?:\.\d{2})?)', price_text)
                    if num_match:
                        price = f"Rs {num_match.group(1)}"
                    else:
                        price = price_text
            
            # Brand
            # Search for brand in tags, or guess from title
            brand = "Unknown"
            # Try to look for brand links on product page
            brand_elem = soup.find('a', href=lambda h: h and 'product-brand' in h)
            if brand_elem:
                brand = brand_elem.text.strip()
            else:
                # Guess from title
                title_upper = title.upper()
                brands_list = ["DELL", "HP", "LENOVO", "ASUS", "ACER", "APPLE", "MSI", "TOSHIBA", "MICROSOFT"]
                for b in brands_list:
                    if b in title_upper:
                        brand = b
                        break
            
            # Description / Specs
            desc_elem = soup.find('div', id='tab-description') or soup.find('div', class_='woocommerce-product-details__short-description')
            description = desc_elem.text.strip() if desc_elem else ""
            
            full_text = f"{title} {description}".lower()
            
            # RAM extraction
            ram_match = re.search(r'(\d+)\s*(?:gb|mb)\s*ram', full_text)
            if not ram_match:
                ram_match = re.search(r'ram\s*(\d+)\s*(?:gb|mb)', full_text)
            ram = f"{ram_match.group(1)}GB" if ram_match else "Unknown"
            
            # Storage extraction
            storage_match = re.search(r'(\d+)\s*(?:gb|tb)\s*(?:hdd|ssd|storage|nvme|m\.2)', full_text)
            storage = f"{storage_match.group(1)} {storage_match.group(0).split()[-1].upper()}" if storage_match else "Unknown"
            
            detailed_data.append({
                "Title": title,
                "Price": price,
                "Location_Category": "MD Computers",
                "Link": link,
                "Brand": brand,
                "Model": "Unknown", # Will be cleaned by preprocessing
                "Condition": "Used",
                "RAM": ram,
                "Storage": storage,
                "Description": description
            })
            
            time.sleep(1) # polite delay
        except Exception as e:
            print(f"Error scraping details for {link}: {e}")
            
    if detailed_data:
        df = pd.DataFrame(detailed_data)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'raw', 'md_laptops_data.csv'))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nSUCCESS! Saved {len(detailed_data)} rich laptop records from MD Computers to {out_path}")
    else:
        print("\nNo data extracted.")

if __name__ == '__main__':
    scrape_md_laptops()
