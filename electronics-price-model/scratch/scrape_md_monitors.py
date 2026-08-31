import requests
from bs4 import BeautifulSoup
import csv
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

base_url = "https://mdcomputers.lk/product-category/monitors/used-monitors/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

product_links = []
print("Fetching category pages from MD Computers...")

for page in range(1, 6):
    if page == 1:
        url = base_url
    else:
        url = f"{base_url}page/{page}/"
        
    print(f"Fetching page {page}: {url}")
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code != 200:
            print(f"Ended pagination at page {page} (Status: {res.status_code})")
            break
            
        soup = BeautifulSoup(res.content, 'html.parser')
        products = soup.find_all('li', class_=lambda c: c and 'type-product' in c)
        
        if not products:
            print(f"No products found on page {page}. Ending loop.")
            break
            
        print(f"Found {len(products)} products on page {page}.")
        for p in products:
            link_elem = p.find('a', href=True)
            if link_elem:
                link = link_elem['href']
                if link not in product_links:
                    product_links.append(link)
                    
        time.sleep(1)
    except Exception as e:
        print(f"Error fetching page {page}: {e}")
        break

print(f"Total product links found: {len(product_links)}")

scraped_data = []

def fetch_product_details(link):
    try:
        res = requests.get(link, headers=headers, timeout=12)
        if res.status_code != 200:
            return None
            
        soup = BeautifulSoup(res.content, 'html.parser')
        
        # Title
        title_elem = soup.find('h1', class_='product_title')
        title = title_elem.text.strip() if title_elem else "Unknown"
        
        # Price
        price_elem = soup.find('p', class_='price') or soup.find('span', class_='price')
        price = "Unknown"
        if price_elem:
            ins_tag = price_elem.find('ins')
            if ins_tag:
                price_text = ins_tag.text.strip()
            else:
                del_tag = price_elem.find('del')
                if del_tag:
                    price_text = price_elem.text.replace(del_tag.text, '').strip()
                else:
                    price_text = price_elem.text.strip()
            
            price_match = re.search(r'(Rs[:\.]\s*[\d,]+(?:\.\d{2})?)', price_text, re.IGNORECASE)
            if price_match:
                price = price_match.group(1).replace('Rs:', 'Rs ').replace('Rs.', 'Rs ').strip()
            else:
                num_match = re.search(r'([\d,]+(?:\.\d{2})?)', price_text)
                if num_match:
                    price = f"Rs {num_match.group(1)}"
                else:
                    price = price_text
                    
        # Description
        desc_elem = soup.find('div', id='tab-description') or soup.find('div', class_='woocommerce-product-details__short-description')
        description = desc_elem.text.strip() if desc_elem else ""
        description = re.sub(r'\s+', ' ', description)
        
        # Brand
        brand = "Other"
        brand_elem = soup.find('a', href=lambda h: h and 'product-brand' in h)
        if brand_elem:
            brand = brand_elem.text.strip().upper()
        else:
            title_upper = title.upper()
            brands_list = ["DELL", "HP", "SAMSUNG", "LG", "ASUS", "ACER", "BENQ", "VIEWSONIC", "PHILIPS", "MSI", "AOC", "LENOVO", "EIZO", "HKC"]
            for b in brands_list:
                if b in title_upper:
                    brand = b
                    break
                    
        return {
            'Title': title,
            'Price': price,
            'Location_Category': 'MD Computers',
            'Link': link,
            'Brand': brand,
            'Condition': 'Used',
            'Description': description
        }
    except Exception as e:
        print(f"Error fetching product details for {link}: {e}")
        return None

print("Scraping monitor details concurrently...")
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(fetch_product_details, link): link for link in product_links}
    for idx, future in enumerate(as_completed(futures)):
        res = future.result()
        if res:
            scraped_data.append(res)
        if (idx + 1) % 5 == 0 or idx == len(product_links) - 1:
            print(f"Progress: [{idx+1}/{len(product_links)}] monitors fetched.")

# Save to CSV
csv_file = "scratch/scraped_md_monitors.csv"
keys = ['Title', 'Price', 'Location_Category', 'Link', 'Brand', 'Condition', 'Description']
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(scraped_data)

print(f"Successfully saved {len(scraped_data)} MD Computers monitors to {csv_file}")
