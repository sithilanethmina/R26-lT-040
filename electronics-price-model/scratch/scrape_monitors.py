import requests
from bs4 import BeautifulSoup
import csv
import time
import re

base_url = "https://ikman.lk/en/ads/sri-lanka/computer-accessories?enum.item_type=monitor&enum.condition=used"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

scraped_data = []

print("Starting scraper for Monitors...")

for page in range(1, 12):
    # Order arguments properly so ikman doesn't return 0 results
    url = f"https://ikman.lk/en/ads/sri-lanka/computer-accessories?page={page}&enum.item_type=monitor&enum.condition=used"
    print(f"Fetching page {page}: {url}")
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"Failed to fetch page {page}, status code: {response.status_code}")
            break
            
        soup = BeautifulSoup(response.text, 'html.parser')
        items = soup.find_all('li', class_=lambda c: c and 'gtm-normal-ad' in c)
        
        if not items:
            print(f"No listings found on page {page}. Ending page loop.")
            break
            
        print(f"Found {len(items)} ad cards on page {page}.")
        
        for item in items:
            link_el = item.find('a', data_testid="ad-card-link")
            if not link_el:
                link_el = item.find('a', class_=lambda c: c and 'card-link' in c)
            if not link_el or not link_el.get('href'):
                continue
                
            href = link_el['href']
            if '/en/ad/' not in href:
                continue
                
            title_el = item.find('h2', class_=lambda c: c and 'title' in c)
            title = title_el.text.strip() if title_el else ""
            
            price_el = item.find('div', class_=lambda c: c and 'price' in c)
            price = price_el.text.strip() if price_el else ""
            
            cond_el = item.find('div', class_=lambda c: c and 'details' in c)
            condition = cond_el.text.strip() if cond_el else "Used"
            
            loc_el = item.find('div', class_=lambda c: c and 'description' in c)
            location = loc_el.text.strip() if loc_el else "Sri Lanka, Monitor"
            
            ad_url = f"https://ikman.lk{href}"
            
            scraped_data.append({
                'Title': title,
                'Price': price,
                'Location_Category': location,
                'Link': ad_url,
                'Brand': '',
                'Model': '',
                'Condition': condition,
                'Description_Link': ad_url
            })
            
    except Exception as e:
        print(f"Error fetching page {page}: {e}")
        break

print(f"Scraped metadata for {len(scraped_data)} monitors. Fetching descriptions concurrently...")

from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_description(entry):
    ad_url = entry['Description_Link']
    try:
        res = requests.get(ad_url, headers=headers, timeout=10)
        if res.status_code == 200:
            sub_soup = BeautifulSoup(res.text, 'html.parser')
            desc_div = sub_soup.find('div', class_=lambda c: c and 'description--1nRbz' in c)
            if not desc_div:
                desc_div = sub_soup.find('div', class_=lambda c: c and 'description-section' in c)
            
            if desc_div:
                description_text = desc_div.get_text(separator=' ').strip()
                description_text = re.sub(r'\s+', ' ', description_text)
                return entry, description_text
    except Exception:
        pass
    return entry, ""

with ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(fetch_description, entry): entry for entry in scraped_data}
    for idx, future in enumerate(as_completed(futures)):
        entry, desc = future.result()
        entry['Description'] = desc
        del entry['Description_Link']
        if (idx + 1) % 10 == 0 or idx == len(scraped_data) - 1:
            print(f"Progress: [{idx+1}/{len(scraped_data)}] descriptions fetched.")

# Save to CSV
csv_file = "scratch/scraped_monitors.csv"
keys = ['Title', 'Price', 'Location_Category', 'Link', 'Brand', 'Model', 'Condition', 'Description']
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(scraped_data)

print(f"Successfully saved {len(scraped_data)} records to {csv_file}")
