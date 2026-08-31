import requests
from bs4 import BeautifulSoup
import csv
import time
import re
import os

base_url = "https://ikman.lk/en/ads/sri-lanka/computers-tablets?sort=relevance&buy_now=0&urgent=0&query=samsung&enum.item_type=tablet&enum.condition=used"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

scraped_data = []

print("Starting scraper for Samsung Tablets...")

for page in range(1, 10):
    url = f"https://ikman.lk/en/ads/sri-lanka/computers-tablets?sort=relevance&buy_now=0&urgent=0&query=samsung&page={page}&enum.item_type=tablet&enum.condition=used"
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
            # Skip if it is not an ad link
            if '/en/ad/' not in href:
                continue
                
            title_el = item.find('h2', class_=lambda c: c and 'title' in c)
            title = title_el.text.strip() if title_el else ""
            
            # Make sure it's a Samsung tablet
            if 'samsung' not in title.lower() and 'samsung' not in href.lower():
                continue
                
            price_el = item.find('div', class_=lambda c: c and 'price' in c)
            price = price_el.text.strip() if price_el else ""
            
            cond_el = item.find('div', class_=lambda c: c and 'details' in c)
            condition = cond_el.text.strip() if cond_el else "Used"
            
            loc_el = item.find('div', class_=lambda c: c and 'description' in c)
            location = loc_el.text.strip() if loc_el else "Sri Lanka, Tablet"
            
            ad_url = f"https://ikman.lk{href}"
            
            scraped_data.append({
                'Title': title,
                'Price': price,
                'Location_Category': location,
                'Link': ad_url,
                'Brand': 'SAMSUNG',
                'Model': '',
                'Condition': condition,
                'Description_Link': ad_url
            })
            
    except Exception as e:
        print(f"Error fetching page {page}: {e}")
        break

print(f"Scraped metadata for {len(scraped_data)} Samsung tablets. Fetching descriptions...")

# Now fetch descriptions
for idx, entry in enumerate(scraped_data):
    ad_url = entry['Description_Link']
    print(f"[{idx+1}/{len(scraped_data)}] Fetching description: {ad_url}")
    try:
        time.sleep(0.3)  # Gentle delay
        res = requests.get(ad_url, headers=headers, timeout=10)
        if res.status_code == 200:
            sub_soup = BeautifulSoup(res.text, 'html.parser')
            desc_div = sub_soup.find('div', class_=lambda c: c and 'description--1nRbz' in c)
            if not desc_div:
                desc_div = sub_soup.find('div', class_=lambda c: c and 'description-section' in c)
            
            if desc_div:
                description_text = desc_div.get_text(separator=' ').strip()
                # Clean up multiple whitespaces
                description_text = re.sub(r'\s+', ' ', description_text)
                entry['Description'] = description_text
            else:
                entry['Description'] = ""
        else:
            entry['Description'] = ""
    except Exception as e:
        print(f"Error fetching detail description for {ad_url}: {e}")
        entry['Description'] = ""

    # Clean the Description_Link property out, keep only Link
    del entry['Description_Link']

# Save to CSV
csv_file = "scratch/scraped_samsung_tablets.csv"
keys = ['Title', 'Price', 'Location_Category', 'Link', 'Brand', 'Model', 'Condition', 'Description']
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(scraped_data)

print(f"Successfully saved {len(scraped_data)} records to {csv_file}")
