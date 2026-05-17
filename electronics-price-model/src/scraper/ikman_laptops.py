import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

def scrape_ikman_laptops(pages=1):
    base_url = "https://ikman.lk/en/ads?query=laptops"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    all_laptops = []

    for page in range(1, pages + 1):
        print(f"Scraping page {page}...")
        url = f"{base_url}&page={page}"
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Failed to fetch page {page}. Status code: {response.status_code}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Ikman ads are usually inside list items or specific divs.
        # We will try to find the standard ad containers.
        ad_containers = soup.find_all('li', class_=lambda c: c and 'normal--' in c) 
        if not ad_containers:
            ad_containers = soup.find_all('div', class_=lambda c: c and 'ad-item' in c)
        if not ad_containers:
            # Another common class for list items
            ad_containers = soup.select('ul.list--3NxGO > li')

        print(f"Found {len(ad_containers)} ads on page {page}")

        for ad in ad_containers:
            try:
                # Extract title
                title_elem = ad.find('h2')
                title = title_elem.text.strip() if title_elem else "N/A"

                # Extract price
                price_elem = ad.find('div', class_=lambda c: c and 'price--' in c)
                if not price_elem:
                     # try another way
                     price_elem = ad.find('div', string=lambda s: s and 'Rs' in s)
                price = price_elem.text.strip() if price_elem else "N/A"

                # Extract location and category
                loc_cat_elem = ad.find('div', class_=lambda c: c and 'description--' in c)
                loc_cat = loc_cat_elem.text.strip() if loc_cat_elem else "N/A"

                # Extract link
                link_elem = ad.find('a', href=True)
                link = "https://ikman.lk" + link_elem['href'] if link_elem else "N/A"

                all_laptops.append({
                    "Title": title,
                    "Price": price,
                    "Location_Category": loc_cat,
                    "Link": link
                })
            except Exception as e:
                print(f"Error parsing ad: {e}")
                continue

        # Polite delay
        time.sleep(2)

    if all_laptops:
        df = pd.DataFrame(all_laptops)
        output_path = os.path.join("..", "..", "data", "raw", "laptops_data.csv")
        # Ensure we are saving in the right directory assuming we run from src/scraper
        df.to_csv(output_path, index=False)
        print(f"Successfully saved {len(all_laptops)} records to {output_path}")
    else:
        print("No data extracted. Might need to update HTML selectors.")

if __name__ == "__main__":
    scrape_ikman_laptops(pages=3)
