import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os
import re
import cloudscraper

def scrape_ikman_large():
    base_url = "https://ikman.lk/en/ads?query=laptops"
    
    # We will use rotating User-Agents to prevent getting blocked easily
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    ]

    all_laptops = []
    
    scraper = cloudscraper.create_scraper() # returns a CloudScraper instance
    
    # STEP 1: Scrape search pages to get all laptop URLs
    print("STEP 1: Gathering all laptop links from Search Pages...")
    # We will try to get 150 pages to reach ~4000 laptops
    for page in range(1, 151):
        print(f"-> Reading Page {page}...")
        url = f"{base_url}&page={page}"
        headers = {"User-Agent": random.choice(user_agents)}
        
        try:
            response = scraper.get(url, headers=headers, timeout=15)
            if response.status_code != 200:
                print(f"Warning: Page {page} blocked or failed. Stopping search extraction.")
                break
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the ad list items
            ad_containers = soup.select('ul.list--3NxGO > li')
            if not ad_containers:
                 # Alternative layout
                 ad_containers = soup.find_all('li', class_=lambda c: c and 'normal--' in c)
                 
            if len(ad_containers) == 0:
                print("No ads found on this page. Stopping search extraction.")
                break
                
            for ad in ad_containers:
                link_elem = ad.find('a', href=True)
                if link_elem and '/ad/' in link_elem['href']:
                    full_link = "https://ikman.lk" + link_elem['href']
                    if full_link not in all_laptops:
                        all_laptops.append(full_link)
                        
            # Sleep randomly between 3 to 6 seconds to look human
            time.sleep(random.uniform(3, 6))
            
        except Exception as e:
            print(f"Error on page {page}: {e}")
            break
            
    print(f"\nSuccessfully found {len(all_laptops)} unique laptop links!")
    
    if not all_laptops:
        return
        
    # STEP 2: Visit each link and extract specifications
    print("\nSTEP 2: Extracting deep specifications (RAM, Storage, Brand) for each laptop...")
    detailed_data = []
    
    for idx, link in enumerate(all_laptops):
        print(f"Scraping {idx+1}/{len(all_laptops)}: {link}")
        headers = {"User-Agent": random.choice(user_agents)}
        
        try:
            res = scraper.get(link, headers=headers, timeout=15)
            if res.status_code != 200:
                print("Blocked. Retrying after 10 seconds...")
                time.sleep(10)
                res = scraper.get(link, headers=headers, timeout=15)
                if res.status_code != 200:
                    continue
            
            soup = BeautifulSoup(res.content, 'html.parser')
            
            # Title
            title_elem = soup.find('h1')
            title = title_elem.text.strip() if title_elem else "Unknown"
            
            # Price
            price_elem = soup.find('div', class_=lambda c: c and 'amount--' in c)
            price = price_elem.text.strip() if price_elem else "Unknown"
            
            # Location
            loc_elem = soup.find('a', class_=lambda c: c and 'subtitle-wrapper--' in c)
            location = loc_elem.text.strip() if loc_elem else "Unknown"
            
            # Properties
            props = {}
            for div in soup.find_all('div', class_=lambda c: c and 'word-break--' in c):
                text = div.text.strip()
                if ':' in text:
                    parts = text.split(':', 1)
                    props[parts[0].strip()] = parts[1].strip()
                    
            # Description
            desc_elem = soup.find('div', class_=lambda c: c and 'description--' in c)
            description = desc_elem.text.strip() if desc_elem else ""
            
            # REGEX Extraction
            full_text = f"{title} {description} {str(props)}".lower()
            
            ram_match = re.search(r'(\d+)\s*(?:gb|mb)\s*ram', full_text)
            if not ram_match:
                ram_match = re.search(r'ram\s*(\d+)\s*(?:gb|mb)', full_text)
            ram = f"{ram_match.group(1)}GB" if ram_match else "Unknown"
            
            storage_match = re.search(r'(\d+)\s*(?:gb|tb)\s*(?:hdd|ssd|storage|nvme|m\.2)', full_text)
            storage = f"{storage_match.group(1)} {storage_match.group(0).split()[-1].upper()}" if storage_match else "Unknown"
            
            detailed_data.append({
                "Title": title,
                "Price": price,
                "Location_Category": location,
                "Link": link,
                "Brand": props.get("Brand", "Unknown"),
                "Model": props.get("Model", "Unknown"),
                "Condition": props.get("Condition", "Unknown"),
                "RAM": ram,
                "Storage": storage,
                "Description": description
            })
            
            # Wait 2 seconds to not get banned
            time.sleep(2)
            
        except Exception as e:
            print(f"Failed to scrape: {e}")
            
        # Every 20 laptops, save progress so we don't lose data
        if (idx + 1) % 20 == 0:
            pd.DataFrame(detailed_data).to_csv('data/raw/laptops_large_dataset.csv', index=False)
            
    # Final save
    if detailed_data:
        out_df = pd.DataFrame(detailed_data)
        out_path = 'data/raw/laptops_large_dataset.csv'
        out_df.to_csv(out_path, index=False)
        print(f"\nSUCCESS! Saved {len(detailed_data)} rich laptop records to {out_path}")

if __name__ == "__main__":
    scrape_ikman_large()
