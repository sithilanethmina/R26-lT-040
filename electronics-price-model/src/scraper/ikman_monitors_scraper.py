import time
import random
import re
import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup

def scrape_monitors():
    base_url = "https://ikman.lk/en/ads?query=monitor"
    
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    ]

    all_monitors = []
    scraper = cloudscraper.create_scraper()
    
    print("STEP 1: Gathering Monitor links from Search Pages...")
    # Scrape 3 pages to get around 50 real monitor ads
    for page in range(1, 4):
        print(f"-> Reading Page {page}...")
        url = f"{base_url}&page={page}"
        headers = {"User-Agent": random.choice(user_agents)}
        
        try:
            response = scraper.get(url, headers=headers, timeout=15)
            if response.status_code != 200:
                print(f"Warning: Page {page} blocked. Stopping search.")
                break
                
            soup = BeautifulSoup(response.content, 'html.parser')
            ad_containers = soup.select('ul.list--3NxGO > li')
            if not ad_containers:
                 ad_containers = soup.find_all('li', class_=lambda c: c and 'normal--' in c)
                 
            for ad in ad_containers:
                link_elem = ad.find('a', href=True)
                if link_elem and '/ad/' in link_elem['href']:
                    full_link = "https://ikman.lk" + link_elem['href']
                    if full_link not in all_monitors:
                        all_monitors.append(full_link)
                        
            time.sleep(random.uniform(3, 5))
            
        except Exception as e:
            print(f"Error on page {page}: {e}")
            break
            
    print(f"\nFound {len(all_monitors)} monitor links!")
    
    detailed_data = []
    print("\nSTEP 2: Extracting Monitor specifications...")
    
    for idx, link in enumerate(all_monitors):
        print(f"Scraping {idx+1}/{len(all_monitors)}: {link}")
        headers = {"User-Agent": random.choice(user_agents)}
        
        try:
            res = scraper.get(link, headers=headers, timeout=15)
            if res.status_code != 200:
                time.sleep(5)
                continue
            
            soup = BeautifulSoup(res.content, 'html.parser')
            
            title_elem = soup.find('h1')
            title = title_elem.text.strip() if title_elem else "Unknown"
            
            price_elem = soup.find('div', class_=lambda c: c and 'amount--' in c)
            price = price_elem.text.strip() if price_elem else "Unknown"
            
            loc_elem = soup.find('a', class_=lambda c: c and 'subtitle-wrapper--' in c)
            location = loc_elem.text.strip() if loc_elem else "Unknown"
            
            props = {}
            for div in soup.find_all('div', class_=lambda c: c and 'word-break--' in c):
                text = div.text.strip()
                if ':' in text:
                    parts = text.split(':', 1)
                    props[parts[0].strip()] = parts[1].strip()
                    
            desc_elem = soup.find('div', class_=lambda c: c and 'description--' in c)
            description = desc_elem.text.strip() if desc_elem else ""
            
            full_text = f"{title} {description} {str(props)}".lower()
            
            # Extract Size (e.g. 24", 27 inch)
            size_match = re.search(r'(\d{2})[\'"]|(\d{2})\s*inch', full_text)
            size = size_match.group(1) or size_match.group(2) if size_match else "Unknown"
            if size != "Unknown": size += " Inch"
            
            # Extract Refresh Rate (e.g. 144hz, 60hz)
            hz_match = re.search(r'(\d{2,3})\s*hz', full_text)
            hz = f"{hz_match.group(1)}Hz" if hz_match else "Unknown"
            
            # Extract Resolution
            res_val = "Unknown"
            if "4k" in full_text or "2160" in full_text: res_val = "4K"
            elif "2k" in full_text or "1440" in full_text: res_val = "2K"
            elif "fhd" in full_text or "1080" in full_text: res_val = "FHD"
            
            detailed_data.append({
                "Title": title,
                "Price": price,
                "Location_Category": location,
                "Link": link,
                "Brand": props.get("Brand", "Unknown"),
                "Condition": props.get("Condition", "Unknown"),
                "Size": size,
                "Refresh_Rate": hz,
                "Resolution": res_val,
                "Description": description
            })
            
            time.sleep(2)
            
        except Exception as e:
            print(f"Failed to scrape: {e}")
            
    if detailed_data:
        out_df = pd.DataFrame(detailed_data)
        out_path = 'data/raw/monitors_dataset.csv'
        out_df.to_csv(out_path, index=False)
        print(f"\nSUCCESS! Saved {len(detailed_data)} monitor records to {out_path}")

if __name__ == "__main__":
    scrape_monitors()
