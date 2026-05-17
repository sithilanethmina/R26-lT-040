import pandas as pd
import time
import re
import os
from playwright.sync_api import sync_playwright

def scrape_ikman_playwright(max_pages=20):
    url = "https://ikman.lk/en/ads/sri-lanka/computers-tablets/laptop"
    
    detailed_data = []

    with sync_playwright() as p:
        # Launch browser in headed mode to look like a real user and bypass Cloudflare
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        # Navigate to main category
        print("Navigating to Ikman Laptops category...")
        page.goto(url)
        time.sleep(3)
        
        all_links = []
        
        for i in range(1, max_pages + 1):
            print(f"Extracting links from page {i}...")
            
            # Find ad containers.
            # Using a more robust selector that finds any link pointing to an ad
            ad_elements = page.locator('a[href*="/en/ad/"]')
            count = ad_elements.count()
            
            for j in range(count):
                href = ad_elements.nth(j).get_attribute('href')
                if href and '/ad/' in href:
                    full_link = "https://ikman.lk" + href
                    if full_link not in all_links:
                        all_links.append(full_link)
                    
            print(f"Found {count} ads. Total links so far: {len(all_links)}")
            
            # Click next page
            next_btn = page.locator('a[data-testid="pagination-next-link"]')
            if next_btn.count() > 0 and i < max_pages:
                next_btn.click()
                time.sleep(4) # Wait for next page to load
            else:
                print("No more pages found or reached max_pages limit.")
                break
                
        # Now visit each link to get details
        print(f"\nStarting to extract details for {len(all_links)} laptops...")
        for idx, link in enumerate(all_links):
            print(f"Scraping detail {idx+1}/{len(all_links)}: {link}", flush=True)
            try:
                page.goto(link, timeout=30000)
                time.sleep(2) # Give it time to load JS
                
                # Get Title
                title_el = page.locator('h1')
                title = title_el.inner_text() if title_el.count() > 0 else "Unknown"
                
                # Get Price
                price_el = page.locator('div.amount--3NTpl')
                if price_el.count() == 0:
                     price_el = page.locator('div:has-text("Rs")').first
                price = price_el.inner_text() if price_el.count() > 0 else "Unknown"
                
                # Get Properties (Brand, Model, Condition)
                props_elements = page.locator('div.word-break--2nyVq')
                props = {}
                for k in range(props_elements.count()):
                    text = props_elements.nth(k).inner_text()
                    if ":" in text:
                        parts = text.split(":", 1)
                        props[parts[0].strip()] = parts[1].strip()
                        
                # Get Description
                desc_el = page.locator('div.description--1nRbz')
                if desc_el.count() == 0:
                     desc_el = page.locator('div[itemprop="description"]')
                description = desc_el.inner_text() if desc_el.count() > 0 else ""
                
                # Extract RAM and Storage via regex
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
                    "Link": link,
                    "Brand": props.get("Brand", "Unknown"),
                    "Model": props.get("Model", "Unknown"),
                    "Condition": props.get("Condition", "Unknown"),
                    "RAM": ram,
                    "Storage": storage,
                    "Description": description
                })
                
            except Exception as e:
                print(f"Error on {link}: {e}")
                
        # Save to CSV
        if detailed_data:
            out_df = pd.DataFrame(detailed_data)
            out_path = '../../data/raw/laptops_large_dataset.csv'
            out_df.to_csv(out_path, index=False)
            print(f"\nSaved {len(detailed_data)} records to {out_path}!")
            
        browser.close()

if __name__ == "__main__":
    # You can increase max_pages to 50 or 100 to get thousands of laptops
    scrape_ikman_playwright(max_pages=20)
