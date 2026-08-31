import asyncio
import json
import re
import os
import sys
import time
import argparse
import pandas as pd
import httpx
from bs4 import BeautifulSoup

class IkmanUsedLaptopsScraper:
    def __init__(self, max_concurrent=6, delay_between_batches=0.5):
        # Precise category filter: item_type=laptop AND condition=used
        self.base_url = "https://ikman.lk/en/ads/sri-lanka/computers-tablets?enum.item_type=laptop&enum.condition=used"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.delay = delay_between_batches
        self.all_records = []
        self.seen_ids = set()

    def extract_initial_data(self, html_text: str) -> dict:
        """Extracts and parses window.initialData from HTML."""
        soup = BeautifulSoup(html_text, "html.parser")
        for s in soup.find_all("script"):
            if s.string and "window.initialData" in s.string:
                raw_text = s.string.strip()
                prefix = "window.initialData = "
                idx = raw_text.find(prefix)
                if idx != -1:
                    json_str = raw_text[idx + len(prefix):].rstrip(";")
                    try:
                        return json.loads(json_str)
                    except Exception:
                        return {}
        return {}

    def extract_specs_from_title(self, title: str) -> dict:
        """Extracts brand, RAM, and Storage automatically from ad title using regex."""
        title_lower = title.lower()
        
        # Brand detection
        brands = ["dell", "hp", "lenovo", "asus", "acer", "apple", "macbook", "msi", "toshiba", "samsung", "huawei", "microsoft", "surface", "razer", "fujitsu"]
        detected_brand = "Other"
        for b in brands:
            if b in title_lower:
                detected_brand = "Apple" if b == "macbook" else b.capitalize()
                break
        
        # RAM detection
        ram_match = re.search(r'(\d+)\s*(?:gb|mb)\s*ram', title_lower)
        if not ram_match:
            ram_match = re.search(r'ram\s*(\d+)\s*(?:gb|mb)', title_lower)
        if not ram_match:
            ram_match = re.search(r'(\d+)\s*gb(?:\s*\|\s*|\s*,\s*|\s+)(?:ddr\d|ssd|hdd|nvme|i\d|core)', title_lower)
        ram = f"{ram_match.group(1)}GB" if ram_match else "Unknown"

        # Storage detection
        storage_match = re.search(r'(\d+)\s*(?:gb|tb)\s*(?:hdd|ssd|storage|nvme|m\.2|emmc)', title_lower)
        storage = f"{storage_match.group(1)} {storage_match.group(0).split()[-1].upper()}" if storage_match else "Unknown"

        # Processor detection (i3, i5, i7, i9, Ryzen)
        proc_match = re.search(r'(core\s*i[3579]|i[3579]|ryzen\s*[3579]|m1|m2|m3|celeron|pentium)', title_lower)
        processor = proc_match.group(1).upper() if proc_match else "Unknown"

        return {
            "brand": detected_brand,
            "ram": ram,
            "storage": storage,
            "processor": processor
        }

    async def fetch_page(self, client: httpx.AsyncClient, page_num: int, retries=3):
        """Fetches a specific page of used laptops with retry logic."""
        async with self.semaphore:
            url = f"{self.base_url}&page={page_num}"
            for attempt in range(retries):
                try:
                    resp = await client.get(url, headers=self.headers, timeout=20.0)
                    if resp.status_code == 200:
                        data = self.extract_initial_data(resp.text)
                        ads_data = data.get("serp", {}).get("ads", {}).get("data", {})
                        ads_list = ads_data.get("ads", [])
                        
                        items = []
                        for ad in ads_list:
                            ad_id = ad.get("id")
                            if not ad_id or ad_id in self.seen_ids:
                                continue
                            self.seen_ids.add(ad_id)

                            title = ad.get("title", "")
                            raw_price = ad.get("price", "0")
                            price_digits = re.sub(r"[^\d]", "", str(raw_price))
                            specs = self.extract_specs_from_title(title)

                            items.append({
                                "id": ad_id,
                                "title": title,
                                "brand": specs["brand"],
                                "processor": specs["processor"],
                                "ram": specs["ram"],
                                "storage": specs["storage"],
                                "price_raw": raw_price,
                                "price": int(price_digits) if price_digits else None,
                                "condition": ad.get("details", "Used"),
                                "location": ad.get("location", "Unknown"),
                                "slug": ad.get("slug"),
                                "link": f"https://ikman.lk/en/ad/{ad.get('slug')}" if ad.get("slug") else None,
                                "shop_name": ad.get("shopName", "Individual"),
                                "is_member": ad.get("isMember", False),
                                "timestamp": ad.get("timeStamp")
                            })
                        
                        print(f"  [+] Page {page_num:03d}: Fetched {len(items)} ads (Attempt {attempt+1})")
                        if self.delay > 0:
                            await asyncio.sleep(self.delay)
                        return items
                    elif resp.status_code == 404:
                        print(f"  [-] Page {page_num:03d}: Reached end of listings (404).")
                        return []
                    else:
                        print(f"  [!] Page {page_num:03d}: Status {resp.status_code}, retrying...")
                except Exception as e:
                    if attempt == retries - 1:
                        print(f"  [X] Page {page_num:03d} failed after {retries} attempts: {e}")
                
                await asyncio.sleep(1.0 * (attempt + 1))
            return []

    async def scrape(self, max_pages=None, save_interval=10, output_path=None):
        start_time = time.time()
        print("=" * 65)
        print("IKMAN USED LAPTOPS PAGINATED EXTRACTION ENGINE")
        print("=" * 65)
        print(f"Target Filter: Computers & Tablets -> Laptop (Condition: Used)")
        print(f"Base URL: {self.base_url}")
        
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            out_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "raw"))
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, "ikman_used_laptops_all.csv")

        async with httpx.AsyncClient(follow_redirects=True) as client:
            # 1. Discover pagination parameters
            print("\n[*] Probing category pagination metadata...")
            probe_resp = await client.get(f"{self.base_url}&page=1", headers=self.headers, timeout=20.0)
            probe_data = self.extract_initial_data(probe_resp.text)
            pagination = probe_data.get("serp", {}).get("ads", {}).get("data", {}).get("paginationData", {})
            
            total_ads = pagination.get("total", 0)
            page_size = pagination.get("pageSize", 25)
            total_pages = (total_ads // page_size) + 1 if page_size else 1
            
            target_pages = total_pages if max_pages is None else min(max_pages, total_pages)
            print(f"[*] Total Available Used Laptop Ads: {total_ads:,}")
            print(f"[*] Total Pages Available: {total_pages:,}")
            print(f"[*] Target Pages to Scrape in this Run: {target_pages:,}")
            print(f"[*] Output Destination: {output_path}\n")
            
            # 2. Batch Processing with Checkpoints
            batch_size = 10
            for start_p in range(1, target_pages + 1, batch_size):
                end_p = min(start_p + batch_size - 1, target_pages)
                print(f"[*] Dispatching Batch: Pages {start_p:03d} to {end_p:03d}...")
                
                tasks = [self.fetch_page(client, p) for p in range(start_p, end_p + 1)]
                batch_results = await asyncio.gather(*tasks)
                
                new_items_count = 0
                for page_items in batch_results:
                    self.all_records.extend(page_items)
                    new_items_count += len(page_items)
                
                # Checkpoint save
                df_current = pd.DataFrame(self.all_records)
                if not df_current.empty:
                    df_current.to_csv(output_path, index=False)
                    print(f"  [SAVE] Checkpoint: Saved {len(df_current):,} total records to CSV ({new_items_count} new in batch)")
        
        df_final = pd.DataFrame(self.all_records)
        elapsed = round(time.time() - start_time, 2)
        print("\n" + "=" * 65)
        print(f"[SUCCESS] COMPLETED in {elapsed} seconds!")
        print(f"Total Used Laptops Scraped: {len(df_final):,}")
        print(f"Output Saved To: {output_path}")
        print("=" * 65)
        return df_final

def main():
    parser = argparse.ArgumentParser(description="Scrape Used Laptops from ikman.lk with high-speed async pagination.")
    parser.add_argument("--pages", type=int, default=20, help="Number of pages to scrape (default: 20, ~500 laptops). Set to 0 or leave empty with --all to scrape everything).")
    parser.add_argument("--all", action="store_true", help="Scrape every single used laptop ad available on ikman.")
    parser.add_argument("--concurrent", type=int, default=6, help="Concurrent async requests limit (default: 6).")
    parser.add_argument("--out", type=str, default=None, help="Custom output CSV path.")
    
    args = parser.parse_args()
    max_pages = None if args.all else args.pages
    
    scraper = IkmanUsedLaptopsScraper(max_concurrent=args.concurrent)
    asyncio.run(scraper.scrape(max_pages=max_pages, output_path=args.out))

if __name__ == "__main__":
    main()
