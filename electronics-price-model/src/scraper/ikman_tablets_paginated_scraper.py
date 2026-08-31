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

class IkmanTabletsScraper:
    def __init__(self, used_only=False, max_concurrent=6, delay_between_batches=0.5):
        # Category filter: Computers & Tablets -> item_type=tablet
        if used_only:
            self.base_url = "https://ikman.lk/en/ads/sri-lanka/computers-tablets?enum.item_type=tablet&enum.condition=used"
        else:
            self.base_url = "https://ikman.lk/en/ads/sri-lanka/computers-tablets?enum.item_type=tablet"
            
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

    def extract_tablet_specs(self, title: str) -> dict:
        """Extracts Brand, Model, Storage, RAM, Connectivity (WiFi/5G/LTE), and Screen Size from title."""
        title_lower = title.lower()
        
        # 1. Brand detection
        brands = [
            "apple", "ipad", "samsung", "lenovo", "huawei", "xiaomi", "redmi", 
            "honor", "amazon", "kindle", "microsoft", "surface", "blackview", 
            "teclast", "realme", "nokia", "oneplus", "oppo", "alcatel", "chuwi"
        ]
        detected_brand = "Other"
        for b in brands:
            if b in title_lower:
                if b in ["ipad", "apple"]:
                    detected_brand = "Apple"
                elif b == "kindle":
                    detected_brand = "Amazon"
                else:
                    detected_brand = b.capitalize()
                break

        # 2. Storage capacity (e.g. 16GB, 32GB, 64GB, 128GB, 256GB, 512GB, 1TB)
        storage_match = re.search(r'(\d+)\s*(?:gb|tb)\s*(?:rom|storage|internal)?\b', title_lower)
        # Check storage vs RAM conflicts (e.g. 8GB/256GB or 8+256GB)
        dual_match = re.search(r'(\d+)\s*(?:gb)?\s*[\/\+]\s*(\d+)\s*(?:gb|tb)', title_lower)
        if dual_match:
            ram = f"{dual_match.group(1)}GB"
            storage = f"{dual_match.group(2)}GB"
        else:
            storage_candidates = re.findall(r'(\d+)\s*(?:gb|tb)', title_lower)
            storage = "Unknown"
            ram = "Unknown"
            if storage_candidates:
                # Tablet storages are typically 16, 32, 64, 128, 256, 512, 1024
                for cand in storage_candidates:
                    val = int(cand)
                    if val in [16, 32, 64, 128, 256, 512, 1024] or val == 1:
                        storage = f"{cand}TB" if val == 1 else f"{cand}GB"
                    elif val in [2, 3, 4, 6, 8, 12, 16] and ram == "Unknown":
                        ram = f"{cand}GB"

        # Explicit RAM matching
        ram_match = re.search(r'(\d+)\s*(?:gb|mb)\s*ram', title_lower)
        if ram_match:
            ram = f"{ram_match.group(1)}GB"

        # 3. Connectivity (WiFi / Cellular / 4G / 5G / LTE / Sim)
        if "5g" in title_lower:
            connectivity = "5G + WiFi"
        elif any(c in title_lower for c in ["4g", "lte", "cellular", "sim", "call"]):
            connectivity = "4G LTE / SIM"
        elif "wifi" in title_lower:
            connectivity = "WiFi Only"
        else:
            connectivity = "WiFi / Standard"

        # 4. Screen Size (e.g. 10.2", 10.9", 11", 12.4", 12.9", 8.3", 8")
        size_match = re.search(r'(\d{1,2}(?:\.\d)?)\s*(?:["\']|inch|\-inch)', title_lower)
        size = f"{size_match.group(1)} Inch" if size_match else "Unknown"

        # 5. Chipset / Generation (M1, M2, M4, Gen 9, Gen 10, S9, S8, etc.)
        gen_match = re.search(r'(m[124]|gen\s*\d+|\b1[0-3]th\s*gen|\b[5-9]th\s*gen)', title_lower)
        generation = gen_match.group(1).upper() if gen_match else "Unknown"

        return {
            "brand": detected_brand,
            "storage": storage,
            "ram": ram,
            "connectivity": connectivity,
            "size": size,
            "generation": generation
        }

    async def fetch_page(self, client: httpx.AsyncClient, page_num: int, retries=3):
        """Fetches a specific page of tablets with retry logic."""
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

                            # Filter out non-tablet ads (e.g., promoted laptops/desktops bleeding into search)
                            prop_desc = ad.get("propertiesDesc", "")
                            if prop_desc and "Tablet" not in prop_desc:
                                continue

                            self.seen_ids.add(ad_id)

                            title = ad.get("title", "")
                            raw_price = ad.get("price", "0")
                            price_digits = re.sub(r"[^\d]", "", str(raw_price))
                            specs = self.extract_tablet_specs(title)

                            items.append({
                                "id": ad_id,
                                "title": title,
                                "brand": specs["brand"],
                                "storage": specs["storage"],
                                "ram": specs["ram"],
                                "connectivity": specs["connectivity"],
                                "size": specs["size"],
                                "generation": specs["generation"],
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
                        
                        print(f"  [+] Page {page_num:03d}: Fetched {len(items)} tablets (Attempt {attempt+1})")
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

    async def scrape(self, max_pages=None, output_path=None):
        start_time = time.time()
        print("=" * 65)
        print("IKMAN TABLETS PAGINATED EXTRACTION ENGINE")
        print("=" * 65)
        print(f"Target Filter: Computers & Tablets -> Tablets")
        print(f"Base URL: {self.base_url}")
        
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            out_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "raw"))
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, "ikman_tablets_all.csv")

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
            print(f"[*] Total Available Tablet Ads: {total_ads:,}")
            print(f"[*] Total Pages Available: {total_pages:,}")
            print(f"[*] Target Pages to Scrape: {target_pages:,}")
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
        print(f"Total Tablets Scraped: {len(df_final):,}")
        print(f"Output Saved To: {output_path}")
        print("=" * 65)
        return df_final

def main():
    parser = argparse.ArgumentParser(description="Scrape Tablets from ikman.lk with high-speed async pagination.")
    parser.add_argument("--pages", type=int, default=None, help="Number of pages to scrape (leave empty with --all to scrape all available pages).")
    parser.add_argument("--all", action="store_true", help="Scrape all available tablet ads on ikman (~4,800+ ads).")
    parser.add_argument("--used-only", action="store_true", help="Filter for Used tablets only.")
    parser.add_argument("--concurrent", type=int, default=6, help="Concurrent async requests limit (default: 6).")
    parser.add_argument("--out", type=str, default=None, help="Custom output CSV path.")
    
    args = parser.parse_args()
    max_pages = None if args.all or args.pages is None else args.pages
    
    scraper = IkmanTabletsScraper(used_only=args.used_only, max_concurrent=args.concurrent)
    asyncio.run(scraper.scrape(max_pages=max_pages, output_path=args.out))

if __name__ == "__main__":
    main()
