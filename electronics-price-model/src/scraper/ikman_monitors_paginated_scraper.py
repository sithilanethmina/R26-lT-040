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

class IkmanMonitorsScraper:
    def __init__(self, used_only=False, max_concurrent=6, delay_between_batches=0.5):
        # Category: Computer Accessories -> item_type=monitor
        if used_only:
            self.base_url = "https://ikman.lk/en/ads/sri-lanka/computer-accessories?enum.item_type=monitor&enum.condition=used"
        else:
            self.base_url = "https://ikman.lk/en/ads/sri-lanka/computer-accessories?enum.item_type=monitor"
            
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

    def extract_monitor_specs(self, title: str) -> dict:
        """Extracts Brand, Screen Size (inch), Refresh Rate (Hz), Resolution, and Panel features from title."""
        title_lower = title.lower()
        
        # 1. Brand detection
        brands = [
            "samsung", "dell", "hp", "lg", "asus", "acer", "benq", "viewsonic", 
            "msi", "aoc", "gigabyte", "philips", "lenovo", "xiaomi", "redmi", 
            "prolink", "armaggeddon", "huawei", "apple", "fujitsu", "epson"
        ]
        detected_brand = "Other"
        for b in brands:
            if b in title_lower:
                detected_brand = b.capitalize() if b != "msi" and b != "aoc" and b != "hp" and b != "lg" else b.upper()
                break

        # 2. Screen Size in Inches (e.g., 24", 27 inch, 21.5", 32 inch)
        size_match = re.search(r'(\d{2}(?:\.\d)?)\s*(?:["\']|inch|\-inch|\s*in\b)', title_lower)
        if not size_match:
            size_match = re.search(r'\b(19|20|21|22|23|24|25|27|28|29|32|34|43|49)\b', title_lower)
        
        size = f"{size_match.group(1)} Inch" if size_match else "Unknown"

        # 3. Refresh Rate in Hz (e.g. 60Hz, 75Hz, 100Hz, 144Hz, 165Hz, 180Hz, 240Hz)
        hz_match = re.search(r'(\d{2,3})\s*hz', title_lower)
        refresh_rate = f"{hz_match.group(1)}Hz" if hz_match else "60Hz"  # Standard default is 60Hz if not specified

        # 4. Resolution (4K, 2K / QHD, 1080p FHD, HD)
        if any(k in title_lower for k in ["4k", "uhd", "2160p", "3840x2160"]):
            resolution = "4K UHD"
        elif any(k in title_lower for k in ["2k", "qhd", "1440p", "wqhd", "2560x1440"]):
            resolution = "2K QHD"
        elif any(k in title_lower for k in ["fhd", "1080p", "full hd", "1920x1080"]):
            resolution = "1080p FHD"
        elif any(k in title_lower for k in ["hd", "720p", "1366x768", "1600x900"]):
            resolution = "HD"
        else:
            resolution = "1080p FHD" if size in ["22 Inch", "24 Inch", "27 Inch"] else "Unknown"

        # 5. Panel Features (IPS, Curved, Gaming, Frameless, OLED, VA)
        is_curved = "curved" in title_lower
        is_gaming = "gaming" in title_lower
        is_ips = "ips" in title_lower
        is_frameless = any(f in title_lower for f in ["frameless", "borderless", "bezel-less"])
        
        panel_type = "IPS" if is_ips else ("OLED" if "oled" in title_lower else ("VA" if " va " in title_lower else "Standard"))

        return {
            "brand": detected_brand,
            "size": size,
            "refresh_rate": refresh_rate,
            "resolution": resolution,
            "panel_type": panel_type,
            "is_curved": is_curved,
            "is_gaming": is_gaming,
            "is_frameless": is_frameless
        }

    async def fetch_page(self, client: httpx.AsyncClient, page_num: int, retries=3):
        """Fetches a specific page of monitors with retry logic."""
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
                            specs = self.extract_monitor_specs(title)

                            items.append({
                                "id": ad_id,
                                "title": title,
                                "brand": specs["brand"],
                                "size": specs["size"],
                                "refresh_rate": specs["refresh_rate"],
                                "resolution": specs["resolution"],
                                "panel_type": specs["panel_type"],
                                "is_curved": specs["is_curved"],
                                "is_gaming": specs["is_gaming"],
                                "is_frameless": specs["is_frameless"],
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
                        
                        print(f"  [+] Page {page_num:03d}: Fetched {len(items)} monitors (Attempt {attempt+1})")
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
        print("IKMAN MONITORS PAGINATED EXTRACTION ENGINE")
        print("=" * 65)
        print(f"Target Filter: Computer Accessories -> Monitors")
        print(f"Base URL: {self.base_url}")
        
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            out_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "raw"))
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, "ikman_monitors_all.csv")

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
            print(f"[*] Total Available Monitor Ads: {total_ads:,}")
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
        print(f"Total Monitors Scraped: {len(df_final):,}")
        print(f"Output Saved To: {output_path}")
        print("=" * 65)
        return df_final

def main():
    parser = argparse.ArgumentParser(description="Scrape Monitors from ikman.lk with high-speed async pagination.")
    parser.add_argument("--pages", type=int, default=None, help="Number of pages to scrape (leave empty with --all to scrape all available pages).")
    parser.add_argument("--all", action="store_true", help="Scrape all available monitor ads on ikman (~2,400+ ads).")
    parser.add_argument("--used-only", action="store_true", help="Filter for Used monitors only.")
    parser.add_argument("--concurrent", type=int, default=6, help="Concurrent async requests limit (default: 6).")
    parser.add_argument("--out", type=str, default=None, help="Custom output CSV path.")
    
    args = parser.parse_args()
    max_pages = None if args.all or args.pages is None else args.pages
    
    scraper = IkmanMonitorsScraper(used_only=args.used_only, max_concurrent=args.concurrent)
    asyncio.run(scraper.scrape(max_pages=max_pages, output_path=args.out))

if __name__ == "__main__":
    main()
