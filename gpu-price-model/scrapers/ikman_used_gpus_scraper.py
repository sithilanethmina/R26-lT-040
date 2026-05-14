import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import time
import random  # Added for randomized delays
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from storage_utils import (
    IKMAN_BASE_URL,
    attach_scrape_metadata,
    collect_previous_identities,
    deduplicate_records,
    dated_snapshot_path,
    filter_new_records,
    list_snapshot_files,
    load_json_records,
    normalize_url,
    write_json_records,
    utc_now_iso,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Expanded headers to look more like a real browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://ikman.lk/en/ads/sri-lanka/computer-accessories",
}


RAW_ALL_OUTPUT = PROJECT_ROOT / "data" / "raw" / "ikman_gpus_raw_all.json"
CLEAN_ALL_OUTPUT = PROJECT_ROOT / "data" / "cleaned" / "ikman_gpus_cleaned_all.json"


DETAIL_PAGE_DATA_CACHE = {}


def fix_mojibake(text):
    """Fixes Sinhala text that has been incorrectly decoded as Latin-1."""
    if not text:
        return text
    try:
        # UTF-8 Sinhala characters start with bytes that look like à¶ (E0 B6) or à· (E0 B7) in Latin-1
        if "à¶" in text or "à·" in text or "\u00e0\u00b6" in text or "\u00e0\u00b7" in text:
            return text.encode('latin-1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return text


def scrape_listing_details_from_page(session, listing_url):
    """Visits the detail page to extract the true Listing ID and the full description."""
    if not listing_url:
        return None, None
    if listing_url in DETAIL_PAGE_DATA_CACHE:
        return DETAIL_PAGE_DATA_CACHE[listing_url]

    try:
        response = session.get(listing_url, timeout=20)
        response.encoding = "utf-8"  # Explicitly set encoding to handle Sinhala text
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"Could not load Ikman detail page: {exc}")
        return None, None

    script_match = re.search(
        r"window\.initialData\s*=\s*(.*?)\s*</script>",
        response.text,
        re.DOTALL,
    )
    if not script_match:
        return None, None

    try:
        payload = json.loads(script_match.group(1))
        ad_data = payload.get("adDetail", {}).get("data", {}).get("ad", {})
        
        listing_id = ad_data.get("id")
        description = ad_data.get("description")
        
        # Fallback for ID if structure changed slightly
        if not listing_id:
             id_match = re.search(r'"id"\s*:\s*"([a-f0-9]{24})"', script_match.group(1))
             listing_id = id_match.group(1) if id_match else None

        DETAIL_PAGE_DATA_CACHE[listing_url] = (listing_id, fix_mojibake(description))
        return listing_id, fix_mojibake(description)
    except Exception as e:
        print(f"Error parsing JSON from {listing_url}: {e}")
        return None, None


def extract_listing_identity(card, session):
    link_elem = card.find("a", href=True)
    listing_url = None
    if link_elem:
        listing_url = normalize_url(link_elem["href"], IKMAN_BASE_URL)

    # We now ALWAYS visit the detail page to get the full description for NLP
    listing_id, full_description = scrape_listing_details_from_page(session, listing_url)

    # If detail page failed, try to get ID from card as fallback
    if not listing_id:
        candidate_values = []
        for attr_value in card.attrs.values():
            if isinstance(attr_value, (list, tuple)):
                candidate_values.extend(str(item) for item in attr_value)
            else:
                candidate_values.append(str(attr_value))
        if link_elem:
            candidate_values.append(str(link_elem))
        candidate_values.append(str(card))

        for value in candidate_values:
            match = re.search(r"\b([a-f0-9]{24})\b", value, re.IGNORECASE)
            if match:
                listing_id = match.group(1)
                break

    return listing_id, listing_url, full_description

def scrape_ikman_multiple_pages(max_pages=1):
    all_gpu_data = []
    
    print(f"--- Starting Multi-Page Scrape (Max Pages: {max_pages}) ---")
    
    # 1. Initialize a Session with a retry strategy
    session = requests.Session()
    
    # Configure retries
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    session.headers.update(headers)
    
    for page in range(1, max_pages + 1):
        url = f"https://ikman.lk/en/ads/sri-lanka/computer-accessories?query=vga&page={page}"
        print(f"Scraping page {page}...")
        
        try:
            # 2. Use session.get() with an increased timeout (20 seconds)
            response = session.get(url, timeout=20)
            response.encoding = "utf-8"
            
            if response.status_code != 200:
                print(f"Stopped at page {page}. Status code: {response.status_code}")
                break
                
            soup = BeautifulSoup(response.text, 'html.parser')
            ad_cards = soup.find_all('li', class_='normal--2QYVk') 
            
            if not ad_cards:
                print("No more ads found. We have reached the end.")
                break
                
            for card in ad_cards:
                try:
                    listing_id, listing_url, full_description = extract_listing_identity(card, session)
                    title_elem = card.find('h2', class_='heading--2eONR')
                    title = fix_mojibake(title_elem.text.strip()) if title_elem else None
                    
                    price_elem = card.find('div', class_='price--3SnqI')
                    price_text = price_elem.text.strip() if price_elem else None
                    
                    # Card snippet description (legacy)
                    desc_elem = card.find('div', class_='description--2-ez3')
                    card_description = fix_mojibake(desc_elem.text.strip()) if desc_elem else None

                    if title and price_text:
                        all_gpu_data.append({
                            'Listing_ID': listing_id,
                            'Listing_URL': listing_url,
                            'Raw_Title': title,
                            'Raw_Price': price_text,
                            'Details': card_description,
                            'Full_Description': full_description
                        })
                    
                    # Add a small delay between detail page hits to avoid rate limiting
                    time.sleep(random.uniform(0.5, 1.5))

                except Exception as e:
                    print(f"Error parsing an ad card: {e}")
                    continue
                    
        # 3. Catch connection errors gracefully to save already collected data
        except requests.exceptions.RequestException as e:
            print(f"\n[!] Connection error on page {page}: {e}")
            print("Saving the data we collected so far before exiting...")
            break 
        
        # 4. Randomized scraping delay between 2.5 and 5.5 seconds
        sleep_time = random.uniform(2.5, 5.5)
        time.sleep(sleep_time)
        
    return pd.DataFrame(all_gpu_data)

def clean_gpu_data(df):
    if df.empty:
        print("No data to clean.")
        return df
        
    print("\n--- Starting Data Cleaning ---")
    
    df_clean = df.copy()
    
    df_clean['Clean_Price_LKR'] = (
        df_clean['Raw_Price']
        .str.replace('Rs', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    df_clean['Clean_Price_LKR'] = pd.to_numeric(df_clean['Clean_Price_LKR'], errors='coerce')

    def extract_model(title):
        pattern = r'(RTX|GTX|GT|RX|Radeon|HD|R[79]|ARC)\s?-?[A-Za-z]?\d{3,4}(?:\s?(?:Ti|XT|XTX|SUPER))?'
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            cleaned_model = match.group(0).upper().replace('-', ' ')
            cleaned_model = re.sub(r'([A-Z]+)(\d+)', r'\1 \2', cleaned_model)
            return cleaned_model.strip()
        return None

    df_clean['Extracted_Model'] = df_clean['Raw_Title'].apply(extract_model)

    df_clean = df_clean.dropna(subset=['Extracted_Model', 'Clean_Price_LKR'])
    
    df_clean['VRAM_GB'] = df_clean['Raw_Title'].str.extract(r'(\d+)\s?[Gg][Bb]')
    
    print(f"Scraped {len(df)} total listings. After cleaning, {len(df_clean)} valid GPUs remain.")
    return df_clean


def build_train_ready_records(frame):
    if frame.empty:
        return []

    columns = [
        "Listing_ID",
        "Listing_URL",
        "Raw_Title",
        "Raw_Price",
        "Details",
        "Full_Description",
        "Clean_Price_LKR",
        "Extracted_Model",
        "VRAM_GB",
    ]
    # Use reindex to safely handle cases where columns are missing (e.g. Full_Description in older raw files)
    cleaned = frame.reindex(columns=columns).copy()
    cleaned = cleaned.where(pd.notna(cleaned), None)
    return cleaned.to_dict(orient="records")


def cleaned_snapshot_path_from_raw(raw_snapshot_path):
    cleaned_name = raw_snapshot_path.name.replace("ikman_gpus_raw_", "ikman_gpus_cleaned_")
    return (
        PROJECT_ROOT
        / "data"
        / "cleaned"
        / "ikman"
        / raw_snapshot_path.relative_to(PROJECT_ROOT / "data" / "raw" / "ikman").parent
        / cleaned_name
    )


def rebuild_ikman_datasets():
    all_unique_raw_records = []
    seen_identities = set()

    for raw_snapshot_path in list_snapshot_files(
        project_root=PROJECT_ROOT,
        dataset_kind="raw",
        source="ikman",
    ):
        snapshot_records = load_json_records(raw_snapshot_path)
        snapshot_records, _ = deduplicate_records(
            source="ikman",
            records=snapshot_records,
            allow_legacy_ikman_fallback=True,
        )
        snapshot_records, _ = filter_new_records(
            source="ikman",
            records=snapshot_records,
            known_identities=seen_identities,
            allow_legacy_ikman_fallback=True,
        )
        write_json_records(raw_snapshot_path, snapshot_records)

        cleaned_snapshot_records = build_train_ready_records(
            clean_gpu_data(pd.DataFrame(snapshot_records))
        )
        write_json_records(
            cleaned_snapshot_path_from_raw(raw_snapshot_path),
            cleaned_snapshot_records,
        )
        all_unique_raw_records.extend(snapshot_records)

    write_json_records(RAW_ALL_OUTPUT, all_unique_raw_records)
    all_cleaned_records = build_train_ready_records(
        clean_gpu_data(pd.DataFrame(all_unique_raw_records))
    )
    write_json_records(CLEAN_ALL_OUTPUT, all_cleaned_records)

if __name__ == "__main__":
    raw_df = scrape_ikman_multiple_pages(max_pages=1) 
    
    if not raw_df.empty:
        scraped_at_utc = utc_now_iso()
        raw_snapshot_output = dated_snapshot_path(
            project_root=PROJECT_ROOT,
            dataset_kind="raw",
            source="ikman",
            stem="ikman_gpus_raw",
            scraped_at_utc=scraped_at_utc,
        )
        raw_records = attach_scrape_metadata(
            raw_df.to_dict(orient="records"),
            source="ikman",
            scraped_at_utc=scraped_at_utc,
        )
        raw_records, raw_run_duplicates = deduplicate_records(
            source="ikman",
            records=raw_records,
            allow_legacy_ikman_fallback=True,
        )
        previous_identities = collect_previous_identities(
            project_root=PROJECT_ROOT,
            dataset_kind="raw",
            source="ikman",
            current_snapshot_path=raw_snapshot_output,
            allow_legacy_ikman_fallback=True,
        )
        raw_records, historical_duplicates = filter_new_records(
            source="ikman",
            records=raw_records,
            known_identities=previous_identities,
            allow_legacy_ikman_fallback=True,
        )
        write_json_records(raw_snapshot_output, raw_records)
        rebuild_ikman_datasets()
        print(f"\n[SUCCESS] Exported raw snapshot to '{raw_snapshot_output}'")
        print(f"[SUCCESS] Removed {raw_run_duplicates} duplicate raw listings inside this run")
        print(f"[SUCCESS] Removed {historical_duplicates} listings already seen in previous snapshots")
        print(f"[SUCCESS] Rebuilt '{RAW_ALL_OUTPUT}' and '{CLEAN_ALL_OUTPUT}' from dated snapshots")
    else:
        print("Scraper returned no data. Check your connection or the HTML classes.")
