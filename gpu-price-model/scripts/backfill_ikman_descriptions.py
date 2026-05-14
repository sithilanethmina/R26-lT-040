import json
import re
import time
import random
import requests
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "cleaned" / "all_scraped_data.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

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

def extract_full_description(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=25)
        response.encoding = "utf-8"  # Ensure Sinhala text is correctly decoded
        response.raise_for_status()
        
        script_match = re.search(r"window\.initialData\s*=\s*(.*?)\s*</script>", response.text, re.DOTALL)
        if not script_match:
            return None
            
        data = json.loads(script_match.group(1))
        description = data.get("adDetail", {}).get("data", {}).get("ad", {}).get("description")
        
        # Apply mojibake fix to the description immediately
        return fix_mojibake(description)
    except Exception as e:
        print(f"\n[Error] Failed for {url}: {e}")
        return None

def main():
    if not DATA_FILE.exists():
        print(f"File not found: {DATA_FILE}")
        return

    print(f"Loading data from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    ikman_listings = [item for item in data if item.get("Source") == "ikman"]
    print(f"Found {len(ikman_listings)} Ikman listings.")

    updated_count = 0
    for item in tqdm(ikman_listings, desc="Backfilling descriptions"):
        url = item.get("Listing_URL")
        if not url:
            continue

        # Check if we already have a full description and if it needs fixing
        current_desc = item.get("Full_Description")
        
        # 1. Fix Mojibake in existing description if present
        if current_desc:
            fixed_desc = fix_mojibake(current_desc)
            if fixed_desc != current_desc:
                item["Full_Description"] = fixed_desc
                updated_count += 1
                # If we fixed it, we might not need to re-fetch unless it's still weird
                # But to be safe, let's keep going to the re-fetch logic if it was really bad
        
        # 2. Re-fetch if missing or if it still looks like Mojibake
        needs_fetch = not item.get("Full_Description") or "à¶" in str(item.get("Full_Description")) or "à·" in str(item.get("Full_Description"))
        
        if needs_fetch:
            # print(f"\n[Info] Fetching/Fixing description for: {url}") # Kept quiet for tqdm
            desc = extract_full_description(url)
            if desc:
                item["Full_Description"] = desc
                updated_count += 1
            
            # Random delay to be polite to Ikman
            time.sleep(random.uniform(1.0, 2.5))
        
        # Save every 50 updates to prevent data loss
        if updated_count > 0 and updated_count % 50 == 0:
            with open(DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    # Final save
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Updated/Backfilled {updated_count} descriptions.")

if __name__ == "__main__":
    main()
