import argparse
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from storage_utils import (
    attach_scrape_metadata,
    merge_and_deduplicate_records,
    utc_now_iso,
)


BASE_URL = "https://mdcomputers.lk/product-category/graphic-card/used-graphic-card/"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://mdcomputers.lk/",
}


def build_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    return session


def parse_price_lkr(price_text):
    if not price_text:
        return None

    match = re.search(r"(\d[\d,]*\.\d{2}|\d[\d,]*)", price_text)
    if not match:
        return None

    cleaned = match.group(1).replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_model(title):
    if not title:
        return None

    pattern = (
        r"(RTX|GTX|GT|RX|RADEON|HD|ARC)\s?-?[A-Z]?\d{3,4}"
        r"(?:\s?(?:TI|XT|XTX|SUPER))?"
    )
    match = re.search(pattern, title, re.IGNORECASE)
    if not match:
        return None

    cleaned_model = match.group(0).upper().replace("-", " ")
    cleaned_model = re.sub(r"([A-Z]+)\s?(\d+)", r"\1 \2", cleaned_model)
    return re.sub(r"\s+", " ", cleaned_model).strip()


def detect_total_pages(soup):
    page_numbers = []
    for link in soup.select("nav.woocommerce-pagination a"):
        href = link.get("href", "")
        match = re.search(r"/page/(\d+)/", href)
        if match:
            page_numbers.append(int(match.group(1)))
    return max(page_numbers) if page_numbers else 1


def parse_product_card(card):
    classes = set(card.get("class", []))
    title_link = card.select_one("h2.woo-loop-product__title a")
    image = card.select_one(".mf-product-thumbnail img")
    brand_link = card.select_one(".meta-brand a")
    price_container = card.select_one(".mf-product-price-box .price")
    price_amount = card.select_one(".mf-product-price-box .woocommerce-Price-amount")
    add_to_cart = card.select_one("a.add_to_cart_button")

    title = title_link.get_text(" ", strip=True) if title_link else None
    if price_amount:
        price_text = price_amount.get_text(" ", strip=True)
    else:
        price_text = price_container.get_text(" ", strip=True) if price_container else None
    product_url = title_link.get("href") if title_link else None

    product_id = None
    if add_to_cart:
        product_id = add_to_cart.get("data-product_id")

    if "outofstock" in classes:
        stock_text = "OUT OF STOCK"
    elif "instock" in classes:
        stock_text = "IN STOCK"
    else:
        stock_text = "UNKNOWN"

    return {
        "Product_ID": product_id,
        "Raw_Title": title,
        "Raw_Price": price_text,
        "Price_LKR": parse_price_lkr(price_text),
        "Category": "GRAPHIC CARD (VGA)",
        "Stock": stock_text,
        "Stock_Status": stock_text,
        "Product_URL": product_url,
        "Image_URL": image.get("src") if image else None,
        "Brand": brand_link.get_text(" ", strip=True) if brand_link else None,
    }


def scrape_md_multiple_pages(max_pages=None, delay_min=1.5, delay_max=3.0):
    max_pages_label = "auto" if max_pages is None else str(max_pages)
    print(f"--- Starting MD scrape (max pages: {max_pages_label}) ---")

    session = build_session()
    all_gpu_data = []
    seen_urls = set()
    page = 1
    resolved_max_pages = max_pages

    while True:
        if resolved_max_pages is not None and page > resolved_max_pages:
            break

        url = BASE_URL if page == 1 else f"{BASE_URL}page/{page}/"
        print(f"Scraping page {page}: {url}")

        try:
            response = session.get(url, timeout=20)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            print(f"[!] Request failed on page {page}: {exc}")
            print("Stopping and keeping data collected so far.")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        cards = soup.select("li.product")

        if not cards:
            print("No product cards found. Reached the end of the category.")
            break

        if page == 1 and resolved_max_pages is None:
            resolved_max_pages = detect_total_pages(soup)
            print(f"Detected approximately {resolved_max_pages} pages.")

        new_items_on_page = 0
        for card in cards:
            item = parse_product_card(card)
            if not item["Product_URL"] or item["Product_URL"] in seen_urls:
                continue

            seen_urls.add(item["Product_URL"])
            all_gpu_data.append(item)
            new_items_on_page += 1

        print(f"Collected {new_items_on_page} new items from page {page}.")

        if new_items_on_page == 0:
            print("No new items were found on this page. Stopping pagination.")
            break

        time.sleep(random.uniform(delay_min, delay_max))
        page += 1

    return pd.DataFrame(all_gpu_data)


def clean_gpu_data(df):
    if df.empty:
        print("No data to clean.")
        return df

    print("\n--- Cleaning MD GPU data ---")
    df_clean = df.copy()

    df_clean["Extracted_Model"] = df_clean["Raw_Title"].apply(extract_model)
    df_clean["VRAM_GB"] = df_clean["Raw_Title"].str.extract(r"(\d+)\s?[Gg][Bb]")
    df_clean["Manufacturer"] = df_clean["Raw_Title"].str.extract(
        r"\b(ASUS|MSI|GIGABYTE|ZOTAC|GALAX|PALIT|SAPPHIRE|EMTEK|FORSA|RANDOM BRAND)\b",
        expand=False,
    )
    df_clean["Stock"] = df_clean["Stock_Status"].fillna("UNKNOWN")

    df_clean = df_clean.dropna(subset=["Extracted_Model", "Price_LKR"])

    print(f"Scraped {len(df)} raw listings. {len(df_clean)} listings remain after cleaning.")
    return df_clean


def main():
    parser = argparse.ArgumentParser(
        description="Scrape used GPU listings from MD Computers."
    )
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--raw-output", default=str(PROJECT_ROOT / "data" / "raw" / "md_gpus_raw_all.json"))
    parser.add_argument("--clean-output", default=str(PROJECT_ROOT / "data" / "cleaned" / "md_gpus_cleaned_all.json"))
    parser.add_argument("--delay-min", type=float, default=1.5)
    parser.add_argument("--delay-max", type=float, default=3.0)
    args = parser.parse_args()

    raw_df = scrape_md_multiple_pages(
        max_pages=args.max_pages,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
    )

    if raw_df.empty:
        print("Scraper returned no data. Check the HTML structure or connectivity.")
        return

    scraped_at_utc = utc_now_iso()
    raw_records = attach_scrape_metadata(
        raw_df.to_dict(orient="records"),
        source="md",
        scraped_at_utc=scraped_at_utc,
    )
    _, raw_removed = merge_and_deduplicate_records(
        source="md",
        output_path=Path(args.raw_output),
        new_records=raw_records,
    )
    print(f"[SUCCESS] Merged raw data into '{args.raw_output}' ({raw_removed} duplicates removed)")

    cleaned_df = clean_gpu_data(raw_df)
    cleaned_records = attach_scrape_metadata(
        cleaned_df.to_dict(orient="records"),
        source="md",
        scraped_at_utc=scraped_at_utc,
    )
    _, clean_removed = merge_and_deduplicate_records(
        source="md",
        output_path=Path(args.clean_output),
        new_records=cleaned_records,
    )
    print(f"[SUCCESS] Merged cleaned data into '{args.clean_output}' ({clean_removed} duplicates removed)")


if __name__ == "__main__":
    main()
