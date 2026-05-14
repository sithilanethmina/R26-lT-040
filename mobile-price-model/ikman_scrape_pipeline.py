"""
Scrape Ikman mobile-phone listings, merge them into the raw dataset, and retrain.

The downstream training script already owns cleaning, duplicate removal,
predefined phone-spec corrections, ML-ready JSON export, and model artifacts.
This pipeline adds the missing upstream step:

1. Read search result pages from https://ikman.lk/en/ads/sri-lanka/mobiles.
2. Visit each ad detail page and extract structured window.initialData JSON.
3. Keep only Mobile Phones ads, normalize the fields used by the trainer.
4. Merge records into ikman_mobile_phones_processed.json without duplicates.
5. Run train_mobile_price_models.py so ML-ready data and models refresh.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LISTINGS_URL = "https://ikman.lk/en/ads/sri-lanka/mobiles"
RAW_DATA_FILE = BASE_DIR / "ikman_mobile_phones_processed.json"
TRAINING_SCRIPT = BASE_DIR / "train_mobile_price_models.py"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

INITIAL_DATA_PATTERN = re.compile(r"window\.initialData\s*=\s*(\{.*?\})\s*</script>", re.DOTALL)
MOBILE_PHONE_CATEGORY_NAMES = {"mobile phones"}
KNOWN_STORAGE_GB_VALUES = {8, 16, 32, 64, 128, 256, 512, 1024, 2048}
WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


@dataclass
class MergeStats:
    existing_records: int
    scraped_records: int
    final_records: int
    added_records: int
    updated_records: int
    duplicate_existing_records: int
    duplicate_scraped_records: int


@dataclass
class ScrapeStats:
    search_pages_read: int = 0
    listing_urls_found: int = 0
    unique_listing_urls: int = 0
    detail_pages_read: int = 0
    mobile_phone_records: int = 0
    skipped_non_phone_ads: int = 0
    failed_detail_pages: int = 0


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_search_url(base_url: str, page: int) -> str:
    parsed = urlparse(base_url)
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key not in {"page", "device"}
    ]

    # Ikman can return an unsupported lightweight page without this hint.
    query_items.append(("device", "desktop"))
    if page > 1:
        query_items.append(("page", str(page)))

    return urlunparse(parsed._replace(query=urlencode(query_items)))


def fetch_text(url: str, timeout: float, retries: int) -> str:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    for attempt in range(1, retries + 2):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="replace")
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt > retries:
                raise
            wait_seconds = min(10.0, 1.5 * attempt)
            logging.warning(
                "Request failed (%s). Retrying in %.1fs: %s",
                exc,
                wait_seconds,
                url,
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"Failed to fetch URL after retries: {url}")


def extract_initial_data(html_text: str, url: str) -> dict[str, Any]:
    match = INITIAL_DATA_PATTERN.search(html_text)
    if not match:
        raise ValueError(f"Could not find window.initialData in {url}")
    return json.loads(match.group(1))


def extract_search_ads(initial_data: dict[str, Any]) -> list[dict[str, Any]]:
    ads_state = initial_data.get("serp", {}).get("ads", {})
    if ads_state.get("type") != "Success":
        return []

    data = ads_state.get("data", {})
    ads: list[dict[str, Any]] = []
    for collection_name in ("topAds", "ads"):
        collection = data.get(collection_name, [])
        if isinstance(collection, list):
            ads.extend(item for item in collection if isinstance(item, dict))

    return dedupe_dicts_by_key(ads, lambda ad: str(ad.get("id") or ad.get("slug") or ad.get("url") or ""))


def build_detail_url(summary: dict[str, Any], search_url: str) -> Optional[str]:
    slug = summary.get("slug")
    if slug:
        return urljoin(search_url, f"/en/ad/{slug}")

    url = summary.get("url")
    if url:
        return urljoin(search_url, str(url))

    return None


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


def dedupe_dicts_by_key(
    values: Iterable[dict[str, Any]],
    key_func: Any,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique_values: list[dict[str, Any]] = []
    for value in values:
        key = key_func(value)
        if not key or key in seen:
            continue
        seen.add(key)
        unique_values.append(value)
    return unique_values


def extract_detail_ad(initial_data: dict[str, Any]) -> Optional[dict[str, Any]]:
    detail_state = initial_data.get("adDetail", {})
    if detail_state.get("type") != "Success":
        return None
    ad = detail_state.get("data", {}).get("ad")
    return ad if isinstance(ad, dict) else None


def normalize_property_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def build_property_map(ad: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for item in ad.get("properties", []) or []:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        for raw_key in (item.get("key"), item.get("label")):
            key = normalize_property_key(raw_key)
            if key:
                properties[key] = value
    return properties


def parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).replace(",", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def parse_price_amount(value: Any) -> Optional[float]:
    price = parse_number(value)
    if price is None or price <= 0:
        return None
    return price


def infer_currency(value: Any) -> str:
    text = str(value or "").lower()
    if "rs" in text or "lkr" in text:
        return "LKR"
    return "LKR"


def extract_gb_candidates(text: str) -> list[float]:
    candidates: list[float] = []
    lowered = text.lower()

    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*tb\b", lowered):
        candidates.append(float(match.group(1)) * 1024)

    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*gb\b", lowered):
        candidates.append(float(match.group(1)))

    return candidates


def extract_storage_gb(*values: Any) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        candidates = [
            candidate
            for candidate in extract_gb_candidates(str(value))
            if int(candidate) in KNOWN_STORAGE_GB_VALUES or candidate >= 16
        ]
        if candidates:
            return max(candidates)
    return None


def extract_ram_gb(*values: Any) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        text = str(value).lower()

        ram_matches = [
            float(match.group(1))
            for match in re.finditer(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|r\b)", text)
        ]
        ram_matches.extend(
            float(match.group(1))
            for match in re.finditer(r"\bram\s*[:=/|-]?\s*(\d+(?:\.\d+)?)\s*gb\b", text)
        )
        if ram_matches:
            plausible = [value for value in ram_matches if 0 < value <= 24]
            if plausible:
                return max(plausible)

        slash_match = re.search(r"\b(\d+(?:\.\d+)?)\s*gb\s*[/|+,-]\s*(\d{2,4})\s*gb\b", text)
        if slash_match:
            ram_value = float(slash_match.group(1))
            if 0 < ram_value <= 24:
                return ram_value

    return None


def parse_boolean_feature(value: Any, true_markers: tuple[str, ...], false_markers: tuple[str, ...] = ()) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).lower()
    if any(marker in text for marker in false_markers):
        return False
    if any(marker in text for marker in true_markers):
        return True
    return None


def bool_to_float(value: Optional[bool]) -> Optional[float]:
    if value is None:
        return None
    return float(int(value))


def number_word_to_float(value: str) -> Optional[float]:
    if value.isdigit():
        return float(value)
    return float(WORD_NUMBERS[value]) if value in WORD_NUMBERS else None


def extract_warranty_days(*values: Any) -> Optional[float]:
    text = "\n".join(str(value) for value in values if value)
    if not text:
        return None

    lowered = text.lower()
    if "no warranty" in lowered or "without warranty" in lowered:
        return 0.0

    token_pattern = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    unit_pattern = r"(year|yr|month|mon|week|day)s?"
    matches = list(re.finditer(rf"{token_pattern}\s*{unit_pattern}", lowered))
    warranty_days: list[float] = []

    for match in matches:
        context = lowered[max(0, match.start() - 35) : match.end() + 35]
        if "warranty" not in context:
            continue
        number = number_word_to_float(match.group(1))
        if number is None:
            continue
        unit = match.group(2)
        if unit in {"year", "yr"}:
            warranty_days.append(number * 365)
        elif unit in {"month", "mon"}:
            warranty_days.append(number * 30)
        elif unit == "week":
            warranty_days.append(number * 7)
        elif unit == "day":
            warranty_days.append(number)

    if warranty_days:
        return max(warranty_days)
    return None


def is_mobile_phone_ad(ad: dict[str, Any]) -> bool:
    category = ad.get("category") or {}
    category_name = str(category.get("name") or "").strip().lower()
    return category_name in MOBILE_PHONE_CATEGORY_NAMES


def nested_name(value: Any) -> Optional[str]:
    return str(value.get("name")) if isinstance(value, dict) and value.get("name") else None


def build_location(ad: dict[str, Any]) -> Optional[str]:
    area_name = nested_name(ad.get("area"))
    location = ad.get("location") or {}
    location_name = nested_name(location)
    parent_name = nested_name(location.get("parent")) if isinstance(location, dict) else None

    parts = [part for part in (area_name, location_name, parent_name) if part]
    if not parts:
        return None

    unique_parts = dedupe_preserve_order(parts)
    return ", ".join(unique_parts)


def seller_name(ad: dict[str, Any]) -> Optional[str]:
    shop = ad.get("shop") or {}
    if isinstance(shop, dict) and shop.get("name"):
        return str(shop["name"])

    contact_card = ad.get("contactCard") or {}
    if isinstance(contact_card, dict) and contact_card.get("name"):
        return str(contact_card["name"])

    return None


def seller_type(ad: dict[str, Any]) -> str:
    if ad.get("shop"):
        return "shop"
    if ad.get("isMember"):
        return "member"
    return "private"


def extract_record_from_detail(ad: dict[str, Any], detail_url: str) -> Optional[dict[str, Any]]:
    if not is_mobile_phone_ad(ad):
        return None

    properties = build_property_map(ad)
    title = ad.get("title")
    description = ad.get("description")
    edition = properties.get("edition")
    money = ad.get("money") or {}
    price_label = money.get("amount") if isinstance(money, dict) else None
    listed_price = parse_price_amount(price_label)
    if listed_price is None:
        logging.debug("Skipping ad without a parseable price: %s", detail_url)
        return None

    storage_gb = extract_storage_gb(properties.get("memory"), properties.get("storage"), edition, title, description)
    ram_gb = extract_ram_gb(properties.get("ram"), edition, title, description)
    network_type = properties.get("network")
    sim_support = properties.get("sim_support")

    has_5g = parse_boolean_feature(network_type, true_markers=("5g",))
    if has_5g is None:
        has_5g = parse_boolean_feature(f"{title} {edition} {description}", true_markers=("5g",))

    dual_sim = parse_boolean_feature(
        sim_support,
        true_markers=("dual",),
        false_markers=("single",),
    )
    if dual_sim is None:
        dual_sim = parse_boolean_feature(f"{title} {edition} {description}", true_markers=("dual sim", "dualsim"))

    has_esim = parse_boolean_feature(
        f"{title} {edition} {description}",
        true_markers=("esim", "e-sim", "e sim"),
    )

    canonical_url = urljoin("https://ikman.lk", f"/en/ad/{ad.get('slug')}") if ad.get("slug") else detail_url
    warranty_days = extract_warranty_days(description, title, edition)
    pta_approved = parse_boolean_feature(
        f"{title} {description}",
        true_markers=("pta approved", "trc approved"),
    )

    return {
        "brand": properties.get("brand"),
        "condition": properties.get("condition"),
        "currency": infer_currency(price_label),
        "dual_sim": bool_to_float(dual_sim),
        "has_5g": bool_to_float(has_5g),
        "has_esim": bool_to_float(has_esim),
        "is_verified": bool(ad.get("isVerified")) if ad.get("isVerified") is not None else None,
        "listed_price": listed_price,
        "listing_id": ad.get("id"),
        "location": build_location(ad),
        "model": properties.get("model"),
        "negotiable": bool(money.get("negotiable")) if isinstance(money, dict) else None,
        "network_type": network_type,
        "posted_at": ad.get("adDate"),
        "pta_approved": bool_to_float(pta_approved),
        "ram_gb": ram_gb,
        "seller_name": seller_name(ad),
        "seller_type": seller_type(ad),
        "sim_support": sim_support,
        "storage_gb": storage_gb,
        "title": title,
        "url": canonical_url,
        "warranty_days": warranty_days,
        "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def scrape_ikman_records(
    listings_url: str,
    pages: int,
    delay_seconds: float,
    timeout: float,
    retries: int,
    detail_limit: Optional[int] = None,
) -> tuple[list[dict[str, Any]], ScrapeStats]:
    stats = ScrapeStats()
    detail_urls: list[str] = []

    for page in range(1, pages + 1):
        search_url = build_search_url(listings_url, page)
        logging.info("Reading search page %s/%s: %s", page, pages, search_url)
        html_text = fetch_text(search_url, timeout=timeout, retries=retries)
        stats.search_pages_read += 1
        initial_data = extract_initial_data(html_text, search_url)
        summaries = extract_search_ads(initial_data)
        page_urls = [build_detail_url(summary, search_url) for summary in summaries]
        page_urls = [url for url in page_urls if url]
        detail_urls.extend(page_urls)
        logging.info("Found %s listing URLs on page %s.", f"{len(page_urls):,}", page)
        if delay_seconds > 0 and page < pages:
            time.sleep(delay_seconds)

    stats.listing_urls_found = len(detail_urls)
    detail_urls = dedupe_preserve_order(detail_urls)
    if detail_limit is not None:
        detail_urls = detail_urls[:detail_limit]
    stats.unique_listing_urls = len(detail_urls)

    records: list[dict[str, Any]] = []
    for index, detail_url in enumerate(detail_urls, start=1):
        logging.info("Reading ad detail %s/%s: %s", index, len(detail_urls), detail_url)
        try:
            html_text = fetch_text(detail_url, timeout=timeout, retries=retries)
            stats.detail_pages_read += 1
            initial_data = extract_initial_data(html_text, detail_url)
            ad = extract_detail_ad(initial_data)
            if not ad:
                stats.failed_detail_pages += 1
                logging.warning("No ad detail payload found: %s", detail_url)
                continue

            record = extract_record_from_detail(ad, detail_url)
            if record is None:
                stats.skipped_non_phone_ads += 1
                continue

            records.append(record)
            stats.mobile_phone_records += 1
        except Exception as exc:
            stats.failed_detail_pages += 1
            logging.warning("Skipping detail page after error: %s | %s", detail_url, exc)

        if delay_seconds > 0 and index < len(detail_urls):
            time.sleep(delay_seconds)

    return records, stats


def load_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

    raise ValueError(f"Unsupported JSON dataset structure: {path}")


def normalized_url_key(value: Any) -> Optional[str]:
    if not value:
        return None
    parsed = urlparse(str(value))
    path = parsed.path.rstrip("/").lower()
    if not path:
        return None
    return path


def record_key(record: dict[str, Any]) -> str:
    listing_id = str(record.get("listing_id") or record.get("id") or "").strip()
    if listing_id:
        return f"id:{listing_id}"

    url_key = normalized_url_key(record.get("url"))
    if url_key:
        return f"url:{url_key}"

    fallback_parts = [
        str(record.get("title") or "").strip().lower(),
        str(record.get("location") or "").strip().lower(),
        str(record.get("listed_price") or "").strip().lower(),
    ]
    return "fallback:" + "|".join(fallback_parts)


def has_value(value: Any) -> bool:
    return value is not None and value != ""


def merge_prefer_new_non_empty(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    merged = dict(old)
    for key, value in new.items():
        if has_value(value):
            merged[key] = value
        elif key not in merged:
            merged[key] = value
    return merged


def merge_records(existing_records: list[dict[str, Any]], scraped_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], MergeStats]:
    records_by_key: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []

    for record in existing_records:
        key = record_key(record)
        if key not in records_by_key:
            ordered_keys.append(key)
        records_by_key[key] = merge_prefer_new_non_empty(records_by_key.get(key, {}), record)

    duplicate_existing_records = len(existing_records) - len(ordered_keys)
    added_records = 0
    updated_records = 0
    duplicate_scraped_records = 0
    seen_scraped_keys: set[str] = set()

    for record in scraped_records:
        key = record_key(record)
        if key in seen_scraped_keys:
            duplicate_scraped_records += 1
        seen_scraped_keys.add(key)

        if key in records_by_key:
            merged = merge_prefer_new_non_empty(records_by_key[key], record)
            if merged != records_by_key[key]:
                updated_records += 1
            records_by_key[key] = merged
        else:
            added_records += 1
            ordered_keys.append(key)
            records_by_key[key] = record

    merged_records = [records_by_key[key] for key in ordered_keys]
    stats = MergeStats(
        existing_records=len(existing_records),
        scraped_records=len(scraped_records),
        final_records=len(merged_records),
        added_records=added_records,
        updated_records=updated_records,
        duplicate_existing_records=duplicate_existing_records,
        duplicate_scraped_records=duplicate_scraped_records,
    )
    return merged_records, stats


def save_json_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)
    logging.info("Saved merged raw dataset: %s", path)


def run_training() -> None:
    logging.info("Starting automatic model training: %s", TRAINING_SCRIPT.name)
    subprocess.run([sys.executable, str(TRAINING_SCRIPT)], cwd=BASE_DIR, check=True)
    logging.info("Automatic model training completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Ikman mobile-phone ads, update the raw dataset, and retrain models.",
    )
    parser.add_argument("--url", default=DEFAULT_LISTINGS_URL, help="Ikman listings URL to scrape.")
    parser.add_argument("--pages", type=int, default=2, help="Number of search pages to scrape.")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per request.")
    parser.add_argument("--detail-limit", type=int, default=None, help="Optional cap for detail pages, useful for tests.")
    parser.add_argument("--output", type=Path, default=RAW_DATA_FILE, help="Raw processed dataset JSON path.")
    parser.add_argument("--skip-train", action="store_true", help="Only scrape and merge; do not retrain.")
    parser.add_argument("--dry-run", action="store_true", help="Scrape and report stats without writing or training.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    os.chdir(BASE_DIR)

    if args.pages < 1:
        raise ValueError("--pages must be at least 1")
    if args.detail_limit is not None and args.detail_limit < 1:
        raise ValueError("--detail-limit must be at least 1 when provided")

    scraped_records, scrape_stats = scrape_ikman_records(
        listings_url=args.url,
        pages=args.pages,
        delay_seconds=max(0.0, args.delay),
        timeout=args.timeout,
        retries=max(0, args.retries),
        detail_limit=args.detail_limit,
    )

    existing_records = load_json_records(args.output)
    merged_records, merge_stats = merge_records(existing_records, scraped_records)

    logging.info(
        "Scrape summary | pages=%s urls=%s unique_urls=%s details=%s phones=%s non_phone=%s failed=%s",
        scrape_stats.search_pages_read,
        scrape_stats.listing_urls_found,
        scrape_stats.unique_listing_urls,
        scrape_stats.detail_pages_read,
        scrape_stats.mobile_phone_records,
        scrape_stats.skipped_non_phone_ads,
        scrape_stats.failed_detail_pages,
    )
    logging.info(
        "Merge summary  | existing=%s scraped=%s added=%s updated=%s final=%s existing_dupes=%s scraped_dupes=%s",
        merge_stats.existing_records,
        merge_stats.scraped_records,
        merge_stats.added_records,
        merge_stats.updated_records,
        merge_stats.final_records,
        merge_stats.duplicate_existing_records,
        merge_stats.duplicate_scraped_records,
    )

    if args.dry_run:
        logging.info("Dry run selected; no files were written and training was skipped.")
        return

    save_json_records(args.output, merged_records)

    if args.skip_train:
        logging.info("Skipping training because --skip-train was provided.")
        return

    run_training()


if __name__ == "__main__":
    main()
