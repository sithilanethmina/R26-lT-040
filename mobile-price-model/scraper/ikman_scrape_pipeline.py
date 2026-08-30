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
import math
import os
import random
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
from http.client import IncompleteRead


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LISTINGS_URL = "https://ikman.lk/en/ads/sri-lanka/mobile-phones"
RAW_DATA_FILE = BASE_DIR / "data" / "raw" / "ikman_mobile_phones_processed.json"
DEFAULT_URLS_FILE = BASE_DIR / "data" / "raw" / "ikman_listing_urls.json"
TRAINING_SCRIPT = BASE_DIR / "src" / "train.py"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

INITIAL_DATA_PATTERN = re.compile(r"window\.initialData\s*=\s*(\{.*?\})\s*</script>", re.DOTALL)
MOBILE_PHONE_CATEGORY_NAMES = {"mobile phones", "mobiles", "phones"}
INCREMENTAL_SAVE_INTERVAL = 25  # Save progress every N detail pages
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
        except Exception as exc:
            if attempt > retries:
                raise
            wait_seconds = min(15.0, 2.0 * attempt)
            logging.warning(
                "Request failed (%s: %s). Retry %s/%s in %.1fs: %s",
                type(exc).__name__,
                exc,
                attempt,
                retries,
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


def extract_pagination_data(initial_data: dict[str, Any]) -> dict[str, int]:
    """Extract pagination metadata from window.initialData to determine total pages."""
    ads_state = initial_data.get("serp", {}).get("ads", {})
    if ads_state.get("type") != "Success":
        return {}
    data = ads_state.get("data", {})
    pagination = data.get("paginationData", {})
    return {
        "total": int(pagination.get("total", 0)),
        "pageSize": int(pagination.get("pageSize", 25)),
        "activePage": int(pagination.get("activePage", 1)),
    }


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


def extract_battery_health_percent(*values: Any) -> Optional[float]:
    text = "\n".join(str(value) for value in values if value)
    if not text:
        return None

    lowered = text.lower()
    candidates: list[float] = []

    for match in re.finditer(r"(?:battery\s*health|bh)\s*[:=\-]?\s*(\d{1,3})(?:\s*%)?", lowered):
        value = float(match.group(1))
        if 0.0 <= value <= 100.0:
            candidates.append(value)

    for match in re.finditer(r"(\d{1,3})\s*%\s*(?:battery\s*health|bh)", lowered):
        value = float(match.group(1))
        if 0.0 <= value <= 100.0:
            candidates.append(value)

    for match in re.finditer(r"battery\s*[:=\-]?\s*(\d{1,3})\s*%", lowered):
        value = float(match.group(1))
        if 0.0 <= value <= 100.0:
            candidates.append(value)

    if not candidates:
        return None

    return max(candidates)


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
    condition_raw = (properties.get("condition") or "").strip().lower()
    if condition_raw != "used":
        logging.debug("Skipping non-used ad (condition=%s): %s", properties.get("condition"), detail_url)
        return None
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
    battery_health_percent = extract_battery_health_percent(
        properties.get("battery_health"),
        properties.get("battery_condition"),
        properties.get("battery"),
        title,
        edition,
        description,
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
        "battery_health_percent": battery_health_percent,
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


def delay_with_jitter(base_delay: float) -> None:
    """Sleep with random jitter (0.5x to 1.5x base delay) to appear more human-like."""
    if base_delay <= 0:
        return
    jittered = base_delay * random.uniform(0.5, 1.5)
    time.sleep(jittered)


def save_urls_file(path: Path, urls: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(urls, file, indent=2, ensure_ascii=False)
    logging.info("Saved %s listing URLs to %s", f"{len(urls):,}", path)


def load_urls_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        return [str(u) for u in data if isinstance(u, str)]
    return []


def collect_listing_urls(
    listings_url: str,
    pages: int,
    delay_seconds: float,
    timeout: float,
    retries: int,
    urls_file: Optional[Path] = DEFAULT_URLS_FILE,
) -> tuple[list[str], ScrapeStats]:
    """Collect listing URLs across search pages and save them to disk incrementally."""
    stats = ScrapeStats()
    detail_urls: list[str] = []
    
    # Load existing URLs if any to append/dedupe
    if urls_file and urls_file.exists():
        existing_urls = load_urls_file(urls_file)
        if existing_urls:
            detail_urls.extend(existing_urls)
            logging.info("Loaded %s pre-existing URLs from %s", f"{len(existing_urls):,}", urls_file)

    total_pages = pages
    auto_detect = pages <= 0

    page = 0
    consecutive_failures = 0
    max_consecutive_failures = 5
    consecutive_empty_pages = 0
    max_consecutive_empty_pages = 20

    try:
        while True:
            page += 1
            if not auto_detect and page > total_pages:
                break

            search_url = build_search_url(listings_url, page)
            page_label = f"{page}/{total_pages:,}" if not auto_detect or total_pages < 99999 else f"{page}/?"
            logging.info("Reading search page %s: %s", page_label, search_url)

            try:
                html_text = fetch_text(search_url, timeout=timeout, retries=retries)
            except Exception as exc:
                consecutive_failures += 1
                logging.warning(
                    "Failed to fetch search page %s (%s/%s consecutive failures): %s",
                    page, consecutive_failures, max_consecutive_failures, exc,
                )
                if consecutive_failures >= max_consecutive_failures:
                    logging.error("Too many consecutive failures. Stopping search phase.")
                    break
                delay_with_jitter(delay_seconds * 2)
                continue

            consecutive_failures = 0
            stats.search_pages_read += 1
            initial_data = extract_initial_data(html_text, search_url)

            if page == 1 and auto_detect:
                pagination = extract_pagination_data(initial_data)
                total_ads = pagination.get("total", 0)
                page_size = pagination.get("pageSize", 25)
                if total_ads > 0 and page_size > 0:
                    total_pages = math.ceil(total_ads / page_size)
                    logging.info(
                        "Auto-detected pagination: %s total ads, %s per page, %s pages",
                        f"{total_ads:,}", page_size, f"{total_pages:,}",
                    )
                else:
                    logging.warning("Could not auto-detect pagination. Will stop when no ads found.")
                    total_pages = 99999

            summaries = extract_search_ads(initial_data)
            if not summaries:
                consecutive_empty_pages += 1
                if total_pages < 99999 and page < total_pages:
                    logging.debug("Empty page %s — continuing since total_pages=%s.", page, total_pages)
                elif consecutive_empty_pages >= max_consecutive_empty_pages:
                    logging.info("%s consecutive empty pages at page %s — stopping search.", consecutive_empty_pages, page)
                    break
                delay_with_jitter(delay_seconds * 0.3)
                continue

            consecutive_empty_pages = 0
            page_urls = [build_detail_url(summary, search_url) for summary in summaries]
            page_urls = [url for url in page_urls if url]
            detail_urls.extend(page_urls)
            logging.info("Found %s listing URLs on page %s.", f"{len(page_urls):,}", page)

            # Save URLs incrementally every 20 pages
            if urls_file and page % 20 == 0:
                saved_deduped = dedupe_preserve_order(detail_urls)
                save_urls_file(urls_file, saved_deduped)

            if auto_detect and page >= total_pages:
                logging.info("Reached last page (%s).", total_pages)
                break

            delay_with_jitter(delay_seconds)
    except KeyboardInterrupt:
        logging.info("URL collection interrupted by user. Saving collected URLs...")
    finally:
        stats.listing_urls_found = len(detail_urls)
        detail_urls = dedupe_preserve_order(detail_urls)
        stats.unique_listing_urls = len(detail_urls)
        if urls_file:
            save_urls_file(urls_file, detail_urls)

    return detail_urls, stats


def scrape_details_from_urls(
    detail_urls: list[str],
    delay_seconds: float,
    timeout: float,
    retries: int,
    detail_limit: Optional[int] = None,
    output_path: Optional[Path] = None,
    skip_existing: bool = True,
) -> tuple[list[dict[str, Any]], ScrapeStats]:
    """Visit each detail URL, extract phone attributes, and incrementally save records."""
    stats = ScrapeStats()
    detail_urls = dedupe_preserve_order(detail_urls)
    
    # Filter out URLs already in output file
    if skip_existing and output_path and output_path.exists():
        existing_records = load_json_records(output_path)
        existing_urls = {
            normalized_url_key(r.get("url"))
            for r in existing_records
            if r.get("url")
        }
        initial_count = len(detail_urls)
        detail_urls = [
            u for u in detail_urls
            if normalized_url_key(u) not in existing_urls
        ]
        logging.info(
            "Filtered out %s already scraped URLs. %s remaining to scrape.",
            f"{initial_count - len(detail_urls):,}",
            f"{len(detail_urls):,}",
        )

    if detail_limit is not None:
        detail_urls = detail_urls[:detail_limit]
    stats.unique_listing_urls = len(detail_urls)

    records: list[dict[str, Any]] = []
    try:
        for index, detail_url in enumerate(detail_urls, start=1):
            logging.info("Reading ad detail %s/%s: %s", f"{index:,}", f"{len(detail_urls):,}", detail_url)
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

            # Incremental save every N detail pages to prevent data loss
            if output_path and index % INCREMENTAL_SAVE_INTERVAL == 0 and records:
                _incremental_save(output_path, records)
                logging.info(
                    "Incremental save: %s records saved so far (%s/%s detail pages done).",
                    f"{len(records):,}", f"{index:,}", f"{len(detail_urls):,}",
                )

            if index < len(detail_urls):
                delay_with_jitter(delay_seconds)
    except KeyboardInterrupt:
        logging.info("Detail scraping interrupted by user. Saving processed records...")
    finally:
        if output_path and records:
            _incremental_save(output_path, records)

    return records, stats


def scrape_ikman_records(
    listings_url: str,
    pages: int,
    delay_seconds: float,
    timeout: float,
    retries: int,
    detail_limit: Optional[int] = None,
    output_path: Optional[Path] = None,
    urls_file: Optional[Path] = DEFAULT_URLS_FILE,
    skip_existing: bool = True,
    queries: Optional[list[str]] = None,
) -> tuple[list[dict[str, Any]], ScrapeStats]:
    if queries:
        all_detail_urls: list[str] = []
        combined_stats = ScrapeStats()
        for q in queries:
            query_url = f"https://ikman.lk/en/ads/sri-lanka/mobile-phones?sort=relevance&buy_now=0&urgent=0&query={urlencode({'q': q})[2:]}&enum.condition=used"
            logging.info("--- Collecting URLs for query: %s ---", q)
            q_urls, q_stats = collect_listing_urls(
                listings_url=query_url,
                pages=pages,
                delay_seconds=delay_seconds,
                timeout=timeout,
                retries=retries,
                urls_file=urls_file,
            )
            all_detail_urls.extend(q_urls)
            combined_stats.search_pages_read += q_stats.search_pages_read
            combined_stats.listing_urls_found += q_stats.listing_urls_found
        detail_urls = dedupe_preserve_order(all_detail_urls)
        search_stats = combined_stats
        search_stats.unique_listing_urls = len(detail_urls)
    else:
        detail_urls, search_stats = collect_listing_urls(
            listings_url=listings_url,
            pages=pages,
            delay_seconds=delay_seconds,
            timeout=timeout,
            retries=retries,
            urls_file=urls_file,
        )
    logging.info(
        "Search phase complete. Found %s unique listing URLs from %s search pages.",
        f"{len(detail_urls):,}", search_stats.search_pages_read,
    )

    records, detail_stats = scrape_details_from_urls(
        detail_urls=detail_urls,
        delay_seconds=delay_seconds,
        timeout=timeout,
        retries=retries,
        detail_limit=detail_limit,
        output_path=output_path,
        skip_existing=skip_existing,
    )
    detail_stats.search_pages_read = search_stats.search_pages_read
    detail_stats.listing_urls_found = search_stats.listing_urls_found
    return records, detail_stats


def _incremental_save(output_path: Path, scraped_records: list[dict[str, Any]]) -> None:
    """Merge scraped records with existing data and save incrementally."""
    try:
        existing_records = load_json_records(output_path)
        merged_records, _ = merge_records(existing_records, scraped_records)
        save_json_records(output_path, merged_records)
    except Exception as exc:
        logging.warning("Incremental save failed (will retry later): %s", exc)


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
        if (record.get("condition") or "").strip().lower() != "used":
            continue
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
        if (record.get("condition") or "").strip().lower() != "used":
            continue
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
    logging.info("Starting automatic model training: python -m src.train")
    subprocess.run([sys.executable, "-m", "src.train"], cwd=BASE_DIR, check=True)
    logging.info("Automatic model training completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Ikman mobile-phone ads, update the raw dataset, and retrain models.",
    )
    parser.add_argument("--url", default=DEFAULT_LISTINGS_URL, help="Ikman listings URL to scrape.")
    parser.add_argument("--query", default=None, help="Single query keyword to search (e.g. 'iPhone 17').")
    parser.add_argument("--queries", nargs="+", default=None, help="Multiple search query keywords to search sequentially.")
    parser.add_argument(
        "--pages", type=int, default=0,
        help="Number of search pages to scrape. Use 0 (default) to auto-detect and scrape ALL pages.",
    )
    parser.add_argument("--all-pages", action="store_true", help="Scrape all available pages (same as --pages 0).")
    parser.add_argument("--delay", type=float, default=1.0, help="Base delay between requests in seconds (jitter is added automatically).")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per request.")
    parser.add_argument("--detail-limit", type=int, default=None, help="Optional cap for detail pages, useful for tests.")
    parser.add_argument("--output", type=Path, default=RAW_DATA_FILE, help="Raw processed dataset JSON path.")
    parser.add_argument("--urls-file", type=Path, default=DEFAULT_URLS_FILE, help="Path to JSON file storing collected listing URLs.")
    parser.add_argument("--collect-urls-only", action="store_true", help="Only scan search pages and save listing URLs to --urls-file; do not scrape details.")
    parser.add_argument("--scrape-from-urls", action="store_true", help="Read listing URLs from --urls-file and scrape ad details directly.")
    parser.add_argument("--no-skip-existing", action="store_true", help="Do not skip URLs that already exist in the output dataset.")
    parser.add_argument("--skip-train", action="store_true", help="Only scrape and merge; do not retrain.")
    parser.add_argument("--dry-run", action="store_true", help="Scrape and report stats without writing or training.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    os.chdir(BASE_DIR)

    if args.all_pages:
        args.pages = 0

    if args.detail_limit is not None and args.detail_limit < 1:
        raise ValueError("--detail-limit must be at least 1 when provided")

    effective_output = None if args.dry_run else args.output
    skip_existing = not args.no_skip_existing

    queries = None
    if args.queries:
        queries = args.queries
    elif args.query:
        queries = [args.query]

    # Mode 1: Collect URLs only and stop
    if args.collect_urls_only:
        logging.info("Running in --collect-urls-only mode. URLs will be saved to: %s", args.urls_file)
        if queries:
            all_urls = []
            combined_stats = ScrapeStats()
            for q in queries:
                q_url = f"https://ikman.lk/en/ads/sri-lanka/mobile-phones?sort=relevance&buy_now=0&urgent=0&query={urlencode({'q': q})[2:]}&enum.condition=used"
                logging.info("Collecting URLs for: %s", q)
                urls, s_stats = collect_listing_urls(
                    listings_url=q_url,
                    pages=args.pages,
                    delay_seconds=max(0.0, args.delay),
                    timeout=args.timeout,
                    retries=max(0, args.retries),
                    urls_file=args.urls_file,
                )
                all_urls.extend(urls)
                combined_stats.search_pages_read += s_stats.search_pages_read
                combined_stats.listing_urls_found += s_stats.listing_urls_found
            urls = dedupe_preserve_order(all_urls)
            save_urls_file(args.urls_file, urls)
            search_stats = combined_stats
        else:
            urls, search_stats = collect_listing_urls(
                listings_url=args.url,
                pages=args.pages,
                delay_seconds=max(0.0, args.delay),
                timeout=args.timeout,
                retries=max(0, args.retries),
                urls_file=args.urls_file,
            )
        logging.info(
            "URL collection finished: %s unique listing URLs saved to %s across %s search pages.",
            f"{len(urls):,}", args.urls_file, search_stats.search_pages_read,
        )
        return

    # Mode 2: Scrape details from saved URLs
    if args.scrape_from_urls:
        if not args.urls_file.exists():
            raise FileNotFoundError(f"URLs file not found at {args.urls_file}. Run with --collect-urls-only first.")
        urls = load_urls_file(args.urls_file)
        logging.info("Loaded %s listing URLs from %s", f"{len(urls):,}", args.urls_file)
        scraped_records, scrape_stats = scrape_details_from_urls(
            detail_urls=urls,
            delay_seconds=max(0.0, args.delay),
            timeout=args.timeout,
            retries=max(0, args.retries),
            detail_limit=args.detail_limit,
            output_path=effective_output,
            skip_existing=skip_existing,
        )
    else:
        # Standard mode: Collect URLs and scrape details
        scraped_records, scrape_stats = scrape_ikman_records(
            listings_url=args.url,
            pages=args.pages,
            delay_seconds=max(0.0, args.delay),
            timeout=args.timeout,
            retries=max(0, args.retries),
            detail_limit=args.detail_limit,
            output_path=effective_output,
            urls_file=args.urls_file,
            skip_existing=skip_existing,
            queries=queries,
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

