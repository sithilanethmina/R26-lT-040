from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin


IKMAN_BASE_URL = "https://ikman.lk"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_iso_timestamp(timestamp: str) -> datetime:
    normalized = timestamp.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def dated_snapshot_path(
    *,
    project_root: Path,
    dataset_kind: str,
    source: str,
    stem: str,
    scraped_at_utc: str,
) -> Path:
    snapshot_dt = parse_iso_timestamp(scraped_at_utc)
    date_dir = project_root / "data" / dataset_kind / source / snapshot_dt.strftime("%Y") / snapshot_dt.strftime("%m") / snapshot_dt.strftime("%d")
    filename = f"{stem}_{snapshot_dt.strftime('%Y%m%dT%H%M%SZ')}.json"
    return date_dir / filename


def snapshot_base_dir(*, project_root: Path, dataset_kind: str, source: str) -> Path:
    return project_root / "data" / dataset_kind / source


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_url(url: Any, base_url: str | None = None) -> str:
    text = normalize_text(url)
    if not text:
        return ""
    if base_url:
        text = urljoin(base_url, text)
    return text.rstrip("/")


def _legacy_ikman_key(record: dict[str, Any]) -> str:
    title = normalize_text(record.get("Raw_Title")).lower()
    price = str(record.get("Price_LKR", record.get("Clean_Price_LKR", ""))).strip()
    details = normalize_text(record.get("Details")).lower()
    return f"{title}|{price}|{details}"


def record_identity(
    source: str,
    record: dict[str, Any],
    *,
    allow_legacy_ikman_fallback: bool = False,
) -> str | None:
    if source == "ikman":
        listing_id = normalize_text(record.get("Listing_ID"))
        if listing_id:
            return f"id:{listing_id}"

        listing_url = normalize_url(record.get("Listing_URL"), IKMAN_BASE_URL)
        if listing_url:
            price = normalize_text(record.get("Price_LKR", record.get("Clean_Price_LKR", record.get("Raw_Price", ""))))
            title = normalize_text(record.get("Raw_Title", ""))
            model = normalize_text(record.get("Extracted_Model", ""))
            details = normalize_text(record.get("Details", ""))
            return f"url:{listing_url}|price:{price}|model:{model}|title:{title}|details:{details}"

        if allow_legacy_ikman_fallback:
            return f"legacy:{_legacy_ikman_key(record)}"
        return None

    product_id = normalize_text(record.get("Product_ID"))
    if product_id:
        return f"id:{product_id}"

    product_url = normalize_url(record.get("Product_URL"))
    if product_url:
        price = normalize_text(record.get("Price_LKR", record.get("Raw_Price", "")))
        title = normalize_text(record.get("Raw_Title", ""))
        model = normalize_text(record.get("Extracted_Model", ""))
        return f"url:{product_url}|price:{price}|model:{model}|title:{title}"
        
    return None


def load_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return data


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def write_json_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(records), handle, indent=4, ensure_ascii=False, allow_nan=False)


def deduplicate_records(
    *,
    source: str,
    records: list[dict[str, Any]],
    allow_legacy_ikman_fallback: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    deduped: list[dict[str, Any]] = []
    key_to_index: dict[str, int] = {}

    for record in records:
        identity = record_identity(
            source,
            record,
            allow_legacy_ikman_fallback=allow_legacy_ikman_fallback,
        )
        if identity is None:
            deduped.append(record)
            continue

        existing_index = key_to_index.get(identity)
        if existing_index is None:
            key_to_index[identity] = len(deduped)
            deduped.append(record)
        else:
            deduped[existing_index] = record

    dropped_duplicates = max(len(records) - len(deduped), 0)
    return deduped, dropped_duplicates


def merge_and_deduplicate_records(
    *,
    source: str,
    output_path: Path,
    new_records: list[dict[str, Any]],
    allow_legacy_ikman_fallback: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    combined = load_json_records(output_path) + list(new_records)
    deduped, dropped_duplicates = deduplicate_records(
        source=source,
        records=combined,
        allow_legacy_ikman_fallback=allow_legacy_ikman_fallback,
    )
    write_json_records(output_path, deduped)
    return deduped, dropped_duplicates


def list_snapshot_files(*, project_root: Path, dataset_kind: str, source: str) -> list[Path]:
    base_dir = snapshot_base_dir(project_root=project_root, dataset_kind=dataset_kind, source=source)
    if not base_dir.exists():
        return []
    return sorted(base_dir.rglob("*.json"))


def collect_previous_identities(
    *,
    project_root: Path,
    dataset_kind: str,
    source: str,
    current_snapshot_path: Path | None = None,
    allow_legacy_ikman_fallback: bool = False,
) -> set[str]:
    identities: set[str] = set()

    for path in list_snapshot_files(project_root=project_root, dataset_kind=dataset_kind, source=source):
        if current_snapshot_path is not None and path.resolve() == current_snapshot_path.resolve():
            continue

        for record in load_json_records(path):
            identity = record_identity(
                source,
                record,
                allow_legacy_ikman_fallback=allow_legacy_ikman_fallback,
            )
            if identity is not None:
                identities.add(identity)

    return identities


def filter_new_records(
    *,
    source: str,
    records: list[dict[str, Any]],
    known_identities: set[str],
    allow_legacy_ikman_fallback: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    new_records: list[dict[str, Any]] = []
    skipped = 0

    for record in records:
        identity = record_identity(
            source,
            record,
            allow_legacy_ikman_fallback=allow_legacy_ikman_fallback,
        )

        if identity is not None and identity in known_identities:
            skipped += 1
            continue

        if identity is not None:
            known_identities.add(identity)
        new_records.append(record)

    return new_records, skipped


def attach_scrape_metadata(
    records: list[dict[str, Any]],
    *,
    source: str,
    scraped_at_utc: str,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []

    for record in records:
        item = dict(record)
        item["Scraped_At_UTC"] = scraped_at_utc

        if source == "ikman":
            item["Listing_URL"] = normalize_url(item.get("Listing_URL"), IKMAN_BASE_URL) or None
            item["Listing_ID"] = normalize_text(item.get("Listing_ID")) or None
        else:
            item["Product_URL"] = normalize_url(item.get("Product_URL")) or None

        enriched.append(item)

    return enriched
