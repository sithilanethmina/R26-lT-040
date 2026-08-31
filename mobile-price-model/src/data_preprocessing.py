"""
Data loading, parsing, cleaning, and standardisation for scraped ikman.lk phone data.

This module handles the entire transformation:
raw Ikman listings (title, slug, price, location, description) →
structured, cleaned, and feature-engineered DataFrame ready for ML training.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    BOOLEAN_COLUMNS,
    CLEANED_DATA_CSV,
    CLEANED_DATA_FILE,
    OUTPUT_COLUMNS,
    OUTPUTS_DIR,
    RAW_DATA_FILE,
    REQUIRED_RAW_COLUMNS,
    TARGET_COLUMN,
    TRAINING_CONDITION,
)
from .feature_engineering import add_engineered_features
from .phone_specs import (
    IPHONE_RAM_GB_BY_NORMALIZED_MODEL,
    MAX_PHONE_RAM_GB,
    get_android_capabilities,
    get_android_valid_ram,
    get_iphone_capabilities,
    normalized_model_key,
    snap_ram_to_nearest_valid,
)

logger = logging.getLogger(__name__)


# ── Non-Phone / Accessory Filter Patterns ───────────────────────────────────
NON_PHONE_PATTERN = re.compile(
    r"(?i)\b(?:airpods?|earphones?|earbuds?|headset|headphone|buds\s*pro|watch|iwatch|ultra\s*watch|"
    r"series\s*[1-9]\s*watch|ipad|tablet|tab\s*[as]\d?|tab\s*plus|cover|case|pouch|charger|cable|"
    r"adapter|tempered|display\s*only|housing|battery\s*only|back\s*glass|spare\s*parts?|"
    r"parts\s*only|board\s*only|motherboard|icloud\s*locked|bypass|dead\s*phone|for\s*parts|"
    r"not\s*working|box\s*only|power\s*bank|speaker|vr\s*box)\b"
)

# Feature / Button Phone Pattern Blacklist
BUTTON_PHONE_PATTERN = re.compile(
    r"(?i)\b(?:105|106|110|120|130|150|215|216|220|225|230|3310|5310|6300|8110|e10|b310e|b110e|"
    r"guru\s*music|guru\s*fm|sm-b\d+|c1\s*plus|it\d{3,4}|ke\d{3,4}|button\s*phone|keypad|torch\s*phone)\b"
)

# Baseline storage (GB) when omitted in listing for known iPhone models
IPHONE_BASE_STORAGE: dict[str, float] = {
    "iPhone 3GS": 16.0, "iPhone 4": 16.0, "iPhone 4S": 16.0,
    "iPhone 5": 16.0, "iPhone 5S": 16.0, "iPhone 6": 64.0, "iPhone 6 Plus": 64.0,
    "iPhone 6S": 64.0, "iPhone 6S Plus": 64.0, "iPhone SE": 64.0,
    "iPhone 7": 64.0, "iPhone 7 Plus": 128.0, "iPhone 8": 64.0, "iPhone 8 Plus": 128.0,
    "iPhone X": 64.0, "iPhone XR": 128.0, "iPhone XS": 128.0, "iPhone XS Max": 128.0,
    "iPhone 11": 128.0, "iPhone 11 Pro": 256.0, "iPhone 11 Pro Max": 256.0,
    "iPhone SE 2": 64.0, "iPhone SE 3": 64.0,
    "iPhone 12": 128.0, "iPhone 12 mini": 128.0, "iPhone 12 Pro": 128.0, "iPhone 12 Pro Max": 128.0,
    "iPhone 13": 128.0, "iPhone 13 mini": 128.0, "iPhone 13 Pro": 128.0, "iPhone 13 Pro Max": 128.0,
    "iPhone 14": 128.0, "iPhone 14 Plus": 128.0, "iPhone 14 Pro": 128.0, "iPhone 14 Pro Max": 128.0,
    "iPhone 15": 128.0, "iPhone 15 Plus": 128.0, "iPhone 15 Pro": 128.0, "iPhone 15 Pro Max": 256.0,
    "iPhone 16": 128.0, "iPhone 16 Plus": 128.0, "iPhone 16 Pro": 128.0, "iPhone 16 Pro Max": 256.0,
    "iPhone 16e": 128.0, "iPhone 17": 128.0, "iPhone 17 Pro": 256.0, "iPhone 17 Pro Max": 256.0,
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(file_path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load JSON or CSV data into a DataFrame with multiple fallback strategies."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path.resolve()}")

    logger.info("Loading dataset: %s", file_path)
    if str(file_path).endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        try:
            df = pd.read_json(file_path)
        except ValueError:
            logger.warning("Standard read_json failed; trying json_normalize fallback.")
            with file_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                df = pd.json_normalize(raw)
            elif isinstance(raw, dict):
                lists = [v for v in raw.values() if isinstance(v, list)]
                df = pd.json_normalize(lists[0]) if lists else pd.json_normalize(raw)
            else:
                raise ValueError("Unsupported JSON structure.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    logger.info("Loaded %s rows × %s columns.", f"{len(df):,}", len(df.columns))
    return df


# ── Parsing helpers ──────────────────────────────────────────────────────────

def _to_snake_case(name: Any) -> str:
    text = str(name).strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def parse_numeric(value: Any) -> float:
    """Extract the first number from a string like '128 GB' or 'Rs. 95,000'."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower().replace(",", "")
    if text in {"", "none", "null", "nan", "unknown", "n/a", "na"}:
        return np.nan
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group(0)) if m else np.nan


def parse_warranty_days(value: Any) -> float:
    """Parse warranty and convert common units → days."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    if text in {"", "none", "null", "nan", "unknown", "n/a", "na"}:
        return np.nan
    if "no warranty" in text or text in {"no", "without warranty", "expired"}:
        return 0.0
    number = parse_numeric(text)
    if pd.isna(number):
        return np.nan
    if "year" in text:
        return number * 365
    if "month" in text:
        return number * 30
    if "week" in text:
        return number * 7
    return number


def parse_battery_health(value: Any) -> float:
    """Parse battery health percentages; reject mAh capacity values."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        v = float(value)
        return v if 0.0 <= v <= 100.0 else np.nan
    text = str(value).strip().lower().replace(",", "")
    if text in {"", "none", "null", "nan", "unknown", "n/a", "na"} or "mah" in text:
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        v = float(m.group(1))
        return v if 0.0 <= v <= 100.0 else np.nan
    if "battery" in text and "health" in text:
        v = parse_numeric(text)
        if not pd.isna(v) and 0.0 <= v <= 100.0:
            return v
    return np.nan


def parse_boolean(value: Any) -> float:
    """Convert yes/no/true/false/1/0 → 1.0 / 0.0."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return 1.0 if float(value) > 0 else 0.0
    text = str(value).strip().lower()
    text = re.sub(r"[\s_-]+", " ", text)
    TRUE = {"1", "yes", "y", "true", "t", "available", "supported", "support",
            "enabled", "dual sim", "5g", "esim"}
    FALSE = {"0", "no", "n", "false", "f", "none", "not available",
             "not supported", "unsupported", "disabled", "single sim", "no 5g", "no esim"}
    if text in TRUE:
        return 1.0
    if text in FALSE:
        return 0.0
    if "dual" in text and "sim" in text:
        return 1.0
    if "single" in text and "sim" in text:
        return 0.0
    if "not" in text or "no " in text:
        return 0.0
    return np.nan


def parse_storage_from_text(text: str) -> float:
    """Extract storage capacity in GB from listing text/slug."""
    m_tb = re.search(r"(\d+)\s*(?:tb|tera)", text, re.IGNORECASE)
    if m_tb:
        return float(m_tb.group(1)) * 1024.0

    # Pattern like 8GB/256GB or 8/256 or 8+256
    m_pair = re.search(r"(\d+)\s*(?:gb)?\s*[/+\s]\s*(\d+)\s*gb", text, re.IGNORECASE)
    if m_pair:
        r, s = float(m_pair.group(1)), float(m_pair.group(2))
        if s in [16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]:
            return s
        if r in [16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]:
            return r

    m_all = re.findall(r"(\d+)\s*gb\b", text, re.IGNORECASE)
    if m_all:
        nums = [float(x) for x in m_all]
        standard_storages = [x for x in nums if x in [16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]]
        if standard_storages:
            return max(standard_storages)
        valid = [x for x in nums if x >= 16.0]
        if valid:
            return max(valid)

    return np.nan


def parse_ram_from_text(text: str) -> float:
    """Extract RAM in GB from listing text/slug."""
    m_ram = re.search(r"(\d+(?:\.\d+)?)\s*gb\s*ram", text, re.IGNORECASE)
    if m_ram:
        v = float(m_ram.group(1))
        if v in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 18.0, 24.0]:
            return v

    m_pair = re.search(r"(\d+(?:\.\d+)?)\s*(?:gb)?\s*[/+\s]\s*(\d+)\s*gb", text, re.IGNORECASE)
    if m_pair:
        r, s = float(m_pair.group(1)), float(m_pair.group(2))
        if r in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 18.0, 24.0] and s in [16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]:
            return r

    m_all = re.findall(r"(\d+)\s*gb\b", text, re.IGNORECASE)
    if len(m_all) >= 2:
        nums = [float(x) for x in m_all]
        rams = [x for x in nums if x in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 18.0, 24.0]]
        if rams and min(rams) != max(nums):
            return min(rams)

    return np.nan


# ── Raw Spec Extraction Engine ───────────────────────────────────────────────

def extract_phone_specs_from_raw(title: str, slug: str = "", desc: str = "") -> Tuple[Optional[str], Optional[str]]:
    """
    Extract (brand, model) from title, slug, and description.
    Filters out non-phones, accessories, tablets, and button phones.
    """
    full_text = f"{title} {slug} {desc}".strip()

    # 1. Non-phone filter
    if NON_PHONE_PATTERN.search(title):
        return None, None

    # 2. Apple / iPhone
    if "iphone" in full_text.lower() or "apple" in full_text.lower():
        if BUTTON_PHONE_PATTERN.search(title):
            return None, None
        m = re.search(
            r"(?:Apple\s*)?iPhone\s*(1[1-7]|SE\s*[23]?|[678]\s*Plus|[678]|X[RS]?\s*Max|X[RS]?|3GS|5S?|4S?)\s*(Pro\s*Max|Pro|Plus|mini|e)?",
            full_text,
            re.IGNORECASE,
        )
        if m:
            gen = re.sub(r"\s+", " ", m.group(1).strip())
            sub = (" " + m.group(2).strip()) if m.group(2) else ""
            model_name = f"iPhone {gen}{sub}".strip()
            model_name = re.sub(r"SE\s*2\b", "SE 2", model_name, flags=re.IGNORECASE)
            model_name = re.sub(r"SE\s*3\b", "SE 3", model_name, flags=re.IGNORECASE)
            model_name = re.sub(r"XS\s*Max\b", "XS Max", model_name, flags=re.IGNORECASE)
            return "Apple", model_name
        return None, None

    # 3. Samsung
    if "samsung" in full_text.lower() or "galaxy" in full_text.lower():
        if BUTTON_PHONE_PATTERN.search(title):
            return None, None
        m = re.search(
            r"(?:Samsung\s*)?(?:Galaxy\s*)?(S2[0-6]\s*(?:Ultra|Plus|\+|FE|e)?|S10\s*(?:Plus|\+|Lite|e)?|S[1-9]\s*(?:Plus|\+|Edge)?|Note\s*20\s*(?:Ultra|Plus|\+)?|Note\s*10\s*(?:Plus|\+|Lite)?|Note\s*[1-9]\s*(?:Plus|\+)?|Z\s*Fold\s*[1-6]?|Z\s*Flip\s*[1-6]?|A0[1-7][se]?|A1[0-7][se]?|A2[0-6][se]?|A3[0-6][se]?|A5[0-6][se]?|A7[0-3][se]?|M0[1-5][se]?|M1[0-7][se]?|M2[0-3][se]?|M3[0-6][se]?|M5[1-5][se]?|F[0-5][0-9]|J[1-8]\s*(?:Plus|\+|Prime|Core)?)",
            full_text,
            re.IGNORECASE,
        )
        if m:
            model_name = "Galaxy " + re.sub(r"\s+", " ", m.group(1).strip())
            return "Samsung", model_name
        return None, None

    # 4. Xiaomi / Redmi / POCO
    if any(k in full_text.lower() for k in ["xiaomi", "redmi", "poco", "mi "]):
        m = re.search(
            r"(Redmi\s*Note\s*\d{1,2}\s*(?:Pro\s*Plus|\+|Pro|Max|S|T)?|Redmi\s*K\d{1,2}\s*(?:Pro|Gaming)?|Redmi\s*A[1-5]\s*(?:Plus|\+)?|Redmi\s*1[0-5][AC]?|Redmi\s*[1-9][AC]?|Poco\s*[XFM]\d{1,2}\s*(?:Pro|GT|NFC)?|Poco\s*C\d{1,2}|Mi\s*1[0-4]\s*(?:Ultra|Pro|Lite|T|NE)?|Mi\s*[1-9]\s*(?:Ultra|Pro|Lite|T|SE)?|Xiaomi\s*1[1-5]\s*(?:Ultra|Pro|Lite|T)?|Xiaomi\s*1[1-5][T]?)",
            full_text,
            re.IGNORECASE,
        )
        if m:
            model_name = re.sub(r"\s+", " ", m.group(1).strip())
            return "Xiaomi", model_name
        return None, None

    # 5. Google Pixel
    if "pixel" in full_text.lower() or "google" in full_text.lower():
        m = re.search(r"(?:Google\s*)?Pixel\s*([1-9]|10)\s*(Pro\s*XL|Pro|Fold|a)?", full_text, re.IGNORECASE)
        if m:
            sub = (" " + m.group(2).strip()) if m.group(2) else ""
            model_name = f"Pixel {m.group(1).strip()}{sub}".strip()
            return "Google", model_name
        return None, None

    # 6. Vivo
    if "vivo" in full_text.lower():
        m = re.search(r"(?:Vivo\s*)?(Y\d{1,3}[a-z]?|V\d{1,3}[a-z]?\s*(?:Pro|e)?|X\d{1,3}\s*(?:Pro|\+)?|T\d\s*(?:Pro|x)?)", full_text, re.IGNORECASE)
        if m:
            return "Vivo", m.group(1).strip().upper()
        return None, None

    # 7. Oppo
    if "oppo" in full_text.lower():
        m = re.search(r"(?:Oppo\s*)?(A\d{1,3}[a-z]?\s*(?:Plus|\+|s)?|Reno\s*\d{1,2}\s*(?:Pro\s*Plus|\+|Pro|F)?|Find\s*X\d\s*(?:Pro)?)", full_text, re.IGNORECASE)
        if m:
            return "Oppo", re.sub(r"\s+", " ", m.group(1).strip())
        return None, None

    # 8. Realme
    if "realme" in full_text.lower():
        m = re.search(r"(?:Realme\s*)?(GT\s*\d?\s*(?:Pro|Master|Neo)?|C\d{1,2}[a-z]?|Note\s*\d{1,2}|\d{1,2}\s*(?:Pro\s*Plus|\+|Pro|x|i)?|Narzo\s*\d{1,2}[a-z]?)", full_text, re.IGNORECASE)
        if m:
            return "Realme", re.sub(r"\s+", " ", m.group(1).strip())
        return None, None

    # 9. OnePlus
    if "oneplus" in full_text.lower() or "one plus" in full_text.lower():
        m = re.search(r"(?:OnePlus\s*|One\s*Plus\s*)?(1[0-5]|7|8|9|Nord\s*(?:CE)?\s*\d?|1\dr|Open)\s*(Pro|T|R)?", full_text, re.IGNORECASE)
        if m:
            sub = (" " + m.group(2).strip()) if m.group(2) else ""
            model_name = f"OnePlus {m.group(1).strip()}{sub}".strip()
            return "OnePlus", model_name
        return None, None

    # 10. Honor
    if "honor" in full_text.lower():
        m = re.search(r"(?:Honor\s*)?(X[5-9][a-z]?|\d{2,3}\s*(?:Pro|Lite)?|Magic\s*\d\s*(?:Pro|Lite)?|Play\s*\d{1,2})", full_text, re.IGNORECASE)
        if m:
            return "Honor", m.group(1).strip()
        return None, None

    # 11. Huawei
    if "huawei" in full_text.lower():
        m = re.search(r"(?:Huawei\s*)?(P[1-6]0\s*(?:Pro|Lite)?|Mate\s*[1-6]0\s*(?:Pro)?|Nova\s*\d{1,2}\s*(?:Pro|i|SE)?|Y[5-9][a-z]?)", full_text, re.IGNORECASE)
        if m:
            return "Huawei", m.group(1).strip()
        return None, None

    # 12. ZTE
    if "zte" in full_text.lower():
        m = re.search(r"(?:ZTE\s*)?(Blade\s*[AV]\d{1,2}|Nubia\s*(?:Focus|Neo|Music|V\d{2}|Z\d{2})|Axon\s*\d{1,2})", full_text, re.IGNORECASE)
        if m:
            return "ZTE", m.group(1).strip()
        return None, None

    # 13. Infinix
    if "infinix" in full_text.lower():
        m = re.search(r"(?:Infinix\s*)?(Hot\s*\d{1,2}[a-z]?|Note\s*\d{1,2}\s*(?:Pro|\+)?|Smart\s*\d{1,2}|Zero\s*\d{1,2}\s*(?:Ultra|Pro)?)", full_text, re.IGNORECASE)
        if m:
            return "Infinix", m.group(1).strip()
        return None, None

    # 14. Tecno
    if "tecno" in full_text.lower():
        m = re.search(r"(?:Tecno\s*)?(Spark\s*\d{1,2}[a-z]?|Camon\s*\d{1,2}\s*(?:Pro)?|Pova\s*\d{1,2}|Pop\s*\d{1,2})", full_text, re.IGNORECASE)
        if m:
            return "Tecno", m.group(1).strip()
        return None, None

    # 15. Sony
    if "sony" in full_text.lower() or "xperia" in full_text.lower():
        m = re.search(r"(?:Sony\s*)?(?:Xperia\s*)?(1|5|10|XZ\d?|XA\d?|Z\d?)\s*(VI|V|IV|III|II|I|\d)?", full_text, re.IGNORECASE)
        if m:
            sub = (" " + m.group(2).strip()) if m.group(2) else ""
            model_name = f"Xperia {m.group(1).strip()}{sub}".strip()
            return "Sony", model_name
        return None, None

    # 16. Motorola
    if "motorola" in full_text.lower() or "moto" in full_text.lower():
        m = re.search(r"(?:Moto(?:rola)?\s*)?(G\d{1,2}[a-z]?|E\d{1,2}[a-z]?|Edge\s*\d{1,2}\s*(?:Pro|Fusion|Ultra)?)", full_text, re.IGNORECASE)
        if m:
            return "Motorola", m.group(1).strip()
        return None, None

    # 17. Nothing
    if "nothing" in full_text.lower():
        m = re.search(r"(?:Nothing\s*)?(?:Phone\s*)?(1|2|2a|3a?)", full_text, re.IGNORECASE)
        if m:
            return "Nothing", f"Phone ({m.group(1).strip()})"
        return None, None

    return None, None


def parse_raw_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Parse raw scraped Ikman listings (title, slug, price) into structured schema."""
    if "title" not in df.columns:
        return df

    logger.info("Parsing raw Ikman listings into structured specifications...")
    parsed_rows = []

    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        slug = str(row.get("slug", "")).strip()
        desc = str(row.get("description", "")).strip()

        brand, model = extract_phone_specs_from_raw(title, slug, desc)
        if not brand or not model:
            continue

        full_text = f"{title} {slug} {desc}"
        storage_gb = parse_storage_from_text(full_text)
        ram_gb = parse_ram_from_text(full_text)

        # Baseline storage lookup for iPhones if not specified
        if pd.isna(storage_gb) and brand == "Apple":
            storage_gb = IPHONE_BASE_STORAGE.get(model, 128.0)
        elif pd.isna(storage_gb):
            # Reasonable default baseline for modern Android smartphones
            storage_gb = 64.0

        # Price parsing
        price_val = row.get("price")
        if pd.isna(price_val) or price_val is None:
            price_val = parse_numeric(row.get("price_raw"))
        else:
            price_val = parse_numeric(price_val)

        if pd.isna(price_val) or price_val <= 0:
            continue

        # Battery health parsing
        battery_health = parse_battery_health(full_text)

        parsed_rows.append({
            "brand": brand,
            "model": model,
            "storage_gb": storage_gb,
            "ram_gb": ram_gb,
            "battery_health_percent": battery_health,
            "warranty_days": 0.0,
            "condition": "used",
            "currency": "LKR",
            "dual_sim": np.nan,
            "has_5g": np.nan,
            "has_esim": np.nan,
            "listed_price": float(price_val),
            "location": row.get("location", ""),
            "is_member": bool(row.get("is_member", False)),
        })

    parsed_df = pd.DataFrame(parsed_rows)
    logger.info("Successfully parsed %s valid smartphone listings from %s raw records.",
                f"{len(parsed_df):,}", f"{len(df):,}")
    return parsed_df


# ── Standardisation ──────────────────────────────────────────────────────────

def _clean_text(value: Any, unknown: str = "Unknown") -> str:
    if pd.isna(value):
        return unknown
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "nan", "unknown", "n/a", "na"}:
        return unknown
    return re.sub(r"\s+", " ", text)


def standardize_brand(value: Any) -> str:
    text = _clean_text(value)
    lowered = text.lower().strip()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)

    if compact in {"apple", "iphone", "iphones"} or "iphone" in lowered:
        return "Apple"

    aliases = {
        "samsung": "Samsung", "oppo": "Oppo", "vivo": "Vivo",
        "xiaomi": "Xiaomi", "mi": "Xiaomi", "redmi": "Xiaomi", "poco": "Xiaomi",
        "realme": "Realme", "huawei": "Huawei", "honor": "Honor",
        "nokia": "Nokia", "oneplus": "OnePlus", "google": "Google",
        "pixel": "Google", "sony": "Sony", "motorola": "Motorola",
        "moto": "Motorola", "infinix": "Infinix", "tecno": "Tecno",
        "itel": "Itel", "lg": "LG", "htc": "HTC", "asus": "Asus",
        "lenovo": "Lenovo", "zte": "ZTE", "nothing": "Nothing",
    }
    if lowered in aliases:
        return aliases[lowered]
    if compact in aliases:
        return aliases[compact]
    for alias, norm in aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return norm
    return text.title() if text != "Unknown" else "Unknown"


def standardize_condition(value: Any) -> str:
    text = _clean_text(value, "unknown").lower()
    text = re.sub(r"[\s_-]+", " ", text)
    if text in {"unknown", "none", "null", "nan", "n/a", "na"}:
        return "unknown"
    if text in {"new", "brand new", "brandnew", "sealed", "unused"}:
        return "new"
    if any(t in text for t in ["reconditioned", "refurbished", "refurb", "renewed"]):
        return "reconditioned"
    if any(t in text for t in ["used", "second hand", "pre owned", "preowned",
                                "like new", "mint", "excellent", "good"]):
        return "used"
    return "unknown"


def standardize_currency(value: Any) -> str:
    text = _clean_text(value).strip()
    compact = re.sub(r"[^a-z]+", "", text.lower())
    if text == "Unknown":
        return "Unknown"
    if compact in {"lkr", "rs", "slrs", "srilankanrupee", "srilankanrupees"}:
        return "LKR"
    if "lkr" in compact or "rupee" in compact or text.lower() in {"rs.", "rs/-"}:
        return "LKR"
    return text.upper()


# ── Spec overrides ───────────────────────────────────────────────────────────

def apply_iphone_ram(df: pd.DataFrame) -> pd.DataFrame:
    """Override scraped iPhone RAM values with known model specifications."""
    keys = df["model"].apply(normalized_model_key)
    predefined = keys.map(IPHONE_RAM_GB_BY_NORMALIZED_MODEL)
    iphone_mask = (df["brand"] == "Apple") | df["model"].str.contains("iPhone", case=False, na=False)
    mask = iphone_mask & predefined.notna()
    changed = int((df.loc[mask, "ram_gb"] != predefined.loc[mask]).sum())
    if changed:
        df.loc[mask, "ram_gb"] = predefined.loc[mask].astype(float)
        logger.info("Corrected %s iPhone RAM values.", f"{changed:,}")
    return df


def apply_android_ram(df: pd.DataFrame) -> pd.DataFrame:
    """Override scraped Android RAM values with known model specifications."""
    android_mask = df["phone_type"] == "android"
    if not android_mask.any():
        return df

    changed = 0
    for idx in df.index[android_mask]:
        brand = str(df.at[idx, "brand"]).strip()
        model = str(df.at[idx, "model"]).strip()
        current_ram = df.at[idx, "ram_gb"]

        valid_rams = get_android_valid_ram(brand, model)
        if valid_rams is not None:
            if pd.isna(current_ram):
                df.at[idx, "ram_gb"] = valid_rams[0]
                changed += 1
            else:
                corrected = snap_ram_to_nearest_valid(float(current_ram), valid_rams)
                if corrected != current_ram:
                    df.at[idx, "ram_gb"] = corrected
                    changed += 1
        elif pd.notna(current_ram) and float(current_ram) > MAX_PHONE_RAM_GB:
            df.at[idx, "ram_gb"] = MAX_PHONE_RAM_GB
            changed += 1
        elif pd.isna(current_ram):
            df.at[idx, "ram_gb"] = 4.0  # standard baseline for Android
            changed += 1

    if changed:
        logger.info("Corrected / assigned %s Android RAM values.", f"{changed:,}")
    return df


def sanitize_ram_values(df: pd.DataFrame) -> pd.DataFrame:
    """Final safety net: cap any remaining RAM values above MAX_PHONE_RAM_GB."""
    ram = pd.to_numeric(df["ram_gb"], errors="coerce")
    over_cap = ram > MAX_PHONE_RAM_GB
    count = int(over_cap.sum())
    if count:
        df.loc[over_cap, "ram_gb"] = MAX_PHONE_RAM_GB
        logger.info("Capped %s remaining RAM values above %.0f GB.", count, MAX_PHONE_RAM_GB)
    return df


def apply_phone_capabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Override noisy capability columns with predefined values."""
    if df.empty:
        return df

    def _get_caps(row):
        brand = _clean_text(row.get("brand"))
        model = _clean_text(row.get("model"))
        key = normalized_model_key(model)
        if brand == "Apple" or "iphone" in str(model).lower():
            return get_iphone_capabilities(key)
        return get_android_capabilities(brand, model)

    caps = df.apply(_get_caps, axis=1, result_type="expand")
    for col in BOOLEAN_COLUMNS:
        old = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
        new = caps[col].astype(int)
        changed = int((old != new).sum())
        df[col] = new
        logger.info("Corrected %s %s values.", f"{changed:,}", col)
    return df


# ── IQR outlier removal ─────────────────────────────────────────────────────

def remove_outliers_iqr(
    df: pd.DataFrame,
    group_col: str = "phone_type",
    price_col: str = TARGET_COLUMN,
    multiplier: float = 3.5,
) -> pd.DataFrame:
    """Remove extreme price outliers per phone type using IQR."""
    groups = []
    total_removed = 0
    for name, gdf in df.groupby(group_col, dropna=False):
        if len(gdf) < 4:
            groups.append(gdf)
            continue
        q1, q3 = gdf[price_col].quantile(0.25), gdf[price_col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr <= 0:
            groups.append(gdf)
            continue
        lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
        mask = gdf[price_col].between(lo, hi, inclusive="both")
        removed = int((~mask).sum())
        total_removed += removed
        logger.info("IQR %-7s | Q1=%.0f Q3=%.0f | bounds=[%.0f, %.0f] | removed=%s",
                     name, q1, q3, lo, hi, removed)
        groups.append(gdf.loc[mask])
    logger.info("Total IQR outliers removed: %s", total_removed)
    return pd.concat(groups, ignore_index=True) if groups else df.iloc[0:0].copy()


# ── Suspicious price removal ────────────────────────────────────────────────

_MODEL_MIN_PRICE_LKR: dict[str, float] = {
    # iPhones
    "iPhone 17 Pro Max": 350000,
    "iPhone 17 Pro": 300000,
    "iPhone 17": 200000,
    "iPhone 16 Pro Max": 240000,
    "iPhone 16 Pro": 190000,
    "iPhone 16 Plus": 160000,
    "iPhone 16": 140000,
    "iPhone 16e": 100000,
    "iPhone 15 Pro Max": 190000,
    "iPhone 15 Pro": 160000,
    "iPhone 15 Plus": 130000,
    "iPhone 15": 115000,
    "iPhone 14 Pro Max": 140000,
    "iPhone 14 Pro": 120000,
    "iPhone 14 Plus": 95000,
    "iPhone 14": 80000,
    "iPhone 13 Pro Max": 110000,
    "iPhone 13 Pro": 95000,
    "iPhone 13": 65000,
    "iPhone 13 mini": 55000,
    "iPhone 12 Pro Max": 75000,
    "iPhone 12 Pro": 65000,
    "iPhone 12": 45000,
    "iPhone 12 mini": 35000,
    "iPhone 11 Pro Max": 55000,
    "iPhone 11 Pro": 45000,
    "iPhone 11": 32000,
    "iPhone XS Max": 28000,
    "iPhone XS": 22000,
    "iPhone XR": 22000,
    "iPhone X": 18000,
    "iPhone SE 3": 35000,
    "iPhone SE 2": 18000,
    "iPhone 8 Plus": 14000,
    "iPhone 8": 10000,
    "iPhone 7 Plus": 9000,
    "iPhone 7": 7000,
    # Samsung flagships
    "Galaxy S25 Ultra": 200000,
    "Galaxy S25+": 150000,
    "Galaxy S25": 120000,
    "Galaxy S24 Ultra": 160000,
    "Galaxy S24+": 110000,
    "Galaxy S24": 85000,
    "Galaxy S23 Ultra": 120000,
    "Galaxy S23+": 85000,
    "Galaxy S23": 65000,
    "Galaxy S22 Ultra": 85000,
    "Galaxy S22+": 60000,
    "Galaxy S22": 50000,
    "Galaxy S21 Ultra": 55000,
    "Galaxy S21 Plus": 40000,
    "Galaxy S21": 30000,
    "Galaxy Z Fold6": 240000,
    "Galaxy Z Fold5": 170000,
    "Galaxy Z Fold4": 120000,
    "Galaxy Z Flip6": 140000,
    "Galaxy Z Flip5": 90000,
    "Galaxy Z Flip4": 65000,
    # Google Pixel
    "Pixel 9 Pro XL": 170000,
    "Pixel 9 Pro": 140000,
    "Pixel 9": 95000,
    "Pixel 8 Pro": 95000,
    "Pixel 8": 65000,
    "Pixel 7 Pro": 55000,
    "Pixel 7": 40000,
}


def remove_suspicious_prices(
    df: pd.DataFrame,
    price_col: str = TARGET_COLUMN,
    iqr_multiplier: float = 2.5,
    min_group_size: int = 5,
) -> pd.DataFrame:
    """Remove listings with suspiciously low prices."""
    flagged_indices: set[int] = set()

    # 1. Model-specific floor thresholds
    threshold_count = 0
    for model_name, min_price in _MODEL_MIN_PRICE_LKR.items():
        mask = (df["model"] == model_name) & (df[price_col] < min_price)
        hits = df.index[mask].tolist()
        if hits:
            flagged_indices.update(hits)
            threshold_count += len(hits)

    logger.info("Model-threshold flagged: %s records.", f"{threshold_count:,}")

    # 2. Per-brand+model IQR (low-end only)
    iqr_count = 0
    for (brand, model), gdf in df.groupby(["brand", "model"], dropna=False):
        if len(gdf) < min_group_size:
            continue
        prices = gdf[price_col]
        q1 = prices.quantile(0.25)
        q3 = prices.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr <= 0:
            continue
        lower_bound = q1 - iqr_multiplier * iqr
        outlier_mask = prices < lower_bound
        if not outlier_mask.any():
            continue
        hits = gdf.index[outlier_mask].tolist()
        new_hits = [i for i in hits if i not in flagged_indices]
        if new_hits:
            flagged_indices.update(new_hits)
            iqr_count += len(new_hits)

    logger.info("Per-group IQR flagged: %s additional records.", f"{iqr_count:,}")
    logger.info("Total suspicious removed: %s (of %s).",
                f"{len(flagged_indices):,}", f"{len(df):,}")

    if flagged_indices:
        df = df.drop(index=list(flagged_indices)).reset_index(drop=True)

    return df


def filter_smartphones_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out non-smartphones and phantom concept models."""
    before = len(df)

    # 1. Feature / button phone pattern blacklist
    button_mask = df["model"].str.contains(BUTTON_PHONE_PATTERN, regex=True, na=False) & (
        df["brand"].isin(["Nokia", "Samsung", "Itel", "Tecno"])
    )
    df = df[~button_mask].copy()

    # 2. Drop phantom unreleased models
    phantom_mask = df["model"].isin(["iPhone Air", "iPhone 17e", "iPhone 17 Pro", "iPhone 17 Pro Max", "iPhone 17"])
    df = df[~phantom_mask].copy()

    # 3. Require valid smartphone memory
    has_valid_storage = df["storage_gb"].notna() & (df["storage_gb"] >= 16.0)
    has_valid_ram = df["ram_gb"].notna() & (df["ram_gb"] >= 1.0)

    valid_smartphone = has_valid_storage | has_valid_ram
    df = df[valid_smartphone].copy()

    logger.info("Smartphone memory & feature phone filter: %s retained, %s removed.",
                f"{len(df):,}", f"{before - len(df):,}")
    return df


def export_brand_model_lookup(df: pd.DataFrame) -> None:
    """Export unique supported brands and models to outputs/mobile_brand_model_lookup.json."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    lookup_file = OUTPUTS_DIR / "mobile_brand_model_lookup.json"
    brand_models: dict[str, list[str]] = {}
    for brand, group in df.groupby("brand"):
        models = sorted(group["model"].dropna().unique().tolist())
        if models:
            brand_models[brand] = models
    with open(lookup_file, "w", encoding="utf-8") as f:
        json.dump(brand_models, f, indent=2)
    logger.info("Exported brand-model lookup (%d brands) to %s", len(brand_models), lookup_file)


# ── Main preprocessing function ─────────────────────────────────────────────

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean, extract, standardise, and engineer features from raw scraped data.
    Saves clean dataset to data/processed/ikman_mobile_phones_ml_ready.json and .csv.
    """
    logger.info("Starting mobile phone data preprocessing pipeline.")
    df = df.copy()

    # 1. Parse raw scraped format if needed
    if "title" in df.columns and ("brand" not in df.columns or "model" not in df.columns):
        df = parse_raw_listings(df)

    df.columns = [_to_snake_case(c) for c in df.columns]

    # 2. Ensure required columns exist
    for col in REQUIRED_RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # 3. Parse target price
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(parse_numeric)
    before = len(df)
    df = df[df[TARGET_COLUMN].notna() & (df[TARGET_COLUMN] > 0)].copy()
    logger.info("Removed %s invalid price rows.", f"{before - len(df):,}")

    # 4. Standardise text columns
    df["currency"] = df["currency"].apply(standardize_currency)
    before = len(df)
    df = df[df["currency"] == "LKR"].copy()
    logger.info("Kept LKR only: %s retained, %s removed.", f"{len(df):,}", f"{before - len(df):,}")

    # 5. Parse numeric columns
    df["storage_gb"] = df["storage_gb"].apply(parse_numeric)
    df["ram_gb"] = df["ram_gb"].apply(parse_numeric)
    df["battery_health_percent"] = df["battery_health_percent"].apply(parse_battery_health)
    df["warranty_days"] = df["warranty_days"].apply(parse_warranty_days)
    df["warranty_days"] = df["warranty_days"].fillna(0.0)

    # 6. Parse booleans
    for col in BOOLEAN_COLUMNS:
        df[col] = df[col].apply(parse_boolean)

    # 7. Standardise categoricals
    df["brand"] = df["brand"].apply(standardize_brand)
    df["model"] = df["model"].apply(_clean_text)
    df["condition"] = df["condition"].apply(standardize_condition)

    # 8. Filter to training condition
    before = len(df)
    df = df[df["condition"] == TRAINING_CONDITION].copy()
    logger.info("Kept condition='%s': %s retained, %s removed.",
                TRAINING_CONDITION, f"{len(df):,}", f"{before - len(df):,}")

    # 9. Set Phone type
    df["phone_type"] = np.where(df["brand"] == "Apple", "iphone", "android")

    # 10. Apply known hardware specs
    df = apply_iphone_ram(df)
    df = apply_android_ram(df)
    df = sanitize_ram_values(df)
    df = apply_phone_capabilities(df)

    # 11. Add engineered features
    df = add_engineered_features(df)

    # 12. Filter strictly for real smartphones
    df = filter_smartphones_only(df)

    # 13. Remove price outliers (broad IQR)
    df = remove_outliers_iqr(df)

    # 14. Remove suspicious / fake low prices
    df = remove_suspicious_prices(df)

    # Keep output columns
    final_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    df = df[final_cols].reset_index(drop=True)

    # 15. Save cleaned data to disk
    CLEANED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(CLEANED_DATA_FILE, orient="records", indent=2)
    df.to_csv(CLEANED_DATA_CSV, index=False)
    logger.info("Saved cleaned dataset to %s and %s", CLEANED_DATA_FILE, CLEANED_DATA_CSV)

    # Export lookup file
    export_brand_model_lookup(df)

    logger.info("Preprocessing complete! Final clean records: %s", f"{len(df):,}")
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    raw_df = load_data(RAW_DATA_FILE)
    cleaned_df = preprocess_data(raw_df)
    print("\n" + "=" * 60)
    print("DATA CLEANING COMPLETE")
    print("=" * 60)
    print(f"Total Cleaned Records : {len(cleaned_df):,}")
    print(f"iPhone Records        : {(cleaned_df['phone_type'] == 'iphone').sum():,}")
    print(f"Android Records       : {(cleaned_df['phone_type'] == 'android').sum():,}")
    print(f"Unique Brands         : {cleaned_df['brand'].nunique()}")
    print(f"Unique Models         : {cleaned_df['model'].nunique()}")
    print(f"Price Range (LKR)     : Rs. {cleaned_df['listed_price'].min():,.0f} - Rs. {cleaned_df['listed_price'].max():,.0f}")
    print(f"Median Price (LKR)    : Rs. {cleaned_df['listed_price'].median():,.0f}")
    print("=" * 60)
