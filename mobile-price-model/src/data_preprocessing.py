"""
Data loading, cleaning, and standardisation for scraped ikman.lk phone data.

This module handles everything from raw JSON → clean DataFrame ready for
feature engineering. It does NOT impute missing numeric values (that is done
inside the sklearn Pipeline to prevent data leakage).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    BOOLEAN_COLUMNS,
    RAW_DATA_FILE,
    REQUIRED_RAW_COLUMNS,
    TARGET_COLUMN,
    TRAINING_CONDITION,
)
from .phone_specs import (
    IPHONE_RAM_GB_BY_NORMALIZED_MODEL,
    get_android_capabilities,
    get_iphone_capabilities,
    normalized_model_key,
)

logger = logging.getLogger(__name__)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(file_path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load JSON data into a DataFrame with multiple fallback strategies."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path.resolve()}")

    logger.info("Loading dataset: %s", file_path)
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
    multiplier: float = 1.5,
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


# ── Main preprocessing function ─────────────────────────────────────────────

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise raw scraped data → ready for feature engineering.

    NOTE: Does NOT impute missing numerics. That happens inside the sklearn
    Pipeline to prevent data leakage between train and test.
    """
    logger.info("Starting preprocessing.")
    df = df.copy()
    df.columns = [_to_snake_case(c) for c in df.columns]

    # Ensure required columns exist
    for col in REQUIRED_RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[REQUIRED_RAW_COLUMNS].copy()

    # Dedup raw
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Removed %s raw duplicates.", f"{before - len(df):,}")

    # Parse target
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(parse_numeric)
    before = len(df)
    df = df[df[TARGET_COLUMN].notna() & (df[TARGET_COLUMN] > 0)].copy()
    logger.info("Removed %s invalid price rows.", f"{before - len(df):,}")

    # Standardise text columns
    df["currency"] = df["currency"].apply(standardize_currency)
    before = len(df)
    df = df[df["currency"] == "LKR"].copy()
    logger.info("Kept LKR only: %s retained, %s removed.", f"{len(df):,}", f"{before - len(df):,}")

    # Parse numeric columns
    df["storage_gb"] = df["storage_gb"].apply(parse_numeric)
    df["ram_gb"] = df["ram_gb"].apply(parse_numeric)
    df["battery_health_percent"] = df["battery_health_percent"].apply(parse_battery_health)
    df["warranty_days"] = df["warranty_days"].apply(parse_warranty_days)
    df["warranty_days"] = df["warranty_days"].fillna(0.0)

    # Parse booleans
    for col in BOOLEAN_COLUMNS:
        df[col] = df[col].apply(parse_boolean)

    # Standardise categoricals
    df["brand"] = df["brand"].apply(standardize_brand)
    df["model"] = df["model"].apply(_clean_text)
    df["condition"] = df["condition"].apply(standardize_condition)

    # Filter to training condition
    before = len(df)
    df = df[df["condition"] == TRAINING_CONDITION].copy()
    logger.info("Kept condition='%s': %s retained, %s removed.",
                TRAINING_CONDITION, f"{len(df):,}", f"{before - len(df):,}")

    # Phone type
    df["phone_type"] = np.where(df["brand"] == "Apple", "iphone", "android")

    # Apply known specs
    df = apply_iphone_ram(df)
    df = apply_phone_capabilities(df)

    # Dedup after standardisation
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Removed %s post-standardisation duplicates.", f"{before - len(df):,}")

    # Remove price outliers
    df = remove_outliers_iqr(df)

    logger.info("Preprocessing done. Final records: %s", f"{len(df):,}")
    return df.reset_index(drop=True)
