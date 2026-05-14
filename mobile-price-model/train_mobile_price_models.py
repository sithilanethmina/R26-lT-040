"""
Train fair price prediction models for used smartphones in Sri Lanka.

This script:
1. Loads scraped ikman mobile phone data from JSON.
2. Cleans and standardizes the research-ready columns.
3. Saves an ML-ready cleaned JSON dataset.
4. Trains separate Random Forest and XGBoost regressors for iPhones and Android phones.
5. Creates XGBoost fair-price predictions for each exact phone model/storage category.
6. Saves model artifacts, evaluation metrics, and fair-price guidance.

Target variable:
    listed_price

Important:
    listed_price is never used as an input feature.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


INPUT_FILE = Path("ikman_mobile_phones_processed.json")
TARGET_PHONES_FILE_NAME = "fair_price_prediction_target_phones_simple.json"
TARGET_PHONES_INPUT_FILE = Path(TARGET_PHONES_FILE_NAME)
TARGET_PHONES_INPUT_FALLBACK_FILES = [
    Path("../Scraper/data") / TARGET_PHONES_FILE_NAME,
]
CLEANED_OUTPUT_FILE = Path("ikman_mobile_phones_ml_ready.json")
EVALUATION_OUTPUT_FILE = Path("model_evaluation_results.json")
FAIR_PRICE_OUTPUT_FILE = Path("fair_price_predictions.json")
FAIR_PRICE_CSV_OUTPUT_FILE = Path("fair_price_predictions.csv")
MODEL_DIR = Path("models")

TARGET_COLUMN = "listed_price"
TRAINING_CONDITION = "used"

REQUIRED_COLUMNS = [
    "brand",
    "model",
    "condition",
    "currency",
    "dual_sim",
    "has_5g",
    "has_esim",
    "warranty_days",
    "storage_gb",
    "ram_gb",
    "listed_price",
]

FEATURE_COLUMNS = [
    "brand",
    "model",
    "condition",
    "currency",
    "dual_sim",
    "has_5g",
    "has_esim",
    "warranty_days",
    "storage_gb",
    "ram_gb",
]

CATEGORICAL_COLUMNS = ["brand", "model", "condition", "currency"]
BOOLEAN_COLUMNS = ["dual_sim", "has_5g", "has_esim"]
NUMERIC_COLUMNS = ["dual_sim", "has_5g", "has_esim", "warranty_days", "storage_gb", "ram_gb"]
MEDIAN_IMPUTED_NUMERIC_COLUMNS = ["dual_sim", "has_5g", "has_esim", "storage_gb", "ram_gb"]

MODEL_OUTPUT_PATHS = [
    MODEL_DIR / "random_forest_iphone.pkl",
    MODEL_DIR / "random_forest_android.pkl",
    MODEL_DIR / "xgboost_iphone.pkl",
    MODEL_DIR / "xgboost_android.pkl",
]

XGBOOST_MODEL_KEYS_BY_PHONE_TYPE = {
    "iphone": "xgboost_iphone",
    "android": "xgboost_android",
}

FAIR_PRICE_GROUP_COLUMNS = ["phone_type", "brand", "model", "storage_gb"]
OUTPUT_COLUMNS = REQUIRED_COLUMNS + ["phone_type"]

RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_ROWS_REQUIRED = 10
MIN_ROWS_WARNING = 50

IPHONE_RAM_GB_BY_MODEL = {
    "iPhone 3GS": 0.25,
    "iPhone 5S": 1.0,
    "iPhone 6": 1.0,
    "iPhone 6 Plus": 1.0,
    "iPhone 6S": 2.0,
    "iPhone 6S Plus": 2.0,
    "iPhone 7": 2.0,
    "iPhone 7 Plus": 3.0,
    "iPhone 8": 2.0,
    "iPhone 8 Plus": 3.0,
    "iPhone X": 3.0,
    "iPhone XR": 3.0,
    "iPhone XS": 4.0,
    "iPhone XS Max": 4.0,
    "iPhone 11": 4.0,
    "iPhone 11 Pro": 4.0,
    "iPhone 11 Pro Max": 4.0,
    "iPhone SE": 2.0,
    "iPhone SE 2": 3.0,
    "iPhone SE 3": 4.0,
    "iPhone 12": 4.0,
    "iPhone 12 mini": 4.0,
    "iPhone 12 Pro": 6.0,
    "iPhone 12 Pro Max": 6.0,
    "iPhone 13": 4.0,
    "iPhone 13 mini": 4.0,
    "iPhone 13 Pro": 6.0,
    "iPhone 13 Pro Max": 6.0,
    "iPhone 14": 6.0,
    "iPhone 14 Plus": 6.0,
    "iPhone 14 Pro": 6.0,
    "iPhone 14 Pro Max": 6.0,
    "iPhone 15": 6.0,
    "iPhone 15 Plus": 6.0,
    "iPhone 15 Pro": 8.0,
    "iPhone 15 Pro Max": 8.0,
    "iPhone 16": 8.0,
    "iPhone 16 Plus": 8.0,
    "iPhone 16 Pro": 8.0,
    "iPhone 16 Pro Max": 8.0,
    "iPhone 16e": 8.0,
    "iPhone 17": 8.0,
    "iPhone Air": 12.0,
    "iPhone 17 Pro": 12.0,
    "iPhone 17 Pro Max": 12.0,
    "iPhone 17e": 8.0,
}

IPHONE_RAM_GB_BY_NORMALIZED_MODEL = {
    re.sub(r"[^a-z0-9]+", "", model.lower()): ram_gb
    for model, ram_gb in IPHONE_RAM_GB_BY_MODEL.items()
}

IPHONE_DUAL_SIM_MODELS = {
    "iPhone XR",
    "iPhone XS",
    "iPhone XS Max",
    "iPhone 11",
    "iPhone 11 Pro",
    "iPhone 11 Pro Max",
    "iPhone SE 2",
    "iPhone SE 3",
    "iPhone 12",
    "iPhone 12 mini",
    "iPhone 12 Pro",
    "iPhone 12 Pro Max",
    "iPhone 13",
    "iPhone 13 mini",
    "iPhone 13 Pro",
    "iPhone 13 Pro Max",
    "iPhone 14",
    "iPhone 14 Plus",
    "iPhone 14 Pro",
    "iPhone 14 Pro Max",
    "iPhone 15",
    "iPhone 15 Plus",
    "iPhone 15 Pro",
    "iPhone 15 Pro Max",
    "iPhone 16",
    "iPhone 16 Plus",
    "iPhone 16 Pro",
    "iPhone 16 Pro Max",
    "iPhone 16e",
    "iPhone 17",
    "iPhone Air",
    "iPhone 17 Pro",
    "iPhone 17 Pro Max",
    "iPhone 17e",
}

IPHONE_5G_MODELS = {
    "iPhone SE 3",
    "iPhone 12",
    "iPhone 12 mini",
    "iPhone 12 Pro",
    "iPhone 12 Pro Max",
    "iPhone 13",
    "iPhone 13 mini",
    "iPhone 13 Pro",
    "iPhone 13 Pro Max",
    "iPhone 14",
    "iPhone 14 Plus",
    "iPhone 14 Pro",
    "iPhone 14 Pro Max",
    "iPhone 15",
    "iPhone 15 Plus",
    "iPhone 15 Pro",
    "iPhone 15 Pro Max",
    "iPhone 16",
    "iPhone 16 Plus",
    "iPhone 16 Pro",
    "iPhone 16 Pro Max",
    "iPhone 16e",
    "iPhone 17",
    "iPhone Air",
    "iPhone 17 Pro",
    "iPhone 17 Pro Max",
    "iPhone 17e",
}

IPHONE_DUAL_SIM_MODELS_NORMALIZED = {
    re.sub(r"[^a-z0-9]+", "", model.lower())
    for model in IPHONE_DUAL_SIM_MODELS
}
IPHONE_5G_MODELS_NORMALIZED = {
    re.sub(r"[^a-z0-9]+", "", model.lower())
    for model in IPHONE_5G_MODELS
}

ANDROID_5G_BRAND_MODEL_PATTERNS = {
    "Asus": [r"\brog\b"],
    "Black View": [r"\bbl9000\s*pro\b"],
    "Google": [r"\bpixel\s*(?:5|5a|6|6a|6\s*pro|7|7a|7\s*pro|8|8a|8\s*pro|9|9a|9\s*pro|9\s*pro\s*xl)\b"],
    "Honor": [r"\b400\b", r"\b400\s*pro\b", r"\bmagic\s*7\s*pro\b", r"\bx9a\b", r"\bx9c\b"],
    "Huawei": [r"\bnova\s*7\s*se\b", r"\bp40\b"],
    "Iqoo": [r"\bz3\b", r"\bz6\s*lite\b", r"\bneo\s*10r\b"],
    "LG": [r"\bv50s?\b", r"\bv60\b", r"\bvelvet\b"],
    "Meizu": [r"\blucky\s*08\b"],
    "Nothing": [r"\bcmf\s*phone\b", r"\bphone\s*(?:1|2|2a|3a)\b"],
    "OnePlus": [r"\b(?:8|8t|9|9r|9\s*pro|10r|10t|10\s*pro|11|11r|12r|15)\b", r"\bnord\b"],
    "Oppo": [r"\breno\s*(?:4\s*z|5\s*z|6|7\s*a|8|12|12\s*pro)\b", r"\bf23\b", r"\ba78\b"],
    "Realme": [r"\bgt\b", r"\bx50\s*pro\b", r"\b11\b"],
    "Samsung": [
        r"\bgalaxy\s*(?:a25|a33|a34|a35|a36|a42|a52s|a53|a54|a55|a56)\b",
        r"\bgalaxy\s*(?:m14|m15|m23|m33|m53)\b",
        r"\bgalaxy\s*(?:note\s*20|note\s*20\s*ultra)\b",
        r"\bgalaxy\s*s(?:20|21|22|23|24)",
        r"\bz\s*flip\s*4\b",
    ],
    "Sony": [r"\bxperia\s*(?:1|5|10)\s*(?:ii|iii|iv)\b"],
    "Vivo": [r"\bv29e\b", r"\bv25\b", r"\bv21e?\b", r"\bx70\s*pro\b", r"\by100a\b"],
    "Xiaomi": [
        r"\b5g\b",
        r"\bpoco\s*(?:f5|m3\s*pro|m4\s*pro|x5\s*pro)\b",
        r"\bredmi\s*note\s*(?:10t|11e|11r|13\s*pro\s*plus|14\s*pro\s*plus)\b",
        r"\bmi\s*11x\s*pro\b",
        r"\bmi\s*9t\s*pro\b",
    ],
}

ANDROID_ESIM_BRAND_MODEL_PATTERNS = {
    "Google": [r"\bpixel\s*(?:3|3a|3a\s*xl|3\s*xl|4|4a|4\s*xl|5|5a|6|6a|6\s*pro|7|7a|7\s*pro|8|8a|8\s*pro|9|9a|9\s*pro|9\s*pro\s*xl)\b"],
    "Honor": [r"\bmagic\s*7\s*pro\b"],
    "Samsung": [
        r"\bgalaxy\s*(?:note\s*20|note\s*20\s*ultra)\b",
        r"\bgalaxy\s*s(?:20|21|22|23|24)",
        r"\bz\s*flip\s*4\b",
    ],
    "Sony": [r"\bxperia\s*(?:1|5|10)\s*iv\b"],
}


def setup_logging() -> None:
    """Configure professional terminal logs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def to_snake_case(column_name: Any) -> str:
    """Convert a column name to snake_case."""
    text = str(column_name).strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower()


def load_data(file_path: Path = INPUT_FILE) -> pd.DataFrame:
    """
    Safely load JSON data using pandas.

    The primary path uses pandas.read_json. A json.load + pandas.json_normalize
    fallback is included for nested or irregular JSON exports.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path.resolve()}")

    logging.info("Loading input dataset: %s", file_path)

    try:
        df = pd.read_json(file_path)
    except ValueError:
        logging.warning("Standard pandas.read_json failed. Trying line-delimited JSON.")
        try:
            df = pd.read_json(file_path, lines=True)
        except ValueError:
            logging.warning("Line-delimited JSON failed. Trying json_normalize fallback.")
            with file_path.open("r", encoding="utf-8") as file:
                raw_data = json.load(file)
            if isinstance(raw_data, list):
                df = pd.json_normalize(raw_data)
            elif isinstance(raw_data, dict):
                list_values = [value for value in raw_data.values() if isinstance(value, list)]
                df = pd.json_normalize(list_values[0]) if list_values else pd.json_normalize(raw_data)
            else:
                raise ValueError("Unsupported JSON structure. Expected a list or dictionary.")

    if df.empty:
        raise ValueError("Input dataset is empty. Cannot train models.")

    logging.info("Loaded %s rows and %s columns.", f"{len(df):,}", f"{len(df.columns):,}")
    return df


def parse_numeric_value(value: Any) -> float:
    """Parse a numeric value from scraped strings such as '128 GB' or 'Rs. 95,000'."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, bool):
        return float(int(value))

    text = str(value).strip().lower().replace(",", "")
    if text in {"", "none", "null", "nan", "unknown", "n/a", "na"}:
        return np.nan

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else np.nan


def parse_warranty_days(value: Any) -> float:
    """Parse warranty values and convert common units to days."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)

    text = str(value).strip().lower()
    if text in {"", "none", "null", "nan", "unknown", "n/a", "na"}:
        return np.nan
    if "no warranty" in text or text in {"no", "without warranty", "expired"}:
        return 0.0

    number = parse_numeric_value(text)
    if pd.isna(number):
        return np.nan

    if "year" in text:
        return number * 365
    if "month" in text:
        return number * 30
    if "week" in text:
        return number * 7
    return number


def parse_boolean(value: Any) -> float:
    """Convert yes/no, true/false, and 1/0 style values to 1 and 0."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return np.nan
        return 1.0 if float(value) > 0 else 0.0

    text = str(value).strip().lower()
    text = re.sub(r"[\s_-]+", " ", text)

    true_values = {
        "1",
        "yes",
        "y",
        "true",
        "t",
        "available",
        "supported",
        "support",
        "enabled",
        "dual sim",
        "5g",
        "esim",
    }
    false_values = {
        "0",
        "no",
        "n",
        "false",
        "f",
        "none",
        "not available",
        "not supported",
        "unsupported",
        "disabled",
        "single sim",
        "no 5g",
        "no esim",
    }

    if text in true_values:
        return 1.0
    if text in false_values:
        return 0.0
    if "dual" in text and "sim" in text:
        return 1.0
    if "single" in text and "sim" in text:
        return 0.0
    if "not" in text or "no " in text:
        return 0.0
    return np.nan


def clean_text_category(value: Any, unknown_value: str = "Unknown") -> str:
    """Clean a categorical text value while preserving useful model names."""
    if pd.isna(value):
        return unknown_value

    text = str(value).strip()
    if text.lower() in {"", "none", "null", "nan", "unknown", "n/a", "na"}:
        return unknown_value

    return re.sub(r"\s+", " ", text)


def standardize_brand(value: Any) -> str:
    """Standardize common smartphone brand names."""
    text = clean_text_category(value)
    lowered = text.lower().strip()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)

    if compact in {"apple", "iphone", "iphones"} or "iphone" in lowered:
        return "Apple"

    brand_aliases = {
        "samsung": "Samsung",
        "oppo": "Oppo",
        "vivo": "Vivo",
        "xiaomi": "Xiaomi",
        "mi": "Xiaomi",
        "redmi": "Redmi",
        "poco": "Poco",
        "realme": "Realme",
        "huawei": "Huawei",
        "honor": "Honor",
        "nokia": "Nokia",
        "oneplus": "OnePlus",
        "one plus": "OnePlus",
        "google": "Google",
        "pixel": "Google",
        "sony": "Sony",
        "motorola": "Motorola",
        "moto": "Motorola",
        "infinix": "Infinix",
        "tecno": "Tecno",
        "itel": "Itel",
        "lg": "LG",
        "htc": "HTC",
        "asus": "Asus",
        "lenovo": "Lenovo",
        "zte": "ZTE",
        "nothing": "Nothing",
    }

    if lowered in brand_aliases:
        return brand_aliases[lowered]
    if compact in brand_aliases:
        return brand_aliases[compact]

    for alias, normalized in brand_aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return normalized

    return text.title() if text != "Unknown" else "Unknown"


def standardize_condition(value: Any) -> str:
    """Map condition values to new, used, reconditioned, or unknown."""
    text = clean_text_category(value, unknown_value="unknown").lower()
    normalized = re.sub(r"[\s_-]+", " ", text)

    if normalized in {"unknown", "none", "null", "nan", "n/a", "na"}:
        return "unknown"
    if normalized in {"new", "brand new", "brandnew", "sealed", "unused"}:
        return "new"
    if any(term in normalized for term in ["reconditioned", "refurbished", "refurb", "renewed"]):
        return "reconditioned"
    if any(
        term in normalized
        for term in ["used", "second hand", "pre owned", "preowned", "like new", "mint", "excellent", "good"]
    ):
        return "used"

    return "unknown"


def standardize_currency(value: Any) -> str:
    """Standardize currency values and identify LKR records."""
    text = clean_text_category(value).strip()
    lowered = text.lower()
    compact = re.sub(r"[^a-z]+", "", lowered)

    if text == "Unknown":
        return "Unknown"
    if compact in {"lkr", "rs", "slrs", "srilankanrupee", "srilankanrupees"}:
        return "LKR"
    if "lkr" in compact or "rupee" in compact or lowered in {"rs.", "rs/-", "රු"}:
        return "LKR"
    if compact in {"usd", "dollar", "dollars"}:
        return "USD"
    return text.upper()


def fill_numeric_medians(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Fill missing numeric values using column medians, with a safe zero fallback."""
    for column in columns:
        median_value = df[column].median(skipna=True)
        if pd.isna(median_value):
            median_value = 0.0
            logging.warning("Column '%s' has no numeric median. Filled missing values with 0.", column)
        df[column] = df[column].fillna(median_value)

    for column in BOOLEAN_COLUMNS:
        df[column] = (df[column].astype(float) >= 0.5).astype(int)

    return df


def fill_missing_warranty_days(df: pd.DataFrame) -> pd.DataFrame:
    """Treat listings without a parsed warranty as zero warranty days."""
    missing_count = int(df["warranty_days"].isna().sum())
    if missing_count:
        df["warranty_days"] = df["warranty_days"].fillna(0.0)
        logging.info("Filled %s missing/unknown warranty_days values with 0.", f"{missing_count:,}")
    return df


def normalized_model_key(value: Any) -> str:
    """Normalize model names for matching against predefined specs."""
    return re.sub(r"[^a-z0-9]+", "", clean_text_category(value).lower())


def apply_predefined_iphone_ram(df: pd.DataFrame) -> pd.DataFrame:
    """Override scraped iPhone RAM values with known model specifications."""
    model_keys = df["model"].apply(normalized_model_key)
    predefined_ram = model_keys.map(IPHONE_RAM_GB_BY_NORMALIZED_MODEL)
    iphone_mask = (df["brand"] == "Apple") | df["model"].str.contains("iPhone", case=False, na=False)
    correction_mask = iphone_mask & predefined_ram.notna()
    changed_count = int((df.loc[correction_mask, "ram_gb"] != predefined_ram.loc[correction_mask]).sum())

    if changed_count:
        df.loc[correction_mask, "ram_gb"] = predefined_ram.loc[correction_mask].astype(float)
        logging.info("Corrected %s iPhone RAM values using predefined model specs.", f"{changed_count:,}")
    else:
        logging.info("No iPhone RAM corrections were needed.")

    return df


def model_matches_patterns(model: Any, patterns: Iterable[str]) -> bool:
    """Return whether a cleaned model name matches any predefined capability pattern."""
    model_text = clean_text_category(model).lower()
    model_text = re.sub(r"[\s_/|+-]+", " ", model_text)
    return any(re.search(pattern, model_text) for pattern in patterns)


def predefined_phone_capabilities(brand: Any, model: Any) -> Dict[str, int]:
    """Return deterministic dual-SIM, 5G, and eSIM values for a phone model."""
    brand_text = clean_text_category(brand)
    model_text = clean_text_category(model)
    model_key = normalized_model_key(model_text)

    if brand_text == "Apple" or "iphone" in model_text.lower():
        return {
            "dual_sim": int(model_key in IPHONE_DUAL_SIM_MODELS_NORMALIZED),
            "has_5g": int(model_key in IPHONE_5G_MODELS_NORMALIZED),
            "has_esim": int(model_key in IPHONE_DUAL_SIM_MODELS_NORMALIZED),
        }

    has_5g = int(
        "5g" in model_text.lower()
        or model_matches_patterns(model_text, ANDROID_5G_BRAND_MODEL_PATTERNS.get(brand_text, []))
    )
    has_esim = int(model_matches_patterns(model_text, ANDROID_ESIM_BRAND_MODEL_PATTERNS.get(brand_text, [])))

    return {
        "dual_sim": 1,
        "has_5g": has_5g,
        "has_esim": has_esim,
    }


def apply_predefined_phone_capabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Override noisy capability columns with predefined values from brand/model specs."""
    if df.empty:
        return df

    capabilities = df.apply(
        lambda row: predefined_phone_capabilities(row.get("brand"), row.get("model")),
        axis=1,
        result_type="expand",
    )

    for column in BOOLEAN_COLUMNS:
        old_values = pd.to_numeric(df[column], errors="coerce").fillna(-1).astype(int)
        new_values = capabilities[column].astype(int)
        changed_count = int((old_values != new_values).sum())
        df[column] = new_values
        logging.info("Corrected %s %s values using predefined phone capabilities.", f"{changed_count:,}", column)

    return df


def normalize_phone_type(row: pd.Series) -> str:
    """Use brand/model identity to keep the phone_type split consistent."""
    brand = str(row.get("brand", "")).strip()
    model = str(row.get("model", "")).strip()
    if brand == "Apple" or "iphone" in model.lower():
        return "iphone"
    return "android"


def prepare_ml_ready_records(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    Normalize records that already use the ML-ready schema.

    This is used for extra target-phone records so future training runs do not
    drop the manually added data.
    """
    logging.info("Preparing ML-ready records from %s.", source_label)

    df = df.copy()
    df.columns = [to_snake_case(column) for column in df.columns]

    missing_columns = [column for column in OUTPUT_COLUMNS if column not in df.columns]
    if missing_columns:
        logging.warning("Missing columns in %s will be created as nulls: %s", source_label, missing_columns)
        for column in missing_columns:
            df[column] = np.nan

    df = df[OUTPUT_COLUMNS].copy()

    before_count = len(df)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(parse_numeric_value)
    df = df[df[TARGET_COLUMN].notna() & (df[TARGET_COLUMN] > 0)].copy()
    logging.info(
        "%s: removed %s rows with missing, zero, negative, or invalid listed_price.",
        source_label,
        f"{before_count - len(df):,}",
    )

    df["currency"] = df["currency"].apply(standardize_currency)
    lkr_count = int((df["currency"] == "LKR").sum())
    if lkr_count > 0:
        before_count = len(df)
        df = df[df["currency"] == "LKR"].copy()
        logging.info(
            "%s: kept only LKR records: %s retained, %s removed.",
            source_label,
            f"{len(df):,}",
            f"{before_count - len(df):,}",
        )

    df["storage_gb"] = df["storage_gb"].apply(parse_numeric_value)
    df["ram_gb"] = df["ram_gb"].apply(parse_numeric_value)
    df["warranty_days"] = df["warranty_days"].apply(parse_warranty_days)
    df = fill_missing_warranty_days(df)

    for column in BOOLEAN_COLUMNS:
        df[column] = df[column].apply(parse_boolean)

    df = fill_numeric_medians(df, MEDIAN_IMPUTED_NUMERIC_COLUMNS)

    df["brand"] = df["brand"].apply(standardize_brand)
    df["model"] = df["model"].apply(clean_text_category)
    df["condition"] = df["condition"].apply(standardize_condition)
    df["currency"] = df["currency"].apply(lambda value: clean_text_category(value).upper())

    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].replace("", "Unknown").fillna("Unknown")

    df["phone_type"] = df.apply(normalize_phone_type, axis=1)
    df = apply_predefined_iphone_ram(df)
    df = apply_predefined_phone_capabilities(df)

    before_count = len(df)
    df = df.drop_duplicates()
    logging.info("%s: removed %s duplicate rows after normalization.", source_label, f"{before_count - len(df):,}")

    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def resolve_target_phone_file(target_path: Path = TARGET_PHONES_INPUT_FILE) -> Optional[Path]:
    """Find the target-phone file from the project folder or known scraper export folder."""
    candidates = [target_path, *TARGET_PHONES_INPUT_FALLBACK_FILES]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def merge_target_phone_records(
    cleaned_df: pd.DataFrame,
    target_path: Path = TARGET_PHONES_INPUT_FILE,
) -> pd.DataFrame:
    """Merge extra target-phone records into the cleaned ML-ready dataset."""
    resolved_target_path = resolve_target_phone_file(target_path)
    if resolved_target_path is None:
        logging.warning("Target-phone file not found, skipping merge: %s", target_path)
        return cleaned_df

    target_df = load_data(resolved_target_path)
    target_df = prepare_ml_ready_records(target_df, source_label=str(resolved_target_path))

    before_count = len(cleaned_df)
    combined_df = pd.concat([cleaned_df[OUTPUT_COLUMNS], target_df[OUTPUT_COLUMNS]], ignore_index=True)
    combined_df = apply_predefined_iphone_ram(combined_df)
    combined_df = apply_predefined_phone_capabilities(combined_df)

    before_dedupe_count = len(combined_df)
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)
    logging.info(
        "Merged target-phone records: base=%s target=%s final=%s duplicates_removed=%s.",
        f"{before_count:,}",
        f"{len(target_df):,}",
        f"{len(combined_df):,}",
        f"{before_dedupe_count - len(combined_df):,}",
    )

    return combined_df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the scraped dataset.

    Returns an ML-ready DataFrame containing the required model features,
    listed_price target, and phone_type split column.
    """
    logging.info("Starting preprocessing.")

    df = df.copy()
    df.columns = [to_snake_case(column) for column in df.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        logging.warning("Missing expected columns will be created as nulls: %s", missing_columns)
        for column in missing_columns:
            df[column] = np.nan

    df = df[REQUIRED_COLUMNS].copy()

    before_count = len(df)
    df = df.drop_duplicates()
    logging.info("Removed %s duplicate rows before cleaning.", f"{before_count - len(df):,}")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(parse_numeric_value)
    before_count = len(df)
    df = df[df[TARGET_COLUMN].notna() & (df[TARGET_COLUMN] > 0)].copy()
    logging.info("Removed %s rows with missing, zero, negative, or invalid listed_price.", f"{before_count - len(df):,}")

    df["currency"] = df["currency"].apply(standardize_currency)
    lkr_count = int((df["currency"] == "LKR").sum())
    if lkr_count > 0:
        before_count = len(df)
        df = df[df["currency"] == "LKR"].copy()
        logging.info("Kept only LKR records: %s retained, %s removed.", f"{len(df):,}", f"{before_count - len(df):,}")
    else:
        logging.warning("No LKR records found after currency standardization. Keeping all currencies.")

    df["storage_gb"] = df["storage_gb"].apply(parse_numeric_value)
    df["ram_gb"] = df["ram_gb"].apply(parse_numeric_value)
    df["warranty_days"] = df["warranty_days"].apply(parse_warranty_days)
    df = fill_missing_warranty_days(df)

    for column in BOOLEAN_COLUMNS:
        df[column] = df[column].apply(parse_boolean)

    df = fill_numeric_medians(df, MEDIAN_IMPUTED_NUMERIC_COLUMNS)

    df["brand"] = df["brand"].apply(standardize_brand)
    df["model"] = df["model"].apply(clean_text_category)
    df["condition"] = df["condition"].apply(standardize_condition)
    df["currency"] = df["currency"].apply(lambda value: clean_text_category(value).upper())

    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].replace("", "Unknown").fillna("Unknown")

    before_count = len(df)
    df = df[df["condition"] == TRAINING_CONDITION].copy()
    logging.info(
        "Kept only condition='%s' records: %s retained, %s removed.",
        TRAINING_CONDITION,
        f"{len(df):,}",
        f"{before_count - len(df):,}",
    )
    if df.empty:
        raise ValueError(f"No records remain after filtering condition='{TRAINING_CONDITION}'.")

    df["phone_type"] = np.where(df["brand"] == "Apple", "iphone", "android")
    df = apply_predefined_iphone_ram(df)
    df = apply_predefined_phone_capabilities(df)

    before_count = len(df)
    df = df.drop_duplicates()
    logging.info("Removed %s duplicate rows after standardization.", f"{before_count - len(df):,}")

    df = remove_outliers_iqr(df, group_column="phone_type", price_column=TARGET_COLUMN)

    df = df[OUTPUT_COLUMNS].reset_index(drop=True)

    logging.info("Preprocessing completed. Final cleaned records: %s", f"{len(df):,}")
    return df


def remove_outliers_iqr(
    df: pd.DataFrame,
    group_column: str = "phone_type",
    price_column: str = TARGET_COLUMN,
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """Remove extreme price outliers separately for each phone type using the IQR method."""
    cleaned_groups = []
    total_removed = 0

    for group_name, group_df in df.groupby(group_column, dropna=False):
        if len(group_df) < 4:
            logging.warning("Skipping IQR outlier removal for '%s': fewer than 4 records.", group_name)
            cleaned_groups.append(group_df)
            continue

        q1 = group_df[price_column].quantile(0.25)
        q3 = group_df[price_column].quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr <= 0:
            logging.warning("Skipping IQR outlier removal for '%s': non-positive IQR.", group_name)
            cleaned_groups.append(group_df)
            continue

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        mask = group_df[price_column].between(lower_bound, upper_bound, inclusive="both")
        removed = int((~mask).sum())
        total_removed += removed

        logging.info(
            "IQR outlier removal for %-7s | Q1: %.2f | Q3: %.2f | Bounds: %.2f to %.2f | Removed: %s",
            str(group_name),
            q1,
            q3,
            lower_bound,
            upper_bound,
            f"{removed:,}",
        )
        cleaned_groups.append(group_df.loc[mask])

    cleaned_df = pd.concat(cleaned_groups, ignore_index=True) if cleaned_groups else df.iloc[0:0].copy()
    logging.info("Total IQR price outliers removed: %s", f"{total_removed:,}")
    return cleaned_df


def split_phone_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split cleaned data into iPhone and Android datasets."""
    iphone_df = df[df["phone_type"] == "iphone"].copy()
    android_df = df[df["phone_type"] == "android"].copy()

    logging.info("iPhone records available for modeling: %s", f"{len(iphone_df):,}")
    logging.info("Android records available for modeling: %s", f"{len(android_df):,}")

    if len(iphone_df) < MIN_ROWS_WARNING:
        logging.warning("iPhone dataset has fewer than %s rows. Model results may be unreliable.", MIN_ROWS_WARNING)
    if len(android_df) < MIN_ROWS_WARNING:
        logging.warning("Android dataset has fewer than %s rows. Model results may be unreliable.", MIN_ROWS_WARNING)

    return iphone_df, android_df


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a version-compatible OneHotEncoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    """Build sklearn preprocessing for categorical and numeric model features."""
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )


def train_model(
    dataset: pd.DataFrame,
    estimator: Any,
    model_name: str,
    phone_type: str,
) -> Tuple[Optional[Pipeline], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:
    """Train a full sklearn Pipeline and return the fitted model plus train/test data."""
    if len(dataset) < MIN_ROWS_REQUIRED:
        logging.warning(
            "Skipping %s model for %s: only %s rows available. Minimum required is %s.",
            model_name,
            phone_type,
            len(dataset),
            MIN_ROWS_REQUIRED,
        )
        return None, None, None, None, None

    X = dataset[FEATURE_COLUMNS].copy()
    y = dataset[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )

    logging.info("Training %s model for %s phones.", model_name, phone_type)
    pipeline.fit(X_train, y_train)
    logging.info("Finished training %s model for %s phones.", model_name, phone_type)

    return pipeline, X_train, X_test, y_train, y_test


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    phone_type: str,
    total_records: int,
    train_records: int,
) -> Dict[str, Any]:
    """Evaluate a trained model using MAE, RMSE, R2, and MAPE."""
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions) * 100

    metrics = {
        "model_name": model_name,
        "phone_type": phone_type,
        "total_records": int(total_records),
        "train_records": int(train_records),
        "test_records": int(len(y_test)),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2_score": float(r2),
        "mape_percent": float(mape),
    }

    print_evaluation_summary(metrics)
    return metrics


def print_evaluation_summary(metrics: Dict[str, Any]) -> None:
    """Print a professional evaluation block for a trained model."""
    separator = "-" * 78
    logging.info(separator)
    logging.info("Evaluation Summary | %s | %s", metrics["phone_type"].upper(), metrics["model_name"])
    logging.info("Records           | total=%s train=%s test=%s", metrics["total_records"], metrics["train_records"], metrics["test_records"])
    logging.info("MAE               | LKR %s", f"{metrics['mae']:,.2f}")
    logging.info("RMSE              | LKR %s", f"{metrics['rmse']:,.2f}")
    logging.info("R2 Score          | %.4f", metrics["r2_score"])
    logging.info("MAPE              | %.2f%%", metrics["mape_percent"])
    logging.info(separator)


def save_model(model: Pipeline, output_path: Path) -> None:
    """Save a fitted model pipeline using joblib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logging.info("Saved model: %s", output_path)


def delete_existing_model_artifacts() -> None:
    """Delete previously trained model files so regenerated outputs are fresh."""
    removed_count = 0
    model_paths = sorted(set(MODEL_OUTPUT_PATHS).union(MODEL_DIR.glob("*.pkl")))
    for model_path in model_paths:
        if model_path.exists():
            model_path.unlink()
            removed_count += 1
            logging.info("Deleted previous model artifact: %s", model_path)

    if removed_count == 0:
        logging.info("No previous model artifacts found to delete.")


def get_random_forest_regressor() -> RandomForestRegressor:
    """Create a Random Forest regressor suitable for tabular price prediction."""
    return RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
    )


def get_xgboost_regressor() -> Optional[Any]:
    """Create an XGBoost regressor if xgboost is installed."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        logging.warning("XGBoost is not installed. Install it with: pip install xgboost")
        return None

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.90,
        colsample_bytree=0.90,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="rmse",
        tree_method="hist",
    )


def save_cleaned_dataset(df: pd.DataFrame, output_path: Path = CLEANED_OUTPUT_FILE) -> None:
    """Save the final cleaned dataset as JSON."""
    df.to_json(output_path, orient="records", indent=2, force_ascii=False)
    logging.info("Saved cleaned ML-ready dataset: %s", output_path)


def build_comparison_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Create a model comparison table sorted by phone type and MAE."""
    if not results:
        return pd.DataFrame()

    table = pd.DataFrame(results.values())
    table = table[
        [
            "phone_type",
            "model_name",
            "total_records",
            "test_records",
            "mae",
            "rmse",
            "r2_score",
            "mape_percent",
        ]
    ].copy()
    table = table.sort_values(["phone_type", "mae"], ascending=[True, True]).reset_index(drop=True)
    return table


def recommend_best_models(comparison_table: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Recommend the best model for each phone type.

    Primary criterion is lowest MAE. R2 is included as the quality cross-check.
    If MAE and R2 disagree, the recommendation uses the best combined rank.
    """
    recommendations: Dict[str, Dict[str, Any]] = {}
    if comparison_table.empty:
        return recommendations

    for phone_type, group in comparison_table.groupby("phone_type"):
        ranked = group.copy()
        ranked["mae_rank"] = ranked["mae"].rank(method="min", ascending=True)
        ranked["r2_rank"] = ranked["r2_score"].rank(method="min", ascending=False)
        ranked["combined_rank"] = ranked["mae_rank"] + ranked["r2_rank"]

        best_mae_row = ranked.sort_values("mae", ascending=True).iloc[0]
        best_r2_row = ranked.sort_values("r2_score", ascending=False).iloc[0]
        recommended_row = ranked.sort_values(["combined_rank", "mae"], ascending=[True, True]).iloc[0]

        recommendations[str(phone_type)] = {
            "recommended_model": str(recommended_row["model_name"]),
            "recommended_model_mae": float(recommended_row["mae"]),
            "recommended_model_r2_score": float(recommended_row["r2_score"]),
            "lowest_mae_model": str(best_mae_row["model_name"]),
            "lowest_mae": float(best_mae_row["mae"]),
            "highest_r2_model": str(best_r2_row["model_name"]),
            "highest_r2_score": float(best_r2_row["r2_score"]),
        }

    return recommendations


def round_lkr(value: Any) -> Optional[float]:
    """Round a numeric LKR value for clean JSON/CSV output."""
    if value is None or pd.isna(value):
        return None
    return round(float(value), 2)


def round_feature_value(value: Any) -> Optional[float]:
    """Round representative feature values without hiding useful decimals."""
    if value is None or pd.isna(value):
        return None
    return round(float(value), 2)


def most_frequent_value(series: pd.Series, default: Any) -> Any:
    """Return a stable mode-like value with a fallback for empty groups."""
    cleaned = series.dropna()
    if cleaned.empty:
        return default

    modes = cleaned.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]
    return cleaned.iloc[0]


def median_value(series: pd.Series, default: float = 0.0) -> float:
    """Return a numeric median with a safe fallback."""
    value = pd.to_numeric(series, errors="coerce").median(skipna=True)
    return float(default) if pd.isna(value) else float(value)


def representative_boolean_value(series: pd.Series, default: int = 0) -> int:
    """Return the most common boolean-style value as 0 or 1."""
    value = most_frequent_value(series, default)
    if pd.isna(value):
        return int(default)
    return int(float(value) >= 0.5)


def format_storage_label(storage_gb: Any) -> str:
    """Format storage values for readable category keys."""
    if storage_gb is None or pd.isna(storage_gb):
        return "unknown storage"

    storage_value = float(storage_gb)
    if storage_value.is_integer():
        return f"{int(storage_value)}GB"
    return f"{storage_value:g}GB"


def build_category_key(brand: str, model: str, storage_gb: Any) -> str:
    """Build a readable category key such as 'Apple iPhone 12 Pro | 256GB'."""
    brand_text = clean_text_category(brand)
    model_text = clean_text_category(model)
    name = model_text if brand_text.lower() in model_text.lower() else f"{brand_text} {model_text}"
    return f"{name} | {format_storage_label(storage_gb)}"


def classify_fair_price_confidence(sample_count: int) -> str:
    """Classify category confidence from observed market sample size."""
    if sample_count >= 20:
        return "high"
    if sample_count >= 8:
        return "medium"
    if sample_count >= 3:
        return "low"
    return "very_low"


def build_representative_category_features(group_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create one representative model input row for a phone model/storage category.

    Category identity is exact brand + model + storage. The remaining features
    use common or median values from that category so the prediction represents
    the observed market instead of a single listing.
    """
    return {
        "brand": str(most_frequent_value(group_df["brand"], "Unknown")),
        "model": str(most_frequent_value(group_df["model"], "Unknown")),
        "condition": TRAINING_CONDITION,
        "currency": "LKR",
        "dual_sim": representative_boolean_value(group_df["dual_sim"], default=0),
        "has_5g": representative_boolean_value(group_df["has_5g"], default=0),
        "has_esim": representative_boolean_value(group_df["has_esim"], default=0),
        "warranty_days": median_value(group_df["warranty_days"], default=0.0),
        "storage_gb": median_value(group_df["storage_gb"], default=0.0),
        "ram_gb": median_value(group_df["ram_gb"], default=0.0),
    }


def build_fair_price_predictions(
    cleaned_df: pd.DataFrame,
    trained_models: Dict[str, Pipeline],
    results: Dict[str, Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Predict fair prices for every observed exact model/storage category."""
    records: list[Dict[str, Any]] = []

    for category_values, group_df in cleaned_df.groupby(FAIR_PRICE_GROUP_COLUMNS, dropna=False, sort=True):
        category = dict(zip(FAIR_PRICE_GROUP_COLUMNS, category_values))
        phone_type = str(category["phone_type"])
        model_key = XGBOOST_MODEL_KEYS_BY_PHONE_TYPE.get(phone_type)

        if model_key is None or model_key not in trained_models:
            logging.warning(
                "Skipping fair-price category %s because no trained XGBoost model is available.",
                category,
            )
            continue

        representative_features = build_representative_category_features(group_df)
        prediction_input = pd.DataFrame([representative_features], columns=FEATURE_COLUMNS)
        prediction = max(0.0, float(trained_models[model_key].predict(prediction_input)[0]))

        model_mae = results.get(model_key, {}).get("mae")
        range_low = max(0.0, prediction - float(model_mae)) if model_mae is not None else None
        range_high = prediction + float(model_mae) if model_mae is not None else None

        observed_prices = group_df[TARGET_COLUMN]
        sample_count = int(len(group_df))
        brand = str(category["brand"])
        model = str(category["model"])
        storage_gb = float(category["storage_gb"]) if not pd.isna(category["storage_gb"]) else None

        records.append(
            {
                "phone_type": phone_type,
                "brand": brand,
                "model": model,
                "storage_gb": round_feature_value(storage_gb),
                "category_key": build_category_key(brand, model, storage_gb),
                "fair_price_lkr": round_lkr(prediction),
                "fair_price_range_low_lkr": round_lkr(range_low),
                "fair_price_range_high_lkr": round_lkr(range_high),
                "sample_count": sample_count,
                "confidence": classify_fair_price_confidence(sample_count),
                "observed_median_lkr": round_lkr(observed_prices.median()),
                "observed_q1_lkr": round_lkr(observed_prices.quantile(0.25)),
                "observed_q3_lkr": round_lkr(observed_prices.quantile(0.75)),
                "observed_min_lkr": round_lkr(observed_prices.min()),
                "observed_max_lkr": round_lkr(observed_prices.max()),
                "representative_ram_gb": round_feature_value(representative_features["ram_gb"]),
                "representative_warranty_days": round_feature_value(representative_features["warranty_days"]),
                "representative_dual_sim": int(representative_features["dual_sim"]),
                "representative_has_5g": int(representative_features["has_5g"]),
                "representative_has_esim": int(representative_features["has_esim"]),
                "model_used": "XGBoost Regressor",
                "model_artifact": str(MODEL_DIR / f"xgboost_{phone_type}.pkl"),
                "model_mae_lkr": round_lkr(model_mae),
            }
        )

    records = sorted(
        records,
        key=lambda row: (
            row["phone_type"],
            row["brand"],
            row["model"],
            -1 if row["storage_gb"] is None else row["storage_gb"],
        ),
    )

    logging.info("Built fair-price predictions for %s exact model/storage categories.", f"{len(records):,}")
    return records


def save_fair_price_predictions(
    fair_price_records: list[Dict[str, Any]],
    json_output_path: Path = FAIR_PRICE_OUTPUT_FILE,
    csv_output_path: Path = FAIR_PRICE_CSV_OUTPUT_FILE,
) -> Dict[str, str]:
    """Save fair-price predictions as JSON and CSV for app/API and spreadsheet use."""
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target": TARGET_COLUMN,
        "definition": (
            "XGBoost predicted fair used-phone price in LKR for each exact "
            "phone_type + brand + model + storage_gb category observed in the cleaned dataset."
        ),
        "grouping": FAIR_PRICE_GROUP_COLUMNS,
        "records": fair_price_records,
    }

    with json_output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    pd.DataFrame(fair_price_records).to_csv(csv_output_path, index=False, encoding="utf-8")

    logging.info("Saved fair-price JSON: %s", json_output_path)
    logging.info("Saved fair-price CSV : %s", csv_output_path)

    return {
        "json": str(json_output_path),
        "csv": str(csv_output_path),
    }


def print_final_summary(
    iphone_count: int,
    android_count: int,
    comparison_table: pd.DataFrame,
    recommendations: Dict[str, Dict[str, Any]],
) -> None:
    """Print the final research-run summary."""
    logging.info("=" * 78)
    logging.info("Final Dataset Summary")
    logging.info("iPhone records used : %s", f"{iphone_count:,}")
    logging.info("Android records used: %s", f"{android_count:,}")
    logging.info("=" * 78)

    if comparison_table.empty:
        logging.warning("No model comparison table available because no models were trained.")
        return

    display_table = comparison_table.copy()
    display_table["mae"] = display_table["mae"].map(lambda value: f"{value:,.2f}")
    display_table["rmse"] = display_table["rmse"].map(lambda value: f"{value:,.2f}")
    display_table["r2_score"] = display_table["r2_score"].map(lambda value: f"{value:.4f}")
    display_table["mape_percent"] = display_table["mape_percent"].map(lambda value: f"{value:.2f}%")

    logging.info("Final Model Comparison Table")
    logging.info("\n%s", display_table.to_string(index=False))

    for phone_type, recommendation in recommendations.items():
        logging.info(
            "Recommendation for %s: %s (MAE: LKR %s, R2: %.4f). Lowest MAE: %s. Highest R2: %s.",
            phone_type,
            recommendation["recommended_model"],
            f"{recommendation['recommended_model_mae']:,.2f}",
            recommendation["recommended_model_r2_score"],
            recommendation["lowest_mae_model"],
            recommendation["highest_r2_model"],
        )


def train_and_evaluate_all_models(
    iphone_df: pd.DataFrame,
    android_df: pd.DataFrame,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Pipeline]]:
    """Train all required model/data combinations and return metrics, skips, and fitted models."""
    results: Dict[str, Dict[str, Any]] = {}
    skipped_models: Dict[str, Dict[str, Any]] = {}
    trained_models: Dict[str, Pipeline] = {}

    model_specs = [
        {
            "key": "random_forest_iphone",
            "model_name": "Random Forest Regressor",
            "phone_type": "iphone",
            "dataset": iphone_df,
            "estimator": get_random_forest_regressor(),
            "output_path": MODEL_DIR / "random_forest_iphone.pkl",
        },
        {
            "key": "random_forest_android",
            "model_name": "Random Forest Regressor",
            "phone_type": "android",
            "dataset": android_df,
            "estimator": get_random_forest_regressor(),
            "output_path": MODEL_DIR / "random_forest_android.pkl",
        },
    ]

    xgboost_estimator_iphone = get_xgboost_regressor()
    xgboost_estimator_android = get_xgboost_regressor() if xgboost_estimator_iphone is not None else None

    model_specs.extend(
        [
            {
                "key": "xgboost_iphone",
                "model_name": "XGBoost Regressor",
                "phone_type": "iphone",
                "dataset": iphone_df,
                "estimator": xgboost_estimator_iphone,
                "output_path": MODEL_DIR / "xgboost_iphone.pkl",
            },
            {
                "key": "xgboost_android",
                "model_name": "XGBoost Regressor",
                "phone_type": "android",
                "dataset": android_df,
                "estimator": xgboost_estimator_android,
                "output_path": MODEL_DIR / "xgboost_android.pkl",
            },
        ]
    )

    for spec in model_specs:
        key = spec["key"]
        estimator = spec["estimator"]
        dataset = spec["dataset"]

        if estimator is None:
            skipped_models[key] = {
                "reason": "XGBoost is not installed.",
                "install_command": "pip install xgboost",
            }
            continue

        fitted_model, X_train, X_test, y_train, y_test = train_model(
            dataset=dataset,
            estimator=estimator,
            model_name=spec["model_name"],
            phone_type=spec["phone_type"],
        )

        if fitted_model is None or X_train is None or X_test is None or y_train is None or y_test is None:
            skipped_models[key] = {
                "reason": "Too few records available for reliable train/test split.",
                "available_records": int(len(dataset)),
                "minimum_required_records": MIN_ROWS_REQUIRED,
            }
            continue

        metrics = evaluate_model(
            model=fitted_model,
            X_test=X_test,
            y_test=y_test,
            model_name=spec["model_name"],
            phone_type=spec["phone_type"],
            total_records=len(dataset),
            train_records=len(y_train),
        )
        results[key] = metrics
        trained_models[key] = fitted_model
        save_model(fitted_model, spec["output_path"])

    return results, skipped_models, trained_models


def save_evaluation_results(
    results: Dict[str, Dict[str, Any]],
    skipped_models: Dict[str, Dict[str, Any]],
    recommendations: Dict[str, Dict[str, Any]],
    fair_price_output_files: Optional[Dict[str, str]] = None,
    output_path: Path = EVALUATION_OUTPUT_FILE,
) -> None:
    """Save model evaluation results and recommendations as JSON."""
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        "training_condition_filter": TRAINING_CONDITION,
        "results": results,
        "skipped_models": skipped_models,
        "recommendations": recommendations,
        "fair_price_output_files": fair_price_output_files or {},
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    logging.info("Saved evaluation results: %s", output_path)


def main() -> None:
    """Run the full preprocessing, training, evaluation, and artifact-saving pipeline."""
    setup_logging()
    logging.info("Starting mobile phone fair price model training pipeline.")

    try:
        raw_df = load_data(INPUT_FILE)
        cleaned_df = preprocess_data(raw_df)
        cleaned_df = merge_target_phone_records(cleaned_df, TARGET_PHONES_INPUT_FILE)
        save_cleaned_dataset(cleaned_df, CLEANED_OUTPUT_FILE)

        training_df = cleaned_df[cleaned_df["condition"] == TRAINING_CONDITION].copy()
        if training_df.empty:
            raise ValueError(f"No records remain for model training with condition='{TRAINING_CONDITION}'.")

        iphone_df, android_df = split_phone_types(training_df)
        delete_existing_model_artifacts()
        results, skipped_models, trained_models = train_and_evaluate_all_models(iphone_df, android_df)

        comparison_table = build_comparison_table(results)
        recommendations = recommend_best_models(comparison_table)

        fair_price_records = build_fair_price_predictions(training_df, trained_models, results)
        fair_price_output_files = save_fair_price_predictions(fair_price_records)

        save_evaluation_results(
            results,
            skipped_models,
            recommendations,
            fair_price_output_files=fair_price_output_files,
            output_path=EVALUATION_OUTPUT_FILE,
        )
        print_final_summary(len(iphone_df), len(android_df), comparison_table, recommendations)

        logging.info("Pipeline completed successfully.")

    except Exception as exc:
        logging.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
