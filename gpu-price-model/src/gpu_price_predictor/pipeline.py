from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
DEFAULT_ALIAS_PATH = REFERENCE_DIR / "gpu_model_aliases.csv"
DEFAULT_SPEC_PATH = REFERENCE_DIR / "gpu_specs.csv"
DEFAULT_DATASETS = {
    "ikman": PROJECT_ROOT / "data" / "cleaned" / "ikman_gpus_cleaned_all.json",
    "msk": PROJECT_ROOT / "data" / "cleaned" / "msk_gpus_cleaned_all.json",
    "md": PROJECT_ROOT / "data" / "cleaned" / "md_gpus_cleaned_all.json",
}

UNKNOWN = "Unknown"


# --- STRING & DATA NORMALIZATION UTILITIES ---
# Stored in: src/gpu_price_predictor/pipeline.py
# Used by: scrapers, cleaning scripts, and training pipeline

def normalize_whitespace(value: Any) -> str:
    """Removes extra spaces and tabs from strings for clean data."""
    if value is None:
        return UNKNOWN

    text = str(value).strip()
    if not text:
        return UNKNOWN

    return re.sub(r"\s+", " ", text)


def normalize_model(model: Any) -> str:
    """
    Standardizes GPU names (e.g., 'Nvidia GeForce RTX 3060' -> 'RTX 3060').
    Used to match scraped data with benchmark databases.
    """
    text = normalize_whitespace(model).upper()
    if text == UNKNOWN.upper():
        return UNKNOWN

    text = text.replace("-", " ")
    text = re.sub(r"\bGEFORCE\b", "", text)
    text = re.sub(r"\bRADEON\b", "", text)
    text = re.sub(r"\bNVIDIA\b", "", text)
    text = re.sub(r"\bAMD\b", "", text)
    text = re.sub(r"\bINTEL ARC\b", "ARC", text)
    text = re.sub(r"([A-Z]+)\s*(\d{3,4})([A-Z]*)", r"\1 \2 \3", text)
    text = re.sub(r"\b(TI|XT|XTX|SUPER)\b", r" \1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_vram(vram: Any, title: Any) -> float:
    """Extracts VRAM size (GB) from text or title using Regex."""
    if vram is not None and str(vram).strip():
        match = re.search(r"\d+", str(vram))
        if match:
            return float(match.group(0))

    title_text = str(title or "")
    match = re.search(r"(\d+)\s*[Gg][Bb]", title_text)
    if match:
        return float(match.group(1))

    return math.nan


def derive_series_family(model: str) -> str:
    """Identifies the series (e.g., RTX, GTX, RX) from the model name."""
    if model == UNKNOWN:
        return UNKNOWN

    match = re.match(r"([A-Z]+)", model)
    return match.group(1) if match else UNKNOWN


def derive_model_number(model: str) -> float:
    """Extracts the numeric part of a model (e.g., '3060' from 'RTX 3060')."""
    if model == UNKNOWN:
        return math.nan

    match = re.search(r"(\d{3,4})", model)
    if not match:
        return math.nan

    return float(match.group(1))


def derive_ti_flag(model: str) -> str:
    """Checks if a GPU is a 'Ti' variant (NVIDIA specific)."""
    if model == UNKNOWN:
        return UNKNOWN
    return "Yes" if " TI" in f" {model} " else "No"


def safe_price(value: Any) -> float:
    """Converts price strings/objects to clean float values in LKR."""
    if value is None or value == "":
        return math.nan

    try:
        price = float(value)
    except (TypeError, ValueError):
        return math.nan

    return price if price > 0 else math.nan


def extract_location(details: Any) -> str:
    """Parses location data (e.g., 'Colombo 5') from listing details."""
    text = normalize_whitespace(details)
    if text == UNKNOWN:
        return UNKNOWN
    return text.split(",")[0].strip() or UNKNOWN


def choose_manufacturer(row: pd.Series) -> str:
    """Determines the card manufacturer (e.g., ASUS, MSI) from various fields."""
    for key in ("Manufacturer", "Brand"):
        value = normalize_whitespace(row.get(key))
        if value != UNKNOWN:
            return value.upper()

    title = normalize_whitespace(row.get("Raw_Title")).upper()
    known_brands = [
        "ASUS",
        "MSI",
        "GIGABYTE",
        "ZOTAC",
        "GALAX",
        "PALIT",
        "SAPPHIRE",
        "EMTEK",
        "FORSA",
        "EVGA",
        "RANDOM BRAND",
    ]
    for brand in known_brands:
        if brand in title:
            return brand

    return UNKNOWN


# --- DATA LOADING & DEDUPLICATION ---
# Stored in: src/gpu_price_predictor/pipeline.py

def load_json_records(path: Path) -> list[dict[str, Any]]:
    """Loads records from a JSON file safely."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path.name} does not contain a list of records.")
    return data


def normalize_record_url(value: Any) -> str:
    text = normalize_whitespace(value)
    if text == UNKNOWN:
        return ""
    return text.rstrip("/")


def record_identity_for_pipeline(source: str, record: dict[str, Any]) -> str | None:
    """Creates a unique ID for each listing to prevent duplicate entries."""
    if source == "ikman":
        listing_id = normalize_whitespace(record.get("Listing_ID"))
        if listing_id != UNKNOWN:
            return f"id:{listing_id}"

        listing_url = normalize_record_url(record.get("Listing_URL"))
        if listing_url:
            return f"url:{listing_url}"

        title = normalize_whitespace(record.get("Raw_Title")).lower()
        price = str(record.get("Price_LKR", record.get("Clean_Price_LKR", ""))).strip()
        details = normalize_whitespace(record.get("Details")).lower()
        if title != UNKNOWN.lower() and price:
            return f"legacy:{title}|{price}|{details}"
        return None

    product_url = normalize_record_url(record.get("Product_URL"))
    if product_url:
        return f"url:{product_url}"

    title = normalize_whitespace(record.get("Raw_Title")).lower()
    price = str(record.get("Price_LKR", record.get("Clean_Price_LKR", ""))).strip()
    details = normalize_whitespace(record.get("Details")).lower()
    if title != UNKNOWN.lower() and price:
        return f"fallback:{title}|{price}|{details}"
    return None


def deduplicate_source_records(source: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Removes identical listings from a specific data source."""
    deduped: list[dict[str, Any]] = []
    key_to_index: dict[str, int] = {}
    missing_identity = 0

    for record in records:
        identity = record_identity_for_pipeline(source, record)
        if identity is None:
            missing_identity += 1
            deduped.append(record)
            continue

        existing_index = key_to_index.get(identity)
        if existing_index is None:
            key_to_index[identity] = len(deduped)
            deduped.append(record)
        else:
            deduped[existing_index] = record

    if missing_identity:
        raise ValueError(
            f"{source} cleaned data contains records without a usable dedup identity. "
            "Run the updated scraper to refresh the canonical dataset."
        )

    return deduped


def normalize_source_frame(source: str, path: Path) -> pd.DataFrame:
    """Converts raw scraped JSON into a standardized Pandas DataFrame."""
    raw_records = deduplicate_source_records(source, load_json_records(path))
    frame = pd.DataFrame(raw_records)
    if frame.empty:
        return frame

    frame["source"] = source
    frame["raw_title"] = frame.get("Raw_Title", pd.Series(dtype="object")).apply(normalize_whitespace)
    frame["price_lkr"] = frame.get(
        "Price_LKR",
        frame.get("Clean_Price_LKR", pd.Series(dtype="float")),
    ).apply(safe_price)
    frame["model"] = frame.get("Extracted_Model", pd.Series(dtype="object")).apply(normalize_model)
    frame["vram_gb"] = frame.apply(
        lambda row: parse_vram(row.get("VRAM_GB"), row.get("Raw_Title")),
        axis=1,
    )
    frame["manufacturer"] = frame.apply(choose_manufacturer, axis=1)
    frame["stock"] = frame.get(
        "Stock",
        frame.get("Stock_Status", pd.Series(dtype="object")),
    ).apply(normalize_whitespace).str.upper()
    frame["brand"] = frame.get("Brand", pd.Series(dtype="object")).apply(normalize_whitespace).str.upper()
    frame["location"] = frame.get("Details", pd.Series(dtype="object")).apply(extract_location)

    normalized = frame[
        [
            "source",
            "raw_title",
            "price_lkr",
            "model",
            "vram_gb",
            "manufacturer",
            "stock",
            "brand",
            "location",
        ]
    ].copy()

    normalized["brand"] = normalized["brand"].replace("", UNKNOWN)
    normalized["stock"] = normalized["stock"].replace(UNKNOWN.upper(), UNKNOWN)
    return normalized


# --- REFERENCE MAPPING UTILITIES ---
# Stored in: src/gpu_price_predictor/pipeline.py

def load_alias_map(path: Path = DEFAULT_ALIAS_PATH) -> dict[str, str]:
    """Loads mapping of GPU nicknames to official names."""
    aliases = pd.read_csv(path)
    required = {"alias", "canonical_model"}
    if not required.issubset(aliases.columns):
        raise ValueError(f"{path.name} must include columns: {sorted(required)}")

    alias_map = {
        normalize_model(row["alias"]): normalize_model(row["canonical_model"])
        for _, row in aliases.iterrows()
    }
    return alias_map


def apply_alias_map(model: str, alias_map: dict[str, str]) -> str:
    normalized = normalize_model(model)
    if normalized == UNKNOWN:
        return UNKNOWN
    return alias_map.get(normalized, normalized)


def load_spec_table(path: Path = DEFAULT_SPEC_PATH) -> pd.DataFrame:
    """Loads the main GPU technical specifications table."""
    specs = pd.read_csv(path)
    required = {
        "model",
        "vendor",
        "release_year",
        "memory_size_mb",
        "memory_type",
        "buswidth_bits",
        "gpu_clockspeed_mhz",
        "memory_clockspeed_mhz",
        "max_bandwidth_mb_s",
        "process_size_nm",
        "transistors_million",
        "external_power",
    }
    if not required.issubset(specs.columns):
        raise ValueError(f"{path.name} must include columns: {sorted(required)}")

    specs = specs.copy()
    specs["model"] = specs["model"].apply(normalize_model)
    duplicate_models = specs["model"][specs["model"].duplicated()].unique().tolist()
    if duplicate_models:
        raise ValueError(
            f"{path.name} contains duplicate canonical models: {', '.join(sorted(duplicate_models))}"
        )
    return specs


def ensure_path_list(path_config: Path | list[Path] | tuple[Path, ...]) -> list[Path]:
    if isinstance(path_config, Path):
        return [path_config]
    return list(path_config)


# --- STATISTICAL UTILITIES ---
# Stored in: src/gpu_price_predictor/pipeline.py

def compute_iqr_summary(frame: pd.DataFrame, column: str) -> dict[str, float | int]:
    """Calculates Interquartile Range to find and remove price outliers."""
    q1 = float(frame[column].quantile(0.25))
    q3 = float(frame[column].quantile(0.75))
    iqr = q3 - q1
    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)
    mask = (frame[column] < lower) | (frame[column] > upper)
    return {
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "lower_bound": round(lower, 2),
        "upper_bound": round(upper, 2),
        "outlier_rows": int(mask.sum()),
    }


NUMERIC_SPEC_COLUMNS = [
    "vram_gb",
    "model_number",
    "release_year",
    "memory_size_mb",
    "buswidth_bits",
    "gpu_clockspeed_mhz",
    "memory_clockspeed_mhz",
    "max_bandwidth_mb_s",
    "process_size_nm",
    "transistors_million",
    "shader_cores_or_stream_processors",
    "boost_clock_mhz",
]

CATEGORICAL_SPEC_COLUMNS = [
    "model",
    "manufacturer",
    "brand",
    "series_family",
    "ti_variant",
    "vram_gb_missing",
    "vendor",
    "memory_type",
    "external_power",
]

FEATURE_COLUMNS = NUMERIC_SPEC_COLUMNS + CATEGORICAL_SPEC_COLUMNS
TARGET_COLUMN = "price_lkr"


# ── V2 Feature Definitions ────────────────────────────────────────────────────

FEATURE_COLUMNS_V2_NUMERIC = [
    "vram_gb",
    "G3Dmark",
    "G2Dmark",
    "log_G3Dmark",
    "fp32_gflops",
    "tdp_watts",
    "memory_bandwidth_gb_s",
    "shader_units",
    "gpu_base_clock_mhz",
    "boost_clock_mhz",
    "perf_per_watt",
    "gpu_age_years",
    "gpu_generation",
    "model_number",
    "ti_variant",
]

FEATURE_COLUMNS_V2_CATEGORICAL = [
    "series_family",
    "brand",
    "architecture",
]

FEATURE_COLUMNS_V2 = FEATURE_COLUMNS_V2_NUMERIC + FEATURE_COLUMNS_V2_CATEGORICAL

TARGET_COLUMN_V2 = "log_price_lkr"


def derive_gpu_generation(model_name: str) -> int:
    """
    Maps a GPU model name to an ordinal generation integer (e.g. RTX 3060 -> 5).
    Crucial feature for the ML model to understand 'new' vs 'old' tech.

    NVIDIA:  GT/GTS=1, GTX 9xx=2, GTX 10xx/16xx=3,
             RTX 20xx=4, RTX 30xx=5, RTX 40xx=6, RTX 50xx=7
    AMD:     RX 4xx/5xx/5xxx=3, RX 6xxx=4, RX 7xxx=5
    default: 0
    """
    num_match = re.search(r"\b(\d{3,4})\b", str(model_name))
    if not num_match:
        return 0
    num = float(num_match.group(1))

    family_match = re.search(r"\b(RTX|GTX|GTS|GT|RX|HD)\b", str(model_name), re.IGNORECASE)
    family = family_match.group(1).upper() if family_match else ""

    if family in ("GTX", "GTS", "GT"):
        if 900 <= num <= 999:
            return 2
        if 1000 <= num <= 1699:
            return 3
        return 1

    if family == "RTX":
        if 2000 <= num < 3000:
            return 4
        if 3000 <= num < 4000:
            return 5
        if 4000 <= num < 5000:
            return 6
        if 5000 <= num < 6000:
            return 7
        return 4

    if family == "RX":
        if num < 1000:
            return 3
        if 5000 <= num < 6000:
            return 3
        if 6000 <= num < 7000:
            return 4
        if 7000 <= num < 8000:
            return 5
        return 3

    return 0


@dataclass
class TrainingDatasetBundle:
    dataset: pd.DataFrame
    unmatched_models: pd.DataFrame
    alias_table: pd.DataFrame
    spec_table: pd.DataFrame
    iqr_summary: dict[str, float | int]


# --- TRAINING DATASET BUILDERS ---
# Stored in: src/gpu_price_predictor/pipeline.py
# Imported by: scripts/train_model_v2.py

def build_training_dataset_bundle(
    dataset_paths: dict[str, Path | list[Path] | tuple[Path, ...]] | None = None,
    alias_path: Path = DEFAULT_ALIAS_PATH,
    spec_path: Path = DEFAULT_SPEC_PATH,
) -> TrainingDatasetBundle:
    """
    The main engine for creating a training-ready dataset.
    Combines scraped data + benchmarks + outlier removal.
    """
    dataset_paths = dataset_paths or DEFAULT_DATASETS
    alias_map = load_alias_map(alias_path)
    spec_table = load_spec_table(spec_path)
    frames = []

    for source, path_config in dataset_paths.items():
        for path in ensure_path_list(path_config):
            if not path.exists():
                raise FileNotFoundError(f"Missing dataset: {path}")
            frames.append(normalize_source_frame(source, path))

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["price_lkr"])
    merged = merged[merged["model"] != UNKNOWN].copy()
    merged["normalized_model"] = merged["model"].apply(normalize_model)
    merged["model"] = merged["normalized_model"].apply(lambda value: apply_alias_map(value, alias_map))
    merged["manufacturer"] = merged["manufacturer"].replace("", UNKNOWN).fillna(UNKNOWN)
    merged["brand"] = merged["brand"].replace("", UNKNOWN).fillna(UNKNOWN)
    merged["stock"] = merged["stock"].replace("", UNKNOWN).fillna(UNKNOWN)
    merged["location"] = merged["location"].replace("", UNKNOWN).fillna(UNKNOWN)
    merged["series_family"] = merged["model"].apply(derive_series_family)
    merged["model_number"] = merged["model"].apply(derive_model_number)
    merged["ti_variant"] = merged["model"].apply(derive_ti_flag)
    merged["vram_gb_missing"] = np.where(np.isnan(merged["vram_gb"]), "Yes", "No")

    enriched = merged.merge(spec_table, how="left", on="model")

    unmatched = enriched[enriched["vendor"].isna()].copy()
    unmatched_audit = unmatched[
        [
            "source",
            "raw_title",
            "normalized_model",
            "model",
            "price_lkr",
            "manufacturer",
            "brand",
            "location",
        ]
    ].rename(columns={"model": "canonical_model"})

    matched = enriched[enriched["vendor"].notna()].copy()
    matched["vram_gb"] = matched["vram_gb"].fillna(matched["memory_size_mb"] / 1024.0)
    matched["vram_gb_missing"] = np.where(np.isnan(matched["vram_gb"]), "Yes", "No")

    numeric_fill_columns = [
        "vram_gb",
        "model_number",
        "release_year",
        "memory_size_mb",
        "buswidth_bits",
        "gpu_clockspeed_mhz",
        "memory_clockspeed_mhz",
        "max_bandwidth_mb_s",
        "process_size_nm",
        "transistors_million",
        "shader_cores_or_stream_processors",
        "boost_clock_mhz",
    ]
    for column in numeric_fill_columns:
        matched[column] = pd.to_numeric(matched[column], errors="coerce")
        matched[column] = matched[column].fillna(matched[column].median())

    matched["memory_type"] = matched["memory_type"].fillna(UNKNOWN)
    matched["external_power"] = matched["external_power"].fillna(UNKNOWN)
    matched["vendor"] = matched["vendor"].fillna(UNKNOWN)
    matched = matched.sort_values(["source", "model", "price_lkr"]).reset_index(drop=True)

    # --- Group-wise Outlier Removal (Per-Model IQR) ---
    # Instead of comparing ALL GPUs together, we calculate the normal price
    # range for EACH model separately. This catches broken/faulty cards that
    # are priced far below the market value for that specific model.
    # e.g. A "broken RTX 3060" at 22,000 LKR will be flagged because other
    #      RTX 3060 listings are typically 85,000–110,000 LKR.
    def _filter_model_outliers(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) < 4:
            # Too few listings to reliably detect outliers; keep all
            return group
        q1 = group[TARGET_COLUMN].quantile(0.25)
        q3 = group[TARGET_COLUMN].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return group[(group[TARGET_COLUMN] >= lower) & (group[TARGET_COLUMN] <= upper)]

    before_count = len(matched)
    matched = matched.groupby("model", group_keys=False).apply(_filter_model_outliers)
    matched = matched.reset_index(drop=True)
    dropped = before_count - len(matched)
    if dropped:
        print(f"  [IQR] Removed {dropped} per-model price outliers.")

    alias_table = pd.read_csv(alias_path).copy()
    alias_table["alias"] = alias_table["alias"].apply(normalize_model)
    alias_table["canonical_model"] = alias_table["canonical_model"].apply(normalize_model)

    return TrainingDatasetBundle(
        dataset=matched,
        unmatched_models=unmatched_audit.reset_index(drop=True),
        alias_table=alias_table,
        spec_table=spec_table.reset_index(drop=True),
        iqr_summary=compute_iqr_summary(matched, TARGET_COLUMN),
    )


def build_training_dataset(
    dataset_paths: dict[str, Path | list[Path] | tuple[Path, ...]] | None = None,
    alias_path: Path = DEFAULT_ALIAS_PATH,
    spec_path: Path = DEFAULT_SPEC_PATH,
) -> pd.DataFrame:
    return build_training_dataset_bundle(
        dataset_paths=dataset_paths,
        alias_path=alias_path,
        spec_path=spec_path,
    ).dataset


def prepare_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[FEATURE_COLUMNS].copy()


# --- ML EVALUATION METRICS ---
# Stored in: src/gpu_price_predictor/pipeline.py
# Imported by: src/gpu_price_predictor/model_training.py & scripts/train_model_v2.py

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculates accuracy metrics (MAE, RMSE, MAPE, R2)."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    safe_true = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / safe_true)) * 100)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 0.0 if ss_tot == 0 else 1 - (ss_res / ss_tot)
    return {
        "mae_lkr": round(mae, 2),
        "rmse_lkr": round(rmse, 2),
        "mape_percent": round(mape, 2),
        "r2": round(float(r2), 4),
        "within_10pct": round(within10pct_accuracy(y_true, y_pred), 2),
    }


def within10pct_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the % of predictions that are within 10% of the actual price."""
    safe = np.where(y_true == 0, 1.0, y_true)
    within = np.abs((y_true - y_pred) / safe) <= 0.10
    return float(within.mean() * 100)


def baseline_median_by_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    model_medians = train_df.groupby("model")[TARGET_COLUMN].median()
    global_median = float(train_df[TARGET_COLUMN].median())
    preds = test_df["model"].map(model_medians).fillna(global_median)
    return preds.to_numpy(dtype=float)


def build_segment_metrics(frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, dict[str, float] | int]]:
    metrics: dict[str, dict[str, dict[str, float] | int]] = {}
    segment_config = {
        "vendor": frame["vendor"].fillna(UNKNOWN),
        "source": frame["source"].fillna(UNKNOWN),
        # Broad price bands make it easy to explain the model's behavior in the report.
        "price_band": pd.cut(
            frame["price_lkr"],
            bins=[-np.inf, 20000, 60000, np.inf],
            labels=["low", "mid", "high"],
        ).astype(str),
    }

    for segment_name, labels in segment_config.items():
        metrics[segment_name] = {}
        label_series = pd.Series(labels, index=frame.index)
        for label in sorted(label_series.dropna().unique().tolist()):
            mask = label_series == label
            if int(mask.sum()) == 0:
                continue
            metrics[segment_name][label] = {
                "rows": int(mask.sum()),
                **evaluate_predictions(y_true[mask.to_numpy()], y_pred[mask.to_numpy()]),
            }
    return metrics


def build_stratify_labels(frame: pd.DataFrame) -> pd.Series | None:
    price_bins = pd.qcut(frame["price_lkr"], q=4, duplicates="drop")
    labels = frame["vendor"].astype(str) + "|" + price_bins.astype(str)
    counts = labels.value_counts()
    if counts.min() < 2:
        return None
    return labels


@dataclass
class PredictionArtifacts:
    merged_dataset_path: Path
    metrics_path: Path
    model_path: Path


# --- CENTRALIZED INFERENCE PREPROCESSING ---

def build_inference_feature_frame(
    model_name: str,
    vram_gb: float,
    brand: str = "Any",
    enriched_df: pd.DataFrame | None = None,
    custom_specs: dict | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Centralized, robust inference feature engineering.
    Converts raw user input + optional dataset/spec lookups into a complete,
    feature-aligned DataFrame compatible with all trained ML models.
    """
    feature_cols = feature_columns or FEATURE_COLUMNS_V2
    inf: dict = {col: np.nan for col in feature_cols}

    norm_name = normalize_model(model_name)

    inf.update({
        "vram_gb": float(vram_gb) if vram_gb else 4.0,
        "series_family": derive_series_family(norm_name),
        "model_number": derive_model_number(norm_name),
        "ti_variant": 1 if "TI" in norm_name.upper() else 0,
        "gpu_generation": derive_gpu_generation(norm_name),
    })

    # Enrich from dataset if model matches historical listings
    if enriched_df is not None and not enriched_df.empty:
        model_col = None
        for col_candidate in ["extracted_model", "model", "norm_model"]:
            if col_candidate in enriched_df.columns:
                model_col = col_candidate
                break

        matches = pd.DataFrame()
        if model_col:
            norm_target = normalize_model(model_name)
            matches = enriched_df[
                enriched_df[model_col].astype(str).apply(normalize_model) == norm_target
            ]

        if not matches.empty:
            num_cols = [
                "G3Dmark", "G2Dmark", "log_G3Dmark", "fp32_gflops", "tdp_watts",
                "memory_bandwidth_gb_s", "shader_units", "gpu_base_clock_mhz",
                "boost_clock_mhz", "perf_per_watt", "gpu_age_years",
            ]
            for col in num_cols:
                if col in matches.columns and col in feature_cols:
                    v = matches[col].dropna().median()
                    if pd.notna(v):
                        inf[col] = float(v)

            if "architecture" in matches.columns and "architecture" in feature_cols:
                mode = matches["architecture"].mode()
                if not mode.empty:
                    inf["architecture"] = mode.iloc[0]

            # Brand handling: if brand is "Any" or "Unknown", use modal brand from matches
            if brand in ("Any", "Unknown", None, ""):
                if "brand" in matches.columns:
                    valid_brands = matches[~matches["brand"].isin(["Unknown", "Any"])]["brand"]
                    if not valid_brands.empty:
                        b_mode = valid_brands.mode()
                        if not b_mode.empty:
                            inf["brand"] = b_mode.iloc[0]

    # Overlay custom specs if provided (e.g. from lookup_gpu_specs in Tab 2)
    if custom_specs and isinstance(custom_specs, dict):
        for k in ["G3Dmark", "G2Dmark", "fp32_gflops", "tdp_watts", "memory_bandwidth_gb_s",
                  "shader_units", "gpu_base_clock_mhz", "boost_clock_mhz"]:
            if custom_specs.get(k) and pd.notna(custom_specs.get(k)):
                inf[k] = float(custom_specs[k])

        if custom_specs.get("architecture"):
            inf["architecture"] = str(custom_specs["architecture"])

        if custom_specs.get("release_year"):
            inf["gpu_age_years"] = float(2026 - int(custom_specs["release_year"]))

    # Final safeguards & default fallbacks
    if brand not in ("Any", "Unknown", None, ""):
        inf["brand"] = str(brand)
    elif pd.isna(inf.get("brand")) or inf.get("brand") in ("Any", "Unknown"):
        inf["brand"] = "ASUS"  # Default to major brand to avoid -1 ordinal encoding penalty

    if pd.isna(inf.get("architecture")):
        inf["architecture"] = "Unknown"

    if pd.notna(inf.get("G3Dmark")) and (pd.isna(inf.get("log_G3Dmark")) or inf.get("log_G3Dmark") == 0.0):
        inf["log_G3Dmark"] = float(np.log1p(inf["G3Dmark"]))

    if pd.notna(inf.get("G3Dmark")) and pd.notna(inf.get("tdp_watts")) and inf.get("tdp_watts") > 0:
        if pd.isna(inf.get("perf_per_watt")):
            inf["perf_per_watt"] = float(inf["G3Dmark"]) / float(inf["tdp_watts"])

    return pd.DataFrame([inf])[feature_cols]
