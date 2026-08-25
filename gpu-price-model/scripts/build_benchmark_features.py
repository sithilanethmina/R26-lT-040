"""
Phase 1 - Build Benchmark Features
====================================
Produces: data/final/gpu_enriched_dataset.csv

Steps
-----
1. Load training_data_v1.json + training_data_v2.json  ->  combined market listings
2. Fuzzy-join listings -> GPU_benchmarks_v7.csv          ->  G3Dmark, G2Dmark, TDP, testDate
3. Fuzzy-join listings -> gpu_1986-2026.csv              ->  fp32_gflops, memory_bandwidth_gb_s,
                                                             shader_units, gpu_base_clock_mhz,
                                                             boost_clock_mhz, architecture,
                                                             release_year (ground-truth)
4. Engineer derived features:
       gpu_age_years  = 2026 - release_year
       gpu_generation = regex from model number
       perf_per_watt  = G3Dmark / TDP  (or fp32_gflops / TDP as fallback)
       log_G3Dmark    = log(G3Dmark + 1)
       model_number   = 3-4 digit number extracted from model name
       series_family  = GTX / RTX / RX / GT etc.
       ti_variant     = 1 if "Ti"/"TI" in name else 0
5. Global IQR outlier filter on price_lkr
6. Save to data/final/gpu_enriched_dataset.csv
7. Print: match rates, feature coverage stats, sample rows
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import process as fz_process, fuzz

# -- Paths ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "final"

LISTING_V1 = DATA_DIR / "training_data_v1.json"
LISTING_V2 = DATA_DIR / "training_data_v2.json"
LISTING_V3 = DATA_DIR / "training_data_v3.json"
BENCHMARKS_CSV = DATA_DIR / "GPU_benchmarks_v7.csv"
SPECS_CSV = DATA_DIR / "gpu_1986-2026.csv"
OUTPUT_CSV = DATA_DIR / "gpu_enriched_dataset.csv"

# -- Constants -----------------------------------------------------------------
CURRENT_YEAR = 2026
FUZZY_THRESHOLD_BENCH = 85   # threshold for benchmark join
FUZZY_THRESHOLD_SPEC = 82    # slightly looser for spec join (longer names)

# -- Helpers -------------------------------------------------------------------

def normalize_model_name(raw: str) -> str:
    """
    Expand short listing model names to match benchmark/spec naming style.

    Examples
    --------
    "GTX 960"      ->  "GeForce GTX 960"
    "RTX 3060 TI"  ->  "GeForce RTX 3060 Ti"
    "RX 5700 XT"   ->  "Radeon RX 5700 XT"
    "GT 1030"      ->  "GeForce GT 1030"
    """
    s = str(raw).strip()

    # Standardise Ti/TI casing
    s = re.sub(r'\bTI\b', 'Ti', s, flags=re.IGNORECASE)
    s = re.sub(r'\bTi\b', 'Ti', s)

    # Expand NVIDIA prefixes
    if re.match(r'^(GTX|RTX|GT)\s', s, re.IGNORECASE):
        s = "GeForce " + s
    # Expand AMD prefixes
    elif re.match(r'^RX\s', s, re.IGNORECASE):
        s = "Radeon " + s

    return s.strip()


def parse_numeric(value_str, unit_hint="") -> float | None:
    """
    Extract the first float from a string like '9.7 TFLOPS', '200 W', '672.0 GB/s',
    '2048 MHz', etc.  Converts TFLOPS -> GFLOPS automatically.
    Returns None if unparseable.
    """
    if pd.isna(value_str) or str(value_str).strip() in ("", "unknown", "N/A"):
        return None
    s = str(value_str)
    # Remove commas (e.g. "1,234.5")
    s = s.replace(",", "")
    match = re.search(r"[\d]+(?:\.\d+)?", s)
    if not match:
        return None
    val = float(match.group())
    # TFLOPS -> GFLOPS
    if "TFLOP" in s.upper():
        val *= 1000.0
    return val


def derive_series_family(model_name: str) -> str:
    """Return GTX / RTX / RX / GT / GTS / HD / other from normalized model name."""
    m = re.search(r'\b(RTX|GTX|GTS|GT|RX|HD)\b', model_name, re.IGNORECASE)
    return m.group(1).upper() if m else "OTHER"


def derive_model_number(model_name: str) -> float | None:
    """Extract the 3-4 digit model number, e.g. 1060, 3070, 6700."""
    match = re.search(r'\b(\d{3,4})\b', model_name)
    return float(match.group(1)) if match else None


def derive_gpu_generation(model_name: str) -> int:
    """
    Map GPU generation to an ordinal int.

    NVIDIA:  GTX 9xx=2, GTX 10xx=3, GTX 16xx=3, RTX 20xx=4,
             RTX 30xx=5, RTX 40xx=6, RTX 50xx=7
    AMD:     RX 4xx/5xx=3, RX 5xxx=3, RX 6xxx=4, RX 7xxx=5
    GT:      1
    default: 0
    """
    num = derive_model_number(model_name)
    if num is None:
        return 0

    family = derive_series_family(model_name)

    if family in ("GTX", "GTS", "GT"):
        if 900 <= num <= 999:
            return 2
        if 1000 <= num <= 1099 or 1600 <= num <= 1699:
            return 3
        return 1

    if family == "RTX":
        if 2000 <= num <= 2099:
            return 4
        if 3000 <= num <= 3099:
            return 5
        if 4000 <= num <= 4099:
            return 6
        if 5000 <= num <= 5099:
            return 7
        return 4  # fallback for RTX

    if family == "RX":
        if num < 1000:
            return 3   # RX 4xx / 5xx
        if 5000 <= num <= 5999:
            return 3
        if 6000 <= num <= 6999:
            return 4
        if 7000 <= num <= 7999:
            return 5
        return 3

    return 0


def is_ti_variant(model_name: str) -> int:
    return 1 if re.search(r'\bTi\b', model_name, re.IGNORECASE) else 0


def apply_iqr_filter(df: pd.DataFrame, col: str = "price_lkr") -> pd.DataFrame:
    """
    Applies per-model IQR outlier removal instead of global quantile truncation.
    This preserves high-end GPUs (>135k LKR) while filtering broken/faulty cards
    that are priced far below market for their specific model.
    """
    before_count = len(df)
    df = df[df[col] >= 3000].copy()

    def _filter_group(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) < 4:
            return group
        q1 = group[col].quantile(0.25)
        q3 = group[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return group[(group[col] >= lower) & (group[col] <= upper)]

    df = df.groupby("extracted_model", group_keys=False).apply(_filter_group).reset_index(drop=True)
    n_removed = before_count - len(df)
    print(f"  Per-model IQR filter: removed {n_removed} price outlier rows")
    return df


def fuzzy_join(
    query_series: pd.Series,
    choices: list[str],
    threshold: int,
    scorer=fuzz.token_sort_ratio,
) -> pd.Series:
    """
    For each value in query_series, find the best match in choices above threshold.
    Returns a Series of the matched choice strings (NaN where no match found).
    """
    results = []
    for q in query_series:
        match = fz_process.extractOne(q, choices, scorer=scorer, score_cutoff=threshold)
        results.append(match[0] if match else None)
    return pd.Series(results, index=query_series.index)


# -- 1. Load Listings ----------------------------------------------------------

def load_listings() -> pd.DataFrame:
    print("=" * 60)
    print("Step 1: Loading market listings ...")
    v1 = pd.DataFrame(json.loads(LISTING_V1.read_text(encoding="utf-8"))) if LISTING_V1.exists() else pd.DataFrame()
    v2 = pd.DataFrame(json.loads(LISTING_V2.read_text(encoding="utf-8"))) if LISTING_V2.exists() else pd.DataFrame()
    v3 = pd.DataFrame(json.loads(LISTING_V3.read_text(encoding="utf-8"))) if LISTING_V3.exists() else pd.DataFrame()
    df = pd.concat([v1, v2, v3], ignore_index=True)

    # Normalise column names
    df.rename(columns={
        "Price_LKR": "price_lkr",
        "Extracted_Model": "extracted_model",
        "VRAM_GB": "vram_gb",
        "Brand": "brand",
        "Listing_ID": "listing_id",
        "Product_ID": "listing_id",
        "Listing_URL": "listing_url",
        "Product_URL": "listing_url",
        "Raw_Title": "raw_title",
        "Scraped_At_UTC": "scraped_at_utc",
    }, inplace=True)

    # Drop rows with missing price or model
    before = len(df)
    df.dropna(subset=["price_lkr", "extracted_model"], inplace=True)
    df = df[df["price_lkr"] > 0].copy()
    print(f"  Loaded {before} total rows -> {len(df)} after dropping nulls/zero-price")

    # Multi-tiered Deduplication to eliminate duplicate listing leakage
    before_dedup = len(df)

    # Tier 1: Deduplicate by unique Listing ID if present (keep latest scrape)
    if "listing_id" in df.columns and df["listing_id"].notna().any():
        if "scraped_at_utc" in df.columns and df["scraped_at_utc"].notna().any():
            df.sort_values("scraped_at_utc", ascending=True, inplace=True)
        has_id = df["listing_id"].notna()
        dedup_id = df[has_id].drop_duplicates(subset=["listing_id"], keep="last")
        df = pd.concat([dedup_id, df[~has_id]], ignore_index=True)

    # Tier 2: Deduplicate by unique Listing URL if present
    if "listing_url" in df.columns and df["listing_url"].notna().any():
        has_url = df["listing_url"].notna()
        dedup_url = df[has_url].drop_duplicates(subset=["listing_url"], keep="last")
        df = pd.concat([dedup_url, df[~has_url]], ignore_index=True)

    # Tier 3: Deduplicate identical title + price
    if "raw_title" in df.columns and df["raw_title"].notna().any():
        has_title = df["raw_title"].notna()
        dedup_title = df[has_title].drop_duplicates(subset=["raw_title", "price_lkr"], keep="last")
        df = pd.concat([dedup_title, df[~has_title]], ignore_index=True)

    # Tier 4: Exact attribute collision (model, price, vram, brand)
    df.drop_duplicates(subset=["extracted_model", "price_lkr", "vram_gb", "brand"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Deduplication: removed {before_dedup - len(df)} duplicate listings -> {len(df)} unique records")

    # Create normalised model column for fuzzy matching
    df["norm_model"] = df["extracted_model"].apply(normalize_model_name)

    return df


# -- 2. Fuzzy-join to PassMark Benchmarks ------------------------------------

def join_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    print("\nStep 2: Fuzzy-joining to GPU_benchmarks_v7.csv ...")
    bdf = pd.read_csv(BENCHMARKS_CSV)
    bdf.columns = bdf.columns.str.strip()

    bench_names = bdf["gpuName"].dropna().unique().tolist()

    # Match each unique normalised model to a benchmark GPU name
    unique_models = df["norm_model"].unique()
    print(f"  Unique model names to match: {len(unique_models)}")

    match_map: dict[str, str | None] = {}
    for nm in unique_models:
        hit = fz_process.extractOne(nm, bench_names, scorer=fuzz.token_sort_ratio,
                                    score_cutoff=FUZZY_THRESHOLD_BENCH)
        match_map[nm] = hit[0] if hit else None

    matched = sum(1 for v in match_map.values() if v is not None)
    print(f"  Benchmark match rate: {matched}/{len(unique_models)} "
          f"({matched / len(unique_models) * 100:.1f}%)")

    if matched / len(unique_models) < 0.60:
        print("  [!]  Match rate below 60% - consider lowering FUZZY_THRESHOLD_BENCH to 75.")

    df["bench_match"] = df["norm_model"].map(match_map)

    # Merge on matched name
    bdf_slim = bdf[["gpuName", "G3Dmark", "G2Dmark", "TDP", "testDate"]].copy()
    bdf_slim.columns = ["bench_match", "G3Dmark", "G2Dmark", "bench_tdp", "release_year_bench"]
    bdf_slim = bdf_slim.drop_duplicates(subset=["bench_match"])

    df = df.merge(bdf_slim, on="bench_match", how="left")

    # Parse numeric fields
    df["G3Dmark"] = pd.to_numeric(df["G3Dmark"], errors="coerce")
    df["G2Dmark"] = pd.to_numeric(df["G2Dmark"], errors="coerce")
    df["bench_tdp"] = pd.to_numeric(df["bench_tdp"], errors="coerce")
    df["release_year_bench"] = pd.to_numeric(df["release_year_bench"], errors="coerce")

    bench_coverage = df["G3Dmark"].notna().mean() * 100
    print(f"  G3Dmark coverage: {bench_coverage:.1f}%")

    return df


# -- 3. Fuzzy-join to TechPowerUp Specs --------------------------------------

def join_specs(df: pd.DataFrame) -> pd.DataFrame:
    print("\nStep 3: Fuzzy-joining to gpu_1986-2026.csv for hardware specs ...")

    # Read only the columns we need to save memory
    SPEC_COLS = [
        "Name",
        "Graphics Processor__Architecture",
        "Graphics Card__Release Date",
        "Memory__Bandwidth",
        "Render Config__Shading Units",
        "Clock Speeds__Base Clock",
        "Clock Speeds__Boost Clock",
        "Board Design__TDP",
        "Theoretical Performance__FP32 (float)",
    ]
    sdf = pd.read_csv(SPECS_CSV, usecols=lambda c: c in SPEC_COLS, low_memory=False)

    # Keep desktop/dedicated cards only (Name column is the card name, not the GPU chip)
    spec_names = sdf["Name"].dropna().unique().tolist()

    unique_models = df["norm_model"].unique()
    print(f"  Unique model names to match: {len(unique_models)}")

    match_map: dict[str, str | None] = {}
    for nm in unique_models:
        hit = fz_process.extractOne(nm, spec_names, scorer=fuzz.token_sort_ratio,
                                    score_cutoff=FUZZY_THRESHOLD_SPEC)
        match_map[nm] = hit[0] if hit else None

    matched = sum(1 for v in match_map.values() if v is not None)
    print(f"  Spec match rate: {matched}/{len(unique_models)} "
          f"({matched / len(unique_models) * 100:.1f}%)")

    df["spec_match"] = df["norm_model"].map(match_map)

    # Slim down spec df
    sdf_slim = sdf.rename(columns={
        "Name": "spec_match",
        "Graphics Processor__Architecture": "architecture",
        "Graphics Card__Release Date": "_release_date_raw",
        "Memory__Bandwidth": "_bandwidth_raw",
        "Render Config__Shading Units": "_shader_raw",
        "Clock Speeds__Base Clock": "_base_clock_raw",
        "Clock Speeds__Boost Clock": "_boost_clock_raw",
        "Board Design__TDP": "_spec_tdp_raw",
        "Theoretical Performance__FP32 (float)": "_fp32_raw",
    }).drop_duplicates(subset=["spec_match"])

    df = df.merge(sdf_slim, on="spec_match", how="left")

    # Parse fields
    def extract_year(s):
        if pd.isna(s):
            return None
        m = re.search(r'\b(19|20)\d{2}\b', str(s))
        return int(m.group()) if m else None

    df["release_year_spec"] = df["_release_date_raw"].apply(extract_year)
    df["memory_bandwidth_gb_s"] = df["_bandwidth_raw"].apply(parse_numeric)
    df["shader_units"] = pd.to_numeric(df["_shader_raw"], errors="coerce")
    df["gpu_base_clock_mhz"] = df["_base_clock_raw"].apply(parse_numeric)
    df["boost_clock_mhz"] = df["_boost_clock_raw"].apply(parse_numeric)
    df["spec_tdp_watts"] = df["_spec_tdp_raw"].apply(parse_numeric)
    df["fp32_gflops"] = df["_fp32_raw"].apply(parse_numeric)

    # Drop raw parse columns
    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True)

    # Print coverage
    for feat in ["architecture", "release_year_spec", "memory_bandwidth_gb_s",
                 "shader_units", "fp32_gflops", "spec_tdp_watts"]:
        cov = df[feat].notna().mean() * 100
        print(f"    {feat:<28}: {cov:.1f}%")

    return df


def derive_tier_class(model_name: str) -> str:
    """Extract performance tier class (10, 30, 50, 60, 70, 80, 90, Other) from GPU model name."""
    s = str(model_name).upper()
    if re.search(r'\b(90|3090|4090)\b', s):
        return "90"
    if re.search(r'\b(80|1080|2080|3080|4080|580|680|780|5800|6800|7800)\b', s):
        return "80"
    if re.search(r'\b(70|1070|2070|3070|4070|570|670|770|5700|6700|7700)\b', s):
        return "70"
    if re.search(r'\b(60|1060|2060|3060|4060|560|660|760|5600|6600|7600)\b', s):
        return "60"
    if re.search(r'\b(50|1050|1650|3050|4050|550|650|750|5500|6500|7500)\b', s):
        return "50"
    if re.search(r'\b(30|730|630|430|1630)\b', s):
        return "30"
    if re.search(r'\b(10|710|610)\b', s):
        return "10"
    return "Other"


# -- 4. Engineer Derived Features ---------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\nStep 4: Engineering derived features ...")

    # --- Release year: prefer spec, fallback to bench ---
    df["release_year"] = df["release_year_spec"].combine_first(df["release_year_bench"])
    df["gpu_age_years"] = CURRENT_YEAR - df["release_year"]
    # Clamp age to sensible range
    df["gpu_age_years"] = df["gpu_age_years"].clip(0, 40)

    # --- TDP: prefer spec (more accurate), fallback to benchmark CSV TDP ---
    df["tdp_watts"] = df["spec_tdp_watts"].combine_first(df["bench_tdp"])

    # --- Performance score: prefer G3Dmark, fallback to fp32_gflops ---
    df["perf_score"] = df["G3Dmark"].combine_first(df["fp32_gflops"])

    # --- Log G3Dmark ---
    df["log_G3Dmark"] = np.log1p(df["G3Dmark"].fillna(0))

    # --- Perf per watt ---
    df["perf_per_watt"] = df["perf_score"] / df["tdp_watts"].replace(0, np.nan)

    # --- Tier class & Model number ---
    df["tier_class"] = df["extracted_model"].apply(derive_tier_class)
    df["model_number"] = df["extracted_model"].apply(derive_model_number)

    # --- Series family ---
    df["series_family"] = df["extracted_model"].apply(derive_series_family)

    # --- GPU generation ---
    df["gpu_generation"] = df["norm_model"].apply(derive_gpu_generation)

    # --- Ti variant ---
    df["ti_variant"] = df["extracted_model"].apply(is_ti_variant)

    # --- Log price (target) ---
    df["log_price_lkr"] = np.log1p(df["price_lkr"])

    print("  Features engineered.")
    for feat in ["gpu_age_years", "tdp_watts", "perf_score", "log_G3Dmark",
                 "perf_per_watt", "tier_class", "model_number", "gpu_generation"]:
        cov = df[feat].notna().mean() * 100
        print(f"    {feat:<28}: {cov:.1f}% filled")

    return df


# -- 5. Finalise Dataset -------------------------------------------------------

FINAL_COLUMNS = [
    # Target
    "price_lkr", "log_price_lkr",
    # Listing fields
    "extracted_model", "norm_model", "vram_gb", "brand",
    # Benchmark join
    "G3Dmark", "G2Dmark", "log_G3Dmark",
    # Spec fields
    "fp32_gflops", "memory_bandwidth_gb_s", "shader_units",
    "gpu_base_clock_mhz", "boost_clock_mhz",
    # Combined / derived
    "tdp_watts", "perf_score", "perf_per_watt",
    "release_year", "gpu_age_years",
    "architecture",
    "tier_class", "model_number", "series_family", "gpu_generation", "ti_variant",
    # Diagnostic & Metadata
    "bench_match", "spec_match", "listing_id", "listing_url", "raw_title", "scraped_at_utc",
]


def finalise(df: pd.DataFrame) -> pd.DataFrame:
    print("\nStep 5: Finalising dataset (retaining all valid price listings without pre-split leakage) ...")
    df = df[df["price_lkr"] >= 3000].copy()

    # Keep only final columns that exist
    cols = [c for c in FINAL_COLUMNS if c in df.columns]
    df = df[cols].copy()
    return df


# -- 6. Save and Report --------------------------------------------------------

def save_and_report(df: pd.DataFrame) -> None:
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OK] Saved {len(df)} rows -> {OUTPUT_CSV}")

    print("\n-- Feature Coverage Summary --------------------------------------")
    FEATURE_COLS = [
        "vram_gb", "G3Dmark", "G2Dmark", "log_G3Dmark",
        "fp32_gflops", "tdp_watts", "memory_bandwidth_gb_s",
        "shader_units", "gpu_base_clock_mhz", "boost_clock_mhz",
        "perf_per_watt", "gpu_age_years", "gpu_generation",
        "model_number", "series_family", "ti_variant", "architecture",
    ]
    for col in FEATURE_COLS:
        if col in df.columns:
            pct = df[col].notna().mean() * 100
            bar = "#" * int(pct // 5) + "." * (20 - int(pct // 5))
            print(f"  {col:<28} {bar} {pct:5.1f}%")

    print("\n-- Price Distribution --------------------------------------------")
    print(df["price_lkr"].describe().to_string())

    print("\n-- Series Family Distribution ------------------------------------")
    if "series_family" in df.columns:
        print(df["series_family"].value_counts().to_string())

    print("\n-- GPU Generation Distribution -----------------------------------")
    if "gpu_generation" in df.columns:
        print(df["gpu_generation"].value_counts().sort_index().to_string())

    print("\n-- Sample Rows (first 5) -----------------------------------------")
    preview_cols = ["extracted_model", "price_lkr", "vram_gb", "brand",
                    "G3Dmark", "tdp_watts", "gpu_age_years", "series_family", "architecture"]
    preview = [c for c in preview_cols if c in df.columns]
    print(df[preview].head().to_string(index=False))

    print("\n-- Unmatched Models (benchmark) ----------------------------------")
    model_col = "extracted_model" if "extracted_model" in df.columns else "norm_model"
    if "bench_match" in df.columns and model_col in df.columns:
        unmatched = (
            df[df["bench_match"].isna()][model_col]
            .value_counts()
            .head(20)
        )
        if len(unmatched):
            print(unmatched.to_string())
        else:
            print("  All models matched! ")

    print("\n-- Unmatched Models (specs) --------------------------------------")
    if "spec_match" in df.columns and model_col in df.columns:
        unmatched_spec = (
            df[df["spec_match"].isna()][model_col]
            .value_counts()
            .head(20)
        )
        if len(unmatched_spec):
            print(unmatched_spec.to_string())
        else:
            print("  All models matched!")


# -- Main ----------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  GPU Price Predictor V2 - Phase 1: Build Features")
    print("=" * 60 + "\n")

    # Validate inputs
    for path in [LISTING_V1, LISTING_V2, BENCHMARKS_CSV, SPECS_CSV]:
        if not path.exists():
            print(f"[!] Missing file: {path}")
            sys.exit(1)

    df = load_listings()
    df = join_benchmarks(df)
    df = join_specs(df)
    df = engineer_features(df)
    df = finalise(df)
    save_and_report(df)


if __name__ == "__main__":
    main()
