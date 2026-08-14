"""
Feature engineering for the mobile price model.

Creates derived features that capture phone value better than raw model names:
- model_tier: numeric tier reflecting generation/premium level (1-10)
- brand_tier: numeric tier reflecting brand price segment (1-3)
- phone_age_years: estimated age based on model release year
- is_flagship: binary flag for premium/flagship models
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .phone_specs import (
    IPHONE_MODEL_TIER_NORMALIZED,
    IPHONE_RELEASE_YEAR_NORMALIZED,
    normalized_model_key,
)

CURRENT_YEAR = datetime.now().year

# ── Brand tiers ──────────────────────────────────────────────────────────────
# 3 = premium, 2 = mid-range, 1 = budget
BRAND_TIER_MAP = {
    "Apple": 3, "Samsung": 3, "Google": 3, "Sony": 3,
    "OnePlus": 2, "Oppo": 2, "Vivo": 2, "Xiaomi": 2,
    "Realme": 2, "Huawei": 2, "Honor": 2, "Nothing": 2,
    "Motorola": 2, "Nokia": 2, "Asus": 2, "LG": 2, "HTC": 2,
    "Infinix": 1, "Tecno": 1, "Itel": 1, "ZTE": 1, "Lenovo": 1,
}

# ── Android model tier patterns ──────────────────────────────────────────────
# Maps regex patterns to (tier, is_flagship) tuples
ANDROID_TIER_PATTERNS: list[tuple[str, int, bool]] = [
    # Samsung flagships
    (r"galaxy\s*s2[3456]\s*ultra", 10, True),
    (r"galaxy\s*s2[3456]\s*(?:plus|\+)", 9, True),
    (r"galaxy\s*s2[3456](?!\s*ultra|\s*plus|\s*\+)", 8, True),
    (r"galaxy\s*s2[012]\s*ultra", 9, True),
    (r"galaxy\s*s2[012]\s*(?:plus|\+|fe)", 8, True),
    (r"galaxy\s*s2[012](?!\s*ultra|\s*plus|\s*\+|\s*fe)", 7, True),
    (r"galaxy\s*z\s*(?:fold|flip)", 9, True),
    (r"galaxy\s*note\s*20\s*ultra", 9, True),
    (r"galaxy\s*note\s*20", 8, True),
    # Samsung mid-range
    (r"galaxy\s*a[5-9]\d", 5, False),
    (r"galaxy\s*a[1-4]\d", 3, False),
    (r"galaxy\s*m\d", 3, False),
    (r"galaxy\s*f\d", 2, False),
    # OnePlus
    (r"oneplus\s*1[2-5](?!\s*r)", 8, True),
    (r"oneplus\s*1[01](?!\s*r)", 7, True),
    (r"oneplus\s*(?:nord|1\dr)", 5, False),
    # Google Pixel
    (r"pixel\s*[89]\s*pro", 9, True),
    (r"pixel\s*[89]", 7, True),
    (r"pixel\s*[67]\s*pro", 8, True),
    (r"pixel\s*[67]a?", 6, False),
    # Xiaomi flagships
    (r"mi\s*1[1-4]\s*ultra", 9, True),
    (r"poco\s*f[5-9]", 7, True),
    (r"poco\s*[xm]\d", 4, False),
    (r"redmi\s*note\s*1[3-5]\s*pro", 5, False),
    (r"redmi\s*note\s*\d", 4, False),
    (r"redmi\s*1[2-9]c?", 3, False),
    (r"redmi\s*a\d", 2, False),
    # Oppo/Vivo/Realme
    (r"reno\s*1[0-2]\s*pro", 7, True),
    (r"reno\s*\d", 5, False),
    (r"realme\s*gt", 7, True),
    (r"iqoo\s*neo", 6, True),
    # Budget brands
    (r"(?:infinix|tecno|itel)", 2, False),
]


def _get_android_tier_flagship(model_text: str) -> tuple[int, bool]:
    """Match an Android model name against known tier patterns."""
    text = model_text.lower().strip()
    text = re.sub(r"[\s_/|+-]+", " ", text)
    for pattern, tier, is_flag in ANDROID_TIER_PATTERNS:
        if re.search(pattern, text):
            return tier, is_flag
    return 3, False  # default mid-tier


def compute_model_tier(row: Any) -> int:
    """Assign a 1-10 numeric tier reflecting generation and premium level."""
    brand = str(row.get("brand", "")).strip()
    model = str(row.get("model", "")).strip()
    key = normalized_model_key(model)

    if brand == "Apple" or "iphone" in model.lower():
        return IPHONE_MODEL_TIER_NORMALIZED.get(key, 5)

    tier, _ = _get_android_tier_flagship(model)
    return tier


def compute_brand_tier(brand: Any) -> int:
    """Map brand to a 1-3 price tier (3=premium, 1=budget)."""
    return BRAND_TIER_MAP.get(str(brand).strip(), 2)


def compute_phone_age(row: Any) -> float:
    """Estimate phone age in years from model release year."""
    brand = str(row.get("brand", "")).strip()
    model = str(row.get("model", "")).strip()
    key = normalized_model_key(model)

    if brand == "Apple" or "iphone" in model.lower():
        year = IPHONE_RELEASE_YEAR_NORMALIZED.get(key)
        if year:
            return max(0.0, float(CURRENT_YEAR - year))

    # For Android, try to extract year from model name patterns
    m = re.search(r"20(1[5-9]|2[0-9])", model)
    if m:
        year = int(m.group(0))
        return max(0.0, float(CURRENT_YEAR - year))

    # Heuristic: use storage/RAM as rough age proxy — can't determine exactly
    return 3.0  # default assumption


def compute_is_flagship(row: Any) -> int:
    """Binary flag: 1 for flagship/premium models, 0 for mid/budget."""
    brand = str(row.get("brand", "")).strip()
    model = str(row.get("model", "")).strip()
    key = normalized_model_key(model)

    if brand == "Apple" or "iphone" in model.lower():
        tier = IPHONE_MODEL_TIER_NORMALIZED.get(key, 5)
        return int(tier >= 8)  # Pro/Pro Max models

    _, is_flag = _get_android_tier_flagship(model)
    return int(is_flag)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features to the DataFrame."""
    df = df.copy()
    df["model_tier"] = df.apply(compute_model_tier, axis=1)
    df["brand_tier"] = df["brand"].apply(compute_brand_tier)
    df["phone_age_years"] = df.apply(compute_phone_age, axis=1)
    df["is_flagship"] = df.apply(compute_is_flagship, axis=1)
    return df
