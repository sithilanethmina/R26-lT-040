"""
Predefined phone specifications: RAM, 5G, eSIM, dual-SIM lookups.

These are used to override noisy scraped values with known-correct hardware specs.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable

# ── iPhone RAM by model ──────────────────────────────────────────────────────
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

# ── iPhone release years (for phone_age_years feature) ───────────────────────
IPHONE_RELEASE_YEAR = {
    "iPhone 3GS": 2009,
    "iPhone 5S": 2013,
    "iPhone 6": 2014,
    "iPhone 6 Plus": 2014,
    "iPhone 6S": 2015,
    "iPhone 6S Plus": 2015,
    "iPhone 7": 2016,
    "iPhone 7 Plus": 2016,
    "iPhone 8": 2017,
    "iPhone 8 Plus": 2017,
    "iPhone X": 2017,
    "iPhone XR": 2018,
    "iPhone XS": 2018,
    "iPhone XS Max": 2018,
    "iPhone 11": 2019,
    "iPhone 11 Pro": 2019,
    "iPhone 11 Pro Max": 2019,
    "iPhone SE": 2016,
    "iPhone SE 2": 2020,
    "iPhone SE 3": 2022,
    "iPhone 12": 2020,
    "iPhone 12 mini": 2020,
    "iPhone 12 Pro": 2020,
    "iPhone 12 Pro Max": 2020,
    "iPhone 13": 2021,
    "iPhone 13 mini": 2021,
    "iPhone 13 Pro": 2021,
    "iPhone 13 Pro Max": 2021,
    "iPhone 14": 2022,
    "iPhone 14 Plus": 2022,
    "iPhone 14 Pro": 2022,
    "iPhone 14 Pro Max": 2022,
    "iPhone 15": 2023,
    "iPhone 15 Plus": 2023,
    "iPhone 15 Pro": 2023,
    "iPhone 15 Pro Max": 2023,
    "iPhone 16": 2024,
    "iPhone 16 Plus": 2024,
    "iPhone 16 Pro": 2024,
    "iPhone 16 Pro Max": 2024,
    "iPhone 16e": 2025,
    "iPhone 17": 2025,
    "iPhone Air": 2025,
    "iPhone 17 Pro": 2025,
    "iPhone 17 Pro Max": 2025,
    "iPhone 17e": 2025,
}

IPHONE_RELEASE_YEAR_NORMALIZED = {
    re.sub(r"[^a-z0-9]+", "", model.lower()): year
    for model, year in IPHONE_RELEASE_YEAR.items()
}

# ── iPhone dual-SIM & 5G sets ────────────────────────────────────────────────
IPHONE_DUAL_SIM_MODELS = {
    "iPhone XR", "iPhone XS", "iPhone XS Max",
    "iPhone 11", "iPhone 11 Pro", "iPhone 11 Pro Max",
    "iPhone SE 2", "iPhone SE 3",
    "iPhone 12", "iPhone 12 mini", "iPhone 12 Pro", "iPhone 12 Pro Max",
    "iPhone 13", "iPhone 13 mini", "iPhone 13 Pro", "iPhone 13 Pro Max",
    "iPhone 14", "iPhone 14 Plus", "iPhone 14 Pro", "iPhone 14 Pro Max",
    "iPhone 15", "iPhone 15 Plus", "iPhone 15 Pro", "iPhone 15 Pro Max",
    "iPhone 16", "iPhone 16 Plus", "iPhone 16 Pro", "iPhone 16 Pro Max",
    "iPhone 16e", "iPhone 17", "iPhone Air",
    "iPhone 17 Pro", "iPhone 17 Pro Max", "iPhone 17e",
}

IPHONE_5G_MODELS = {
    "iPhone SE 3",
    "iPhone 12", "iPhone 12 mini", "iPhone 12 Pro", "iPhone 12 Pro Max",
    "iPhone 13", "iPhone 13 mini", "iPhone 13 Pro", "iPhone 13 Pro Max",
    "iPhone 14", "iPhone 14 Plus", "iPhone 14 Pro", "iPhone 14 Pro Max",
    "iPhone 15", "iPhone 15 Plus", "iPhone 15 Pro", "iPhone 15 Pro Max",
    "iPhone 16", "iPhone 16 Plus", "iPhone 16 Pro", "iPhone 16 Pro Max",
    "iPhone 16e", "iPhone 17", "iPhone Air",
    "iPhone 17 Pro", "iPhone 17 Pro Max", "iPhone 17e",
}

IPHONE_DUAL_SIM_MODELS_NORMALIZED = {
    re.sub(r"[^a-z0-9]+", "", m.lower()) for m in IPHONE_DUAL_SIM_MODELS
}
IPHONE_5G_MODELS_NORMALIZED = {
    re.sub(r"[^a-z0-9]+", "", m.lower()) for m in IPHONE_5G_MODELS
}

# ── iPhone model tiers (higher = newer/more premium) ─────────────────────────
IPHONE_MODEL_TIER = {
    "iPhone 3GS": 1, "iPhone 5S": 2,
    "iPhone 6": 3, "iPhone 6 Plus": 3,
    "iPhone 6S": 3, "iPhone 6S Plus": 3,
    "iPhone SE": 3,
    "iPhone 7": 4, "iPhone 7 Plus": 4,
    "iPhone 8": 4, "iPhone 8 Plus": 5,
    "iPhone X": 5,
    "iPhone XR": 5, "iPhone XS": 6, "iPhone XS Max": 6,
    "iPhone 11": 6, "iPhone 11 Pro": 7, "iPhone 11 Pro Max": 7,
    "iPhone SE 2": 5, "iPhone SE 3": 6,
    "iPhone 12": 6, "iPhone 12 mini": 6,
    "iPhone 12 Pro": 7, "iPhone 12 Pro Max": 8,
    "iPhone 13": 7, "iPhone 13 mini": 7,
    "iPhone 13 Pro": 8, "iPhone 13 Pro Max": 8,
    "iPhone 14": 7, "iPhone 14 Plus": 7,
    "iPhone 14 Pro": 8, "iPhone 14 Pro Max": 9,
    "iPhone 15": 8, "iPhone 15 Plus": 8,
    "iPhone 15 Pro": 9, "iPhone 15 Pro Max": 9,
    "iPhone 16": 8, "iPhone 16 Plus": 8,
    "iPhone 16 Pro": 9, "iPhone 16 Pro Max": 10,
    "iPhone 16e": 7, "iPhone 17": 9, "iPhone Air": 9,
    "iPhone 17 Pro": 10, "iPhone 17 Pro Max": 10, "iPhone 17e": 8,
}

IPHONE_MODEL_TIER_NORMALIZED = {
    re.sub(r"[^a-z0-9]+", "", m.lower()): t
    for m, t in IPHONE_MODEL_TIER.items()
}

# ── Android 5G capability patterns ───────────────────────────────────────────
ANDROID_5G_BRAND_MODEL_PATTERNS: Dict[str, list[str]] = {
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
        r"\bgalaxy\s*s(?:20|21|22|23|24|25|26)",
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

# ── Android eSIM capability patterns ─────────────────────────────────────────
ANDROID_ESIM_BRAND_MODEL_PATTERNS: Dict[str, list[str]] = {
    "Google": [r"\bpixel\s*(?:3|3a|3a\s*xl|3\s*xl|4|4a|4\s*xl|5|5a|6|6a|6\s*pro|7|7a|7\s*pro|8|8a|8\s*pro|9|9a|9\s*pro|9\s*pro\s*xl)\b"],
    "Honor": [r"\bmagic\s*7\s*pro\b"],
    "Samsung": [
        r"\bgalaxy\s*(?:note\s*20|note\s*20\s*ultra)\b",
        r"\bgalaxy\s*s(?:20|21|22|23|24|25|26)",
        r"\bz\s*flip\s*4\b",
    ],
    "Sony": [r"\bxperia\s*(?:1|5|10)\s*iv\b"],
}


# ── Helper functions ─────────────────────────────────────────────────────────

def normalized_model_key(value: Any) -> str:
    """Normalize model names for matching against predefined specs."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "nan", "unknown", "n/a", "na"}:
        return ""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def model_matches_patterns(model: Any, patterns: Iterable[str]) -> bool:
    """Return whether a cleaned model name matches any predefined capability pattern."""
    if not patterns:
        return False
    model_text = str(model).strip().lower() if model else ""
    model_text = re.sub(r"[\s_/|+-]+", " ", model_text)
    return any(re.search(pattern, model_text) for pattern in patterns)


def get_iphone_capabilities(model_key: str) -> Dict[str, int]:
    """Get known dual-SIM, 5G, eSIM values for an iPhone model."""
    return {
        "dual_sim": int(model_key in IPHONE_DUAL_SIM_MODELS_NORMALIZED),
        "has_5g": int(model_key in IPHONE_5G_MODELS_NORMALIZED),
        "has_esim": int(model_key in IPHONE_DUAL_SIM_MODELS_NORMALIZED),
    }


def get_android_capabilities(brand: str, model: Any) -> Dict[str, int]:
    """Get known 5G, eSIM values for an Android phone; assume dual-SIM."""
    model_text = str(model).strip() if model else ""
    has_5g = int(
        "5g" in model_text.lower()
        or model_matches_patterns(model_text, ANDROID_5G_BRAND_MODEL_PATTERNS.get(brand, []))
    )
    has_esim = int(
        model_matches_patterns(model_text, ANDROID_ESIM_BRAND_MODEL_PATTERNS.get(brand, []))
    )
    return {"dual_sim": 1, "has_5g": has_5g, "has_esim": has_esim}
