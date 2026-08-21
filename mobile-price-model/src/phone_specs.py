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

# ── Maximum valid RAM for consumer phones ────────────────────────────────────
# As of 2025, no consumer phone ships with more than 24 GB RAM.
# The overwhelming majority are ≤ 16 GB.  We use 24 GB as an absolute hard cap
# and per-model lookups for tighter corrections.
MAX_PHONE_RAM_GB = 24.0

# RAM values that actually exist in real phones (used for sanity checks)
VALID_RAM_VALUES = {0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 18.0, 24.0}

# ── Android RAM by (brand, model) ────────────────────────────────────────────
# Correct RAM values (list of valid RAM variants) for known Android models.
# Used to override noisy scraped values, exactly like IPHONE_RAM_GB_BY_MODEL.
# Each entry maps (brand, model) → list of valid RAM sizes in GB.
ANDROID_RAM_GB: dict[tuple[str, str], list[float]] = {
    # ── Samsung Galaxy S series ──────────────────────────────────────────
    ("Samsung", "Galaxy S4"):       [2.0],
    ("Samsung", "Galaxy S5"):       [2.0],
    ("Samsung", "Galaxy S6"):       [3.0],
    ("Samsung", "Galaxy S6 Edge"):  [3.0],
    ("Samsung", "Galaxy S7"):       [4.0],
    ("Samsung", "Galaxy S7 Edge"):  [4.0],
    ("Samsung", "Galaxy S8"):       [4.0],
    ("Samsung", "Galaxy S8+"):      [4.0],
    ("Samsung", "Galaxy S9"):       [4.0],
    ("Samsung", "Galaxy S9+"):      [6.0],
    ("Samsung", "Galaxy S10"):      [8.0],
    ("Samsung", "Galaxy S10e"):     [6.0],
    ("Samsung", "Galaxy S10 Plus"): [8.0, 12.0],
    ("Samsung", "Galaxy S20"):      [8.0, 12.0],
    ("Samsung", "Galaxy S20+"):     [8.0, 12.0],
    ("Samsung", "Galaxy S20 Ultra"):[12.0, 16.0],
    ("Samsung", "Galaxy S20FE"):    [6.0, 8.0],
    ("Samsung", "Galaxy S21"):      [8.0],
    ("Samsung", "Galaxy S21 FE"):   [6.0, 8.0],
    ("Samsung", "Galaxy S21 Plus"): [8.0],
    ("Samsung", "Galaxy S21 Ultra"):[12.0],
    ("Samsung", "Galaxy S22"):      [8.0],
    ("Samsung", "Galaxy S22+"):     [8.0],
    ("Samsung", "Galaxy S22 Ultra"):[8.0, 12.0],
    ("Samsung", "Galaxy S23"):      [8.0],
    ("Samsung", "Galaxy S23+"):     [8.0],
    ("Samsung", "Galaxy S23 FE"):   [8.0],
    ("Samsung", "Galaxy S23 Ultra"):[8.0, 12.0],
    ("Samsung", "Galaxy S24"):      [8.0],
    ("Samsung", "Galaxy S24 FE"):   [8.0],
    ("Samsung", "Galaxy S24 Ultra"):[12.0],
    ("Samsung", "Galaxy S25"):      [12.0],
    # ── Samsung Galaxy Note series ───────────────────────────────────────
    ("Samsung", "Galaxy Note 4"):   [3.0],
    ("Samsung", "Galaxy Note 8"):   [6.0],
    ("Samsung", "Galaxy Note 9"):   [6.0, 8.0],
    ("Samsung", "Galaxy Note 10"):       [8.0],
    ("Samsung", "Galaxy Note 10 Plus"):  [12.0],
    ("Samsung", "Galaxy Note 20"):       [8.0],
    ("Samsung", "Galaxy Note 20 Ultra"): [12.0],
    # ── Samsung Galaxy Z series ──────────────────────────────────────────
    ("Samsung", "Galaxy Z Flip"):   [8.0],
    ("Samsung", "Galaxy Z Flip3"):  [8.0],
    ("Samsung", "Z Flip 4"):        [8.0],
    ("Samsung", "Galaxy Z Flip 5"): [8.0],
    ("Samsung", "Galaxy Z Flip 6"): [12.0],
    ("Samsung", "Galaxy Z Fold3"):  [12.0],
    ("Samsung", "Galaxy Z Fold 5"): [12.0],
    # ── Samsung Galaxy A series ──────────────────────────────────────────
    ("Samsung", "Galaxy A01"):      [2.0],
    ("Samsung", "Galaxy A02"):      [2.0, 3.0],
    ("Samsung", "Galaxy A02s"):     [3.0, 4.0],
    ("Samsung", "Galaxy A03"):      [3.0, 4.0],
    ("Samsung", "Galaxy A03 Core"): [2.0],
    ("Samsung", "Galaxy A03s"):     [3.0, 4.0],
    ("Samsung", "Galaxy A04"):      [3.0, 4.0],
    ("Samsung", "Galaxy A04e"):     [3.0],
    ("Samsung", "Galaxy A04s"):     [3.0, 4.0],
    ("Samsung", "Galaxy A05"):      [4.0, 6.0],
    ("Samsung", "Galaxy A05s"):     [4.0, 6.0],
    ("Samsung", "Galaxy A06"):      [4.0, 6.0],
    ("Samsung", "Galaxy A10"):      [2.0],
    ("Samsung", "Galaxy A10s"):     [2.0, 3.0],
    ("Samsung", "Galaxy A11"):      [2.0, 3.0],
    ("Samsung", "Galaxy A12"):      [3.0, 4.0, 6.0],
    ("Samsung", "Galaxy A13"):      [3.0, 4.0, 6.0],
    ("Samsung", "Galaxy A14"):      [4.0, 6.0],
    ("Samsung", "Galaxy A15"):      [4.0, 6.0, 8.0],
    ("Samsung", "Galaxy A16"):      [4.0, 6.0, 8.0],
    ("Samsung", "Galaxy A20"):      [3.0],
    ("Samsung", "Galaxy A20s"):     [3.0, 4.0],
    ("Samsung", "Galaxy A21"):      [3.0],
    ("Samsung", "Galaxy A21s"):     [3.0, 4.0, 6.0],
    ("Samsung", "Galaxy A22"):      [4.0, 6.0],
    ("Samsung", "Galaxy A23"):      [4.0, 6.0],
    ("Samsung", "Galaxy A24"):      [4.0, 6.0, 8.0],
    ("Samsung", "Galaxy A25"):      [6.0, 8.0],
    ("Samsung", "Galaxy A30"):      [3.0, 4.0],
    ("Samsung", "Galaxy A30s"):     [3.0, 4.0],
    ("Samsung", "Galaxy A31"):      [4.0, 6.0],
    ("Samsung", "Galaxy A32"):      [4.0, 6.0, 8.0],
    ("Samsung", "Galaxy A33"):      [6.0, 8.0],
    ("Samsung", "Galaxy A34"):      [6.0, 8.0],
    ("Samsung", "Galaxy A35"):      [6.0, 8.0],
    ("Samsung", "Galaxy A36"):      [6.0, 8.0],
    ("Samsung", "Galaxy A42"):      [6.0, 8.0],
    ("Samsung", "Galaxy A50"):      [4.0, 6.0],
    ("Samsung", "Galaxy A51"):      [4.0, 6.0, 8.0],
    ("Samsung", "Galaxy A52"):      [6.0, 8.0],
    ("Samsung", "Galaxy A52s"):     [6.0, 8.0],
    ("Samsung", "Galaxy A53"):      [6.0, 8.0],
    ("Samsung", "Galaxy A54"):      [6.0, 8.0],
    ("Samsung", "Galaxy A55"):      [8.0, 12.0],
    ("Samsung", "Galaxy A56"):      [8.0, 12.0],
    ("Samsung", "Galaxy A6"):       [3.0],
    ("Samsung", "Galaxy A7"):       [4.0],
    ("Samsung", "Galaxy A70"):      [6.0, 8.0],
    ("Samsung", "Galaxy A71"):      [6.0, 8.0],
    ("Samsung", "Galaxy A73"):      [6.0, 8.0],
    ("Samsung", "Galaxy A80"):      [8.0],
    ("Samsung", "Galaxy A82"):      [6.0],
    ("Samsung", "Galaxy A9"):       [6.0, 8.0],
    ("Samsung", "Galaxy A90"):      [6.0],
    # ── Samsung Galaxy J series (old, budget) ────────────────────────────
    ("Samsung", "Galaxy J1"):       [1.0],
    ("Samsung", "Galaxy J2"):       [1.0, 1.5, 2.0],
    ("Samsung", "Galaxy J5"):       [1.5, 2.0],
    ("Samsung", "Galaxy J6"):       [3.0, 4.0],
    ("Samsung", "Galaxy J6+"):      [4.0],
    ("Samsung", "Galaxy J7"):       [2.0, 3.0],
    ("Samsung", "Galaxy J8"):       [3.0, 4.0],
    # ── Samsung Galaxy M series ──────────────────────────────────────────
    ("Samsung", "Galaxy M01"):      [3.0],
    ("Samsung", "Galaxy M01 Core"): [1.0, 2.0],
    ("Samsung", "Galaxy M01s"):     [3.0],
    ("Samsung", "Galaxy M02"):      [2.0, 3.0],
    ("Samsung", "Galaxy M02s"):     [3.0, 4.0],
    ("Samsung", "Galaxy M04"):      [4.0],
    ("Samsung", "Galaxy M05"):      [4.0],
    ("Samsung", "Galaxy M10"):      [2.0, 3.0],
    ("Samsung", "Galaxy M11"):      [3.0, 4.0],
    ("Samsung", "Galaxy M12"):      [3.0, 4.0, 6.0],
    ("Samsung", "Galaxy M13"):      [4.0, 6.0],
    ("Samsung", "Galaxy M14"):      [4.0, 6.0],
    ("Samsung", "Galaxy M15"):      [4.0, 6.0, 8.0],
    ("Samsung", "Galaxy M16"):      [6.0],
    ("Samsung", "Galaxy M17"):      [6.0],
    ("Samsung", "Galaxy M20"):      [3.0, 4.0],
    ("Samsung", "Galaxy M21"):      [4.0, 6.0],
    ("Samsung", "Galaxy M23"):      [4.0, 6.0],
    ("Samsung", "Galaxy M31"):      [6.0, 8.0],
    ("Samsung", "Galaxy M32"):      [4.0, 6.0, 8.0],
    ("Samsung", "Galaxy M33"):      [6.0, 8.0],
    ("Samsung", "Galaxy M34"):      [6.0, 8.0],
    ("Samsung", "Galaxy M36"):      [6.0],
    ("Samsung", "Galaxy M51"):      [6.0, 8.0],
    ("Samsung", "Galaxy M52"):      [6.0, 8.0],
    ("Samsung", "Galaxy M53"):      [6.0, 8.0],
    # ── Samsung Galaxy F series ──────────────────────────────────────────
    ("Samsung", "Galaxy F05"):      [4.0],
    ("Samsung", "Galaxy F06"):      [4.0, 6.0],
    ("Samsung", "Galaxy F16"):      [6.0, 8.0],
    ("Samsung", "Galaxy F23"):      [6.0],
    ("Samsung", "Galaxy F42"):      [6.0, 8.0],
    ("Samsung", "Galaxy F55"):      [8.0, 12.0],
    # ── Samsung misc ─────────────────────────────────────────────────────
    ("Samsung", "Galaxy Jump"):     [6.0],
    ("Samsung", "Galaxy Wide 7"):   [6.0],
    ("Samsung", "M14"):             [4.0, 6.0],

    # ── Xiaomi / Redmi / Poco ────────────────────────────────────────────
    ("Xiaomi", "Redmi 3"):          [2.0, 3.0],
    ("Xiaomi", "Redmi 5"):          [2.0, 3.0, 4.0],
    ("Xiaomi", "Redmi 6A"):         [2.0],
    ("Xiaomi", "Redmi 8"):          [3.0, 4.0],
    ("Xiaomi", "Redmi 8A"):         [2.0, 3.0],
    ("Xiaomi", "Redmi 9"):          [3.0, 4.0],
    ("Xiaomi", "Redmi 9 Power"):    [4.0],
    ("Xiaomi", "Redmi 9A"):         [2.0, 3.0],
    ("Xiaomi", "Redmi 9C"):         [2.0, 3.0, 4.0],
    ("Xiaomi", "Redmi 9t"):         [4.0, 6.0],
    ("Xiaomi", "Redmi 10"):         [4.0, 6.0],
    ("Xiaomi", "Redmi 10A"):        [2.0, 3.0, 4.0],
    ("Xiaomi", "Redmi 10C"):        [4.0, 6.0],
    ("Xiaomi", "Redmi 12"):         [4.0, 6.0, 8.0],
    ("Xiaomi", "Redmi 12C"):        [3.0, 4.0, 6.0],
    ("Xiaomi", "Redmi 13"):         [6.0, 8.0],
    ("Xiaomi", "Redmi 13C"):        [4.0, 6.0, 8.0],
    ("Xiaomi", "Redmi 14"):         [4.0, 6.0, 8.0],
    ("Xiaomi", "Redmi 14C"):        [4.0, 6.0, 8.0],
    ("Xiaomi", "Redmi 15"):         [6.0, 8.0],
    ("Xiaomi", "Redmi 15C"):        [4.0, 6.0, 8.0],
    ("Xiaomi", "Redmi A1"):         [2.0, 3.0],
    ("Xiaomi", "Redmi A2"):         [2.0, 3.0],
    ("Xiaomi", "Redmi A2 plus"):    [3.0, 4.0],
    ("Xiaomi", "Redmi A3"):         [3.0, 4.0],
    ("Xiaomi", "Redmi A5"):         [3.0, 4.0, 6.0],
    ("Xiaomi", "Redmi Go"):         [1.0],
    ("Xiaomi", "Redmi K20 Pro"):    [6.0, 8.0],
    ("Xiaomi", "Redmi Note 4"):     [3.0, 4.0],
    ("Xiaomi", "Redmi Note 6 Pro"): [4.0],
    ("Xiaomi", "Redmi Note 7"):     [3.0, 4.0],
    ("Xiaomi", "Redmi Note 7 Pro"): [4.0, 6.0],
    ("Xiaomi", "Redmi Note 8"):     [4.0, 6.0],
    ("Xiaomi", "Redmi Note 8 Pro"): [6.0, 8.0],
    ("Xiaomi", "Redmi Note 9"):     [3.0, 4.0],
    ("Xiaomi", "Redmi Note 9 Pro"): [4.0, 6.0],
    ("Xiaomi", "Redmi Note 9S"):    [4.0, 6.0],
    ("Xiaomi", "Redmi Note 10"):    [4.0, 6.0],
    ("Xiaomi", "Redmi Note 10 Pro"):     [6.0, 8.0],
    ("Xiaomi", "Redmi Note 10 Pro Max"): [8.0],
    ("Xiaomi", "Redmi Note 10T"):   [4.0],
    ("Xiaomi", "Redmi Note 10s"):   [6.0, 8.0],
    ("Xiaomi", "Redmi Note 11"):    [4.0, 6.0],
    ("Xiaomi", "Redmi Note 11 Pro"):     [6.0, 8.0],
    ("Xiaomi", "Redmi Note 11 Pro+"):    [6.0, 8.0],
    ("Xiaomi", "Redmi Note 11E"):   [4.0, 6.0],
    ("Xiaomi", "Redmi Note 11R"):   [4.0, 6.0],
    ("Xiaomi", "Redmi Note 11S"):   [6.0, 8.0],
    ("Xiaomi", "Redmi Note 12"):    [4.0, 6.0, 8.0],
    ("Xiaomi", "Redmi Note 12 Pro"):     [6.0, 8.0],
    ("Xiaomi", "Redmi Note 12 Pro Plus"):[8.0, 12.0],
    ("Xiaomi", "Redmi Note 13"):    [6.0, 8.0],
    ("Xiaomi", "Redmi Note 13 Pro"):     [8.0, 12.0],
    ("Xiaomi", "Redmi Note 13 Pro Plus"):[8.0, 12.0],
    ("Xiaomi", "Redmi Note 14"):    [6.0, 8.0],
    ("Xiaomi", "Redmi Note 14 Pro"):     [8.0, 12.0],
    ("Xiaomi", "Redmi Note 14 Pro Plus"):[8.0, 12.0],
    ("Xiaomi", "Redmi Note 15"):    [6.0, 8.0],
    ("Xiaomi", "Redmi Note 15 Pro"):     [8.0, 12.0],
    ("Xiaomi", "Redmi Note 15 Pro+"):    [8.0, 12.0],
    ("Xiaomi", "Poco C3"):          [3.0, 4.0],
    ("Xiaomi", "Poco C55"):         [4.0, 6.0],
    ("Xiaomi", "Poco C71"):         [4.0, 6.0],
    ("Xiaomi", "Poco F5"):          [8.0, 12.0],
    ("Xiaomi", "Poco M3"):          [4.0, 6.0],
    ("Xiaomi", "Poco M3 Pro"):      [4.0, 6.0],
    ("Xiaomi", "Poco M4 Pro 5G"):   [4.0, 6.0],
    ("Xiaomi", "Poco M5"):          [4.0, 6.0],
    ("Xiaomi", "Poco M7 Pro"):      [8.0],
    ("Xiaomi", "Poco X3"):          [6.0, 8.0],
    ("Xiaomi", "Poco X3 NFC"):      [6.0, 8.0],
    ("Xiaomi", "Poco X3 Pro"):      [6.0, 8.0],
    ("Xiaomi", "Poco X5 Pro"):      [6.0, 8.0],
    ("Xiaomi", "Mi 8 lite"):        [4.0, 6.0],
    ("Xiaomi", "Mi 9T Pro"):        [6.0, 8.0],
    ("Xiaomi", "Mi 11 Lite"):       [6.0, 8.0],
    ("Xiaomi", "Mi 11X Pro"):       [8.0],
    ("Xiaomi", "Mi A3"):            [4.0, 6.0],
    ("Xiaomi", "Mi Max"):           [3.0, 4.0],
    ("Xiaomi", "Mi Note 3"):        [6.0],
    ("Xiaomi", "11 Lite NE"):       [6.0, 8.0],
    ("Xiaomi", "12"):               [8.0, 12.0],
    ("Xiaomi", "12 Lite"):          [6.0, 8.0],
    ("Xiaomi", "12 Pro"):           [8.0, 12.0],
    ("Xiaomi", "12C"):              [3.0, 4.0],
    ("Xiaomi", "12S"):              [8.0, 12.0],
    ("Xiaomi", "13"):               [8.0, 12.0],
    ("Xiaomi", "13 Pro"):           [12.0],
    ("Xiaomi", "Civi"):             [8.0],

    # ── Oppo ─────────────────────────────────────────────────────────────
    ("Oppo", "A37"):                [2.0],
    ("Oppo", "A3s"):                [2.0, 3.0],
    ("Oppo", "A5"):                 [4.0],
    ("Oppo", "A53"):                [4.0, 6.0],
    ("Oppo", "A54"):                [4.0, 6.0],
    ("Oppo", "A5s"):                [2.0, 3.0, 4.0],
    ("Oppo", "A57"):                [4.0],
    ("Oppo", "A58"):                [6.0, 8.0],
    ("Oppo", "A60"):                [8.0],
    ("Oppo", "A78"):                [8.0],
    ("Oppo", "A9"):                 [4.0, 8.0],
    ("Oppo", "F11"):                [4.0, 6.0],
    ("Oppo", "F11 Pro"):            [6.0],
    ("Oppo", "F17 Pro"):            [8.0],
    ("Oppo", "F19"):                [6.0],
    ("Oppo", "F19 Pro"):            [8.0],
    ("Oppo", "F19 Pro+"):           [8.0],
    ("Oppo", "F21 Pro"):            [8.0],
    ("Oppo", "F23"):                [8.0],
    ("Oppo", "F9"):                 [4.0, 6.0],
    ("Oppo", "Reno 10x Zoom"):      [6.0, 8.0],
    ("Oppo", "Reno 4 Z"):           [8.0],
    ("Oppo", "Reno 5 Z"):           [8.0],
    ("Oppo", "Reno 6"):             [8.0],
    ("Oppo", "Reno7 A"):            [6.0],
    ("Oppo", "Reno 8"):             [8.0],
    ("Oppo", "Reno 12"):            [8.0, 12.0],
    ("Oppo", "Reno 12 Pro"):        [12.0],

    # ── Vivo ─────────────────────────────────────────────────────────────
    ("Vivo", "S1"):                 [4.0, 6.0],
    ("Vivo", "T2"):                 [6.0, 8.0],
    ("Vivo", "T4x"):                [6.0, 8.0],
    ("Vivo", "V20 Pro"):            [8.0],
    ("Vivo", "V21"):                [8.0],
    ("Vivo", "V21e"):               [8.0],
    ("Vivo", "V25"):                [8.0],
    ("Vivo", "V29E"):               [8.0],
    ("Vivo", "V40"):                [8.0, 12.0],
    ("Vivo", "V40e"):               [8.0],
    ("Vivo", "V50"):                [8.0, 12.0],
    ("Vivo", "V60e"):               [8.0],
    ("Vivo", "X70 Pro"):            [8.0, 12.0],
    ("Vivo", "Y03"):                [4.0],
    ("Vivo", "Y04"):                [4.0],
    ("Vivo", "Y100A"):              [8.0],
    ("Vivo", "Y11"):                [3.0],
    ("Vivo", "Y12s"):               [3.0],
    ("Vivo", "Y15S"):               [3.0],
    ("Vivo", "Y17"):                [4.0],
    ("Vivo", "Y17s"):               [4.0, 6.0],
    ("Vivo", "Y19"):                [4.0, 6.0],
    ("Vivo", "Y1s"):                [2.0, 3.0],
    ("Vivo", "Y20"):                [3.0, 4.0],
    ("Vivo", "Y21"):                [4.0],
    ("Vivo", "Y21d"):               [4.0],
    ("Vivo", "Y28"):                [4.0, 6.0, 8.0],
    ("Vivo", "Y29"):                [6.0, 8.0],
    ("Vivo", "Y35"):                [4.0, 8.0],
    ("Vivo", "Y400"):               [8.0],
    ("Vivo", "Y50"):                [8.0],
    ("Vivo", "Y51A"):               [6.0, 8.0],
    ("Vivo", "Y53s"):               [6.0, 8.0],
    ("Vivo", "Y55s"):               [4.0, 8.0],
    ("Vivo", "Y69"):                [3.0],
    ("Vivo", "Y73"):                [8.0],
    ("Vivo", "Y83"):                [4.0],
    ("Vivo", "Y85"):                [4.0],
    ("Vivo", "Y91c"):               [2.0],
    ("Vivo", "Y91i"):               [2.0, 3.0],
    ("Vivo", "Y93"):                [3.0, 4.0],
    ("Vivo", "Y95"):                [4.0, 6.0],

    # ── OnePlus ──────────────────────────────────────────────────────────
    ("OnePlus", "6"):               [6.0, 8.0],
    ("OnePlus", "6T"):              [6.0, 8.0],
    ("OnePlus", "7"):               [6.0, 8.0],
    ("OnePlus", "7 Pro"):           [6.0, 8.0, 12.0],
    ("OnePlus", "7T"):              [8.0],
    ("OnePlus", "7T Pro"):          [8.0],
    ("OnePlus", "8"):               [8.0, 12.0],
    ("OnePlus", "8 Pro"):           [8.0, 12.0],
    ("OnePlus", "8T"):              [8.0, 12.0],
    ("OnePlus", "9"):               [8.0, 12.0],
    ("OnePlus", "9 Pro"):           [8.0, 12.0],
    ("OnePlus", "9R"):              [8.0, 12.0],
    ("OnePlus", "10 Pro"):          [8.0, 12.0],
    ("OnePlus", "10R"):             [8.0, 12.0],
    ("OnePlus", "10T"):             [8.0, 16.0],
    ("OnePlus", "11"):              [8.0, 16.0],
    ("OnePlus", "11R"):             [8.0, 16.0],
    ("OnePlus", "12"):              [12.0, 16.0],
    ("OnePlus", "12R"):             [8.0, 16.0],
    ("OnePlus", "15"):              [12.0, 16.0],
    ("OnePlus", "Nord CE"):         [6.0, 8.0, 12.0],
    ("OnePlus", "Nord"):            [6.0, 8.0, 12.0],

    # ── Huawei ───────────────────────────────────────────────────────────
    ("Huawei", "GR3"):              [2.0],
    ("Huawei", "Mate 10"):          [4.0],
    ("Huawei", "Mate 10 Pro"):      [6.0],
    ("Huawei", "Mate 20"):          [4.0, 6.0],
    ("Huawei", "Mate 20 Lite"):     [4.0],
    ("Huawei", "Mate 20 Pro"):      [6.0, 8.0],
    ("Huawei", "Mate 30 Pro"):      [8.0],
    ("Huawei", "Nova"):             [3.0],
    ("Huawei", "Nova 2i"):          [4.0],
    ("Huawei", "Nova 3i"):          [4.0],
    ("Huawei", "Nova 7 SE"):        [8.0],
    ("Huawei", "Nova 7i"):          [8.0],
    ("Huawei", "P10"):              [4.0],
    ("Huawei", "P20 Lite"):         [4.0],
    ("Huawei", "P20 Pro"):          [6.0],
    ("Huawei", "P30"):              [6.0, 8.0],
    ("Huawei", "P30 Lite"):         [4.0, 6.0],
    ("Huawei", "P30 Pro"):          [8.0],
    ("Huawei", "P40"):              [8.0],
    ("Huawei", "P8 Lite"):          [2.0],
    ("Huawei", "Y5"):               [1.0, 2.0],
    ("Huawei", "Y5 Lite"):          [1.0, 2.0],
    ("Huawei", "Y5 Prime"):         [2.0],
    ("Huawei", "Y6"):               [2.0, 3.0],
    ("Huawei", "Y6 Prime"):         [2.0, 3.0],
    ("Huawei", "Y6p"):              [3.0],
    ("Huawei", "Y7"):               [3.0],
    ("Huawei", "Y7 Prime"):         [3.0],
    ("Huawei", "Y9"):               [4.0],
    ("Huawei", "Y9 Prime"):         [4.0],

    # ── Honor ────────────────────────────────────────────────────────────
    ("Honor", "X7c"):               [6.0, 8.0],
    ("Honor", "X9a"):               [8.0],
    ("Honor", "X9c"):               [8.0, 12.0],
    ("Honor", "Magic 7 Pro"):       [12.0],
    ("Honor", "X6"):                [4.0],
    ("Honor", "X6a"):               [4.0, 6.0],
    ("Honor", "X6b"):               [6.0],
    ("Honor", "X8a"):               [6.0, 8.0],
    ("Honor", "200"):               [8.0, 12.0],
    ("Honor", "200 Smart"):         [4.0],
    ("Honor", "400"):               [8.0, 12.0],
    ("Honor", "400 Pro"):           [12.0],

    # ── Google Pixel ─────────────────────────────────────────────────────
    ("Google", "Pixel 3"):          [4.0],
    ("Google", "Pixel 3a"):         [4.0],
    ("Google", "Pixel 4"):          [6.0],
    ("Google", "Pixel 4a"):         [6.0],
    ("Google", "Pixel 5"):          [8.0],
    ("Google", "Pixel 5a"):         [6.0],
    ("Google", "Pixel 6"):          [8.0],
    ("Google", "Pixel 6a"):         [6.0],
    ("Google", "Pixel 6 Pro"):      [12.0],
    ("Google", "Pixel 7"):          [8.0],
    ("Google", "Pixel 7a"):         [8.0],
    ("Google", "Pixel 7 Pro"):      [12.0],
    ("Google", "Pixel 8"):          [8.0],
    ("Google", "Pixel 8a"):         [8.0],
    ("Google", "Pixel 8 Pro"):      [12.0],
    ("Google", "Pixel 9"):          [12.0],
    ("Google", "Pixel 9a"):         [8.0],
    ("Google", "Pixel 9 Pro"):      [16.0],
    ("Google", "Pixel 9 Pro XL"):   [16.0],

    # ── Realme ───────────────────────────────────────────────────────────
    ("Realme", "C53"):              [6.0, 8.0],
    ("Realme", "C55"):              [6.0, 8.0],
    ("Realme", "C67"):              [6.0, 8.0],
    ("Realme", "11"):               [8.0],
    ("Realme", "GT"):               [8.0, 12.0],

    # ── Nothing ──────────────────────────────────────────────────────────
    ("Nothing", "Phone 1"):         [8.0, 12.0],
    ("Nothing", "Phone 2"):         [8.0, 12.0],
    ("Nothing", "Phone 2a"):        [8.0, 12.0],
    ("Nothing", "Phone 3a"):        [8.0],
    ("Nothing", "CMF Phone 1"):     [6.0, 8.0],
    ("Nothing", "CMF Phone 2 Pro"): [8.0],

    # ── Nokia ────────────────────────────────────────────────────────────
    ("Nokia", "C1"):                [1.0],
    ("Nokia", "C2"):                [1.0, 2.0],
    ("Nokia", "C3"):                [2.0, 3.0],
    ("Nokia", "G21"):               [4.0, 6.0],
    ("Nokia", "G60"):               [4.0, 6.0],

    # ── Sony ─────────────────────────────────────────────────────────────
    ("Sony", "Xperia 1"):           [6.0],
    ("Sony", "Xperia 1 II"):        [8.0],
    ("Sony", "Xperia 1 III"):       [12.0],
    ("Sony", "Xperia 1 IV"):        [12.0],
    ("Sony", "Xperia 5"):           [6.0],
    ("Sony", "Xperia 5 II"):        [8.0],
    ("Sony", "Xperia 5 III"):       [8.0],
    ("Sony", "Xperia 8"):           [4.0],
    ("Sony", "Xperia 10 II"):       [4.0],
    ("Sony", "Xperia 10 IV"):       [6.0],
    ("Sony", "Xperia XZ"):          [3.0],
    ("Sony", "Xperia XZ Premium"):  [4.0],
    ("Sony", "Xperia XZ2"):         [4.0, 6.0],
    ("Sony", "Xperia XZ2 Compact"): [4.0],
    ("Sony", "Xperia XZ2 Premium"): [6.0],
    ("Sony", "Xperia XZ3"):         [4.0],

    # ── Infinix ──────────────────────────────────────────────────────────
    ("Infinix", "Hot 40i"):         [4.0, 8.0],
    ("Infinix", "Hot 60 Pro+"):     [4.0, 8.0],
    ("Infinix", "Hot 60i"):         [4.0, 6.0],
    ("Infinix", "Note 40"):         [8.0],
    ("Infinix", "Note 50"):         [8.0],

    # ── Meizu ────────────────────────────────────────────────────────────
    ("Meizu", "Lucky 08"):          [8.0],

    # ── Asus ─────────────────────────────────────────────────────────────
    ("Asus", "ROG"):                [8.0, 12.0, 16.0],

    # ── Fujitsu ──────────────────────────────────────────────────────────
    ("Fujitsu", "Arrows NX F-01J"): [3.0],

    # ── E-Tel ────────────────────────────────────────────────────────────
    ("E-Tel", "Other model"):       [2.0, 3.0, 4.0],

    # ── LG ───────────────────────────────────────────────────────────────
    ("LG", "G7"):                   [4.0],
    ("LG", "G8"):                   [6.0],
    ("LG", "V30"):                  [4.0],
    ("LG", "V40"):                  [6.0],
    ("LG", "V50"):                  [6.0],
    ("LG", "V60"):                  [8.0],
    ("LG", "Velvet"):               [6.0, 8.0],

    # ── Tecno ────────────────────────────────────────────────────────────
    ("Tecno", "Pop 5"):             [2.0],
    ("Tecno", "Pop 7"):             [2.0, 3.0],
    ("Tecno", "Spark"):             [2.0, 3.0, 4.0],
    ("Tecno", "Spark 10c"):         [4.0],
    ("Tecno", "Spark 6 Go"):        [4.0],
    ("Tecno", "Spark 7"):           [2.0, 3.0],
    ("Tecno", "Spark 8c"):          [2.0, 3.0],
}

# Build a normalized lookup for faster matching:
#   normalized (brand_lower, model_key) → list of valid RAM values
ANDROID_RAM_GB_NORMALIZED: dict[tuple[str, str], list[float]] = {
    (brand.lower(), re.sub(r"[^a-z0-9]+", "", model.lower())): rams
    for (brand, model), rams in ANDROID_RAM_GB.items()
}


def get_android_valid_ram(brand: str, model: str) -> "list[float] | None":
    """Return the list of valid RAM sizes for an Android phone, or None if unknown."""
    key = (brand.lower().strip(), re.sub(r"[^a-z0-9]+", "", str(model).lower()))
    return ANDROID_RAM_GB_NORMALIZED.get(key)


def snap_ram_to_nearest_valid(ram_gb: float, valid_rams: "list[float]") -> float:
    """Snap a potentially noisy RAM value to the nearest entry in valid_rams."""
    if not valid_rams:
        return ram_gb
    return min(valid_rams, key=lambda v: abs(v - ram_gb))


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
