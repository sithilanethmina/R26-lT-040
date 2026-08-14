"""
Centralised configuration: paths, column lists, and training constants.

Every other module imports from here so that changes propagate automatically.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

RAW_DATA_FILE = DATA_RAW_DIR / "ikman_mobile_phones_processed.json"
CLEANED_DATA_FILE = DATA_PROCESSED_DIR / "ikman_mobile_phones_ml_ready.json"
EVALUATION_FILE = OUTPUTS_DIR / "model_evaluation_results.json"
FAIR_PRICE_JSON = OUTPUTS_DIR / "fair_price_predictions.json"
FAIR_PRICE_CSV = OUTPUTS_DIR / "fair_price_predictions.csv"

# ── Target & filter ──────────────────────────────────────────────────────────
TARGET_COLUMN = "listed_price"
TRAINING_CONDITION = "used"

# ── Column lists ─────────────────────────────────────────────────────────────
# Columns expected in the raw scraped JSON
REQUIRED_RAW_COLUMNS = [
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
    "battery_health_percent",
    "listed_price",
]

# Boolean feature columns (0/1)
BOOLEAN_COLUMNS = ["dual_sim", "has_5g", "has_esim"]

# Categorical features fed to the model (after engineering)
CATEGORICAL_FEATURES = ["brand", "model"]

# Numeric features fed to the model
NUMERIC_FEATURES = [
    "storage_gb",
    "ram_gb",
    "warranty_days",
    "battery_health_percent",
    "dual_sim",
    "has_5g",
    "has_esim",
    # engineered
    "model_tier",
    "brand_tier",
    "phone_age_years",
    "is_flagship",
]

# All features (order matters for the sklearn pipeline)
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Columns kept in the cleaned ML-ready JSON
OUTPUT_COLUMNS = [
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
    "battery_health_percent",
    "listed_price",
    "phone_type",
    # engineered features
    "model_tier",
    "brand_tier",
    "phone_age_years",
    "is_flagship",
]

# Columns used to group fair-price predictions
FAIR_PRICE_GROUP_COLUMNS = ["phone_type", "brand", "model", "storage_gb"]

# ── Training hyper-parameters ────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Minimum rows needed to train / warn
MIN_ROWS_REQUIRED = 30
MIN_ROWS_WARNING = 80
