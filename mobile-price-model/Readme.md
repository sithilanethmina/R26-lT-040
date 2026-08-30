# Mobile Price Model

Predict fair used-phone prices in Sri Lanka using machine learning models trained on scraped [ikman.lk](https://ikman.lk) data.

## Overview

This project scrapes mobile phone listings from ikman.lk, cleans and engineers features from the data, and trains regression models (Random Forest, XGBoost, CatBoost, LightGBM) to predict fair market prices for used phones.

**Key features:**
- Separate models for **iPhone** and **Android** phones
- **Battery health** as a price factor for iPhones
- **Feature engineering**: model tier, brand tier, phone age, flagship flag
- **No data leakage**: all preprocessing happens inside the sklearn Pipeline
- **Cross-validated** metrics with **hyperparameter tuning** (RandomizedSearchCV)
- **GPU acceleration**: automatic CUDA detection for XGBoost and CatBoost
- Streamlit web app for interactive predictions
- FastAPI REST API for programmatic access
- **Fair-price catalog**: pre-computed fair prices for 1,400+ phone categories

## Dataset

| Metric | Value |
|--------|-------|
| Total raw used-phone listings | 20,051 |
| ML-ready records (after cleaning) | 12,502 |
| iPhone records | 6,354 |
| Android records | 6,148 |
| Fair-price categories | 1,462 |
| Source | ikman.lk (Sri Lanka) |

## Model Performance

### iPhone — CatBoost (Recommended)

| Metric | Value |
|--------|-------|
| Test MAE | LKR 10,360 |
| Test RMSE | LKR 16,220 |
| R² Score | 0.9487 |
| MAPE | 12.77% |
| 5-Fold CV MAE | LKR 10,360 (±373) |

### Android — Random Forest (Recommended)

| Metric | Value |
|--------|-------|
| Test MAE | LKR 7,278 |
| Test RMSE | LKR 11,453 |
| R² Score | 0.8290 |
| MAPE | 27.72% |
| 5-Fold CV MAE | LKR 7,787 (±342) |

### All Models Compared

| Phone Type | Model | MAE (LKR) | R² | MAPE |
|------------|-------|-----------|----|------|
| iPhone | **CatBoost** ★ | 10,360 | 0.9487 | 12.77% |
| iPhone | XGBoost | 10,499 | 0.9438 | 12.98% |
| iPhone | LightGBM | 10,576 | 0.9427 | 13.08% |
| iPhone | Random Forest | 10,808 | 0.9420 | 13.24% |
| Android | **Random Forest** ★ | 7,278 | 0.8290 | 27.72% |
| Android | XGBoost | 7,330 | 0.8247 | 28.02% |
| Android | LightGBM | 7,389 | 0.8207 | 27.81% |
| Android | CatBoost | 7,424 | 0.8261 | 28.48% |

## Project Structure

```
mobile-price-model/
├── app/
│   └── streamlit_app.py          # Streamlit web UI (prediction + browse)
├── data/
│   ├── raw/                      # Raw scraped JSON data
│   └── processed/                # Cleaned ML-ready data
├── models/                       # Trained model artifacts (.pkl)
│   ├── best_iphone_model.pkl     # CatBoost pipeline
│   └── best_android_model.pkl    # Random Forest pipeline
├── outputs/
│   ├── model_evaluation_results.json  # Full evaluation metrics
│   ├── fair_price_predictions.json    # Fair prices (1,462 categories)
│   ├── fair_price_predictions.csv     # Fair prices (CSV export)
│   ├── mobile_brand_model_lookup.json # Supported brands/models lookup
│   └── mobile_metadata.json           # Model metadata
├── scraper/
│   └── ikman_scrape_pipeline.py  # ikman.lk web scraper
├── src/
│   ├── config.py                 # Paths, constants, feature lists
│   ├── phone_specs.py            # iPhone/Android spec lookups
│   ├── data_preprocessing.py     # Data cleaning and standardization
│   ├── feature_engineering.py    # Derived feature creation
│   ├── train.py                  # Training pipeline (main entry point)
│   ├── evaluate.py               # Model evaluation and comparison
│   └── predict.py                # Prediction module
├── api.py                        # FastAPI REST API
├── streamlit_app.py              # Streamlit entrypoint forwarder
├── requirements.txt
├── Readme.md
└── .gitignore
```

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd mobile-price-model

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1    # Windows PowerShell
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Scrape data (optional)

```bash
python scraper/ikman_scrape_pipeline.py --all-pages --skip-train
```

This scrapes all mobile phone listings from ikman.lk and saves them to `data/raw/`. Only "Used" condition phones are kept; "Brand New" ads are automatically filtered out.

### 2. Train the model

```bash
python -m src.train
```

This will:
1. Load and clean the raw scraped data
2. Engineer features (model tier, brand tier, phone age, etc.)
3. Train multiple models (Random Forest, XGBoost, CatBoost, LightGBM) with hyperparameter tuning
4. Evaluate using 5-fold cross-validation and a held-out test set (80/20 split)
5. Save the best model per phone type to `models/`
6. Generate fair-price predictions for all phone categories

### 3. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

The app provides two tabs:
- **Predict Price**: Select phone specs and get a fair price estimate with confidence range
- **Browse Categories**: View pre-computed fair prices for all 1,462 phone categories

### 4. Run the FastAPI server

```bash
python api.py
```

Endpoints:
- `GET /health` — Health check
- `GET /metadata` — Supported brands and models
- `POST /predict` — Predict fair price for a phone

### 5. Predict from code

```python
from src.predict import predict_price

result = predict_price(
    phone_type="iphone",
    brand="Apple",
    model="iPhone 14 Pro",
    storage_gb=256,
    ram_gb=6,
    battery_health_percent=85,
)
print(f"Fair price: LKR {result['predicted_price']:,.0f}")
print(f"Range: LKR {result['range_low']:,.0f} – {result['range_high']:,.0f}")
```

## Features Used

| Feature | Type | Description |
|---------|------|-------------|
| brand | categorical | Phone brand (Apple, Samsung, etc.) |
| model | categorical | Phone model name |
| storage_gb | numeric | Internal storage in GB |
| ram_gb | numeric | RAM in GB |
| warranty_days | numeric | Remaining warranty in days |
| battery_health_percent | numeric | Battery health % (iPhones) |
| dual_sim | binary | Dual SIM support |
| has_5g | binary | 5G capability |
| has_esim | binary | eSIM support |
| model_tier | engineered | Phone generation/premium tier (1–10) |
| brand_tier | engineered | Brand price segment (1–3) |
| phone_age_years | engineered | Estimated age from release year |
| is_flagship | engineered | Flagship model flag |

## Data Pipeline

```
Raw JSON (ikman.lk scrape)
  → Deduplication & condition filter (Used only)
  → Text standardization (brand, model, condition)
  → Numeric parsing (price, storage, RAM, battery)
  → Boolean parsing (dual SIM, 5G, eSIM)
  → Known-spec overrides (iPhone/Android RAM, capabilities)
  → IQR outlier removal (per phone type)
  → Feature engineering (tier, age, flagship)
  → ML-ready dataset
  → Train/Test split (80/20)
  → sklearn Pipeline (impute → encode/scale → model)
  → Hyperparameter tuning (RandomizedSearchCV, 5-fold CV)
  → Best model selection (combined MAE + R² ranking)
  → Model artifacts + fair-price catalog
```

## License

MIT
