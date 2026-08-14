# Mobile Price Model

Predict fair used-phone prices in Sri Lanka using machine learning models trained on scraped [ikman.lk](https://ikman.lk) data.

## Overview

This project scrapes mobile phone listings from ikman.lk, cleans and engineers features from the data, and trains regression models (Random Forest, XGBoost, Gradient Boosting, LightGBM) to predict fair market prices for used phones.

**Key features:**
- Separate models for **iPhone** and **Android** phones
- **Battery health** as a price factor for iPhones
- **Feature engineering**: model tier, brand tier, phone age, flagship flag
- **No data leakage**: all preprocessing happens inside the sklearn Pipeline
- **Cross-validated** metrics with **hyperparameter tuning**
- Streamlit web app for interactive predictions

## Project Structure

```
mobile-price-model/
├── app/
│   └── streamlit_app.py          # Streamlit web UI
├── data/
│   ├── raw/                      # Raw scraped JSON data
│   └── processed/                # Cleaned ML-ready data
├── models/                       # Trained model artifacts (.pkl)
├── outputs/                      # Evaluation results, fair-price tables
├── scraper/
│   └── ikman_scrape_pipeline.py  # ikman.lk web scraper
├── src/
│   ├── config.py                 # Paths, constants, feature lists
│   ├── phone_specs.py            # iPhone/Android spec lookups
│   ├── data_preprocessing.py     # Data cleaning and standardization
│   ├── feature_engineering.py    # Derived feature creation
│   ├── train.py                  # Training pipeline (main entry point)
│   ├── evaluate.py               # Model evaluation and comparison
│   └── predict.py                # Prediction API
├── requirements.txt
├── README.md
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

This scrapes all mobile phone listings from ikman.lk and saves them to `data/raw/`.

### 2. Train the model

```bash
python -m src.train
```

This will:
1. Load and clean the raw scraped data
2. Engineer features (model tier, brand tier, phone age, etc.)
3. Train multiple models with hyperparameter tuning
4. Evaluate using cross-validation and a held-out test set
5. Save the best model per phone type to `models/`
6. Generate fair-price predictions for all phone categories

### 3. Evaluate

Evaluation results are automatically saved to `outputs/model_evaluation_results.json` after training.

### 4. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

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
```

## Models

| Phone Type | Models Compared |
|-----------|----------------|
| iPhone | Random Forest, XGBoost, Gradient Boosting, LightGBM |
| Android | Random Forest, XGBoost, Gradient Boosting, LightGBM |

The best model is selected based on combined MAE + R² ranking.

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
| model_tier | engineered | Phone generation/premium tier (1-10) |
| brand_tier | engineered | Brand price segment (1-3) |
| phone_age_years | engineered | Estimated age from release year |
| is_flagship | engineered | Flagship model flag |

## License

MIT
