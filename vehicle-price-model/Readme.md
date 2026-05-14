# Toyota Aqua Price Prediction

## 1. Project Overview
This is a beginner-friendly Python machine learning project designed to predict the fair market value of used Toyota Aqua vehicles in Sri Lanka. It trains three different regression models (Decision Tree, Random Forest, and XGBoost) and automatically selects the best one to make predictions.

## 2. What this model predicts
This experiment specifically predicts the **average category-level fair price** of a Toyota Aqua based on just two features:
- **Year Range** (e.g., "2012-2014", "2015-2017")
- **Variant / Grade** (e.g., "S Grade", "G Grade", "Base/Unknown")

Because it only uses these high-level category groupings (and ignores mileage, exact condition, or specific options), it does *not* predict a fully detailed or exact individual vehicle price. Instead, it provides a general, fair baseline price for the selected category.

## 3. Dataset Location
The dataset file should be placed in the `data/` folder and named `clean_aqua_dataset.json`. 
In this project setup, it has already been placed at:
`vehicle-price-prediction/data/clean_aqua_dataset.json`

## 4. Installation Steps

### 5. How to create virtual environment
Open your terminal or command prompt in the project folder (`vehicle-price-prediction`) and run:
```bash
python -m venv venv
```

### 6. How to install requirements
First, activate the virtual environment:
- **Windows**:
  ```bash
  venv\Scripts\activate
  ```
- **Mac/Linux**:
  ```bash
  source venv/bin/activate
  ```

Once activated, install the necessary Python packages:
```bash
pip install -r requirements.txt
```

## 7. How to train models
To run the data cleaning, model training, and evaluation, execute the training script from the root of the project:
```bash
python src/train.py
```
This script will:
- Clean and filter the dataset based on the requested rules.
- Train the models (Decision Tree, Random Forest, XGBoost).
- Compare them using evaluation metrics and save the best model to the `models/` folder.
- Output summary statistics and performance comparisons to the `outputs/` folder.

## 8. How to run prediction
Once the best model is trained, you can use the prediction script to get estimated prices:
```bash
python src/predict.py --year_range "2012-2014" --variant "S Grade"
```

## 9. Evaluation Metrics Explained (in simple words)
- **MAE (Mean Absolute Error):** The average absolute difference between the predicted price and the actual price. If MAE is 100,000, it means our predictions are off by 100,000 LKR on average.
- **MSE (Mean Squared Error):** Similar to MAE, but it squares the differences before averaging them. This heavily penalizes large errors.
- **RMSE (Root Mean Squared Error):** The square root of MSE. It converts the error back into the original units (LKR), making it easier to understand.
- **R2 Score (R-Squared):** A score between 0 and 1 that tells you how well the model explains the variance in the data. A score of 1.0 is perfect, while 0.0 means the model is no better than simply guessing the average price.

## 10. Understanding the Results
**Lower MAE and RMSE are better!** A lower error means the model's predicted prices are closer to the actual real-world prices. Higher R2 Scores are also better, as it indicates the model understands the patterns in the data well.

## 11. Important Note
As mentioned, this experiment only uses `year_range` and `variant`. It predicts average category-level fair price, not a fully detailed vehicle price taking into account specific damages, upgrades, exact mileage, or location. This provides a baseline understanding of fair vehicle pricing.

---

## Toyota Corolla Combined Model

### Overview
This section extends the project to predict used Toyota Corolla prices in Sri Lanka using **one combined regression model** trained on four Corolla variants together.

### Why One Combined Model?
The dataset has different record counts per variant:
- **121**: ~130 rows
- **141**: ~72 rows
- **AE110**: ~46 rows
- **DX/KE72**: ~25 rows

Training four separate models would make the DX/KE72 model too weak. By combining all variants into **one dataset** and using `variant` as a feature, the model learns patterns across all groups while still distinguishing between them at prediction time.

### Corolla Variants Included
| Variant  | Description |
|----------|-------------|
| 121      | Toyota Corolla 121 (compact/modern) |
| 141      | Toyota Corolla 141 (larger/newer) |
| AE110    | Toyota Corolla AE110 series |
| DX/KE72  | Classic DX / KE72 models |

### Features Used for Training
| Feature      | Type        | Description |
|-------------|-------------|-------------|
| model_year   | Numeric     | Year the vehicle was manufactured |
| variant      | Categorical | Which Corolla type (121, 141, AE110, DX/KE72) |
| transmission | Categorical | Automatic or Manual |
| fuel_type    | Categorical | Petrol, Diesel, or Hybrid |

> The model also compares four feature set combinations (A/B/C/D) and picks the best one automatically.

### How to Train the Corolla Model
Run from the project root (with venv activated):
```bash
python src/train_corolla_combined.py
```
This will:
- Load and filter the Corolla dataset (only High-confidence, `keep` rows)
- Flag and exclude suspect "down payment" price rows
- Train Decision Tree, Random Forest, and XGBoost with 4 feature set combos
- Print overall and per-variant performance tables
- Save the best model automatically

### How to Run Corolla Predictions
```bash
# Corolla 121 - Automatic - Petrol - 2005
python src/predict_corolla.py --model_year 2005 --variant 121 --transmission Automatic --fuel_type Petrol

# Corolla 141 - Automatic - Petrol - 2008
python src/predict_corolla.py --model_year 2008 --variant 141 --transmission Automatic --fuel_type Petrol

# Corolla AE110 - Manual - Petrol - 1998
python src/predict_corolla.py --model_year 1998 --variant AE110 --transmission Manual --fuel_type Petrol

# Corolla DX/KE72 - Manual - Petrol - 1986
python src/predict_corolla.py --model_year 1986 --variant "DX/KE72" --transmission Manual --fuel_type Petrol
```

### Output Files
| File | Description |
|------|-------------|
| `outputs/corolla/corolla_variant_counts.csv` | Record count per variant |
| `outputs/corolla/corolla_variant_price_summary.csv` | Min/max/mean/median price per variant |
| `outputs/corolla/corolla_variant_year_price_summary.csv` | Prices broken down by variant and year |
| `outputs/corolla/corolla_overall_model_comparison.csv` | All model + feature set MAE/RMSE/R2 results |
| `outputs/corolla/corolla_per_variant_model_comparison.csv` | Per-variant scores for each model |
| `outputs/corolla/corolla_feature_set_comparison.csv` | Best MAE achieved per feature set |
| `outputs/corolla/corolla_variant_predictions.csv` | Predicted vs actual mean prices per group |
| `outputs/corolla/corolla_test_predictions.csv` | Row-level predictions on test set with error |
| `outputs/corolla/suspect_price_rows.csv` | Rows flagged as possible down-payment listings |
| `outputs/corolla/corolla_cross_validation_results.csv` | 5-fold cross-validation results |
| `models/corolla_combined/best_corolla_combined_model.pkl` | Best saved combined model |

### Important Note
This Corolla model uses richer features (`model_year`, `variant`, `transmission`, `fuel_type`) compared to the Aqua model. Even so, it provides a **category-level fair price estimate** — not an exact individual vehicle valuation. Specific condition, mileage, and options are not used in this experiment.
