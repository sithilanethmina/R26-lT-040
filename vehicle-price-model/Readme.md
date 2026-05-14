# Vehicle Price Prediction Models

## 1. Project Overview
This project contains machine learning models designed to predict the fair market value of various used vehicles in Sri Lanka. It uses regression algorithms (Decision Tree, Random Forest, and XGBoost) to provide category-level price estimates based on historical market data.

Currently, the project includes models for:
- **Toyota Aqua**
- **Toyota Corolla** (Multiple variants combined)
- **Suzuki Alto**

---

## 2. Models Included

### A. Toyota Aqua Model
Predicts the average category-level fair price of a Toyota Aqua based on:
- **Year Range** (e.g., "2012-2014")
- **Variant / Grade** (e.g., "S Grade", "G Grade")

### B. Toyota Corolla Combined Model
A single regression model trained across multiple Corolla generations to learn patterns even from variants with less data.
- **Variants**: 121, 141, AE110, DX/KE72
- **Features**: Model Year, Variant, Transmission, Fuel Type

### C. Suzuki Alto Model
Predicts fair prices for the popular Suzuki Alto models based on manufacturing year and other key features.

---

## 3. Installation Steps

### Create Virtual Environment
Open your terminal in the project folder and run:
```bash
python -m venv venv
```

### Install Requirements
Activate the virtual environment:
- **Windows**: `venv\Scripts\activate`
- **Mac/Linux**: `source venv/bin/activate`

Then install dependencies:
```bash
pip install -r requirements.txt
```

---

## 4. How to Train & Predict

### Training Models
Run the specific training script for the vehicle you want to model:
- **Aqua**: `python src/train.py`
- **Corolla**: `python src/train_corolla_combined.py`
- **Alto**: `python src/train_alto.py`

### Running Predictions
Use the prediction scripts to get estimated prices:
```bash
# Example for Corolla
python src/predict_corolla.py --model_year 2005 --variant 121 --transmission Automatic --fuel_type Petrol
```

---

## 5. Evaluation Metrics
We use the following metrics to ensure model accuracy:
- **MAE (Mean Absolute Error):** Average price difference from actual market value.
- **RMSE (Root Mean Squared Error):** Penalizes larger errors more heavily.
- **R2 Score:** Measures how well the model understands the data (1.0 is perfect).

---

## 6. Important Note
These models provide **category-level fair price estimates**. They provide a general baseline for the selected vehicle type but do not account for individual vehicle condition, exact mileage, or specific aftermarket upgrades.
