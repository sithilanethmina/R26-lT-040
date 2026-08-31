import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

def train_and_benchmark_tablet_models(data_path, models_dir):
    print("=" * 65)
    print("TABLET PRICE REGRESSION: TRAINING & BENCHMARKING ENGINE")
    print("=" * 65)
    
    if not os.path.exists(data_path):
        print(f"[!] Error: Preprocessed data not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    print(f"[*] Loaded Cleaned Dataset: {len(df):,} rows, {df.shape[1]} columns")
    
    # 1. Define Features & Target
    cat_cols = ['Brand_Cleaned', 'Model_Cleaned', 'Connectivity_Cleaned', 'Screen_Size', 'Condition_Cleaned', 'Location_Cleaned']
    num_cols = ['Storage_GB', 'RAM_GB']
    feature_cols = cat_cols + num_cols
    
    X = df[feature_cols].copy()
    y_raw = df['Price_Cleaned'].values
    
    # Log target transformation for variance stabilization
    y_log = np.log1p(y_raw)
    
    # Ensure all categorical columns are string type
    for col in cat_cols:
        X[col] = X[col].astype(str)
        
    # 2. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train_log, y_test_log, y_train_raw, y_test_raw = train_test_split(
        X, y_log, y_raw, test_size=0.2, random_state=42
    )
    
    print(f"[*] Training Set: {len(X_train):,} samples | Test Set: {len(X_test):,} samples\n")
    
    # Pre-encode for models that require numeric inputs (XGBoost, Random Forest)
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    X_train_encoded[cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_test_encoded[cat_cols] = encoder.transform(X_test[cat_cols])
    
    # LightGBM categorical format
    X_train_lgb = X_train.copy()
    X_test_lgb = X_test.copy()
    for col in cat_cols:
        X_train_lgb[col] = X_train_lgb[col].astype('category')
        X_test_lgb[col] = X_test_lgb[col].astype('category')
    
    # 3. Model Definitions
    models = {
        'CatBoost': {
            'model': CatBoostRegressor(
                iterations=700,
                learning_rate=0.06,
                depth=6,
                cat_features=cat_cols,
                verbose=0,
                random_seed=42
            ),
            'X_tr': X_train,
            'X_te': X_test,
            'type': 'native_cat'
        },
        'XGBoost': {
            'model': XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                reg_alpha=0.1,
                reg_lambda=1.0
            ),
            'X_tr': X_train_encoded,
            'X_te': X_test_encoded,
            'type': 'encoded'
        },
        'LightGBM': {
            'model': lgb.LGBMRegressor(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbosity=-1
            ),
            'X_tr': X_train_lgb,
            'X_te': X_test_lgb,
            'type': 'lgb_cat'
        },
        'Random Forest': {
            'model': RandomForestRegressor(
                n_estimators=150,
                max_depth=14,
                random_state=42,
                n_jobs=-1
            ),
            'X_tr': X_train_encoded,
            'X_te': X_test_encoded,
            'type': 'encoded'
        }
    }
    
    os.makedirs(models_dir, exist_ok=True)
    results = {}
    best_model_name = None
    best_r2 = -float('inf')
    
    print("-" * 65)
    print(f"{'Model Name':<16} | {'R2 Score':<10} | {'MAE (LKR)':<14} | {'RMSE (LKR)':<14} | {'MAPE':<8}")
    print("-" * 65)
    
    for name, config in models.items():
        clf = config['model']
        clf.fit(config['X_tr'], y_train_log)
        
        # Predict on test set and inverse-transform to real Rupees
        preds_log = clf.predict(config['X_te'])
        preds_raw = np.expm1(preds_log)
        
        # Calculate real-world metrics in LKR
        r2 = r2_score(y_test_raw, preds_raw)
        mae = mean_absolute_error(y_test_raw, preds_raw)
        rmse = np.sqrt(mean_squared_error(y_test_raw, preds_raw))
        mape = mean_absolute_percentage_error(y_test_raw, preds_raw) * 100
        
        results[name] = {
            'R2': r2,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'model': clf
        }
        
        print(f"{name:<16} | {r2:.4f}     | Rs {mae:,.0f}{'':<5} | Rs {rmse:,.0f}{'':<5} | {mape:.2f}%")
        
        # Save individual model bundle
        slug = name.lower().replace(' ', '_')
        bundle_path = os.path.join(models_dir, f"tablet_{slug}.pkl")
        
        bundle = {
            'model': clf,
            'model_name': name,
            'model_type': config['type'],
            'features': feature_cols,
            'cat_cols': cat_cols,
            'num_cols': num_cols,
            'encoder': encoder if config['type'] == 'encoded' else None,
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        joblib.dump(bundle, bundle_path)
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            
    print("-" * 65)
    print(f"\n[BEST MODEL]: {best_model_name} with R2 = {results[best_model_name]['R2']:.4f} and MAE = Rs {results[best_model_name]['MAE']:,.2f}")
    
    # Save the primary production model
    best_bundle_path = os.path.join(models_dir, "best_tablet_model.pkl")
    prod_bundle_path = os.path.join(models_dir, "tablet_production_model.pkl")
    rf_compat_path = os.path.join(models_dir, "random_forest_tablet_price.pkl")
    
    best_bundle = {
        'model': results[best_model_name]['model'],
        'model_name': best_model_name,
        'model_type': models[best_model_name]['type'],
        'features': feature_cols,
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'encoder': encoder if models[best_model_name]['type'] == 'encoded' else None,
        'r2_score': results[best_model_name]['R2'],
        'mae': results[best_model_name]['MAE'],
        'mape': results[best_model_name]['MAPE']
    }
    
    joblib.dump(best_bundle, best_bundle_path)
    joblib.dump(best_bundle, prod_bundle_path)
    joblib.dump(best_bundle, rf_compat_path)
    print(f"[+] Saved Best Production Model to:")
    print(f"    - {best_bundle_path}")
    print(f"    - {prod_bundle_path}")
    print("=" * 65)
    return results

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "processed", "tablets_cleaned.csv"))
    models_directory = os.path.abspath(os.path.join(script_dir, "..", "..", "models"))
    
    train_and_benchmark_tablet_models(data_file, models_directory)
