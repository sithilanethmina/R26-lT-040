import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def train_random_forest(data_path, model_path):
    print(f"Loading cleaned data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Cleaned data not found. Please run the preprocessing script first.")
        return
        
    print(f"Dataset shape: {df.shape}")
    
    # 1. Feature Engineering: Convert categorical columns to numeric using One-Hot Encoding
    # We will use get_dummies for 'Brand_Cleaned', 'Condition_Cleaned', 'Storage_Type'
    
    features = ['Brand_Cleaned', 'Model_Cleaned', 'CPU_Cleaned', 'Generation_Cleaned', 'Condition_Cleaned', 'RAM_GB', 'Storage_Capacity_GB', 'Storage_Type']
    X = df[features]
    y = df['Price_Cleaned']
    
    print("Encoding categorical variables...")
    X_encoded = pd.get_dummies(X, columns=['Brand_Cleaned', 'Model_Cleaned', 'CPU_Cleaned', 'Condition_Cleaned', 'Storage_Type'], drop_first=True)
    
    # 2. Train/Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} laptops")
    print(f"Testing set: {X_test.shape[0]} laptops")
    
    # 3. Train and compare models
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    best_model = None
    best_r2 = -float('inf')
    best_name = ""
    results = {}

    print("\nTraining and Saving Models Separately:")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
        
        # Save each model individually
        model_slug = name.lower().replace(' ', '_')
        save_path = f'models/laptop_{model_slug}.pkl'
        
        model_data = {
            'model': model,
            'features': list(X_train.columns),
            'accuracy': r2,
            'model_name': name
        }
        joblib.dump(model_data, save_path)
        print(f"{name} Results:")
        print(f"   - R2 Score:   {r2:.4f}")
        print(f"   - MAE:        Rs {mae:,.2f}")
        print(f"   - RMSE:       Rs {rmse:,.2f}")
        print(f"   - Saved to:   {save_path}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name

    print(f"\nTraining complete. Best model was {best_name}.")

if __name__ == "__main__":
    cleaned_data_path = 'data/processed/laptops_cleaned.csv'
    saved_model_path = 'models/random_forest_laptop_price.pkl'
    train_random_forest(cleaned_data_path, saved_model_path)
