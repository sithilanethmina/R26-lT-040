import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import os

def main():
    input_file = 'data/processed/tablets_cleaned.csv'
    model_output_file = 'models/random_forest_tablet_price.pkl'
    
    print(f"Loading cleaned tablet data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        return
        
    print(f"Dataset shape: {df.shape}")
    
    # Separate features and target
    X = df.drop(['Title', 'Price_Cleaned'], axis=1)
    y = df['Price_Cleaned']
    
    # One-hot encode categorical variables
    print("Encoding categorical variables...")
    categorical_cols = ['Brand_Cleaned', 'Model_Cleaned', 'Condition_Cleaned']
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {len(X_train)} tablets")
    print(f"Testing set: {len(X_test)} tablets\n")
    # 3. Train and compare models
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
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
        save_path = f'models/tablet_{model_slug}.pkl'
        
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
    main()
