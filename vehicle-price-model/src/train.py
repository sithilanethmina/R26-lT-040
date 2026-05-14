import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def ensure_directories():
    """Ensure output and model directories exist."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'outputs'), exist_ok=True)

def load_and_clean_data(filepath):
    """Loads and cleans the dataset based on project rules."""
    print("Loading data...")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {filepath}")
        print("Please ensure 'clean_aqua_dataset.json' is in the 'data' folder.")
        exit(1)
    
    # Validate required columns exist
    required_cols = ['price_lkr', 'model_year', 'variant', 'fuel_type']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        exit(1)

    # Clean price_lkr and model_year as numeric values
    df['price_lkr'] = pd.to_numeric(df['price_lkr'], errors='coerce')
    df['model_year'] = pd.to_numeric(df['model_year'], errors='coerce')
    
    # Drop rows where price_lkr or model_year is missing
    df = df.dropna(subset=['price_lkr', 'model_year'])

    # Remove cars priced below 5,000,000 LKR (Floor Price)
    # The 2.1M and 2.25M records are error listings
    df = df[df["price_lkr"] >= 5000000].copy()
    
    # Filter for individual years 2012, 2013, 2014, 2015
    df = df[df['model_year'].isin([2012, 2013, 2014, 2015])].copy()
    
    # Print total records after cleaning
    print(f"\nTotal records after cleaning (2012-2015): {len(df)}")
    
    # Print count for each year
    counts = df['model_year'].value_counts().sort_index()
    print("\nRecords per year:")
    print(counts.to_string())
    
    return df

def train_and_evaluate():
    ensure_directories()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'clean_aqua_dataset.json')
    
    df = load_and_clean_data(data_path)
    
    if len(df) == 0:
        print("Error: No data left after filtering. Cannot train models.")
        exit(1)
        
    # Use only model_year and target
    X = df[['model_year']]
    y = df['price_lkr']
    
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use OneHotEncoder inside a ColumnTransformer (treat year as categorical)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['model_year'])
        ]
    )
    
    # Models to train (Ensemble Trio)
    models = {
        'Random Forest Regressor': RandomForestRegressor(random_state=42, n_estimators=100),
        'XGBRegressor': XGBRegressor(random_state=42, objective='reg:squarederror'),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100)
    }
    
    results = []
    trained_pipelines = {}
    
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        # Use Pipeline for each model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2_Score': r2
        })
        
        trained_pipelines[name] = pipeline
        
        # Save each trained model
        model_filename = name.lower().replace(' ', '_') + '.pkl'
        model_path = os.path.join(base_dir, 'models', model_filename)
        joblib.dump(pipeline, model_path)
        
    results_df = pd.DataFrame(results)
    
    # Print model performance clearly
    print("\nModel Performance Comparison:")
    print(results_df.to_string(index=False))
    
    # Choose the best model using lowest MAE
    best_model_row = results_df.loc[results_df['MAE'].idxmin()]
    best_model_name = best_model_row['Model']
    print(f"\nBest Model: {best_model_name} (Lowest MAE: {best_model_row['MAE']:,.2f})")
    
    # Run Cross-Validation on the best model
    best_pipeline = trained_pipelines[best_model_name]
    run_cross_validation(df, best_pipeline, ['model_year'])

    # Save the best model
    best_pipeline = trained_pipelines[best_model_name]
    joblib.dump(best_pipeline, os.path.join(base_dir, 'models', 'best_model.pkl'))
    
    # Save model comparison
    results_df.to_csv(os.path.join(base_dir, 'outputs', 'model_comparison.csv'), index=False)
    
    # Save group price summary
    summary_df = df.groupby(['model_year'])['price_lkr'].agg(['mean', 'median', 'min', 'max', 'count']).reset_index()
    summary_df.columns = ['model_year', 'mean', 'median', 'min', 'max', 'observed_count']
    summary_df.to_csv(os.path.join(base_dir, 'outputs', 'aqua_year_price_summary.csv'), index=False)
    
    # Predict common category prices for individual years
    valid_years = [{"model_year": y} for y in [2012, 2013, 2014, 2015]]
    pred_df = pd.DataFrame(valid_years)
    pred_df['predicted_price_lkr'] = best_pipeline.predict(pred_df)
    
    # Merge observed counts into predictions for the UI
    pred_df = pred_df.merge(summary_df[['model_year', 'observed_count']], on='model_year', how='left')
    pred_df.to_csv(os.path.join(base_dir, 'outputs', 'aqua_year_predictions.csv'), index=False)
    
    # ─── Print Summary Table for User ───
    print("\nPredicted Fair Prices by Toyota Aqua Year:")
    header = f"  {'Year':<10} {'N':>5}   {'Actual Mean (LKR)':>20}   {'Predicted Fair Price (LKR)':>26}"
    print(header)
    print("  " + "-" * 75)
    for _, r in pred_df.iterrows():
        # Get actual mean for this year from summary
        actual_mean = summary_df[summary_df['model_year'] == r['model_year']]['mean'].values[0]
        print(f"  {int(r['model_year']):<10} {int(r['observed_count']):>5}   {actual_mean:>20,.2f}   {r['predicted_price_lkr']:>26,.2f}")
    
    print("\nSuccess! Models, comparisons, and predictions saved to 'models/' and 'outputs/' folders.")

def run_cross_validation(df, pipeline, features):
    """5-fold CV with year stratification."""
    X = df[features]
    y = df["price_lkr"]

    try:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(
            pipeline, X, y,
            cv=list(skf.split(X, df["model_year"])), 
            scoring={"mae": "neg_mean_absolute_error",
                     "rmse": "neg_root_mean_squared_error",
                     "r2":   "r2"},
            return_train_score=False
        )
        
        cv_df = pd.DataFrame({
            "Fold" : range(1, 6),
            "MAE"  : -cv_results["test_mae"],
            "RMSE" : -cv_results["test_rmse"],
            "R2"   : cv_results["test_r2"],
        })
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cv_df.to_csv(os.path.join(base_dir, 'outputs', 'aqua_cross_validation_results.csv'), index=False)

        print("\n5-Fold Cross-Validation Results (Best Model):")
        print(cv_df.to_string(index=False))

        print(f"\n  CV Mean MAE  : {cv_df['MAE'].mean():>12,.2f} LKR  (±{cv_df['MAE'].std():,.2f})")
        print(f"  CV Mean RMSE : {cv_df['RMSE'].mean():>12,.2f} LKR  (±{cv_df['RMSE'].std():,.2f})")
        print(f"  CV Mean R²   : {cv_df['R2'].mean():>12.4f}       (±{cv_df['R2'].std():.4f})")

    except Exception as e:
        print(f"\n  (Cross-validation skipped: {e})")

if __name__ == "__main__":
    train_and_evaluate()
