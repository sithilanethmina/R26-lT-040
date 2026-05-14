import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
# Model Paths for Corolla & Aqua (single model set each)
MODELS = {
    "Toyota Corolla": {
        "dir": os.path.join(BASE_DIR, "models", "corolla_combined"),
        "metrics": os.path.join(BASE_DIR, "outputs", "corolla", "corolla_overall_model_comparison.csv"),
        "predictions": os.path.join(BASE_DIR, "outputs", "corolla", "corolla_variant_predictions.csv"),
        "files": {
            "Random Forest Regressor": "random_forest_regressor.pkl",
            "XGBoost Regressor": "xgbregressor.pkl",
            "Gradient Boosting": "gradient_boosting_regressor.pkl"
        }
    },
    "Toyota Aqua": {
        "dir": os.path.join(BASE_DIR, "models"),
        "metrics": os.path.join(BASE_DIR, "outputs", "model_comparison.csv"),
        "predictions": os.path.join(BASE_DIR, "outputs", "aqua_year_predictions.csv"),
        "files": {
            "Random Forest Regressor": "random_forest_regressor.pkl",
            "XGBoost Regressor": "xgbregressor.pkl",
            "Gradient Boosting": "gradient_boosting.pkl"
        }
    }
}

# Alto has separate models per group
ALTO_GROUPS = [
    "G1_Manual_2000-2012",
    "G2_Manual_2013-2015",
    "G3_Manual_2016-2019",
    "G4_Auto_lt700_2000-2015",
]
ALTO_MODEL_DIR  = os.path.join(BASE_DIR, "models", "alto")
ALTO_METRICS    = os.path.join(BASE_DIR, "outputs", "alto", "alto_overall_model_comparison.csv")
ALTO_PREDICTIONS = os.path.join(BASE_DIR, "outputs", "alto", "alto_year_predictions.csv")

# --- State ---
loaded_models = {"Toyota Corolla": {}, "Toyota Aqua": {}}
alto_models = {}  # { group_name: { model_name: pipeline } }
alto_best = {}    # { group_name: pipeline }
metrics_dfs = {}
predictions_dfs = {}
alto_metrics_df = None
alto_predictions_df = None


def init_app():
    global alto_metrics_df, alto_predictions_df

    # Load Corolla & Aqua models
    for model_name, config in MODELS.items():
        for name, filename in config["files"].items():
            path = os.path.join(config["dir"], filename)
            if os.path.exists(path):
                try:
                    loaded_models[model_name][name] = joblib.load(path)
                    print(f"Loaded {model_name} model: {name}")
                except Exception as e:
                    print(f"Error loading {model_name} {name}: {e}")

        # Load metrics CSV
        if os.path.exists(config["metrics"]):
            metrics_dfs[model_name] = pd.read_csv(config["metrics"])
            metrics_dfs[model_name]['Model'] = metrics_dfs[model_name]['Model'].replace({
                'XGBRegressor': 'XGBoost Regressor'
            })

        # Load predictions CSV
        if os.path.exists(config["predictions"]):
            predictions_dfs[model_name] = pd.read_csv(config["predictions"])

    # Load Alto group models
    for group in ALTO_GROUPS:
        group_dir = os.path.join(ALTO_MODEL_DIR, group)
        if not os.path.isdir(group_dir):
            print(f"[Alto] Group dir not found: {group}")
            continue

        alto_models[group] = {}
        # Load individual models
        model_files = {
            "Random Forest Regressor": "random_forest_regressor.pkl",
            "XGBoost Regressor": "xgbregressor.pkl",
            "Gradient Boosting": "gradient_boosting.pkl",
        }
        for name, fname in model_files.items():
            path = os.path.join(group_dir, fname)
            if os.path.exists(path):
                try:
                    alto_models[group][name] = joblib.load(path)
                    print(f"Loaded Alto [{group}] model: {name}")
                except Exception as e:
                    print(f"Error loading Alto [{group}] {name}: {e}")

        # Load best model
        best_path = os.path.join(group_dir, "best_model.pkl")
        if os.path.exists(best_path):
            alto_best[group] = joblib.load(best_path)

    # Load Alto metrics & predictions
    if os.path.exists(ALTO_METRICS):
        alto_metrics_df = pd.read_csv(ALTO_METRICS)
        alto_metrics_df['Model'] = alto_metrics_df['Model'].replace({
            'XGBRegressor': 'XGBoost Regressor'
        })

    if os.path.exists(ALTO_PREDICTIONS):
        alto_predictions_df = pd.read_csv(ALTO_PREDICTIONS)


# --- Corolla year range helper ---
def get_year_range(variant, year):
    if variant == "121":
        if 2000 <= year <= 2001: return "2000-2001"
        if 2002 <= year <= 2003: return "2002-2003"
        if 2004 <= year <= 2008: return "2004-2008"
    elif variant == "141":
        if 2007 <= year <= 2009: return "2007-2009"
        if 2010 <= year <= 2013: return "2010-2013"
    elif variant == "AE110":
        if 1994 <= year <= 1996: return "1994-1996"
        if 1997 <= year <= 2000: return "1997-2000"
    elif variant == "DX/KE72":
        if 1980 <= year <= 1984: return "1980-1984"
        if 1985 <= year <= 1990: return "1985-1990"

    if year < 1990: return "Pre-1990"
    if year < 2000: return "1990-1999"
    if year < 2010: return "2000-2009"
    return "2010-Later"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        vehicle_model = data.get('model', 'Toyota Corolla')
        model_year = int(data.get('model_year'))
        variant = data.get('variant')
        transmission = data.get('transmission')
        fuel_type = data.get('fuel_type')

        # --- Suzuki Alto (group-based) ---
        if vehicle_model == "Suzuki Alto":
            group = variant  # variant IS the group name for Alto
            if group not in alto_models or not alto_models[group]:
                return jsonify({"status": "error", "message": f"No models found for Alto group: {group}"}), 400

            # Build input — Alto uses model_year_str (categorical)
            input_df = pd.DataFrame([{"model_year_str": str(model_year)}])
            year_range = str(model_year)

            results = []
            for name, pipe in alto_models[group].items():
                predicted_price = pipe.predict(input_df)[0]

                # Get metrics for this model + group
                mae_val = 0
                r2_val = 0
                if alto_metrics_df is not None:
                    m_df = alto_metrics_df[
                        (alto_metrics_df['Model'] == name) &
                        (alto_metrics_df['group'] == group)
                    ]
                    if not m_df.empty:
                        mae_val = float(m_df['MAE'].values[0])
                        r2_val = float(m_df['R2_Score'].values[0])

                # Get sample size
                sample_size = 0
                if alto_predictions_df is not None:
                    match = alto_predictions_df[
                        (alto_predictions_df['group'] == group) &
                        (alto_predictions_df['model_year'] == model_year)
                    ]
                    if not match.empty and 'observed_count' in match.columns:
                        sample_size = int(match['observed_count'].values[0])

                results.append({
                    "name": name,
                    "predictedPrice": float(predicted_price),
                    "mae": mae_val,
                    "r2": r2_val,
                    "sample_size": sample_size
                })

            return jsonify({
                "status": "success",
                "predictions": results,
                "year_range": year_range
            })

        # --- Toyota Corolla ---
        elif vehicle_model == "Toyota Corolla":
            year_range = get_year_range(variant, model_year)
            input_df = pd.DataFrame([{
                "year_range": year_range,
                "variant": variant,
                "transmission": transmission,
                "fuel_type": fuel_type,
            }])

        # --- Toyota Aqua ---
        else:
            input_df = pd.DataFrame([{"model_year": model_year}])
            year_range = str(model_year)

        # Corolla / Aqua shared prediction logic
        results = []
        metrics_df = metrics_dfs.get(vehicle_model)
        predictions_df = predictions_dfs.get(vehicle_model)

        for name, model_pipe in loaded_models[vehicle_model].items():
            predicted_price = model_pipe.predict(input_df)[0]

            # Find metrics
            m_df = metrics_df[(metrics_df['Model'] == name)]
            if vehicle_model == "Toyota Corolla":
                m_df = m_df[m_df['Feature_Set'] == 'D']

            if m_df.empty and not metrics_df.empty:
                m_df = metrics_df[metrics_df['Model'] == name].iloc[0:1]

            # Sample size
            sample_size = 0
            if predictions_df is not None:
                if vehicle_model == "Toyota Corolla":
                    match = predictions_df[
                        (predictions_df['variant'].astype(str) == str(variant)) &
                        (predictions_df['year_range'] == year_range) &
                        (predictions_df['transmission'] == transmission)
                    ]
                else:  # Aqua
                    match = predictions_df[predictions_df['model_year'] == model_year]

                if not match.empty:
                    sample_size = int(match['observed_count'].values[0]) if 'observed_count' in match.columns else 0

            results.append({
                "name": name,
                "predictedPrice": float(predicted_price),
                "mae": float(m_df['MAE'].values[0]) if not m_df.empty else 0,
                "r2": float(m_df['R2_Score'].values[0]) if not m_df.empty else 0,
                "sample_size": sample_size
            })

        return jsonify({
            "status": "success",
            "predictions": results,
            "year_range": year_range
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/metadata', methods=['GET'])
def get_metadata():
    return jsonify({
        "models": ["Toyota Corolla", "Toyota Aqua", "Suzuki Alto"],
        "variants": ["121", "141", "AE110", "DX/KE72", "Aqua",
                      "G1_Manual_2000-2012", "G2_Manual_2013-2015",
                      "G3_Manual_2016-2019", "G4_Auto_lt700_2000-2015"],
        "transmissions": ["Automatic", "Manual"],
        "fuel_types": ["Petrol", "Diesel", "Hybrid"]
    })

if __name__ == '__main__':
    init_app()
    app.run(debug=True, port=5000)
