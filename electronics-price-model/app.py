from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    from flask import jsonify
    return jsonify({"status": "ok", "service": "electronics_price_predictor"})

# Dictionary to store all loaded models
models_db = {
    'laptop': {},
    'monitor': {},
    'tablet': {}
}

def load_models():
    # Laptop Models
    for algo in ['xgboost', 'random_forest', 'gradient_boosting']:
        path = f'models/laptop_{algo}.pkl'
        if os.path.exists(path):
            try:
                models_db['laptop'][algo] = joblib.load(path)
                print(f"Loaded Laptop {algo} model")
            except Exception as e:
                print(f"Failed to load Laptop {algo} model: {e}")

    # Monitor Models
    for algo in ['xgboost', 'random_forest']:
        path = f'models/monitor_{algo}.pkl'
        if os.path.exists(path):
            try:
                models_db['monitor'][algo] = joblib.load(path)
                print(f"Loaded Monitor {algo} model")
            except Exception as e:
                print(f"Failed to load Monitor {algo} model: {e}")

    # Tablet Models
    for algo in ['xgboost', 'random_forest']:
        path = f'models/tablet_{algo}.pkl'
        if os.path.exists(path):
            try:
                models_db['tablet'][algo] = joblib.load(path)
                print(f"Loaded Tablet {algo} model")
            except Exception as e:
                print(f"Failed to load Tablet {algo} model: {e}")

load_models()

def safe_float(value, default=0.0):
    try:
        return float(value) if value else default
    except (ValueError, TypeError):
        return default

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        category = data.get('category', 'laptop')
        algorithm = data.get('algorithm', 'xgboost')
        
        if category not in models_db or algorithm not in models_db[category]:
            return jsonify({'success': False, 'error': f'Model {algorithm} for {category} not found.'})
        
        model_data = models_db[category][algorithm]
        model = model_data['model']
        features = model_data['features']
        
        # Prepare input data
        input_dict = {}
        
        if category == 'laptop':
            input_dict = {
                'RAM_GB': [safe_float(data.get('ram'))],
                'Storage_Capacity_GB': [safe_float(data.get('storage'))],
                'Generation_Cleaned': [safe_float(data.get('generation'))],
                f"Brand_Cleaned_{data.get('brand', '').upper()}": [1],
                f"Model_Cleaned_{data.get('model', '').upper()}": [1],
                f"CPU_Cleaned_{data.get('cpu', '').upper()}": [1],
                f"Condition_Cleaned_Used": [1],
                f"Storage_Type_{data.get('storageType', '').upper()}": [1]
            }
        elif category == 'monitor':
            input_dict = {
                'Size_Inch': [safe_float(data.get('size'))],
                'Refresh_Rate_Hz': [safe_float(data.get('refreshRate'))],
                f"Brand_Cleaned_{data.get('brand', '').upper()}": [1],
                f"Condition_Cleaned_{data.get('condition', 'Used')}": [1],
                f"Resolution_Cleaned_{data.get('resolution', 'FHD').upper()}": [1]
            }
        elif category == 'tablet':
            input_dict = {
                'RAM_GB': [safe_float(data.get('ram'))],
                'Storage_GB': [safe_float(data.get('storage'))],
                f"Brand_Cleaned_{data.get('brand', '').upper()}": [1],
                f"Model_Cleaned_{data.get('model', '').upper()}": [1],
                f"Condition_Cleaned_Used": [1]
            }

        df_input = pd.DataFrame(input_dict)
        for col in features:
            if col not in df_input.columns:
                df_input[col] = 0
        
        # Ensure the columns are in the exact same order as training
        df_input = df_input[features]
        predicted_price = float(model.predict(df_input)[0])
        
        # Gather all results for comparison
        all_results = {}
        
        # Log to file and terminal for research verification
        log_header = f"\n--- Prediction Request: {category.upper()} ({algorithm}) ---"
        spec_summary = f"Input: {data.get('brand')} {data.get('model')} | Specs: {data.get('ram')}GB RAM, {data.get('storage')}GB {data.get('storageType', '')}"
        price_summary = f"PREDICTED PRICE: {f'Rs {predicted_price:,.2f}'}"
        
        with open('research_prediction_log.txt', 'a') as f:
            f.write(f"{log_header}\n{spec_summary}\n{price_summary}\n")
            print(log_header, flush=True)
            print(spec_summary, flush=True)
            print(price_summary, flush=True)
            
            for algo, m_data in models_db[category].items():
                r2 = float(m_data.get('accuracy', 0))
                all_results[m_data['model_name']] = {'R2': r2}
                status = "[ACTIVE]" if algo == algorithm else "        "
                line = f"{status} {m_data['model_name']}: R2 Score = {r2:.4f}\n"
                f.write(line)
                print(line.strip(), flush=True)
                
            f.write("-" * 40 + "\n")
            print("-" * 40, flush=True)
            
        return jsonify({
            'success': True,
            'price': f"Rs {predicted_price:,.2f}",
            'model_name': model_data['model_name'],
            'accuracy': float(model_data.get('accuracy', 0)),
            'all_results': all_results
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

@app.route('/model_info')
def model_info():
    info = {}
    for cat in models_db:
        info[cat] = {algo: {'accuracy': float(m['accuracy']), 'name': m['model_name']} 
                    for algo, m in models_db[cat].items()}
    return jsonify(info)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8004, use_reloader=False)
