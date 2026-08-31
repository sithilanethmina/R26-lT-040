import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from app import models_db, clean_brand, clean_model, clean_cpu, clean_generation, safe_float

# Prepare inputs
brand_val = "ASUS"
model_val = "Asus Vivobook"
ram_val = 4
storage_val = 256
storage_type_val = "SSD"
cpu_val = "i3"
generation_val = 13
algorithm = "xgboost"

model_data = models_db['laptop'][algorithm]
model = model_data['model']
features = model_data['features']

q_brand = clean_brand(brand_val, model_val)
q_model = clean_model(q_brand, model_val)
q_cpu = cpu_val if cpu_val else clean_cpu(model_val)
q_cpu = str(q_cpu).strip().upper()
q_gen = safe_float(generation_val) if generation_val is not None else clean_generation(model_val)
q_ram = safe_float(ram_val, 8.0)
q_storage = safe_float(storage_val, 256.0)
q_storage_type = "SSD" if "SSD" in str(storage_type_val).upper() or "SSD" in str(model_val).upper() else "HDD"

input_dict = {
    'RAM_GB': [q_ram],
    'Storage_Capacity_GB': [q_storage],
    'Generation_Cleaned': [q_gen],
    f"Brand_Cleaned_{q_brand.upper()}": [1],
    f"Model_Cleaned_{q_model.upper()}": [1],
    f"CPU_Cleaned_{q_cpu.upper()}": [1],
    f"Condition_Cleaned_Used": [1],
    f"Storage_Type_{q_storage_type.upper()}": [1]
}

df_input = pd.DataFrame(input_dict)
for col in features:
    if col not in df_input.columns:
        df_input[col] = 0
df_input = df_input[features]

print("df_input shape:", df_input.shape)
print("df_input non-zero columns:")
for c in df_input.columns:
    if df_input.iloc[0][c] != 0:
        print(f"  {c}: {df_input.iloc[0][c]}")

try:
    ml_price = float(model.predict(df_input)[0])
    print("ML Predicted Price:", ml_price)
except Exception as e:
    print("Error during predict:", e)
