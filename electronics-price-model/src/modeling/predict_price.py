import pandas as pd
import joblib
import os

def predict_laptop_price(brand, condition, ram_gb, storage_gb, storage_type):
    model_path = 'models/best_laptop_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train_model.py first.")
        return
        
    # Load the saved model and the expected feature columns
    model_data = joblib.load(model_path)
    rf_model = model_data['model']
    features = model_data['features']
    
    # Create a DataFrame for the new laptop
    input_data = {
        'RAM_GB': [ram_gb],
        'Storage_Capacity_GB': [storage_gb],
        f'Brand_Cleaned_{brand.upper()}': [1],
        f'Condition_Cleaned_{condition}': [1],
        f'Storage_Type_{storage_type.upper()}': [1]
    }
    
    df_input = pd.DataFrame(input_data)
    
    # Add any missing columns (from the training phase) and set them to 0
    for col in features:
        if col not in df_input.columns:
            df_input[col] = 0
            
    # Ensure the columns are in the exact same order as training
    df_input = df_input[features]
    
    # Predict the price
    predicted_price = rf_model.predict(df_input)[0]
    
    print("\n" + "="*40)
    print(f"LAPTOP SPECIFICATIONS:")
    print(f"Brand: {brand.upper()}")
    print(f"Condition: {condition}")
    print(f"RAM: {ram_gb} GB")
    print(f"Storage: {storage_gb} GB {storage_type.upper()}")
    print("="*40)
    print(f"--> PREDICTED FAIR PRICE: Rs {predicted_price:,.2f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    print("Welcome to the Ikman Laptop Price Predictor!")
    print("Let's predict the price of a laptop.")
    
    # You can change these values to test different laptops!
    test_brand = "HP"
    test_condition = "Used"
    test_ram = 8
    test_storage = 256
    test_storage_type = "SSD"
    
    predict_laptop_price(test_brand, test_condition, test_ram, test_storage, test_storage_type)
