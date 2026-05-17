import joblib
import pandas as pd

model_data = joblib.load('models/random_forest_laptop_price.pkl')
features = model_data['features']
importances = model_data['model'].feature_importances_

df_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
df_imp = df_imp.sort_values(by='Importance', ascending=False)

print("Full Feature Importances:")
print(df_imp)
