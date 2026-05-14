import json
import pandas as pd
import re

DATA_PATH = "data/clean_corolla_dataset_final.json"

def analyze_hidden_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df = df[df["variant"].isin(["121", "141", "AE110", "DX/KE72"])]
    
    total = len(df)
    
    # 1. Analyze Mileage
    df["mileage_km"] = pd.to_numeric(df["mileage_km"], errors="coerce").fillna(0)
    mileage_valid = df[df["mileage_km"] > 0]
    mileage_pct = (len(mileage_valid) / total) * 100
    
    # 2. Extract Grade Keywords
    def extract_grade(text):
        text = str(text).upper()
        if "G GRADE" in text or "G-GRADE" in text or " G " in text: return "G Grade"
        if "X GRADE" in text or "X-GRADE" in text or " X " in text: return "X Grade"
        if "LX" in text: return "LX"
        if "LIMITED" in text: return "Limited"
        if "SE SALOON" in text or "SE-SALOON" in text: return "SE Saloon"
        return "Not Specified"

    df["detected_grade"] = df["seller_description_clean"].apply(extract_grade)
    grade_counts = df["detected_grade"].value_counts()
    grade_pct = ((total - grade_counts.get("Not Specified", 0)) / total) * 100

    print(f"--- Dataset Analysis (N={total}) ---")
    print(f"Mileage Data: {len(mileage_valid)} records have mileage ({mileage_pct:.1f}%)")
    print(f"Grade Data: Found specific grades in {total - grade_counts.get('Not Specified', 0)} records ({grade_pct:.1f}%)")
    print("\nDetected Grades Summary:")
    print(grade_counts)

if __name__ == "__main__":
    analyze_hidden_data()
