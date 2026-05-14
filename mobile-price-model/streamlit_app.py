from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
CLEANED_DATA_FILE = BASE_DIR / "ikman_mobile_phones_ml_ready.json"
EVALUATION_FILE = BASE_DIR / "model_evaluation_results.json"
FAIR_PRICE_FILE = BASE_DIR / "fair_price_predictions.json"
MODEL_DIR = BASE_DIR / "models"
HIDDEN_MODEL_NAMES = {"other model", "unknown"}
HIDDEN_BRANDS_BY_PHONE_TYPE = {
    "android": {"other brand"},
}


def format_lkr(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"LKR {float(value):,.2f}"


@st.cache_data
def load_cleaned_data() -> pd.DataFrame:
    df = pd.read_json(CLEANED_DATA_FILE)
    df["storage_gb"] = pd.to_numeric(df["storage_gb"], errors="coerce")
    df["ram_gb"] = pd.to_numeric(df["ram_gb"], errors="coerce")
    df["warranty_days"] = pd.to_numeric(df["warranty_days"], errors="coerce")
    return df


@st.cache_data
def load_evaluation_data() -> dict[str, Any]:
    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_fair_price_table() -> pd.DataFrame:
    with FAIR_PRICE_FILE.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    table = pd.DataFrame(payload.get("records", []))
    if not table.empty and "storage_gb" in table.columns:
        table["storage_gb"] = pd.to_numeric(table["storage_gb"], errors="coerce")
    return table


def get_recommended_model_info(phone_type: str, evaluation_data: dict[str, Any]) -> tuple[str, Path, dict[str, Any]]:
    recommendation = evaluation_data.get("recommendations", {}).get(phone_type, {})
    model_name = str(recommendation.get("recommended_model", "XGBoost Regressor"))
    model_prefix = "xgboost" if "xgboost" in model_name.lower() else "random_forest"
    model_key = f"{model_prefix}_{phone_type}"
    model_path = MODEL_DIR / f"{model_key}.pkl"
    metrics = evaluation_data.get("results", {}).get(model_key, {})
    return model_name, model_path, metrics


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def safe_mode(series: pd.Series, default: Any) -> Any:
    cleaned = series.dropna()
    if cleaned.empty:
        return default
    modes = cleaned.mode(dropna=True)
    if modes.empty:
        return cleaned.iloc[0]
    return modes.iloc[0]


def safe_median(series: pd.Series, default: float) -> float:
    value = pd.to_numeric(series, errors="coerce").median(skipna=True)
    if pd.isna(value):
        return float(default)
    return float(value)


def closest_index(options: list[float], target: float) -> int:
    if not options:
        return 0
    distances = [abs(option - target) for option in options]
    return int(distances.index(min(distances)))


def visible_model_options(series: pd.Series) -> list[str]:
    return sorted(
        model
        for model in series.dropna().unique().tolist()
        if str(model).strip().lower() not in HIDDEN_MODEL_NAMES
    )


def visible_brand_options(series: pd.Series, phone_type: str) -> list[str]:
    hidden_brands = HIDDEN_BRANDS_BY_PHONE_TYPE.get(phone_type, set())
    return sorted(
        brand
        for brand in series.dropna().unique().tolist()
        if str(brand).strip().lower() not in hidden_brands
    )


def find_exact_category_match(
    fair_price_df: pd.DataFrame,
    phone_type: str,
    brand: str,
    model: str,
    storage_gb: float,
) -> pd.Series | None:
    if fair_price_df.empty:
        return None

    mask = (
        (fair_price_df["phone_type"] == phone_type)
        & (fair_price_df["brand"] == brand)
        & (fair_price_df["model"] == model)
        & np.isclose(fair_price_df["storage_gb"].fillna(-1), storage_gb, atol=0.01)
    )
    matches = fair_price_df.loc[mask]
    if matches.empty:
        return None
    return matches.iloc[0]


def build_prediction_input(
    brand: str,
    model: str,
    storage_gb: float,
    ram_gb: float,
    warranty_days: float,
    dual_sim: bool,
    has_5g: bool,
    has_esim: bool,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "brand": brand,
                "model": model,
                "condition": "used",
                "currency": "LKR",
                "dual_sim": int(dual_sim),
                "has_5g": int(has_5g),
                "has_esim": int(has_esim),
                "warranty_days": float(warranty_days),
                "storage_gb": float(storage_gb),
                "ram_gb": float(ram_gb),
            }
        ]
    )


def main() -> None:
    st.set_page_config(page_title="Mobile Fair Price Predictor", layout="wide")

    st.title("Used Mobile Fair Price Predictor")
    st.caption("Simple Streamlit frontend for your research demo using the trained models and saved fair-price outputs.")

    cleaned_df = load_cleaned_data()
    evaluation_data = load_evaluation_data()
    fair_price_df = load_fair_price_table()

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    summary_col1.metric("Cleaned Records", f"{len(cleaned_df):,}")
    summary_col2.metric("iPhone Records", f"{int((cleaned_df['phone_type'] == 'iphone').sum()):,}")
    summary_col3.metric("Android Records", f"{int((cleaned_df['phone_type'] == 'android').sum()):,}")
    summary_col4.metric("Saved Fair-Price Categories", f"{len(fair_price_df):,}")

    predict_tab, lookup_tab = st.tabs(["Predict Price", "Browse Category Table"])

    with predict_tab:
        left_col, right_col = st.columns([1.25, 1])

        with left_col:
            phone_type = st.selectbox(
                "Phone type",
                options=["android", "iphone"],
                format_func=lambda value: value.title(),
            )

            phone_subset = cleaned_df.loc[cleaned_df["phone_type"] == phone_type].copy()
            brand_options = visible_brand_options(phone_subset["brand"], phone_type)
            if not brand_options:
                st.warning("No visible brands are available after hiding placeholder brand names.")
                st.stop()
            brand = st.selectbox("Brand", options=brand_options)

            brand_subset = phone_subset.loc[phone_subset["brand"] == brand].copy()
            model_options = visible_model_options(brand_subset["model"])
            if not model_options:
                st.warning("No visible models are available for this brand after hiding placeholder model names.")
                st.stop()
            model = st.selectbox("Model", options=model_options)

            model_subset = brand_subset.loc[brand_subset["model"] == model].copy()
            storage_options = sorted(
                float(value)
                for value in model_subset["storage_gb"].dropna().unique().tolist()
            )
            default_storage = storage_options[0] if storage_options else 128.0
            storage_gb = st.selectbox(
                "Storage (GB)",
                options=storage_options or [default_storage],
                index=0,
                format_func=lambda value: f"{int(value) if float(value).is_integer() else value} GB",
            )

            category_subset = model_subset.loc[np.isclose(model_subset["storage_gb"], storage_gb, atol=0.01)].copy()
            default_subset = category_subset if not category_subset.empty else model_subset

            ram_options = sorted(
                float(value)
                for value in default_subset["ram_gb"].dropna().unique().tolist()
            )
            default_ram = safe_median(default_subset["ram_gb"], default=6.0)
            ram_gb = st.selectbox(
                "RAM (GB)",
                options=ram_options or [default_ram],
                index=closest_index(ram_options or [default_ram], default_ram),
                format_func=lambda value: f"{int(value) if float(value).is_integer() else value} GB",
            )

            warranty_default = int(round(safe_median(default_subset["warranty_days"], default=0.0)))
            warranty_days = st.number_input(
                "Warranty (days)",
                min_value=0,
                max_value=3650,
                value=warranty_default,
                step=1,
            )

            feature_col1, feature_col2, feature_col3 = st.columns(3)
            dual_sim = bool(int(safe_mode(default_subset["dual_sim"], 0)))
            has_5g = bool(int(safe_mode(default_subset["has_5g"], 0)))
            has_esim = bool(int(safe_mode(default_subset["has_esim"], 0)))
            with feature_col1:
                st.checkbox("Dual SIM", value=dual_sim, disabled=True)
            with feature_col2:
                st.checkbox("5G", value=has_5g, disabled=True)
            with feature_col3:
                st.checkbox("eSIM", value=has_esim, disabled=True)

            predict_button = st.button("Predict fair price", type="primary", use_container_width=True)

        with right_col:
            model_name, model_path, metrics = get_recommended_model_info(phone_type, evaluation_data)
            st.subheader("Model used")
            st.write(f"Recommended model: **{model_name}**")
            st.write(f"Expected MAE: **{format_lkr(metrics.get('mae'))}**")
            st.write(f"R² score: **{metrics.get('r2_score', 0):.4f}**")
            st.write("Training filter: **used phones in LKR**")

            exact_match = find_exact_category_match(fair_price_df, phone_type, brand, model, float(storage_gb))
            st.subheader("Saved category match")
            if exact_match is None:
                st.info("No exact saved category match was found for the selected combination.")
            else:
                st.write(f"Category: **{exact_match['category_key']}**")
                st.write(f"Saved fair price: **{format_lkr(exact_match['fair_price_lkr'])}**")
                st.write(f"Observed median: **{format_lkr(exact_match['observed_median_lkr'])}**")

        if predict_button:
            if not model_path.exists():
                st.error(f"Model file not found: {model_path.name}")
            else:
                model_pipeline = load_model(str(model_path))
                input_df = build_prediction_input(
                    brand=brand,
                    model=model,
                    storage_gb=float(storage_gb),
                    ram_gb=float(ram_gb),
                    warranty_days=float(warranty_days),
                    dual_sim=dual_sim,
                    has_5g=has_5g,
                    has_esim=has_esim,
                )
                predicted_price = max(0.0, float(model_pipeline.predict(input_df)[0]))
                mae = float(metrics.get("mae", 0.0))
                range_low = max(0.0, predicted_price - mae)
                range_high = predicted_price + mae

                result_col1, result_col2, result_col3 = st.columns(3)
                result_col1.metric("Predicted fair price", format_lkr(predicted_price))
                result_col2.metric("Estimated lower range", format_lkr(range_low))
                result_col3.metric("Estimated upper range", format_lkr(range_high))

                st.success("Prediction completed successfully.")

                details_df = pd.DataFrame(
                    [
                        {
                            "Phone Type": phone_type.title(),
                            "Brand": brand,
                            "Model": model,
                            "Storage (GB)": storage_gb,
                            "RAM (GB)": ram_gb,
                            "Warranty (days)": warranty_days,
                            "Dual SIM": "Yes" if dual_sim else "No",
                            "5G": "Yes" if has_5g else "No",
                            "eSIM": "Yes" if has_esim else "No",
                            "Model Used": model_name,
                        }
                    ]
                )
                st.dataframe(details_df, use_container_width=True, hide_index=True)

    with lookup_tab:
        st.subheader("Saved fair-price categories")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            selected_type = st.selectbox(
                "Filter by phone type",
                options=["all", "android", "iphone"],
                format_func=lambda value: value.title(),
                key="lookup_phone_type",
            )
        with filter_col2:
            brand_choices = ["all"] + sorted(fair_price_df["brand"].dropna().unique().tolist())
            selected_brand = st.selectbox("Filter by brand", options=brand_choices, key="lookup_brand")
        with filter_col3:
            model_search = st.text_input("Search model", placeholder="Type model name")

        filtered_table = fair_price_df.copy()
        if selected_type != "all":
            filtered_table = filtered_table.loc[filtered_table["phone_type"] == selected_type]
        if selected_brand != "all":
            filtered_table = filtered_table.loc[filtered_table["brand"] == selected_brand]
        if model_search.strip():
            filtered_table = filtered_table.loc[
                filtered_table["model"].str.contains(model_search.strip(), case=False, na=False)
            ]

        display_columns = [
            "phone_type",
            "brand",
            "model",
            "storage_gb",
            "fair_price_lkr",
            "fair_price_range_low_lkr",
            "fair_price_range_high_lkr",
            "sample_count",
            "confidence",
            "observed_median_lkr",
        ]
        st.dataframe(
            filtered_table[display_columns].sort_values(
                by=["sample_count", "phone_type", "brand", "model", "storage_gb"],
                ascending=[False, True, True, True, True],
            ),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
