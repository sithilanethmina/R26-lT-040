"""
Streamlit app for mobile phone fair-price prediction.

Loads trained models and provides:
1. A prediction tab — select phone specs and get a fair price estimate
2. A category browser — view pre-computed fair prices for all phone models
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLEANED_DATA_FILE, EVALUATION_FILE, FAIR_PRICE_JSON, MODEL_DIR
from src.feature_engineering import (
    compute_brand_tier,
    compute_is_flagship,
    compute_model_tier,
    compute_phone_age,
)
from src.predict import get_model_info, load_model

HIDDEN_MODEL_NAMES = {"other model", "unknown"}
HIDDEN_BRANDS = {"other brand"}


def format_lkr(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"LKR {float(value):,.2f}"


@st.cache_data
def load_cleaned_data() -> pd.DataFrame:
    df = pd.read_json(CLEANED_DATA_FILE)
    for col in ["storage_gb", "ram_gb", "warranty_days", "battery_health_percent"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data
def load_fair_price_table() -> pd.DataFrame:
    if not FAIR_PRICE_JSON.exists():
        return pd.DataFrame()
    with FAIR_PRICE_JSON.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    table = pd.DataFrame(payload.get("records", []))
    if not table.empty and "storage_gb" in table.columns:
        table["storage_gb"] = pd.to_numeric(table["storage_gb"], errors="coerce")
    return table


def safe_mode(series: pd.Series, default: Any) -> Any:
    cleaned = series.dropna()
    if cleaned.empty:
        return default
    modes = cleaned.mode(dropna=True)
    return modes.iloc[0] if not modes.empty else cleaned.iloc[0]


def safe_median(series: pd.Series, default: float) -> float:
    v = pd.to_numeric(series, errors="coerce").median(skipna=True)
    return float(v) if not pd.isna(v) else default


def closest_idx(options: list[float], target: float) -> int:
    if not options:
        return 0
    return int(min(range(len(options)), key=lambda i: abs(options[i] - target)))


def visible_options(series: pd.Series, hidden: set[str] | None = None) -> list[str]:
    hidden = hidden or set()
    return sorted(v for v in series.dropna().unique() if str(v).strip().lower() not in hidden)


def main() -> None:
    st.set_page_config(page_title="Mobile Fair Price Predictor", layout="wide")
    st.title("📱 Used Mobile Fair Price Predictor")
    st.caption("Predict fair market prices for used smartphones in Sri Lanka using ML models trained on ikman.lk data.")

    try:
        cleaned_df = load_cleaned_data()
        fair_price_df = load_fair_price_table()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(cleaned_df):,}")
    c2.metric("iPhone", f"{int((cleaned_df['phone_type'] == 'iphone').sum()):,}")
    c3.metric("Android", f"{int((cleaned_df['phone_type'] == 'android').sum()):,}")
    c4.metric("Price Categories", f"{len(fair_price_df):,}")

    predict_tab, browse_tab = st.tabs(["🔮 Predict Price", "📋 Browse Categories"])

    # ── Predict tab ──────────────────────────────────────────────────────
    with predict_tab:
        left, right = st.columns([1.3, 1])

        with left:
            phone_type = st.selectbox("Phone Type", ["iphone", "android"],
                                       format_func=str.title)

            subset = cleaned_df[cleaned_df["phone_type"] == phone_type]
            brands = visible_options(subset["brand"], HIDDEN_BRANDS)
            if not brands:
                st.warning("No brands available.")
                st.stop()
            brand = st.selectbox("Brand", brands)

            brand_sub = subset[subset["brand"] == brand]
            models = visible_options(brand_sub["model"], HIDDEN_MODEL_NAMES)
            if not models:
                st.warning("No models available for this brand.")
                st.stop()
            model = st.selectbox("Model", models)

            model_sub = brand_sub[brand_sub["model"] == model]
            storages = sorted(float(v) for v in model_sub["storage_gb"].dropna().unique())
            storage_gb = st.selectbox(
                "Storage (GB)",
                storages or [128.0],
                format_func=lambda v: f"{int(v)} GB" if float(v).is_integer() else f"{v} GB",
            )

            cat_sub = model_sub[np.isclose(model_sub["storage_gb"], storage_gb, atol=0.01)]
            ref = cat_sub if not cat_sub.empty else model_sub

            rams = sorted(float(v) for v in ref["ram_gb"].dropna().unique())
            default_ram = safe_median(ref["ram_gb"], 6.0)
            ram_gb = st.selectbox(
                "RAM (GB)",
                rams or [default_ram],
                index=closest_idx(rams or [default_ram], default_ram),
                format_func=lambda v: f"{int(v)} GB" if float(v).is_integer() else f"{v} GB",
            )

            warranty_days = st.number_input(
                "Warranty (days)", min_value=0, max_value=3650,
                value=int(round(safe_median(ref["warranty_days"], 0.0))), step=1,
            )

            if phone_type == "iphone":
                bh_default = min(max(int(round(safe_median(ref["battery_health_percent"], 85.0))), 0), 100)
                battery_health = st.slider("Battery Health (%)", 0, 100, bh_default)
            else:
                battery_health = safe_median(ref["battery_health_percent"], 90.0)

            # Auto-computed features (disabled checkboxes)
            dual_sim = int(safe_mode(ref["dual_sim"], 0))
            has_5g = int(safe_mode(ref["has_5g"], 0))
            has_esim = int(safe_mode(ref["has_esim"], 0))
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st.checkbox("Dual SIM", value=bool(dual_sim), disabled=True)
            with fc2:
                st.checkbox("5G", value=bool(has_5g), disabled=True)
            with fc3:
                st.checkbox("eSIM", value=bool(has_esim), disabled=True)

            predict_btn = st.button("Predict fair price", type="primary", use_container_width=True)

        with right:
            try:
                info = get_model_info(phone_type)
                st.subheader("Model Info")
                st.write(f"**{info['model_name']}**")
                st.write(f"MAE: **{format_lkr(info['mae'])}**")
                st.write(f"R²: **{info['r2_score']:.4f}**")
                if info.get("cv_mae"):
                    st.write(f"CV MAE: **{format_lkr(info['cv_mae'])}**")
            except FileNotFoundError:
                st.warning("No trained model found. Run training first.")

            # Show saved category match
            if not fair_price_df.empty:
                mask = (
                    (fair_price_df["phone_type"] == phone_type)
                    & (fair_price_df["brand"] == brand)
                    & (fair_price_df["model"] == model)
                    & np.isclose(fair_price_df["storage_gb"].fillna(-1), storage_gb, atol=0.01)
                )
                matches = fair_price_df[mask]
                if not matches.empty:
                    m = matches.iloc[0]
                    st.subheader("Saved Category")
                    st.write(f"Fair price: **{format_lkr(m['fair_price_lkr'])}**")
                    st.write(f"Market median: **{format_lkr(m['observed_median_lkr'])}**")
                    st.write(f"Samples: **{m['sample_count']}** ({m.get('confidence', 'N/A')})")

        if predict_btn:
            try:
                # Compute engineered features
                row = {"brand": brand, "model": model}
                m_tier = compute_model_tier(row)
                b_tier = compute_brand_tier(brand)
                age = compute_phone_age(row)
                flag = compute_is_flagship(row)

                pipeline = load_model(phone_type)
                info = get_model_info(phone_type)

                input_df = pd.DataFrame([{
                    "brand": brand, "model": model,
                    "storage_gb": float(storage_gb), "ram_gb": float(ram_gb),
                    "warranty_days": float(warranty_days),
                    "battery_health_percent": float(battery_health),
                    "dual_sim": dual_sim, "has_5g": has_5g, "has_esim": has_esim,
                    "model_tier": m_tier, "brand_tier": b_tier,
                    "phone_age_years": age, "is_flagship": flag,
                }])

                from src.config import FEATURE_COLUMNS
                input_df = input_df[FEATURE_COLUMNS]

                predicted = max(0.0, float(pipeline.predict(input_df)[0]))
                mae = float(info.get("mae", 0.0))
                lo, hi = max(0.0, predicted - mae), predicted + mae

                r1, r2, r3 = st.columns(3)
                r1.metric("Predicted Fair Price", format_lkr(predicted))
                r2.metric("Range Low", format_lkr(lo))
                r3.metric("Range High", format_lkr(hi))
                st.success("Prediction completed.")

            except FileNotFoundError:
                st.error("Model not found. Please run training first.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    # ── Browse tab ───────────────────────────────────────────────────────
    with browse_tab:
        st.subheader("Fair-Price Category Table")
        if fair_price_df.empty:
            st.info("No fair-price data available. Run training first.")
        else:
            f1, f2, f3 = st.columns(3)
            with f1:
                sel_type = st.selectbox("Phone Type", ["all", "android", "iphone"],
                                         format_func=str.title, key="browse_type")
            with f2:
                brand_opts = ["all"] + sorted(fair_price_df["brand"].dropna().unique().tolist())
                sel_brand = st.selectbox("Brand", brand_opts, key="browse_brand")
            with f3:
                search = st.text_input("Search model", placeholder="Type model name")

            filtered = fair_price_df.copy()
            if sel_type != "all":
                filtered = filtered[filtered["phone_type"] == sel_type]
            if sel_brand != "all":
                filtered = filtered[filtered["brand"] == sel_brand]
            if search.strip():
                filtered = filtered[
                    filtered["model"].str.contains(search.strip(), case=False, na=False)
                ]

            display_cols = [c for c in [
                "phone_type", "brand", "model", "storage_gb",
                "fair_price_lkr", "fair_price_range_low_lkr", "fair_price_range_high_lkr",
                "sample_count", "confidence", "observed_median_lkr",
            ] if c in filtered.columns]

            st.dataframe(
                filtered[display_cols].sort_values(
                    by=["sample_count", "phone_type", "brand"],
                    ascending=[False, True, True],
                ),
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()
