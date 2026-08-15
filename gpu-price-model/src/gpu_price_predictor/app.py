"""GPU Price Predictor — v2.0 Streamlit App (minimalist redesign)"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import streamlit as st
    cache_resource = st.cache_resource
    cache_data = st.cache_data
except Exception:
    st = None
    def cache_resource(func=None, **kwargs):
        return (lambda f: f) if func is None else func
    def cache_data(func=None, **kwargs):
        return (lambda f: f) if func is None else func

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_V2_PATH = ARTIFACTS_DIR / "gpu_price_model_v2.joblib"
MODEL_V1_PATH = ARTIFACTS_DIR / "gpu_price_model.joblib"
ENRICHED_CSV  = PROJECT_ROOT / "data" / "final" / "gpu_enriched_dataset.csv"
BENCH_CSV     = PROJECT_ROOT / "data" / "final" / "GPU_benchmarks_v7.csv"
SPECS_CSV     = PROJECT_ROOT / "data" / "final" / "gpu_1986-2026.csv"
SHAP_PNG      = ARTIFACTS_DIR / "shap_summary_plot.png"

FUZZY_THRESHOLD      = 80
FUZZY_MATCH_CUTOFF   = FUZZY_THRESHOLD + 5   # named constant, not hardcoded offset

# ── Top-level import for gpu_generation (avoid repeated import per call) ──────
try:
    from gpu_price_predictor.pipeline import derive_gpu_generation as _derive_gpu_gen
    _GPU_GEN_AVAILABLE = True
except Exception:
    _GPU_GEN_AVAILABLE = False


# ── CSS — FairPriceLK Design System ───────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');

:root {
    --bg: #FAFAF8;
    --bg-surface: #FFFFFF;
    --bg-result: #F5F4F0;
    --text-primary: #1A1A18;
    --text-secondary: #6B6B66;
    --text-muted: #A3A39F;
    --border: #E5E5E3;
    --border-hover: #D1D1CD;
    --accent: #D97706;
    --accent-hover: #B45309;
    --success: #16A34A;
    --warning: #D97706;
    --danger: #DC2626;
    --radius: 6px;
}

*, html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    box-sizing: border-box;
}

.stApp {
    background-color: #FAFAF8 !important;
    color: #1A1A18 !important;
}

/* ── Global Text Visibility Overrides ── */
.stApp p, .stApp span, .stApp strong, .stApp li {
    color: #1A1A18;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    color: #1A1A18 !important;
    letter-spacing: -0.02em !important;
}

.section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #6B6B66;
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #E5E5E3;
}

/* ── Header Bar ── */
.fp-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 18px;
    background-color: #FFFFFF;
    border: 1px solid #E5E5E3;
    border-radius: 6px;
    margin-bottom: 1.5rem;
}
.fp-logo-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
}
.fp-logo-icon {
    color: #D97706;
    display: flex;
    align-items: center;
}
.fp-title {
    font-size: 16px;
    font-weight: 700;
    color: #1A1A18;
    letter-spacing: -0.01em;
}
.fp-badge {
    font-size: 11px;
    font-weight: 500;
    padding: 2px 8px;
    background-color: #F5F4F0;
    color: #6B6B66;
    border: 1px solid #E5E5E3;
    border-radius: 6px;
    margin-left: 6px;
}
.fp-subtitle {
    font-size: 12px;
    color: #6B6B66;
}

/* ── Form Labels & Controls (Contrast & Readability Audit) ── */
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #6B6B66 !important;
    margin-bottom: 4px !important;
}

/* Text Inputs, Number Inputs, Text Areas */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
.stDateInput input {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E5E3 !important;
    border-radius: 6px !important;
    color: #1A1A18 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}

.stTextInput input:hover,
.stNumberInput input:hover,
.stTextArea textarea:hover {
    border-color: #D1D1CD !important;
}

.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {
    outline: none !important;
    border-color: #D97706 !important;
    box-shadow: 0 0 0 1px #D97706 !important;
}

.stTextInput input::placeholder,
.stNumberInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #A3A39F !important;
    opacity: 1 !important;
}

/* Number Input Stepper Controls */
.stNumberInput button {
    background-color: #FFFFFF !important;
    border-color: #E5E5E3 !important;
    color: #1A1A18 !important;
}
.stNumberInput button:hover {
    background-color: #F5F4F0 !important;
    border-color: #D1D1CD !important;
    color: #D97706 !important;
}
.stNumberInput button svg {
    fill: #1A1A18 !important;
}

/* Selectboxes & Multiselects */
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E5E3 !important;
    border-radius: 6px !important;
    color: #1A1A18 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    min-height: 38px !important;
    transition: border-color 0.15s ease !important;
}

div[data-baseweb="select"]:hover > div {
    border-color: #D1D1CD !important;
}

div[data-baseweb="select"]:focus-within > div {
    border-color: #D97706 !important;
    box-shadow: 0 0 0 1px #D97706 !important;
}

div[data-baseweb="select"] span,
div[data-baseweb="select"] div {
    color: #1A1A18 !important;
}

div[data-baseweb="select"] [data-placeholder="true"],
div[data-baseweb="select"] [data-placeholder] {
    color: #A3A39F !important;
    opacity: 1 !important;
}

/* Dropdown Menu Popover Options */
div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[role="listbox"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E5E3 !important;
    border-radius: 6px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
}

li[role="option"] {
    background-color: #FFFFFF !important;
    color: #1A1A18 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
}

li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background-color: #F5F4F0 !important;
    color: #D97706 !important;
    font-weight: 500 !important;
}

/* Help text & Captions */
[data-testid="stCaptionContainer"], .stCaptionContainer, small, .help-text {
    font-size: 11px !important;
    color: #6B6B66 !important;
}

/* ── Buttons (FairPriceLK Accent) ── */
.stButton > button,
.stButton > button:focus,
.stButton > button:active {
    background-color: #D97706 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 18px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    cursor: pointer !important;
    transition: background-color 0.15s ease !important;
    box-shadow: none !important;
    width: 100% !important;
}

.stButton > button:hover {
    background-color: #B45309 !important;
    color: #FFFFFF !important;
}

.stButton > button p,
.stButton > button span,
.stButton > button div {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* ── KPI Row ── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px;
    margin-bottom: 2rem;
}
.kpi-cell {
    background: #FFFFFF;
    border: 1px solid #E5E5E3;
    border-radius: 6px;
    padding: 14px 16px;
}
.kpi-label {
    font-size: 11px;
    font-weight: 500;
    color: #6B6B66;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.kpi-value {
    font-family: 'DM Sans', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #1A1A18;
    line-height: 1.2;
}
.kpi-delta {
    font-size: 11px;
    color: #6B6B66;
    margin-top: 4px;
}

/* ── FairPriceLK Result Box ── */
.result-box {
    background-color: #F5F4F0;
    border: 1px solid #E5E5E3;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 1.25rem 0;
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.result-header {
    font-size: 11px;
    font-weight: 700;
    color: #6B6B66;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.result-price {
    font-size: 26px;
    font-weight: 700;
    color: #1A1A18;
    letter-spacing: -0.02em;
}
.result-meta {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 4px;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    background-color: #E5E5E3;
    color: #1A1A18;
    border: 1px solid rgba(0, 0, 0, 0.05);
}
.badge.fair {
    background-color: #DCFCE7;
    color: #16A34A;
    border-color: #BBF7D0;
}
.badge.overpriced {
    background-color: #FEF3C7;
    color: #D97706;
    border-color: #FDE68A;
}
.badge.scam {
    background-color: #FEE2E2;
    color: #DC2626;
    border-color: #FECACA;
}
.diff-text {
    font-size: 12px;
    color: #6B6B66;
    font-weight: 500;
}
.result-footer {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid #E5E5E3;
    font-size: 11px;
    color: #6B6B66;
}

/* ── Leaderboard Table ── */
.lb-table {
    width: 100%;
    border-collapse: collapse;
    background: #FFFFFF;
    border: 1px solid #E5E5E3;
    border-radius: 6px;
    overflow: hidden;
    font-size: 13px;
    margin-bottom: 1.5rem;
}
.lb-table th {
    font-size: 11px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #6B6B66;
    font-weight: 700;
    padding: 10px 14px;
    background: #FAFAF8;
    border-bottom: 1px solid #E5E5E3;
    text-align: left;
}
.lb-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #E5E5E3;
    color: #1A1A18;
}
.lb-table tr:last-child td { border-bottom: none; }
.lb-table tr.best-row { background: #FFFDF5; }
.lb-table tr:hover td { background: #F5F4F0; }
.mono { font-family: 'DM Mono', monospace; font-size: 12px; }
.best-pill {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    background: #FEF3C7;
    color: #D97706;
    border: 1px solid #FDE68A;
    padding: 2px 6px;
    border-radius: 4px;
    margin-left: 6px;
    vertical-align: middle;
}
.rank-num {
    font-family: 'DM Mono', monospace;
    color: #A3A39F;
    font-size: 12px;
}
.rank-1 { color: #D97706; font-weight: 700; }

/* ── Metrics Cards ── */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E5E3 !important;
    border-radius: 6px !important;
    padding: 12px 14px !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #6B6B66 !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] div {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    color: #1A1A18 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #E5E5E3 !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #6B6B66 !important;
    padding: 8px 16px !important;
    border-radius: 6px 6px 0 0 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #1A1A18 !important;
    background-color: #F5F4F0 !important;
}
.stTabs [aria-selected="true"] {
    color: #D97706 !important;
    border-bottom: 2px solid #D97706 !important;
    font-weight: 700 !important;
    background: transparent !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E5E3 !important;
    border-radius: 6px !important;
    margin-bottom: 1rem !important;
}
[data-testid="stExpander"] summary {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #1A1A18 !important;
}
[data-testid="stExpander"] summary:hover {
    color: #D97706 !important;
}

/* ── Dataframes ── */
.stDataFrame {
    border: 1px solid #E5E5E3 !important;
    border-radius: 6px !important;
    background: #FFFFFF !important;
}

/* ── Dividers ── */
hr {
    border: none !important;
    border-top: 1px solid #E5E5E3 !important;
    margin: 1.5rem 0 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E5E3 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong {
    color: #1A1A18 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #6B6B66 !important;
    font-size: 13px !important;
}
.dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    background-color: #E5E5E3;
}
.dot.online {
    background-color: #16A34A;
}

/* ── Alerts & Status Messages ── */
.stAlert {
    border-radius: 6px !important;
    font-size: 13px !important;
    border-width: 1px !important;
}
div[data-testid="stAlertContainer"] {
    border-radius: 6px !important;
}
.stAlert [data-testid="stMarkdownContainer"] p {
    color: #1A1A18 !important;
}

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #FAFAF8; }
::-webkit-scrollbar-thumb { background: #D1D1CD; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #A3A39F; }
</style>
"""


# ── Loaders ───────────────────────────────────────────────────────────────────
@cache_resource
def load_artifact():
    if MODEL_V2_PATH.exists():
        return joblib.load(MODEL_V2_PATH), "v2"
    if MODEL_V1_PATH.exists():
        return joblib.load(MODEL_V1_PATH), "v1"
    return None, None


@cache_data
def load_enriched() -> pd.DataFrame | None:
    if ENRICHED_CSV.exists():
        return pd.read_csv(ENRICHED_CSV)
    alt_csv = ARTIFACTS_DIR / "gpu_training_dataset_enriched.csv"
    if alt_csv.exists():
        return pd.read_csv(alt_csv)
    return None


@cache_data
def load_bench_df() -> pd.DataFrame | None:
    if BENCH_CSV.exists():
        df = pd.read_csv(BENCH_CSV)
        df.columns = df.columns.str.strip()   # guard against whitespace in headers
        return df
    return None


@cache_data
def load_specs_df() -> pd.DataFrame | None:
    if not SPECS_CSV.exists():
        return None
    wanted = {
        "Name",
        "Graphics Processor__Architecture",
        "Graphics Card__Release Date",
        "Memory__Bandwidth",
        "Render Config__Shading Units",
        "Clock Speeds__Base Clock",
        "Clock Speeds__Boost Clock",
        "Board Design__TDP",
        "Theoretical Performance__FP32 (float)",
    }
    df = pd.read_csv(SPECS_CSV, low_memory=False)
    df.columns = df.columns.str.strip()       # guard against whitespace in headers
    return df[[c for c in df.columns if c in wanted]]


def _parse_num(val) -> float | None:
    """Extract first float from strings like '9.7 TFLOPS', '256.3 GB/s', '150 W'."""
    if pd.isna(val) or str(val).strip() in ("", "unknown", "N/A"):
        return None
    s = str(val).replace(",", "")
    m = re.search(r"[\d]+(?:\.\d+)?", s)
    if not m:
        return None
    v = float(m.group())
    if "TFLOP" in s.upper():
        v *= 1000.0
    return v


def lookup_gpu_specs(name: str, bench_df, specs_df) -> dict:
    """
    Fuzzy-match GPU name against benchmark and spec CSVs.
    Returns a unified dict of hardware features.
    """
    from rapidfuzz import process as fzp, fuzz

    result: dict = {"found": False}
    if not name.strip():
        return result

    # ── Benchmark CSV ─────────────────────────────────────────────────────────
    if bench_df is not None:
        bench_names = bench_df["gpuName"].dropna().tolist()
        hits = fzp.extract(
            name, bench_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=FUZZY_MATCH_CUTOFF,
            limit=5,
        )
        input_num = _model_number(name)
        best_hit = None
        for hit in hits:
            match_num = _model_number(hit[0])
            if not np.isnan(input_num) and not np.isnan(match_num):
                if input_num != match_num:
                    continue
            best_hit = hit
            break

        if best_hit:
            row = bench_df[bench_df["gpuName"] == best_hit[0]].iloc[0]
            result["bench_name"]  = best_hit[0]
            result["bench_score"] = best_hit[1]
            result["G3Dmark"]     = pd.to_numeric(row.get("G3Dmark"), errors="coerce")
            result["G2Dmark"]     = pd.to_numeric(row.get("G2Dmark"), errors="coerce")
            td = _parse_num(row.get("TDP"))
            if td:
                result["bench_tdp"] = td
            yr_raw = str(row.get("testDate", ""))
            m = re.search(r"\b(20\d{2})\b", yr_raw)
            if m:
                result["release_year_bench"] = int(m.group(1))
            result["found"] = True

    # ── Spec CSV ──────────────────────────────────────────────────────────────
    if specs_df is not None:
        spec_names = specs_df["Name"].dropna().tolist()
        hits2 = fzp.extract(
            name, spec_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=FUZZY_MATCH_CUTOFF,
            limit=5,
        )
        input_num = _model_number(name)
        best_hit2 = None
        for hit in hits2:
            match_num = _model_number(hit[0])
            if not np.isnan(input_num) and not np.isnan(match_num):
                if input_num != match_num:
                    continue
            best_hit2 = hit
            break

        if best_hit2:
            row2 = specs_df[specs_df["Name"] == best_hit2[0]].iloc[0]
            result["spec_name"]               = best_hit2[0]
            result["spec_score"]              = best_hit2[1]
            result["architecture"]            = str(row2.get("Graphics Processor__Architecture", "") or "Unknown")
            result["memory_bandwidth_gb_s"]   = _parse_num(row2.get("Memory__Bandwidth"))
            result["shader_units"]            = _parse_num(row2.get("Render Config__Shading Units"))
            result["gpu_base_clock_mhz"]      = _parse_num(row2.get("Clock Speeds__Base Clock"))
            result["boost_clock_mhz"]         = _parse_num(row2.get("Clock Speeds__Boost Clock"))
            result["spec_tdp"]                = _parse_num(row2.get("Board Design__TDP"))
            result["fp32_gflops"]             = _parse_num(row2.get("Theoretical Performance__FP32 (float)"))
            date_raw = str(row2.get("Graphics Card__Release Date", "") or "")
            m2 = re.search(r"\b(20\d{2})\b", date_raw)
            if m2:
                result["release_year_spec"] = int(m2.group())
            result["found"] = True

    result["tdp_watts"]    = result.get("spec_tdp") or result.get("bench_tdp")
    result["release_year"] = result.get("release_year_spec") or result.get("release_year_bench") or 2020
    return result


# ── Inference helpers ─────────────────────────────────────────────────────────

def _series_family(m: str) -> str:
    hit = re.search(r"\b(RTX|GTX|GTS|GT|RX|HD)\b", str(m), re.IGNORECASE)
    return hit.group(1).upper() if hit else "OTHER"


def _model_number(m: str) -> float:
    hit = re.search(r"\b(\d{3,4})\b", str(m))
    return float(hit.group(1)) if hit else np.nan


def _ti_variant(m: str) -> int:
    return 1 if re.search(r"\bTi\b", str(m), re.IGNORECASE) else 0


def _gpu_generation(m: str) -> int:
    if _GPU_GEN_AVAILABLE:
        try:
            return _derive_gpu_gen(m)
        except Exception:
            pass
    return 0


def predict_all(
    artifact: dict,
    model_name: str,
    vram: float,
    brand: str,
    enriched: pd.DataFrame | None,
    custom_specs: dict | None = None,
) -> dict[str, float]:
    """Run all trained models and return {model_name: predicted_lkr}."""
    feature_cols: list[str] = artifact["feature_columns"]
    all_models: dict = artifact.get("all_models", {})

    from gpu_price_predictor.pipeline import build_inference_feature_frame
    df_inf = build_inference_feature_frame(
        model_name=model_name,
        vram_gb=vram,
        brand=brand,
        enriched_df=enriched,
        custom_specs=custom_specs,
        feature_columns=feature_cols,
    )

    results: dict[str, float] = {}
    for name, pipeline in all_models.items():
        try:
            pred = float(pipeline.predict(df_inf)[0])
            results[name] = max(0.0, float(np.expm1(pred)))
        except Exception:
            pass
    return results


# ── Shared prediction renderer ────────────────────────────────────────────────

def _render_prediction_results(
    predictions: dict[str, float],
    best_name: str,
    eval_results: dict,
    label: str,
    vram: float,
    brand: str,
    listed_price: float = 0.0,
) -> None:
    """Render FairPriceLK prediction card + per-model breakdown."""
    if not predictions:
        st.error("All models failed to produce a prediction. Check that the artifact is valid.")
        return

    sorted_preds = sorted(predictions.items(), key=lambda kv: kv[1])
    best_price = predictions.get(best_name, sorted_preds[0][1])

    # ── Fairness calculation (matching browser-extension popup.js) ──────────────
    if listed_price and listed_price > 0:
        diff = listed_price - best_price
        diff_pct = (diff / best_price) * 100
        diff_formatted = f"Rs. {abs(diff):,.0f}"

        if diff_pct > 15:
            fairness_badge_html = '<span class="badge overpriced">OVERPRICED</span>'
            price_diff_html = f'<span class="diff-text">+{diff_formatted} (+{diff_pct:.1f}%)</span>'
        elif diff_pct < -25:
            fairness_badge_html = '<span class="badge scam">SCAM RISK</span>'
            price_diff_html = f'<span class="diff-text">-{diff_formatted} ({diff_pct:.1f}%)</span>'
        else:
            fairness_badge_html = '<span class="badge fair">FAIR PRICE</span>'
            sign = "+" if diff >= 0 else "-"
            price_diff_html = f'<span class="diff-text">{sign}{diff_formatted} ({abs(diff_pct):.1f}%)</span>'
    else:
        fairness_badge_html = '<span class="badge">ESTIMATED MARKET VALUE</span>'
        price_diff_html = '<span class="diff-text">Enter listed price to evaluate fairness</span>'

    best_mape = eval_results.get(best_name, {}).get("mape_pct", "?")

    st.markdown(f"""
    <div class="result-box">
        <div class="result-header">PREDICTED MARKET VALUE</div>
        <div class="result-price">Rs. {best_price:,.0f}</div>
        <div class="result-meta">
            {fairness_badge_html}
            {price_diff_html}
        </div>
        <div class="result-footer">
            Model used: <strong>{best_name.replace('_', ' ').title()}</strong> (MAPE {best_mape}%) &nbsp;·&nbsp; {label} &nbsp;·&nbsp; {vram:.0f} GB VRAM &nbsp;·&nbsp; {brand}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">All model predictions</div>', unsafe_allow_html=True)
    pred_cols = st.columns(len(sorted_preds))
    for i, (mname, price) in enumerate(sorted_preds):
        m = eval_results.get(mname, {})
        is_best = mname == best_name
        label_str = ("★ " if is_best else "") + mname.replace("_", " ").title()
        with pred_cols[i]:
            st.metric(label_str, f"LKR {price:,.0f}")
            st.caption(f"MAPE {m.get('mape_pct', '?')}% · R²={m.get('r2', '?')}")

    st.divider()
    vc1, vc2 = st.columns(2)
    with vc1:
        st.markdown('<div class="section-label">Price by algorithm</div>', unsafe_allow_html=True)
        pred_df = pd.DataFrame({
            "Algorithm": [n.replace("_", " ").title() for n, _ in sorted_preds],
            "Price (LKR)": [p for _, p in sorted_preds],
        })
        st.bar_chart(pred_df.set_index("Algorithm"))
    with vc2:
        st.markdown('<div class="section-label">MAPE % by algorithm</div>', unsafe_allow_html=True)
        mape_rows = [(n, eval_results[n]["mape_pct"]) for n, _ in sorted_preds if n in eval_results]
        if mape_rows:
            mape_df = pd.DataFrame(mape_rows, columns=["Algorithm", "MAPE %"])
            mape_df["Algorithm"] = mape_df["Algorithm"].str.replace("_", " ").str.title()
            st.bar_chart(mape_df.set_index("Algorithm"))


def _get_model_col(df: pd.DataFrame | None) -> str:
    if df is None:
        return "model"
    for col in ["extracted_model", "norm_model", "model"]:
        if col in df.columns:
            return col
    return df.columns[0] if len(df.columns) > 0 else "model"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="FairPriceLK — GPU Price Predictor",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    artifact, artifact_ver = load_artifact()
    enriched = load_enriched()

    # ── FairPriceLK Header ────────────────────────────────────────────────────
    st.markdown("""
    <div class="fp-header">
        <div class="fp-logo-wrap">
            <div class="fp-logo-icon">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
            </div>
            <span class="fp-title">FairPriceLK</span>
            <span class="fp-badge">GPU Valuation</span>
        </div>
        <div class="fp-subtitle">v2.0 · 6-Model Ensemble · Sri Lanka Market</div>
    </div>
    """, unsafe_allow_html=True)

    if artifact is None:
        st.error("No model artifact found. Run `python scripts/train_model_v2.py` to train.")
        return

    eval_results: dict = artifact.get("evaluation_results", {})
    best_name: str     = artifact.get("best_model_name", "")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            <span class="dot online"></span>
            <strong style="color: #1A1A18; font-size: 14px;">Pipeline Status</strong>
        </div>
        """, unsafe_allow_html=True)
        if enriched is not None:
            st.write(f"Training records: `{len(enriched):,}`")
        if best_name and best_name in eval_results:
            m = eval_results[best_name]
            st.write(f"Active model: `{best_name.replace('_', ' ').title()}`")
            st.write(f"MAPE: `{m.get('mape_pct')}%`")
            st.write(f"R²: `{m.get('r2')}`")
            st.write(f"Within 10%: `{m.get('within_10pct')}%`")
            st.write(f"RMSE: `LKR {m.get('rmse_lkr'):,.0f}`")
        st.divider()
        st.caption("FairPriceLK · GPU Valuation Model · 2026")

    # ── KPI strip ─────────────────────────────────────────────────────────────
    bm = eval_results.get(best_name, {})

    if enriched is not None:
        n_samples = f"{len(enriched):,}"
        model_col = _get_model_col(enriched)
        n_models  = str(enriched[model_col].dropna().nunique())
        avg_price = f"LKR {enriched['price_lkr'].mean():,.0f}" if "price_lkr" in enriched.columns else "—"
    else:
        n_samples = n_models = avg_price = "—"

    best_mape = f"{bm.get('mape_pct', '—')}%"
    within10  = f"{bm.get('within_10pct', '—')}%"

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-cell">
            <div class="kpi-label">Training listings</div>
            <div class="kpi-value">{n_samples}</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">Unique GPU models</div>
            <div class="kpi-value">{n_models}</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">Avg market price</div>
            <div class="kpi-value">{avg_price}</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">Best model MAPE</div>
            <div class="kpi-value">{best_mape}</div>
            <div class="kpi-delta">lower is better</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">Within 10% accuracy</div>
            <div class="kpi-value">{within10}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Leaderboard ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Model leaderboard</div>', unsafe_allow_html=True)

    sorted_models = sorted(eval_results.items(), key=lambda kv: kv[1].get("mape_pct", 99))
    rows_html = ""
    for i, (mname, metrics) in enumerate(sorted_models):
        is_best   = mname == best_name
        rank_cls  = "rank-num rank-1" if i == 0 else "rank-num"
        best_pill = '<span class="best-pill">best</span>' if is_best else ""
        row_cls   = "best-row" if is_best else ""
        rows_html += f"""
        <tr class="{row_cls}">
            <td><span class="{rank_cls}">{i + 1}</span></td>
            <td class="mono">{mname.replace('_', ' ').title()}{best_pill}</td>
            <td class="mono">{metrics.get('mape_pct')}%</td>
            <td class="mono">{metrics.get('r2')}</td>
            <td class="mono">{metrics.get('within_10pct')}%</td>
            <td class="mono">LKR {metrics.get('rmse_lkr', 0):,.0f}</td>
        </tr>"""

    st.markdown(f"""
    <table class="lb-table">
        <thead>
            <tr>
                <th>#</th><th>Model</th><th>MAPE</th>
                <th>R²</th><th>Within 10%</th><th>RMSE</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    # ── Prediction ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-label">Price prediction & fairness checker</div>', unsafe_allow_html=True)

    if enriched is None:
        st.warning("No enriched dataset found. Run `build_benchmark_features.py` first.")
        return

    model_col     = _get_model_col(enriched)
    unique_models = sorted(enriched[model_col].dropna().unique().tolist())
    unique_brands = sorted(enriched["brand"].dropna().unique().tolist()) if "brand" in enriched.columns else []

    tab_listed, tab_custom = st.tabs(["Listed GPU", "Unlisted / Custom GPU"])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 1 — Listed GPU
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_listed:
        fcol1, fcol2 = st.columns([2, 1])
        with fcol1:
            selected_model = st.selectbox("GPU Model", unique_models, key="sel_model")
        mask = enriched[model_col] == selected_model
        typical_vram = float(enriched.loc[mask, "vram_gb"].dropna().median() or 4.0)
        with fcol2:
            selected_brand = st.selectbox("Brand / Manufacturer", ["Any"] + unique_brands, key="sel_brand")

        fcol3, fcol4 = st.columns(2)
        with fcol3:
            selected_vram = st.number_input("VRAM (GB)", min_value=1.0, value=typical_vram, step=1.0, key="listed_vram")
        with fcol4:
            listed_price = st.number_input("Listed / Asking Price (LKR — Optional)", min_value=0.0, value=0.0, step=1000.0, key="listed_price_input", help="Enter seller asking price to evaluate fairness")

        if st.button("Check Price", key="btn_listed"):
            with st.spinner("Evaluating models…"):
                predictions = predict_all(artifact, selected_model, selected_vram,
                                          selected_brand, enriched)
            _render_prediction_results(
                predictions, best_name, eval_results,
                selected_model, selected_vram, selected_brand,
                listed_price=listed_price,
            )

            # Market context
            st.divider()
            st.markdown('<div class="section-label">Market context</div>', unsafe_allow_html=True)
            ctx_mask = enriched[model_col] == selected_model
            if selected_brand != "Any" and "brand" in enriched.columns:
                ctx_mask &= enriched["brand"] == selected_brand
            matches = enriched[ctx_mask]

            if not matches.empty:
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric("Listings found", len(matches))
                with mc2:
                    st.metric("Median price", f"LKR {matches['price_lkr'].median():,.0f}")
                with mc3:
                    spread = matches["price_lkr"].max() - matches["price_lkr"].min()
                    st.metric("Price spread", f"LKR {spread:,.0f}")

                color_col = "brand" if "brand" in matches.columns else None
                st.scatter_chart(matches, x="vram_gb", y="price_lkr", color=color_col)

                with st.expander("View comparable market listings"):
                    show_cols = [c for c in [model_col, "brand", "vram_gb", "price_lkr",
                                             "G3Dmark", "gpu_age_years", "architecture"]
                                 if c in matches.columns]
                    st.dataframe(matches[show_cols].sort_values("price_lkr").reset_index(drop=True),
                                 width="stretch")
            else:
                st.info("No historical listings for this selection. Prediction extrapolated from features.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 2 — Custom / Unlisted GPU
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_custom:
        bench_df = load_bench_df()
        specs_df = load_specs_df()

        all_ref_names = []
        if bench_df is not None:
            all_ref_names.extend(bench_df["gpuName"].dropna().unique().tolist())
        if specs_df is not None:
            all_ref_names.extend(specs_df["Name"].dropna().unique().tolist())
        all_ref_names = sorted(list(set(all_ref_names)))

        r1c1, r1c2 = st.columns([2, 1])
        with r1c1:
            if all_ref_names:
                custom_name = st.selectbox(
                    "Search GPU Model", 
                    options=all_ref_names,
                    index=None,
                    placeholder="Type to search (e.g. RTX 3070)",
                    key="cust_name_select",
                    help="Search across 10,000+ reference GPUs",
                )
                if custom_name is None: custom_name = ""
            else:
                custom_name = st.text_input(
                    "GPU Name", placeholder="e.g. RTX 3070",
                    key="cust_name_input",
                )
        with r1c2:
            custom_brand = st.selectbox("Brand / Manufacturer", ["Unknown"] + unique_brands, key="cust_brand")

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            custom_vram = st.number_input("VRAM (GB)", min_value=1.0, value=8.0, step=1.0, key="cust_vram")
        with r2c2:
            custom_listed_price = st.number_input("Listed / Asking Price (LKR — Optional)", min_value=0.0, value=0.0, step=1000.0, key="cust_listed_price", help="Enter seller asking price to evaluate fairness")

        # Auto-lookup
        if custom_name.strip():
            specs = lookup_gpu_specs(custom_name.strip(), bench_df, specs_df)

            if specs.get("found"):
                src_parts = []
                if specs.get("bench_name"):
                    src_parts.append(f"Benchmark: **{specs['bench_name']}** ({specs.get('bench_score', 0):.0f}%)")
                if specs.get("spec_name"):
                    src_parts.append(f"Specs: **{specs['spec_name']}** ({specs.get('spec_score', 0):.0f}%)")
                st.success("Specs auto-filled · " + " · ".join(src_parts))

                summary: dict[str, str] = {}
                if specs.get("G3Dmark"):
                    summary["G3Dmark"]        = f"{specs['G3Dmark']:,.0f}"
                if specs.get("G2Dmark"):
                    summary["G2Dmark"]        = f"{specs['G2Dmark']:,.0f}"
                if specs.get("tdp_watts"):
                    summary["TDP (W)"]        = f"{specs['tdp_watts']:.0f}"
                if specs.get("fp32_gflops"):
                    summary["FP32 GFLOPS"]    = f"{specs['fp32_gflops']:,.0f}"
                if specs.get("memory_bandwidth_gb_s"):
                    summary["Bandwidth GB/s"] = f"{specs['memory_bandwidth_gb_s']:.1f}"
                if specs.get("shader_units"):
                    summary["Shader units"]   = f"{specs['shader_units']:.0f}"
                if specs.get("architecture"):
                    summary["Architecture"]   = specs["architecture"]
                summary["Release year"]       = str(specs.get("release_year", "?"))
                summary["GPU age (yrs)"]      = str(2026 - int(specs.get("release_year", 2020)))

                s_cols = st.columns(min(len(summary), 6))
                for i, (k, v) in enumerate(summary.items()):
                    with s_cols[i % 6]:
                        st.metric(k, v)

                st.divider()
                if st.button("Check Price from Specs", key="btn_custom"):
                    with st.spinner("Evaluating models…"):
                        custom_predictions = predict_all(
                            artifact, custom_name, custom_vram, custom_brand,
                            enriched, custom_specs=specs
                        )

                    if custom_predictions:
                        _render_prediction_results(
                            custom_predictions, best_name, eval_results,
                            custom_name, custom_vram, custom_brand,
                            listed_price=custom_listed_price,
                        )
                    else:
                        st.error("All models failed. Check that the artifact matches the feature schema.")

            else:
                st.error(
                    f"**'{custom_name}'** not found in reference databases. "
                    "Prediction requires hardware specs — check the GPU name spelling."
                )

    # ── Dataset explorer ──────────────────────────────────────────────────────
    st.divider()
    with st.expander("Explore enriched dataset"):
        show_cols = [c for c in [
            model_col, "brand", "vram_gb", "price_lkr", "G3Dmark",
            "tdp_watts", "gpu_age_years", "series_family", "architecture",
        ] if c in enriched.columns]
        st.dataframe(enriched[show_cols], width="stretch")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;opacity:0.65;font-size:0.75rem;"
        "font-family:DM Sans,sans-serif;color:#6B6B66;'>"
        "FairPriceLK · GPU Price Predictor · v2.0 · 2026"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()