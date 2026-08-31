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
            return 0


def predict_all(
    artifact: dict,
    model_name: str,
    vram: float,
    brand: str,
    enriched: pd.DataFrame | None,
    custom_specs: dict | None = None,
    return_log: bool = False,
) -> dict[str, float]:
    """Run all trained models and return {model_name: predicted_lkr or log_pred}."""
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
            if return_log:
                results[name] = float(pred)
            else:
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
    calibration_data: dict | None = None,
    sample_count: int = 20,
    description: str = "",
    is_shop: bool = False,
) -> None:
    """Render FairPriceLK prediction card + per-model breakdown."""
    if not predictions:
        st.error("All models failed to produce a prediction. Check that the artifact is valid.")
        return

    # Safety Guard 1: Newly Released Generation Restriction (RTX 50-series / Blackwell)
    import re
    if bool(re.search(r'\b(rtx\s*50\d{2}|rx\s*90\d{2})\b', str(label).lower())):
        st.warning(
            f"⚠️ **Newly Released Generation**: "
            f"The **{label}** belongs to a newly released hardware generation. "
            "Secondary market pricing has not yet stabilized in Sri Lanka, so automatic price valuation is restricted to ensure accuracy."
        )
        return

    # Safety Guard 2: Check for minimum sample size threshold (N >= 3)
    MIN_SAMPLE_THRESHOLD = 3
    if sample_count < MIN_SAMPLE_THRESHOLD:
        st.warning(
            f"⚠️ **Insufficient Market Data**: "
            f"Market listings for **{label}** are currently limited in Sri Lanka. "
            "Automatic price valuation is unavailable to ensure accuracy."
        )
        return

    from gpu_price_predictor.pipeline import (
        apply_condition_adjustment,
        calculate_fair_market_range,
        get_fairness_verdict,
    )

    sorted_preds = sorted(predictions.items(), key=lambda kv: kv[1])
    base_price = predictions.get(best_name, sorted_preds[0][1])

    # ── Condition & Warranty Adjustment ─────────────────────────────────────────
    base_log_price = float(np.log1p(base_price))
    condition_adj = apply_condition_adjustment(
        predicted_log_price=base_log_price,
        description=description,
        is_shop=is_shop,
    )
    effective_log_price = condition_adj["adjusted_log_price"]
    adjusted_best_price = condition_adj["adjusted_price_lkr"]
    condition_multiplier = condition_adj["condition_multiplier_pct"]
    applied_factors = condition_adj["applied_factors"]

    # ── Conformal Range Calculation ─────────────────────────────────────────────
    range_info = calculate_fair_market_range(
        predicted_log_price=effective_log_price,
        sample_count=sample_count,
        calibration_data=calibration_data,
        confidence_level="90%"
    )

    lower_price = range_info["lower_price_lkr"]
    upper_price = range_info["upper_price_lkr"]

    # ── Fairness Verdict & Score ────────────────────────────────────────────────
    verdict_info = get_fairness_verdict(listed_price, lower_price, upper_price)
    badge_cls = verdict_info["badge_class"]
    verdict_text = verdict_info["verdict"]
    score = verdict_info.get("fairness_score", 0.0)
    desc = verdict_info.get("description", "")

    if listed_price and listed_price > 0:
        if badge_cls == "great-deal":
            fairness_badge_html = f'<span class="badge" style="background:#ECFDF5; border:1px solid #A7F3D0; color:#047857;">{verdict_text.upper()}</span>'
        elif badge_cls == "fair":
            fairness_badge_html = f'<span class="badge" style="background:#EFF6FF; border:1px solid #BFDBFE; color:#1D4ED8;">{verdict_text.upper()}</span>'
        elif badge_cls == "high":
            fairness_badge_html = f'<span class="badge" style="background:#FFFBEB; border:1px solid #FDE68A; color:#B45309;">{verdict_text.upper()}</span>'
        elif badge_cls == "overpriced":
            fairness_badge_html = f'<span class="badge" style="background:#FEF2F2; border:1px solid #FECACA; color:#B91C1C;">{verdict_text.upper()}</span>'
        elif badge_cls == "suspicious":
            fairness_badge_html = f'<span class="badge" style="background:#FFF1F2; border:1px solid #FECDD3; color:#BE123C;">{verdict_text.upper()}</span>'
        else:
            fairness_badge_html = f'<span class="badge scam">{verdict_text.upper()}</span>'

        diff_lkr = verdict_info.get("price_difference_lkr", 0)
        diff_pct = verdict_info.get("price_difference_pct", 0.0)
        sign = "+" if diff_lkr >= 0 else "-"
        price_diff_html = f'<span class="diff-text">Listing: <strong>Rs. {listed_price:,.0f}</strong> &nbsp;·&nbsp; {sign}Rs. {abs(diff_lkr):,.0f} ({sign}{abs(diff_pct):.1f}%) &nbsp;·&nbsp; Fairness Score: <strong>{score:.0f}/100</strong></span>'
    else:
        fairness_badge_html = '<span class="badge">ESTIMATED FAIR MARKET RANGE</span>'
        price_diff_html = '<span class="diff-text">Enter seller asking price above to evaluate listing fairness</span>'

    # Condition factors HTML pills
    condition_pills_html = ""
    if applied_factors:
        pills = []
        for f in applied_factors:
            fname = f["factor"].replace("_", " ").title()
            pills.append(f'<span style="background:#FEF3C7; border:1px solid #FCD34D; color:#92400E; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600;">{fname} ({f["pct_impact"]})</span>')
        condition_pills_html = f'<div style="margin-top:8px; display:flex; gap:6px; flex-wrap:wrap; align-items:center;"><span style="font-size:11px; font-weight:700; color:#6B6B66;">CONDITION ADJUSTMENTS ({condition_multiplier:+.1f}%):</span> {" ".join(pills)}</div>'

    best_mape = eval_results.get(best_name, {}).get("mape_pct", "?")
    warn_html = ""
    if range_info.get("limited_data_warning"):
        warn_html = ' &nbsp;·&nbsp; <span style="color:#D97706;font-weight:600;">⚠ Limited market data for this GPU</span>'

    st.markdown(f"""
    <div class="result-box">
        <div class="result-header">ESTIMATED FAIR MARKET RANGE (90% CONFIDENCE)</div>
        <div class="result-price">Rs. {lower_price:,.0f} – Rs. {upper_price:,.0f}</div>
        <div class="result-meta">
            {fairness_badge_html}
            {price_diff_html}
        </div>
        {condition_pills_html}
        {f'<div style="font-size:12px;color:#6B6B66;margin-top:6px;">💡 {desc}</div>' if desc else ''}
        <div class="result-footer">
            Primary model: <strong>{best_name.replace('_', ' ').title()}</strong> (MAPE {best_mape}%) &nbsp;·&nbsp; {label} &nbsp;·&nbsp; {vram:.0f} GB VRAM &nbsp;·&nbsp; {brand}{warn_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Technical Model Details & Point Prediction", expanded=False):
        st.write(f"**Base Hardware Point Estimate:** `Rs. {base_price:,.0f}` &nbsp;·&nbsp; **Condition-Adjusted Point Estimate:** `Rs. {adjusted_best_price:,.0f}` ({condition_multiplier:+.1f}%)")
        st.caption("The fair market range is derived using Split Conformal Prediction on empirical out-of-fold log-residuals.")
        
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


def _get_model_training_records_info(artifact: dict | None, enriched: pd.DataFrame | None) -> dict:
    total_len = len(enriched) if enriched is not None else 11280
    train_default = int(round(total_len * 0.8))
    test_default = total_len - train_default

    meta = artifact.get("training_records", {}) if artifact else {}
    models_meta = meta.get("models", {})

    default_models_info = {
        "lightgbm": {
            "name": "LightGBM",
            "train_records": models_meta.get("lightgbm", {}).get("train_records", train_default),
            "test_records": models_meta.get("lightgbm", {}).get("test_records", test_default),
            "tune_records": models_meta.get("lightgbm", {}).get("tune_records", train_default),
            "preprocessing": "Tree-based (No scaling)",
            "notes": f"Tuned with Optuna on {models_meta.get('lightgbm', {}).get('train_records', train_default):,} training records; evaluated on {models_meta.get('lightgbm', {}).get('test_records', test_default):,} test records.",
        },
        "xgboost": {
            "name": "XGBoost",
            "train_records": models_meta.get("xgboost", {}).get("train_records", train_default),
            "test_records": models_meta.get("xgboost", {}).get("test_records", test_default),
            "tune_records": models_meta.get("xgboost", {}).get("tune_records", train_default),
            "preprocessing": "Tree-based (No scaling)",
            "notes": f"Tuned with Optuna on {models_meta.get('xgboost', {}).get('train_records', train_default):,} training records; evaluated on {models_meta.get('xgboost', {}).get('test_records', test_default):,} test records.",
        },
        "random_forest": {
            "name": "Random Forest",
            "train_records": models_meta.get("random_forest", {}).get("train_records", train_default),
            "test_records": models_meta.get("random_forest", {}).get("test_records", test_default),
            "tune_records": models_meta.get("random_forest", {}).get("tune_records", train_default),
            "preprocessing": "Tree-based (SimpleImputer + OrdinalEncoder)",
            "notes": f"Tuned with Optuna on {models_meta.get('random_forest', {}).get('train_records', train_default):,} training records; evaluated on {models_meta.get('random_forest', {}).get('test_records', test_default):,} test records.",
        },
        "knn": {
            "name": "KNN (K-Nearest Neighbors)",
            "train_records": models_meta.get("knn", {}).get("train_records", train_default),
            "test_records": models_meta.get("knn", {}).get("test_records", test_default),
            "tune_records": models_meta.get("knn", {}).get("tune_records", train_default),
            "preprocessing": "StandardScaler Normalized",
            "notes": f"Features normalized using StandardScaler across {models_meta.get('knn', {}).get('train_records', train_default):,} training records.",
        },
        "svr": {
            "name": "SVR (Support Vector Regressor)",
            "train_records": models_meta.get("svr", {}).get("train_records", train_default),
            "test_records": models_meta.get("svr", {}).get("test_records", test_default),
            "tune_records": models_meta.get("svr", {}).get("tune_records", min(train_default, 5000)),
            "preprocessing": "StandardScaler Normalized",
            "notes": f"Hyperparameters tuned on {models_meta.get('svr', {}).get('tune_records', min(train_default, 5000)):,} records for speed; final fit trained on all {models_meta.get('svr', {}).get('train_records', train_default):,} records.",
        },
        "stacking_ensemble": {
            "name": "Stacking Ensemble",
            "train_records": models_meta.get("stacking_ensemble", {}).get("train_records", train_default),
            "test_records": models_meta.get("stacking_ensemble", {}).get("test_records", test_default),
            "tune_records": models_meta.get("stacking_ensemble", {}).get("tune_records", train_default),
            "preprocessing": "Base Estimators (LGBM + RF + KNN) → Ridge Meta-Learner",
            "notes": f"Trained using 5-fold cross-validated meta-features across all {models_meta.get('stacking_ensemble', {}).get('train_records', train_default):,} training records.",
        },
    }
    return {
        "total_records": meta.get("total_records", total_len),
        "train_records": meta.get("train_records", train_default),
        "test_records": meta.get("test_records", test_default),
        "models": default_models_info,
    }


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
    records_info: dict = _get_model_training_records_info(artifact, enriched)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            <span class="dot online"></span>
            <strong style="color: #1A1A18; font-size: 14px;">Pipeline Status</strong>
        </div>
        """, unsafe_allow_html=True)
        if enriched is not None:
            st.write(f"Total dataset records: `{len(enriched):,}`")
            st.write(f"Train split records: `{records_info['train_records']:,}` (80%)")
            st.write(f"Test split records: `{records_info['test_records']:,}` (20%)")
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
        n_samples = f"{records_info['train_records']:,}"
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
            <div class="kpi-delta">80% train split</div>
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
    st.markdown('<div class="section-label">Model leaderboard & dataset records</div>', unsafe_allow_html=True)

    models_meta = records_info.get("models", {})
    sorted_models = sorted(eval_results.items(), key=lambda kv: kv[1].get("mape_pct", 99))
    rows_html = ""
    for i, (mname, metrics) in enumerate(sorted_models):
        is_best   = mname == best_name
        rank_cls  = "rank-num rank-1" if i == 0 else "rank-num"
        best_pill = '<span class="best-pill">best</span>' if is_best else ""
        row_cls   = "best-row" if is_best else ""

        m_meta = models_meta.get(mname, {})
        tr_recs = m_meta.get("train_records", records_info["train_records"])
        te_recs = m_meta.get("test_records", records_info["test_records"])
        tune_recs = m_meta.get("tune_records", tr_recs)

        if mname == "svr" and tune_recs < tr_recs:
            train_display = f"{tr_recs:,} <span style='font-size:10px;color:#D97706;'>(Tuned on {tune_recs:,})</span>"
        else:
            train_display = f"{tr_recs:,}"

        rows_html += f"""
        <tr class="{row_cls}">
            <td><span class="{rank_cls}">{i + 1}</span></td>
            <td class="mono">{mname.replace('_', ' ').title()}{best_pill}</td>
            <td class="mono">{train_display}</td>
            <td class="mono">{te_recs:,}</td>
            <td class="mono">{metrics.get('mape_pct')}%</td>
            <td class="mono">{metrics.get('r2')}</td>
            <td class="mono">{metrics.get('within_10pct')}%</td>
            <td class="mono">LKR {metrics.get('rmse_lkr', 0):,.0f}</td>
        </tr>"""

    st.markdown(f"""
    <table class="lb-table">
        <thead>
            <tr>
                <th>#</th><th>Model</th><th>Train Records</th><th>Test Records</th><th>MAPE</th>
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

        listed_desc = st.text_area(
            "Listing Description / Notes (Optional)",
            placeholder="Paste description or mention condition details (e.g. '06 months company warranty included', 'urgent sale', 'සුපිරි තත්වයේ', 'needs fan repair')",
            key="listed_desc_input",
            help="Bilingual condition extractor automatically parses warranty, urgent sales, repairs, and condition factors",
        )

        if st.button("Check Price", key="btn_listed"):
            with st.spinner("Evaluating models…"):
                predictions = predict_all(artifact, selected_model, selected_vram,
                                          selected_brand, enriched)
            from gpu_price_predictor.pipeline import get_model_sample_count, normalize_model
            norm_target = normalize_model(selected_model)
            ctx_mask = enriched[model_col].astype(str).apply(normalize_model) == norm_target
            if selected_brand != "Any" and "brand" in enriched.columns:
                ctx_mask &= enriched["brand"] == selected_brand
            matches = enriched[ctx_mask]
            s_count = get_model_sample_count(selected_model, enriched)

            _render_prediction_results(
                predictions, best_name, eval_results,
                selected_model, selected_vram, selected_brand,
                listed_price=listed_price,
                calibration_data=artifact.get("conformal_calibration"),
                sample_count=s_count,
                description=listed_desc,
            )

            if not matches.empty:
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric("Listings found", len(matches))
                with mc2:
                    st.metric("Median price", f"LKR {matches['price_lkr'].median():,.0f}")
                with mc3:
                    spread = matches["price_lkr"].max() - matches["price_lkr"].min()
                    st.metric("Price spread", f"LKR {spread:,.0f}")

                st.markdown(
                    "<div class='section-label'>Market price distribution for this GPU model</div>",
                    unsafe_allow_html=True,
                )
                model_prices = matches["price_lkr"].dropna()
                if model_prices.nunique() > 1:
                    # Use a small, data-driven number of bands so sparse models remain readable.
                    n_bins = min(12, max(3, int(np.ceil(np.sqrt(len(model_prices))))))
                    bin_edges = np.linspace(model_prices.min(), model_prices.max(), n_bins + 1)
                    price_bands = pd.cut(model_prices, bins=bin_edges, include_lowest=True)
                    distribution = price_bands.value_counts(sort=False)
                    distribution.index = [
                        f"{band.left:,.0f} - {band.right:,.0f}"
                        for band in distribution.index
                    ]
                    st.bar_chart(distribution.rename("Listings"))
                    st.caption("Each bar shows how many matching listings fall within that LKR price band.")
                else:
                    st.info("There is not enough price variation to show a distribution for this model.")

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

        cust_desc = st.text_area(
            "Listing Description / Notes (Optional)",
            placeholder="Paste description or mention condition details (e.g. '06 months company warranty included', 'urgent sale', 'සුපිරි තත්වයේ', 'needs fan repair')",
            key="cust_desc_input",
            help="Bilingual condition extractor automatically parses warranty, urgent sales, repairs, and condition factors",
        )

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
                        from gpu_price_predictor.pipeline import get_model_sample_count
                        cust_s_count = get_model_sample_count(custom_name, enriched)
                        _render_prediction_results(
                            custom_predictions, best_name, eval_results,
                            custom_name, custom_vram, custom_brand,
                            listed_price=custom_listed_price,
                            calibration_data=artifact.get("conformal_calibration"),
                            sample_count=cust_s_count,
                            description=cust_desc,
                        )
                    else:
                        st.error("All models failed. Check that the artifact matches the feature schema.")

            else:
                st.error(
                    f"**'{custom_name}'** not found in reference databases. "
                    "Prediction requires hardware specs — check the GPU name spelling."
                )

    # ── GPU Model Distribution & Market Analysis ─────────────────────────────
    st.divider()
    st.markdown('<div class="section-label">GPU Model Distribution & Market Intelligence</div>', unsafe_allow_html=True)

    if enriched is not None:
        model_col_main = _get_model_col(enriched)
        
        # 1. Market-wide Model Volume Distribution
        m_counts = enriched[model_col_main].value_counts()
        total_recs = len(enriched)

        dist_c1, dist_c2 = st.columns(2)
        with dist_c1:
            st.markdown("<div class='section-label'>Top 15 Most Listed GPU Models</div>", unsafe_allow_html=True)
            top_15_models = m_counts.head(15)
            st.bar_chart(top_15_models)
            st.caption(f"Top 15 models account for **{top_15_models.sum():,}** listings ({top_15_models.sum()/total_recs*100:.1f}% of total dataset).")

        with dist_c2:
            st.markdown("<div class='section-label'>Market Share by GPU Series / Family</div>", unsafe_allow_html=True)
            if "series_family" in enriched.columns:
                series_counts = enriched["series_family"].value_counts().head(10)
                st.bar_chart(series_counts)
                st.caption("Distribution across major series families (RTX, GTX, RX, GT, ARC, etc.).")
            elif "tier_class" in enriched.columns:
                tier_counts = enriched["tier_class"].value_counts()
                st.bar_chart(tier_counts)
                st.caption("Distribution across performance tiers.")

        # 2. Individual GPU Model Distribution Deep-Dive
        st.markdown("##### 🔍 Inspect Individual GPU Model Distribution")
        sel_dist_model = st.selectbox(
            "Select GPU Model to View Market Price & VRAM Distribution",
            options=sorted(enriched[model_col_main].dropna().unique().tolist()),
            index=0,
            key="sel_gpu_model_dist",
            help="Select any GPU model to inspect its exact price distribution, VRAM variants, and brand market share."
        )

        if sel_dist_model:
            model_subset = enriched[enriched[model_col_main] == sel_dist_model]
            m_prices = model_subset["price_lkr"].dropna()
            m_vrams = model_subset["vram_gb"].dropna()
            m_brands = model_subset["brand"].dropna() if "brand" in model_subset.columns else pd.Series(dtype=object)

            # Metric Cards
            dm1, dm2, dm3, dm4, dm5 = st.columns(5)
            with dm1:
                st.metric("Total Listings", f"{len(model_subset):,}", f"{len(model_subset)/total_recs*100:.1f}% of market")
            with dm2:
                st.metric("Median Resale Price", f"LKR {m_prices.median():,.0f}" if not m_prices.empty else "—")
            with dm3:
                min_p = m_prices.min() if not m_prices.empty else 0
                max_p = m_prices.max() if not m_prices.empty else 0
                st.metric("Price Range", f"LKR {min_p:,.0f} – {max_p:,.0f}")
            with dm4:
                std_p = m_prices.std() if len(m_prices) > 1 else 0
                st.metric("Std Deviation", f"LKR {std_p:,.0f}")
            with dm5:
                typ_vram = m_vrams.median() if not m_vrams.empty else 0
                st.metric("Typical VRAM", f"{typ_vram:.0f} GB" if typ_vram > 0 else "—")

            # Charts for selected model
            mc_col1, mc_col2 = st.columns(2)
            with mc_col1:
                st.markdown(f"<div class='section-label'>Price Distribution for {sel_dist_model}</div>", unsafe_allow_html=True)
                if m_prices.nunique() > 1:
                    n_bins = min(10, max(3, int(np.ceil(np.sqrt(len(m_prices))))))
                    bin_edges = np.linspace(m_prices.min(), m_prices.max(), n_bins + 1)
                    price_bands = pd.cut(m_prices, bins=bin_edges, include_lowest=True)
                    dist_series = price_bands.value_counts(sort=False)
                    dist_series.index = [f"Rs. {b.left:,.0f} - {b.right:,.0f}" for b in dist_series.index]
                    st.bar_chart(dist_series.rename("Listings"))
                    st.caption("Distribution of asking prices across market brackets.")
                else:
                    st.info(f"Fixed price point in dataset: LKR {m_prices.iloc[0]:,.0f} ({len(model_subset)} listings).")

            with mc_col2:
                st.markdown(f"<div class='section-label'>Brand & VRAM Breakdown for {sel_dist_model}</div>", unsafe_allow_html=True)
                if not m_brands.empty and m_brands.nunique() > 1:
                    st.bar_chart(m_brands.value_counts().head(6).rename("Listings by Brand"))
                elif not m_vrams.empty and m_vrams.nunique() > 1:
                    st.bar_chart(m_vrams.value_counts().rename("Listings by VRAM"))
                elif not m_brands.empty:
                    st.info(f"Primary Brand in dataset: **{m_brands.iloc[0]}** ({len(model_subset)} listings)")

            with st.expander(f"View all {len(model_subset)} market listings for {sel_dist_model}"):
                sub_cols = [c for c in [model_col_main, "brand", "vram_gb", "price_lkr", "G3Dmark", "gpu_age_years", "architecture"] if c in model_subset.columns]
                st.dataframe(
                    model_subset[sub_cols].sort_values("price_lkr").reset_index(drop=True),
                    width="stretch",
                    column_config={
                        "price_lkr": st.column_config.NumberColumn("Price (LKR)", format="LKR %,d"),
                        "vram_gb": st.column_config.NumberColumn("VRAM (GB)", format="%.1f GB"),
                    }
                )

        # 3. Complete Model Distribution Master Table
        with st.expander("📊 Complete Model-by-Model Distribution Summary Table"):
            summary_records = []
            for m_name, grp in enriched.groupby(model_col_main):
                p_col = grp["price_lkr"].dropna()
                v_col = grp["vram_gb"].dropna()
                b_list = grp["brand"].dropna().value_counts().head(3).index.tolist() if "brand" in grp.columns else []
                summary_records.append({
                    "GPU Model": m_name,
                    "Listings": len(grp),
                    "Market Share %": round(len(grp) / total_recs * 100, 2),
                    "Median Price (LKR)": int(p_col.median()) if not p_col.empty else 0,
                    "Min Price (LKR)": int(p_col.min()) if not p_col.empty else 0,
                    "Max Price (LKR)": int(p_col.max()) if not p_col.empty else 0,
                    "Std Dev (LKR)": int(p_col.std()) if len(p_col) > 1 else 0,
                    "Typical VRAM": f"{v_col.median():.0f} GB" if not v_col.empty else "—",
                    "Top Brands": ", ".join(b_list) if b_list else "Any",
                })
            sum_df = pd.DataFrame(summary_records).sort_values("Listings", ascending=False).reset_index(drop=True)
            st.dataframe(
                sum_df,
                width="stretch",
                column_config={
                    "Median Price (LKR)": st.column_config.NumberColumn("Median Price (LKR)", format="LKR %,d"),
                    "Min Price (LKR)": st.column_config.NumberColumn("Min Price (LKR)", format="LKR %,d"),
                    "Max Price (LKR)": st.column_config.NumberColumn("Max Price (LKR)", format="LKR %,d"),
                    "Std Dev (LKR)": st.column_config.NumberColumn("Std Dev (LKR)", format="LKR %,d"),
                    "Market Share %": st.column_config.NumberColumn("Market Share", format="%.2f%%"),
                }
            )

    # ── Used Dataset Records by Model Inspector ───────────────────────────────
    st.divider()
    st.markdown('<div class="section-label">Used dataset records per model</div>', unsafe_allow_html=True)

    if enriched is not None:
        model_options = ["All Models (Full Training Set)"] + [m.replace("_", " ").title() for m in eval_results.keys()]
        selected_model_option = st.selectbox(
            "Select Model to Inspect Used Dataset Records",
            options=model_options,
            key="sel_inspect_model",
            help="Choose a trained model algorithm to view its training records, holdout test records, and feature pipeline configuration.",
        )

        selected_key = None
        for k in eval_results.keys():
            if k.replace("_", " ").title() in selected_model_option:
                selected_key = k
                break

        m_info = models_meta.get(selected_key, {}) if selected_key else {}

        tr_cnt = m_info.get("train_records", records_info["train_records"]) if selected_key else records_info["train_records"]
        te_cnt = m_info.get("test_records", records_info["test_records"]) if selected_key else records_info["test_records"]
        prep_method = m_info.get("preprocessing", "18-Feature Enriched Pipeline") if selected_key else "6-Model Ensemble Pipeline"
        model_notes = m_info.get("notes", "Full Sri Lankan GPU marketplace dataset enriched with PassMark benchmarks and techpowerup specs.") if selected_key else f"All 6 models were trained on {records_info['train_records']:,} records (80% split) and evaluated on {records_info['test_records']:,} holdout test records (20% split)."

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Training dataset records", f"{tr_cnt:,}")
        with r2:
            st.metric("Holdout test records", f"{te_cnt:,}")
        with r3:
            st.metric("Total dataset rows", f"{len(enriched):,}")
        with r4:
            st.metric("Feature preprocessing", prep_method)

        st.info(f"💡 **Model Training Configuration ({selected_model_option}):** {model_notes}")

        st.markdown("##### Search & Filter Training Dataset Records")
        fcol_a, fcol_b, fcol_c = st.columns([2, 1, 1])

        with fcol_a:
            search_query = st.text_input(
                "Search GPU Model Name",
                placeholder="e.g. RTX 3060, GTX 1060, RX 580",
                key="inspect_search",
            )

        with fcol_b:
            avail_brands = ["All Brands"] + sorted(enriched["brand"].dropna().unique().tolist()) if "brand" in enriched.columns else ["All Brands"]
            selected_inspect_brand = st.selectbox("Brand / Manufacturer", avail_brands, key="inspect_brand")

        with fcol_c:
            avail_archs = ["All Architectures"] + sorted(enriched["architecture"].dropna().unique().tolist()) if "architecture" in enriched.columns else ["All Architectures"]
            selected_inspect_arch = st.selectbox("Architecture", avail_archs, key="inspect_arch")

        inspect_df = enriched.copy()
        if search_query.strip():
            model_col_name = _get_model_col(inspect_df)
            inspect_df = inspect_df[inspect_df[model_col_name].astype(str).str.contains(search_query.strip(), case=False, na=False)]

        if selected_inspect_brand != "All Brands" and "brand" in inspect_df.columns:
            inspect_df = inspect_df[inspect_df["brand"] == selected_inspect_brand]

        if selected_inspect_arch != "All Architectures" and "architecture" in inspect_df.columns:
            inspect_df = inspect_df[inspect_df["architecture"] == selected_inspect_arch]

        f_cnt = len(inspect_df)
        st.caption(f"Displaying **{f_cnt:,}** records out of **{len(enriched):,}** dataset records used for training and evaluating models.")

        if not inspect_df.empty:
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("<div class='section-label'>Training price vs vram distribution</div>", unsafe_allow_html=True)
                color_col = "brand" if "brand" in inspect_df.columns else None
                st.scatter_chart(inspect_df, x="vram_gb", y="price_lkr", color=color_col)

            with chart_col2:
                st.markdown("<div class='section-label'>Brand distribution in dataset records</div>", unsafe_allow_html=True)
                if "brand" in inspect_df.columns:
                    b_counts = inspect_df["brand"].value_counts().head(8)
                    st.bar_chart(b_counts)

            st.markdown("<div class='section-label'>Used dataset records table</div>", unsafe_allow_html=True)
            display_cols = [c for c in [
                _get_model_col(inspect_df), "brand", "vram_gb", "price_lkr", "G3Dmark",
                "tdp_watts", "gpu_age_years", "architecture", "series_family", "gpu_generation", "ti_variant"
            ] if c in inspect_df.columns]

            st.dataframe(
                inspect_df[display_cols].reset_index(drop=True),
                width="stretch",
                column_config={
                    "price_lkr": st.column_config.NumberColumn("Price (LKR)", format="LKR %,d"),
                    "vram_gb": st.column_config.NumberColumn("VRAM (GB)", format="%.1f GB"),
                    "G3Dmark": st.column_config.NumberColumn("G3Dmark Score", format="%,d"),
                }
            )
        else:
            st.warning("No dataset records found matching the filter criteria.")

    # ── Dataset explorer ──────────────────────────────────────────────────────
    st.divider()
    with st.expander("Explore full enriched dataset raw view"):
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
