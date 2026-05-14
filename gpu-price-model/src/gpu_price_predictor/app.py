"""GPU Price Predictor — v2.0 Streamlit App (minimalist redesign)"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

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


# ── CSS — minimalist, no gradients ────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=DM+Sans:wght@300;400;500;600&display=swap');

*, html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    box-sizing: border-box;
}

.stApp {
    background-color: #F7F6F3;
    color: #1A1A18;
}

/* ── Global Text Visibility Overrides ── */
.stApp p, .stApp span, .stApp label, .stApp strong, .stApp li {
    color: #1A1A18 !important;
}

/* Ensure button text remains orange and doesn't get hit by global dark text rule */
.stButton > button p, 
.stButton > button span, 
.stButton > button div {
    color: inherit !important; 
}
.stButton > button {
    color: #FF4B00 !important;
}
.stButton > button:hover { 
    color: #FFFFFF !important;
}

[data-testid="stWidgetLabel"] p {
    color: #1A1A18 !important;
    font-weight: 500 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
    color: #1A1A18 !important;
}

/* ── Typography ── */
.page-title {
    font-family: 'DM Mono', monospace;
    font-size: 1.75rem;
    font-weight: 500;
    color: #1A1A18;
    letter-spacing: -0.03em;
    margin-bottom: 0.1rem;
}
.page-sub {
    font-size: 0.82rem;
    color: #555550;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}
h2, h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: #1A1A18 !important;
    letter-spacing: -0.02em !important;
}

/* ── Section label ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #555550;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #E4E3DF;
}

/* ── KPI strip ── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1px;
    background: #E4E3DF;
    border: 1px solid #E4E3DF;
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 2.5rem;
}
.kpi-cell {
    background: #F7F6F3;
    padding: 1rem 1.25rem;
}
.kpi-label {
    font-size: 0.72rem;
    color: #555550;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.kpi-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.3rem;
    font-weight: 500;
    color: #1A1A18;
    line-height: 1.2;
}
.kpi-delta {
    font-size: 0.72rem;
    color: #555550;
    margin-top: 0.2rem;
}

/* ── Leaderboard ── */
.lb-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.lb-table th {
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #555550;
    font-weight: 500;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #E4E3DF;
    text-align: left;
}
.lb-table td {
    padding: 0.7rem 0.75rem;
    border-bottom: 1px solid #F0EFeb;
    color: #3A3A36;
}
.lb-table tr:last-child td { border-bottom: none; }
.lb-table tr.best-row td { background: #FFFFF8; }
.lb-table tr:hover td { background: #F0EFeb; }
.mono { font-family: 'DM Mono', monospace; font-size: 0.85rem; }
.best-pill {
    display: inline-block;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border: 1px solid #1A1A18;
    padding: 0.1rem 0.45rem;
    border-radius: 2px;
    margin-left: 0.5rem;
    vertical-align: middle;
}
.rank-num {
    font-family: 'DM Mono', monospace;
    color: #8A8A80;
    font-size: 0.8rem;
}
.rank-1 { color: #1A1A18; font-weight: 600; }

/* ── Prediction result ── */
.pred-block {
    border: 1px solid #E4E3DF;
    border-left: 4px solid #FF4B00;
    border-radius: 4px;
    padding: 1.75rem 2rem;
    margin: 1.5rem 0;
    background: #FAFAF8;
}
.pred-model-label {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #555550;
    margin-bottom: 0.4rem;
    font-family: 'DM Mono', monospace;
}
.pred-price {
    font-family: 'DM Mono', monospace;
    font-size: 2.6rem;
    font-weight: 500;
    color: #1A1A18;
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.pred-context {
    font-size: 0.82rem;
    color: #555550;
    margin-top: 0.5rem;
}

/* ── Info box ── */
.info-box {
    background: #FFFFF8;
    border: 1px solid #E8E6D0;
    border-radius: 4px;
    padding: 1rem 1.25rem;
    font-size: 0.85rem;
    color: #4A4A40;
    margin-bottom: 1.5rem;
    line-height: 1.6;
}
.info-box a { color: #4A4A40; }

/* ── Buttons ── */
.stButton > button {
    background: #1A1A18 !important;
    border: none !important;
    padding: 0.65rem 1.5rem !important;
    border-radius: 4px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.15s ease !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover { 
    background: #FF4B00 !important; 
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #FFFFFF !important;
    border: 1px solid #E4E3DF !important;
    border-radius: 4px !important;
    color: #1A1A18 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Placeholder & Selectbox Text ── */
[data-baseweb="select"] * {
    color: #1A1A18 !important;
}
[data-baseweb="select"] [data-placeholder] {
    color: #1A1A18 !important;
    opacity: 1 !important;
}

/* ── Metrics override ── */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E4E3DF;
    border-radius: 4px;
    padding: 0.9rem 1rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #555550 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 1.2rem !important;
    color: #1A1A18 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #E4E3DF;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #555550 !important;
    padding: 0.5rem 1rem !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #1A1A18 !important;
    border-bottom: 2px solid #1A1A18 !important;
    background: transparent !important;
}

/* ── Divider ── */
hr { border-color: #E4E3DF !important; margin: 2rem 0 !important; }

/* ── Dataframe ── */
.stDataFrame { border: 1px solid #E4E3DF !important; border-radius: 4px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #F0EFeb !important;
    border-right: 1px solid #E4E3DF !important;
    color: #1A1A18 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F7F6F3; }
::-webkit-scrollbar-thumb { background: #C8C7C0; border-radius: 3px; }

/* ── Success / Error / Warning ── */
.stSuccess { background: #F2FAF2 !important; border-color: #B8D8B8 !important; color: #1A1A18 !important; }
.stError   { background: #FFF2F2 !important; border-color: #D8B8B8 !important; color: #1A1A18 !important; }
.stWarning { background: #FFFBF0 !important; border-color: #D8D0A0 !important; color: #1A1A18 !important; }

/* ── Caption & Generic ── */
[data-testid="stCaptionContainer"], .stCaptionContainer {
    color: #555550 !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #555550 !important;
}
</style>
"""


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifact():
    if MODEL_V2_PATH.exists():
        return joblib.load(MODEL_V2_PATH), "v2"
    if MODEL_V1_PATH.exists():
        return joblib.load(MODEL_V1_PATH), "v1"
    return None, None


@st.cache_data
def load_enriched() -> pd.DataFrame | None:
    if ENRICHED_CSV.exists():
        return pd.read_csv(ENRICHED_CSV)
    return None


@st.cache_data
def load_bench_df() -> pd.DataFrame | None:
    if BENCH_CSV.exists():
        df = pd.read_csv(BENCH_CSV)
        df.columns = df.columns.str.strip()   # guard against whitespace in headers
        return df
    return None


@st.cache_data
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
) -> dict[str, float]:
    """Run all trained models and return {model_name: predicted_lkr}."""
    feature_cols: list[str] = artifact["feature_columns"]
    all_models: dict = artifact.get("all_models", {})

    inf: dict = {col: np.nan for col in feature_cols}
    inf.update({
        "vram_gb":        vram,
        "brand":          brand if brand != "Any" else "Unknown",
        "series_family":  _series_family(model_name),
        "model_number":   _model_number(model_name),
        "ti_variant":     _ti_variant(model_name),
        "gpu_generation": _gpu_generation(model_name),
        "architecture":   "Unknown",
        "log_G3Dmark":    0.0,
    })

    # Enrich from the training dataset when the model has prior listings
    if enriched is not None:
        model_col = "extracted_model" if "extracted_model" in enriched.columns else "model"
        matches = enriched[enriched[model_col].str.upper() == model_name.upper()]
        if not matches.empty:
            num_cols = [
                "G3Dmark", "G2Dmark", "log_G3Dmark", "fp32_gflops", "tdp_watts",
                "memory_bandwidth_gb_s", "shader_units", "gpu_base_clock_mhz",
                "boost_clock_mhz", "perf_per_watt", "gpu_age_years",
            ]
            for col in num_cols:
                if col in matches.columns:
                    v = matches[col].dropna().median()
                    if pd.notna(v):
                        inf[col] = v
            if "architecture" in matches.columns:
                mode = matches["architecture"].mode()
                if not mode.empty:
                    inf["architecture"] = mode.iloc[0]

    df_inf = pd.DataFrame([inf])[feature_cols]

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
) -> None:
    """Render prediction card + per-model breakdown. Safe against empty dicts."""
    if not predictions:
        st.error("All models failed to produce a prediction. Check that the artifact is valid.")
        return

    sorted_preds = sorted(predictions.items(), key=lambda kv: kv[1])
    best_price = predictions.get(best_name, sorted_preds[0][1])

    st.markdown(f"""
    <div class="pred-block">
        <div class="pred-model-label">Best model &nbsp;/&nbsp; {best_name.replace('_', ' ').upper()}</div>
        <div class="pred-price">LKR {best_price:,.0f}</div>
        <div class="pred-context">{label} &nbsp;·&nbsp; {vram:.0f} GB VRAM &nbsp;·&nbsp; {brand}</div>
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="GPU Price Predictor",
        page_icon="▣",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    artifact, artifact_ver = load_artifact()
    enriched = load_enriched()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown('<div class="page-title">GPU Price Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">v2.0 &nbsp;·&nbsp; 6-model ensemble &nbsp;·&nbsp; '
        'benchmark-enriched &nbsp;·&nbsp; Sri Lanka market</div>',
        unsafe_allow_html=True,
    )

    if artifact is None:
        st.error("No model artifact found. Run `python scripts/train_model_v2.py` to train.")
        return

    eval_results: dict = artifact.get("evaluation_results", {})
    best_name: str     = artifact.get("best_model_name", "")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("**Pipeline info**")
        if enriched is not None:
            st.write(f"Training rows: `{len(enriched):,}`")
        if best_name and best_name in eval_results:
            m = eval_results[best_name]
            st.write(f"Best model: `{best_name}`")
            st.write(f"MAPE: `{m.get('mape_pct')}%`")
            st.write(f"R²: `{m.get('r2')}`")
            st.write(f"Within 10%: `{m.get('within_10pct')}%`")
            st.write(f"RMSE: `LKR {m.get('rmse_lkr'):,.0f}`")
        st.divider()
        st.caption("v2.0 · 2026")

    # ── KPI strip ─────────────────────────────────────────────────────────────
    bm = eval_results.get(best_name, {})

    if enriched is not None:
        n_samples = f"{len(enriched):,}"
        model_col = "extracted_model" if "extracted_model" in enriched.columns else "model"
        n_models  = str(enriched[model_col].dropna().nunique())
        avg_price = f"LKR {enriched['price_lkr'].mean():,.0f}" if "price_lkr" in enriched.columns else "—"
    else:
        n_samples = n_models = avg_price = "—"

    best_mape    = f"{bm.get('mape_pct', '—')}%"
    within10     = f"{bm.get('within_10pct', '—')}%"

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-cell">
            <div class="kpi-label">Training samples</div>
            <div class="kpi-value">{n_samples}</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">GPU models</div>
            <div class="kpi-value">{n_models}</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">Avg market price</div>
            <div class="kpi-value">{avg_price}</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">Best MAPE</div>
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
            <td class="mono">{mname.replace('_', ' ')}{best_pill}</td>
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
    st.markdown('<div class="section-label">Price prediction</div>', unsafe_allow_html=True)

    if enriched is None:
        st.warning("No enriched dataset found. Run `build_benchmark_features.py` first.")
        return

    model_col     = "extracted_model" if "extracted_model" in enriched.columns else "model"
    unique_models = sorted(enriched[model_col].dropna().unique().tolist())
    unique_brands = sorted(enriched["brand"].dropna().unique().tolist()) if "brand" in enriched.columns else []

    tab_listed, tab_custom = st.tabs(["Listed GPU", "Unlisted / custom GPU"])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 1 — Listed GPU
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_listed:
        pc1, pc2, pc3 = st.columns([2, 1, 1])
        with pc1:
            selected_model = st.selectbox("GPU model", unique_models, key="sel_model")
        mask = enriched[model_col] == selected_model
        typical_vram = float(enriched.loc[mask, "vram_gb"].dropna().median() or 4.0)
        with pc2:
            selected_vram = st.number_input("VRAM (GB)", min_value=1.0, value=typical_vram,
                                            step=1.0, key="listed_vram")
        with pc3:
            selected_brand = st.selectbox("Brand", ["Any"] + unique_brands, key="sel_brand")

        if st.button("Calculate predicted price", key="btn_listed"):
            with st.spinner("Running 6 models…"):
                predictions = predict_all(artifact, selected_model, selected_vram,
                                          selected_brand, enriched)
            _render_prediction_results(predictions, best_name, eval_results,
                                       selected_model, selected_vram, selected_brand)

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

                # FIX: guard color param — only pass when column exists
                color_col = "brand" if "brand" in matches.columns else None
                st.scatter_chart(matches, x="vram_gb", y="price_lkr", color=color_col)

                with st.expander("View comparable listings"):
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

        # st.markdown("""
        # <div class="info-box">
        #     Type any GPU name — hardware specs are auto-filled from the reference databases.
        #     The model predicts price from hardware numbers alone, so it works even for
        #     GPUs with no Sri Lankan market history.
        #     Look up PassMark scores at
        #     <a href="https://www.videocardbenchmark.net" target="_blank">
        #     videocardbenchmark.net</a>.
        # </div>
        # """, unsafe_allow_html=True)

        all_ref_names = []
        if bench_df is not None:
            all_ref_names.extend(bench_df["gpuName"].dropna().unique().tolist())
        if specs_df is not None:
            all_ref_names.extend(specs_df["Name"].dropna().unique().tolist())
        all_ref_names = sorted(list(set(all_ref_names)))

        r1c1, r1c2, r1c3 = st.columns([2, 1, 1])
        with r1c1:
            if all_ref_names:
                custom_name = st.selectbox(
                    "Search GPU model", 
                    options=all_ref_names,
                    index=None,
                    placeholder="Type to search (e.g. RTX 3070)",
                    key="cust_name_select",
                    help="Search across 10,000+ reference GPUs",
                )
                if custom_name is None: custom_name = ""
            else:
                custom_name = st.text_input(
                    "GPU name", placeholder="e.g. RTX 3070",
                    key="cust_name_input",
                )
        with r1c2:
            custom_vram = st.number_input("VRAM (GB)", min_value=1.0, value=8.0, step=1.0,
                                          key="cust_vram")
        with r1c3:
            custom_brand = st.selectbox("Brand / seller", ["Unknown"] + unique_brands,
                                        key="cust_brand")

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
                    summary["G3Dmark"]       = f"{specs['G3Dmark']:,.0f}"
                if specs.get("G2Dmark"):
                    summary["G2Dmark"]       = f"{specs['G2Dmark']:,.0f}"
                if specs.get("tdp_watts"):
                    summary["TDP (W)"]       = f"{specs['tdp_watts']:.0f}"
                if specs.get("fp32_gflops"):
                    summary["FP32 GFLOPS"]   = f"{specs['fp32_gflops']:,.0f}"
                if specs.get("memory_bandwidth_gb_s"):
                    summary["Bandwidth GB/s"] = f"{specs['memory_bandwidth_gb_s']:.1f}"
                if specs.get("shader_units"):
                    summary["Shader units"]  = f"{specs['shader_units']:.0f}"
                if specs.get("architecture"):
                    summary["Architecture"]  = specs["architecture"]
                summary["Release year"]      = str(specs.get("release_year", "?"))
                summary["GPU age (yrs)"]     = str(2026 - int(specs.get("release_year", 2020)))

                s_cols = st.columns(min(len(summary), 6))
                for i, (k, v) in enumerate(summary.items()):
                    with s_cols[i % 6]:
                        st.metric(k, v)

                st.divider()
                if st.button("Predict price from found specs", key="btn_custom"):
                    g3d  = specs.get("G3Dmark")
                    g2d  = specs.get("G2Dmark")
                    tdp  = specs.get("tdp_watts")
                    fp32 = specs.get("fp32_gflops")
                    age  = 2026 - int(specs.get("release_year", 2020))
                    arch = specs.get("architecture") or "Unknown"

                    feature_cols_list: list[str] = artifact["feature_columns"]
                    custom_inf: dict = {col: np.nan for col in feature_cols_list}
                    custom_inf.update({
                        "vram_gb":                float(custom_vram),
                        "brand":                  custom_brand,
                        "G3Dmark":                float(g3d)  if g3d  else np.nan,
                        "G2Dmark":                float(g2d)  if g2d  else np.nan,
                        "log_G3Dmark":            float(np.log1p(g3d)) if g3d else np.nan,
                        "fp32_gflops":            float(fp32) if fp32 else np.nan,
                        "tdp_watts":              float(tdp)  if tdp  else np.nan,
                        "memory_bandwidth_gb_s":  float(specs.get("memory_bandwidth_gb_s") or np.nan),
                        "shader_units":           float(specs.get("shader_units") or np.nan),
                        "gpu_base_clock_mhz":     float(specs.get("gpu_base_clock_mhz") or np.nan),
                        "boost_clock_mhz":        float(specs.get("boost_clock_mhz") or np.nan),
                        "perf_per_watt":          (float(g3d) / float(tdp)) if (g3d and tdp) else np.nan,
                        "gpu_age_years":          float(age),
                        "architecture":           arch,
                        "series_family":          _series_family(custom_name),
                        "model_number":           _model_number(custom_name),
                        "gpu_generation":         _gpu_generation(custom_name),
                        "ti_variant":             _ti_variant(custom_name),
                    })

                    df_custom_inf = pd.DataFrame([custom_inf])[feature_cols_list]
                    custom_predictions: dict[str, float] = {}
                    with st.spinner("Running 6 models…"):
                        for mname, pipeline in artifact.get("all_models", {}).items():
                            try:
                                pred = float(pipeline.predict(df_custom_inf)[0])
                                custom_predictions[mname] = max(0.0, float(np.expm1(pred)))
                            except Exception:
                                pass

                    if custom_predictions:
                        st.success(f"Prediction complete for **{custom_name}** (extrapolated from specs)")
                        _render_prediction_results(
                            custom_predictions, best_name, eval_results,
                            custom_name, custom_vram, custom_brand,
                        )
                        with st.expander("Internal features used"):
                            st.write(custom_inf)
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
        "font-family:DM Mono,monospace;'>"
        "GPU Price Predictor &nbsp;·&nbsp; v2.0 &nbsp;·&nbsp; 2026"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()