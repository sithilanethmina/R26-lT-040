"""SHAP analysis — Phase 5. Uses the LightGBM sub-model (TreeExplainer-compatible)."""
from __future__ import annotations
import pickle
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "data" / "final"

ENRICHED_CSV = DATA_DIR / "gpu_enriched_dataset.csv"
MODEL_PATH = ARTIFACTS_DIR / "gpu_price_model_v2.joblib"
SHAP_PNG = ARTIFACTS_DIR / "shap_summary_plot.png"
SHAP_PKL = ARTIFACTS_DIR / "shap_values.pkl"

SAMPLE_SIZE = 500


def _get_pre_and_model(pipeline):
    """Extract (preprocessor_step, raw_model) from a sklearn Pipeline."""
    steps = list(pipeline.named_steps.values())
    if len(steps) == 1:
        return None, steps[0]
    return steps[0], steps[-1]


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run train_model_v2.py first.")
    if not ENRICHED_CSV.exists():
        raise FileNotFoundError("Run build_benchmark_features.py first.")

    artifact = joblib.load(MODEL_PATH)
    feature_cols: list[str] = artifact["feature_columns"]
    all_models: dict = artifact.get("all_models", {})

    # Prefer LightGBM — TreeExplainer works natively
    shap_model_name = "lightgbm" if "lightgbm" in all_models else list(all_models.keys())[0]
    pipeline = all_models[shap_model_name]
    print(f"SHAP target model: {shap_model_name}")

    df = pd.read_csv(ENRICHED_CSV)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    x = df[feature_cols].copy()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x), min(SAMPLE_SIZE, len(x)), replace=False)
    x_sample = x.iloc[idx].reset_index(drop=True)

    pre, model = _get_pre_and_model(pipeline)
    if pre is not None:
        x_transformed = pre.transform(x_sample)
        try:
            feat_names = pre.get_feature_names_out().tolist()
        except Exception:
            feat_names = feature_cols
    else:
        x_transformed = x_sample.values
        feat_names = feature_cols

    # TreeExplainer works for LightGBM, XGBoost, RandomForest
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_transformed)
    except Exception as e:
        print(f"TreeExplainer failed ({e}), falling back to KernelExplainer (slow) …")
        background = shap.sample(x_transformed, 50)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(x_transformed, nsamples=100)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Summary plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, x_transformed, feature_names=feat_names, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {SHAP_PNG}")

    # Top 10
    mean_abs = np.abs(shap_values).mean(axis=0)
    top = sorted(zip(feat_names, mean_abs), key=lambda kv: kv[1], reverse=True)[:10]
    print("\nTop-10 features by mean |SHAP|:")
    for name, val in top:
        print(f"  {name:<35} {val:.4f}")

    with open(SHAP_PKL, "wb") as f:
        pickle.dump({"shap_values": shap_values, "feature_names": feat_names,
                     "model_used": shap_model_name}, f)
    print(f"Saved: {SHAP_PKL}")


if __name__ == "__main__":
    main()
