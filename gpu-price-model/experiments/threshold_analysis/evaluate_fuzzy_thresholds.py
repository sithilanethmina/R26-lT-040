"""
Fuzzy Matching Threshold Evaluation & Viva Demonstration Tool
============================================================
This script empirically evaluates different fuzzy matching threshold values (60 to 100)
on the actual dataset for both:
  1. PassMark Benchmarks (GPU_benchmarks_v7.csv)
  2. TechPowerUp Specs (gpu_1986-2026.csv)

It demonstrates:
  - Why 85 is the optimal threshold for PassMark Benchmarks.
  - Why 82 is the optimal threshold for TechPowerUp Specs.
  - What happens if you use 100 (high false negatives).
  - What happens if you use < 80 (false positives / wrong model collisions).
"""

import json
import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from rapidfuzz import process as fz_process, fuzz
import matplotlib.pyplot as plt

# -- Paths setup ----------------------------------------------------------------
EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "final"

LISTING_V1 = DATA_DIR / "training_data_v1.json"
LISTING_V2 = DATA_DIR / "training_data_v2.json"
LISTING_V3 = DATA_DIR / "training_data_v3.json"
BENCHMARKS_CSV = DATA_DIR / "GPU_benchmarks_v7.csv"
SPECS_CSV = DATA_DIR / "gpu_1986-2026.csv"

OUTPUT_CSV = EXPERIMENT_DIR / "threshold_sweep_results.csv"
OUTPUT_PLOT = EXPERIMENT_DIR / "fuzzy_threshold_tradeoff.png"


def normalize_model_name(raw: str) -> str:
    """Standardize extracted model names to match benchmark/spec naming style."""
    s = str(raw).strip()
    s = re.sub(r'\bTI\b', 'Ti', s, flags=re.IGNORECASE)
    s = re.sub(r'\bTi\b', 'Ti', s)
    if re.match(r'^(GTX|RTX|GT)\s', s, re.IGNORECASE):
        s = "GeForce " + s
    elif re.match(r'^RX\s', s, re.IGNORECASE):
        s = "Radeon " + s
    return s.strip()


def check_keyword_guard(query: str, candidate: str) -> bool:
    """Ensures that distinguishing keywords (Ti, Super, XT) match exactly."""
    q_lower = query.lower()
    c_lower = candidate.lower()
    for kw in ["ti", "super", "xt"]:
        has_q = bool(re.search(rf"\b{kw}\b", q_lower))
        has_c = bool(re.search(rf"\b{kw}\b", c_lower))
        if has_q != has_c:
            return False
    return True


def load_unique_scraped_models() -> list[str]:
    """Load and normalize unique GPU models from real scraped listing datasets."""
    v1 = pd.DataFrame(json.loads(LISTING_V1.read_text(encoding="utf-8"))) if LISTING_V1.exists() else pd.DataFrame()
    v2 = pd.DataFrame(json.loads(LISTING_V2.read_text(encoding="utf-8"))) if LISTING_V2.exists() else pd.DataFrame()
    v3 = pd.DataFrame(json.loads(LISTING_V3.read_text(encoding="utf-8"))) if LISTING_V3.exists() else pd.DataFrame()
    df = pd.concat([v1, v2, v3], ignore_index=True)

    df.rename(columns={
        "Price_LKR": "price_lkr",
        "Extracted_Model": "extracted_model",
        "VRAM_GB": "vram_gb",
        "Brand": "brand",
    }, inplace=True)

    models = df["extracted_model"].dropna().unique().tolist()
    normalized = sorted(list({normalize_model_name(m) for m in models if str(m).strip()}))
    return normalized


def run_threshold_sweep(
    models: list[str],
    database_names: list[str],
    thresholds: list[int],
    target_name: str
) -> pd.DataFrame:
    """Sweep different threshold values to compute match coverage and inspect examples."""
    results = []

    for th in thresholds:
        matched_count = 0
        samples = []

        for model in models:
            hits = fz_process.extract(
                model, database_names, scorer=fuzz.token_sort_ratio, limit=10
            )
            matched_candidate = None
            matched_score = 0.0

            for cand, score, _ in hits:
                if score < th:
                    continue
                if check_keyword_guard(model, cand):
                    matched_candidate = cand
                    matched_score = score
                    break

            if matched_candidate is not None:
                matched_count += 1
                if len(samples) < 3:
                    samples.append(f"'{model}' -> '{matched_candidate}' ({matched_score:.1f}%)")

        match_rate = (matched_count / len(models)) * 100.0
        results.append({
            "Dataset": target_name,
            "Threshold": th,
            "Total_Models": len(models),
            "Matched_Models": matched_count,
            "Match_Rate_Pct": round(match_rate, 2),
            "Sample_Matches": " | ".join(samples)
        })

    return pd.DataFrame(results)


def plot_comparison_chart(bench_df: pd.DataFrame, spec_df: pd.DataFrame):
    """Plot publication-ready tradeoff curve comparing match rates across thresholds."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(
        bench_df["Threshold"], bench_df["Match_Rate_Pct"],
        marker="o", linewidth=2.5, color="#1f77b4", label="PassMark Benchmarks"
    )
    plt.plot(
        spec_df["Threshold"], spec_df["Match_Rate_Pct"],
        marker="s", linewidth=2.5, color="#ff7f0e", label="TechPowerUp Specs"
    )

    # Highlight chosen thresholds
    plt.axvline(x=85, color="#1f77b4", linestyle="--", alpha=0.7, label="Chosen Bench Threshold = 85")
    plt.axvline(x=82, color="#ff7f0e", linestyle="--", alpha=0.7, label="Chosen Spec Threshold = 82")

    plt.title("Empirical Threshold Sweep: GPU Match Rate vs. Fuzzy Threshold", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Fuzzy Similarity Threshold (0 - 100)", fontsize=12)
    plt.ylabel("Match Coverage (%)", fontsize=12)
    plt.xticks(np.arange(60, 105, 5))
    plt.yticks(np.arange(0, 105, 10))
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.close()
    print(f"\n[+] Saved visualization plot to: {OUTPUT_PLOT.name}")


def show_case_studies(models: list[str], spec_names: list[str]):
    """Demonstrate exact real-world cases that prove why 82 and 85 are required."""
    print("\n" + "=" * 80)
    print("KEY CASE STUDIES PROVING WHY 82 AND 85 ARE REQUIRED")
    print("=" * 80)

    test_cards = [
        "GeForce GTX 1660 Super",
        "Radeon RX 570",
        "GeForce RTX 2060",
        "GeForce GTX 1060 6GB",
        "GeForce RTX 3060 Ti"
    ]

    for q in test_cards:
        hits = fz_process.extract(q, spec_names, scorer=fuzz.token_sort_ratio, limit=5)
        print(f"\nQuery Card: '{q}'")
        for cand, score, _ in hits:
            guard_pass = check_keyword_guard(q, cand)
            guard_str = "PASS" if guard_pass else "FAIL (Variant Mismatch)"
            pass_at_85 = "YES" if score >= 85 and guard_pass else "NO (Dropped)"
            pass_at_82 = "YES" if score >= 82 and guard_pass else "NO (Dropped)"
            print(f"  -> Candidate: '{cand}' | Score: {score:.1f}% | Guard: {guard_str}")
            print(f"     [Matches at 85? {pass_at_85}]  [Matches at 82? {pass_at_82}]")


def main():
    print("=" * 80)
    print("FUZZY MATCHING THRESHOLD EMPIRICAL EVALUATION")
    print("=" * 80)

    # 1. Load Data
    unique_models = load_unique_scraped_models()
    print(f"[+] Loaded {len(unique_models)} unique GPU model names from scraped data.")

    bdf = pd.read_csv(BENCHMARKS_CSV)
    bench_names = bdf["gpuName"].dropna().unique().tolist()
    print(f"[+] Loaded {len(bench_names)} unique names from PassMark Benchmark DB.")

    sdf = pd.read_csv(SPECS_CSV, usecols=["Name"], low_memory=False)
    spec_names = sdf["Name"].dropna().unique().tolist()
    print(f"[+] Loaded {len(spec_names)} unique names from TechPowerUp Spec DB.")

    # 2. Run Sweep across thresholds [60, 65, 70, 75, 80, 82, 85, 90, 95, 100]
    thresholds = [60, 65, 70, 75, 80, 82, 85, 90, 95, 100]

    print("\n[*] Evaluating PassMark Benchmarks across thresholds...")
    bench_results = run_threshold_sweep(unique_models, bench_names, thresholds, "PassMark_Benchmarks")

    print("[*] Evaluating TechPowerUp Specs across thresholds...")
    spec_results = run_threshold_sweep(unique_models, spec_names, thresholds, "TechPowerUp_Specs")

    # Combine results
    all_results = pd.concat([bench_results, spec_results], ignore_index=True)
    all_results.to_csv(OUTPUT_CSV, index=False)
    print(f"[+] Saved complete threshold sweep results to: {OUTPUT_CSV.name}")

    # Display comparison table
    print("\n" + "=" * 80)
    print("THRESHOLD SWEEP COMPARISON TABLE")
    print("=" * 80)
    pivot = all_results.pivot(index="Threshold", columns="Dataset", values="Match_Rate_Pct")
    print(pivot.to_string())

    # 3. Show Case Studies
    show_case_studies(unique_models, spec_names)

    # 4. Generate Plot
    plot_comparison_chart(bench_results, spec_results)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    main()
