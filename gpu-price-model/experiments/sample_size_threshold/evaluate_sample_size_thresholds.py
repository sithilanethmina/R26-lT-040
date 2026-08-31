"""
Empirical Sample Size Threshold Analysis & Valuation Safety Guard
Evaluates listing density, empirical pricing variance, and degrees of freedom
across GPU models in the FairPriceLK dataset.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "final", "gpu_enriched_dataset.csv")
OUTPUT_DIR = os.path.dirname(__file__)

def run_sample_size_analysis():
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    print("=" * 70)
    print("FairPriceLK - GPU Sample Size Threshold & Reliability Analysis")
    print("=" * 70)
    print(f"Total Listings Analyzed: {len(df):,}")
    
    # Analyze by normalized model
    model_col = "norm_model" if "norm_model" in df.columns else "extracted_model"
    counts = df[model_col].value_counts()
    total_unique = len(counts)
    print(f"Unique GPU Models: {total_unique}")
    
    # Distribution of sample sizes
    print("\n--- Listing Count Distribution Across GPU Models ---")
    print(f"Min Samples: {counts.min()}")
    print(f"25th Percentile (Q1): {counts.quantile(0.25):.1f}")
    print(f"Median (50th Percentile): {counts.median():.1f}")
    print(f"Mean Samples: {counts.mean():.1f}")
    print(f"75th Percentile (Q3): {counts.quantile(0.75):.1f}")
    print(f"Max Samples: {counts.max()} ({counts.index[0]})")
    
    # Threshold categories
    n_less_3 = (counts < 3).sum()
    n_3_to_9 = ((counts >= 3) & (counts < 10)).sum()
    n_10_to_29 = ((counts >= 10) & (counts < 30)).sum()
    n_30_plus = (counts >= 30).sum()
    
    print("\n--- Tiered Data Availability Breakdown ---")
    print(f"1. Extreme Scarcity (N < 3 listings):    {n_less_3:2d} models ({n_less_3/total_unique*100:5.1f}%) -> Action: BLOCK PREDICTION")
    print(f"2. Low Data (3 <= N < 10 listings):      {n_3_to_9:2d} models ({n_3_to_9/total_unique*100:5.1f}%) -> Action: WARN (High Uncertainty)")
    print(f"3. Moderate Data (10 <= N < 30 listings):{n_10_to_29:2d} models ({n_10_to_29/total_unique*100:5.1f}%) -> Action: WARN (Mild Uncertainty)")
    print(f"4. Abundant Data (N >= 30 listings):     {n_30_plus:2d} models ({n_30_plus/total_unique*100:5.1f}%) -> Action: FULL CONFIDENCE")

    # Statistical justification table
    records = []
    for n in range(1, 35):
        # Degrees of freedom
        df_val = max(0, n - 1)
        # Sample size multiplier k_n from Conformal Prediction formula
        if n < 10:
            k_n = 1.0 + (1.5 / math.sqrt(n + 1))
            status = "BLOCK (< 3)" if n < 3 else "HIGH UNCERTAINTY"
        elif n < 20:
            k_n = 1.0 + (0.5 / math.sqrt(n))
            status = "MODERATE UNCERTAINTY"
        else:
            k_n = 1.0
            status = "ROBUST SAMPLE"
            
        models_at_least_n = (counts >= n).sum()
        pct_coverage = (models_at_least_n / total_unique) * 100
        
        records.append({
            "sample_size_n": n,
            "degrees_of_freedom": df_val,
            "conformal_penalty_multiplier_kn": round(k_n, 4),
            "eligible_models_count": models_at_least_n,
            "eligible_models_pct": round(pct_coverage, 2),
            "recommended_action": status
        })
        
    results_df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, "sample_size_threshold_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved threshold analysis table to: {csv_path}")

    # Plot sample size vs penalty & model coverage
    try:
        fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=300)

        color1 = "#d9534f"
        ax1.set_xlabel("Sample Size (N listings per GPU model)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Conformal Uncertainty Penalty Multiplier (k_n)", color=color1, fontsize=11, fontweight="bold")
        ax1.plot(results_df["sample_size_n"], results_df["conformal_penalty_multiplier_kn"], color=color1, marker="o", linewidth=2.2, label="Uncertainty Multiplier (k_n)")
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.axvline(x=3, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label="Threshold N=3 (Min for Prediction)")
        ax1.axvline(x=10, color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label="Threshold N=10 (Standard Confidence)")

        ax2 = ax1.twinx()
        color2 = "#0275d8"
        ax2.set_ylabel("GPU Models Retained (%)", color=color2, fontsize=11, fontweight="bold")
        ax2.plot(results_df["sample_size_n"], results_df["eligible_models_pct"], color=color2, marker="s", linewidth=2.2, linestyle="-.", label="Marketplace Model Coverage (%)")
        ax2.tick_params(axis="y", labelcolor=color2)

        # Highlight regions
        ax1.axvspan(1, 2.9, color="#f8d7da", alpha=0.4, label="Blocked Zone (N < 3)")
        ax1.axvspan(3, 9.9, color="#fff3cd", alpha=0.4, label="Warning Zone (3 <= N < 10)")
        ax1.axvspan(10, 35, color="#d4edda", alpha=0.4, label="Confident Zone (N >= 10)")

        plt.title("Sample Size vs. Valuation Uncertainty & Market Coverage Tradeoff", fontsize=13, fontweight="bold", pad=15)
        fig.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "sample_size_tradeoff.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved tradeoff chart to: {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == "__main__":
    run_sample_size_analysis()
