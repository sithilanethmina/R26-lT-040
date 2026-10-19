# FairPriceLK: GPU Valuation Experimental Proof Suite
## Empirical Experiments, Statistical Validations & Academic Proofs
> **Author & Candidate:** Dissanayake K.M.S.N.B (IT22231246)  
> **Degree:** B.Sc. (Hons) in Information Technology / Software Engineering, SLIIT  
> **Project:** FairPriceLK (R26-IT-040) — AI-Driven Second-Hand Marketplace Valuation Platform  
> **Scope:** GPU Valuation Engine, Conformal Price Bounds, API Gateway & Browser Extension  

---

## 1. Executive Summary & Experimental Methodology

This directory contains the **empirical experiments and reproducible proof scripts** developed for the FairPriceLK GPU price prediction engine. Each experiment addresses a specific academic challenge, peer review critique, or statistical validation requirement under the **Design Science Research (DSR)** framework (Peffers et al., 2007).

All experiments use real secondary marketplace data collected from Sri Lankan platforms (Ikman.lk, Facebook Marketplace, retail shops) combined with PassMark physical hardware benchmark specifications.

---

## 2. Master Summary Table of Experimental Proofs

| # | Experiment Subfolder | Research Question / Thesis Defense Point | Key Metric / Statistical Proof | Artifact Files |
| :---: | :--- | :--- | :--- | :--- |
| **1** | [`threshold_analysis/`](./threshold_analysis/) | How to separate genuine title typos from distinct hardware models? | Threshold **82** achieves **F1 = 97.2%** (Precision 98.4%, Recall 96.1%). Thresholds < 75 cause brand collisions (e.g. GTS 450 → GT 430). | `evaluate_fuzzy_thresholds.py`<br>`threshold_sweep_results.csv`<br>`fuzzy_threshold_tradeoff.png` |
| **2** | [`sample_size_threshold/`](./sample_size_threshold/) | What is the minimum market sample size needed before price estimation is valid? | $N < 3$ mathematically lacks degrees of freedom ($df < 2$), causing sample variance errors. High threshold ($N \ge 10$) discards **31.9%** of active Sri Lankan GPU models. | `evaluate_sample_size_thresholds.py`<br>`sample_size_threshold_results.csv`<br>`sample_size_tradeoff.png` |
| **3** | [`conformal_confidence_tradeoff/`](./conformal_confidence_tradeoff/) | Why choose a 90% confidence target over 95% or 99% for market ranges? | 90% confidence delivers **90.02% empirical coverage** with a compact ~28% width. 95% confidence balloons width to **>45%**, making ranges useless for bargaining. | `evaluate_conformal_tradeoff.py`<br>`conformal_coverage_results.csv`<br>`confidence_tradeoff_plot.png` |
| **4** | [`generation_boundary_analysis/`](./generation_boundary_analysis/) | Why restrict automated pricing on newly launched hardware (RTX 50-Series)? | Launch-window listings ($N < 25$) reflect speculative importer markups rather than mature depreciation equilibrium ($0$ yrs market maturity). | `README.md` |
| **5** | [`warranty_hedonic_analysis/`](./warranty_hedonic_analysis/) | Does remaining seller warranty add measurable value, and does marketing fluff matter? | Two-stage OLS proves warranty adds **+0.84% per month** ($\beta = +0.0084, t = 4.62, p < 0.001$, capped at 24 mo / +15%). Subjective terms ("mint") are statistically non-significant ($t=0.88$). | `evaluate_warranty_impact.py`<br>`warranty_regression_results.csv`<br>`warranty_hedonic_curve.png` |
| **6** | [`benchmark_enrichment_proof/`](./benchmark_enrichment_proof/) | Does physical compute grounding solve the cold-start problem on unseen hardware? | On unseen GPUs, benchmark enrichment drops error from **41.85% MAPE** (Text TF-IDF, $R^2 = 0.54$) to **19.64% MAPE** (Benchmark + SVR, $R^2 = 0.9166$). | `evaluate_benchmark_proof.py`<br>`benchmark_vs_text_results.csv`<br>`benchmark_cold_start_proof.png` |
| **7** | [`data_hygiene_anti_leakage/`](./data_hygiene_anti_leakage/) | How do we prove our reported accuracy is honest and free of test-set data leakage? | Pre-split IQR outlier filtering leaks test distributions, falsely reporting 7.42% MAPE. Post-split train-only IQR maintains **0 SHA-256 collisions** and yields an honest **9.84% MAPE**. | `evaluate_leakage_proof.py`<br>`data_hygiene_results.csv`<br>`data_leakage_comparison.png` |

---

## 3. Deep-Dive on Each Proof & Viva Defense Answers

### 1. RapidFuzz Matching Threshold Sweep (`threshold_analysis/`)
* **Core Question:** Sri Lankan classified listings lack structured forms and have unstructured, typo-ridden titles. How do we scientifically decide the RapidFuzz token matching threshold?
* **Experimental Method:** Evaluated 300 test titles across thresholds 60 to 100 with ground-truth GPU labels.
* **Key Finding:** Thresholds below 75 produce catastrophic collisions (e.g., matching low-end GT 430 to GTS 450). Thresholds above 90 reject 28% of valid listings due to minor typos. **Threshold 82** optimizes the harmonic mean, delivering **F1 = 97.2%**.
* **Viva One-Liner:** *"We swept thresholds from 60 to 100; threshold 82 maximizes F1-score at 97.2% while completely preventing dangerous low-end to high-end hardware collisions."*

### 2. Sample Density Safety Cutoff Sweep (`sample_size_threshold/`)
* **Core Question:** When should the system refuse to predict a fair market price for rare hardware?
* **Experimental Method:** Swept sample cutoffs $N \in [1, 15]$ across all 113 unique GPU models in Sri Lanka.
* **Key Finding:** For $N < 3$, degrees of freedom are $df < 2$, making variance estimation mathematically undefined. However, setting $N \ge 10$ discards 31.9% of all GPU models. Thus, $N \ge 3$ is established as the minimum statistical threshold, paired with a Central Limit Theorem penalty multiplier $k(n) = 1.0 + \frac{1.5}{\sqrt{n+1}}$ for $3 \le n < 10$.
* **Viva One-Liner:** *"At N < 3, degrees of freedom are less than 2, so the system blocks automated evaluation. We do not use N = 10 because it would drop 31.9% of Sri Lanka's GPU market."*

### 3. Split Conformal Prediction Coverage vs. Width (`conformal_confidence_tradeoff/`)
* **Core Question:** Why does FairPriceLK use a 90% confidence level instead of the standard 95% or 99%?
* **Experimental Method:** Evaluated empirical coverage and relative interval width across 6 confidence levels ($70\% \dots 95\%$) on 1,200 holdout test listings.
* **Key Finding:** At 90% nominal confidence, empirical coverage is **90.02%** and mean relative interval width is **~28.0%–35.5%**. Moving to 95% balloons the width to **45.03%**, making the price range too broad to be actionable for consumer negotiation.
* **Viva One-Liner:** *"In secondary electronics, 95% confidence requires an interval width over 45% of the total price. At 90%, we guarantee 90.02% coverage while keeping the width tight at ~28%."*

### 4. Generation Boundary & Architectural Guard (`generation_boundary_analysis/`)
* **Core Question:** What happens if someone lists an NVIDIA RTX 50-Series (Blackwell) card?
* **Experimental Method:** Analyzed secondary market liquidity and trading density for launch-window architectures in Sri Lanka.
* **Key Finding:** Newly launched cards have $0$ years of secondary market stabilization. Listings reflect speculative dealer markups and scalper pricing. The API gateway enforces an explicit boundary check and returns `status: "generation_restricted"` rather than outputting a deceptive ML estimate.
* **Viva One-Liner:** *"An ML model can compute a number from VRAM and clock speed, but without an established secondary market, that number has no empirical market validity. We block it to protect buyers."*

### 5. Two-Stage Hedonic Warranty Analysis (`warranty_hedonic_analysis/`)
* **Core Question:** How do we prove that warranty adds real value and justify our +0.84% per month adjustment?
* **Experimental Method:** Trained a baseline price model without warranty data, extracted log price residuals, and ran an Ordinary Least Squares (OLS) hedonic regression on verified remaining warranty months and marketing keywords.
* **Key Finding:** Remaining warranty is statistically significant ($t = 4.62, p < 0.001$) with a coefficient of $\beta = +0.0084$ (+0.84% per month). Subjective marketing claims ("mint condition", "urgent sale") have no statistical significance ($t = 0.88, p = 0.38$).
* **Viva One-Liner:** *"Our two-stage hedonic regression proved that remaining warranty adds a statistically significant +0.84% per month (p < 0.001), whereas marketing buzzwords like 'mint' have zero statistical effect."*

### 6. Benchmark Grounding vs. Text NLP Cold-Start (`benchmark_enrichment_proof/`)
* **Core Question:** How do we prove that continuous benchmark features (PassMark 3D, TDP, VRAM) are better than standard NLP text embeddings?
* **Experimental Method:** Group-KFold split holding out entire GPU models to simulate newly listed hardware never seen during training. Compared a text-only TF-IDF baseline against our benchmark-enriched model.
* **Key Finding:** On unseen GPU architectures, text-only models suffer an unacceptable **41.85% MAPE** ($R^2 = 0.5401$). Grounding the cards in physical compute benchmarks drops error to **19.64% MAPE** ($R^2 = 0.9166$).
* **Viva One-Liner:** *"Text embeddings cannot price a GPU model they have never seen before (41.85% error). Our benchmark enrichment grounds the card in physical computing power, cutting cold-start error in half to 19.64%."*

### 7. Post-Split Data Hygiene & Zero-Leakage Proof (`data_hygiene_anti_leakage/`)
* **Core Question:** How can the panel trust that our 9.84% MAPE is not caused by data leakage?
* **Experimental Method:** Compared pre-split outlier filtering (global IQR) against our post-split train-only IQR filtering, logging SHA-256 cryptographic row hashes.
* **Key Finding:** Pre-split outlier filtering removes hard listings globally, producing an artificially deflated 7.42% MAPE. Post-split train-only IQR keeps the 1,200 holdout test listings completely untouched, yielding **0 SHA-256 row collisions** and an honest, real-world holdout MAPE of **9.84%**.
* **Viva One-Liner:** *"We split 80/20 first and applied IQR bounds strictly to the training fold. Our SHA-256 verification proves zero test set leakage, guaranteeing our 9.84% MAPE reflects genuine real-world accuracy."*

---

## 4. How to Run All Experiment Scripts

To reproduce all 7 experiments and regenerate all figures and CSV tables:

```powershell
# Activate project virtual environment
.\venv\Scripts\activate

# 1. RapidFuzz Threshold Analysis
python gpu-price-model/experiments/threshold_analysis/evaluate_fuzzy_thresholds.py

# 2. Sample Size Cutoff Analysis
python gpu-price-model/experiments/sample_size_threshold/evaluate_sample_size_thresholds.py

# 3. Conformal Coverage Tradeoff Analysis
python gpu-price-model/experiments/conformal_confidence_tradeoff/evaluate_conformal_tradeoff.py

# 4. Warranty Hedonic Regression Analysis
python gpu-price-model/experiments/warranty_hedonic_analysis/evaluate_warranty_impact.py

# 5. Benchmark Cold-Start Grounding Analysis
python gpu-price-model/experiments/benchmark_enrichment_proof/evaluate_benchmark_proof.py

# 6. Post-Split Anti-Leakage Data Hygiene Analysis
python gpu-price-model/experiments/data_hygiene_anti_leakage/evaluate_leakage_proof.py
```
