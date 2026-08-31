# Fuzzy Matching Threshold Evaluation & Viva Demonstration

This experiment directory provides concrete proof and interactive demonstration for the fuzzy matching thresholds used in the GPU Price Prediction pipeline:
* `FUZZY_THRESHOLD_BENCH = 85` (PassMark Benchmarks)
* `FUZZY_THRESHOLD_SPEC = 82` (TechPowerUp Specs)

---

## 1. How to Run the Demonstration Script

In your terminal, navigate to the project root and run:
```bash
python experiments/threshold_analysis/evaluate_fuzzy_thresholds.py
```

---

## 2. What the Script Produces

1. **Terminal Output Table**: Shows match coverage across thresholds from 60% to 100%.
2. **`threshold_sweep_results.csv`**: Full raw evaluation data across every tested threshold.
3. **`fuzzy_threshold_tradeoff.png`**: High-resolution chart showing match coverage vs threshold curve.
4. **Concrete Case Studies**: Real GPU listings showing exact score calculations and how the keyword guard protects against wrong models.

---

## 3. Key Findings for the Viva Panel

| Threshold | PassMark Benchmarks Match % | TechPowerUp Specs Match % | Notes & Panel Explanation |
| :---: | :---: | :---: | :--- |
| **100** | 70.21% | 86.17% | ❌ **30% of Benchmarks Lost**: Exact match fails due to minor naming/spacing differences. |
| **95** | 72.34% | 86.17% | ❌ Still misses ~28% of benchmark records. |
| **90** | 91.49% | 92.55% | Good, but misses valid cards with modifier noise. |
| **85** | **92.55%** | 92.55% | ✅ **Optimal for PassMark**: High match rate, zero model collisions. |
| **82** | **92.55%** | **92.55%** | ✅ **Optimal for Specs**: Captures longer names with extra tokens without false matches. |
| **< 75** | 93.6%+ | 92.5%+ | ❌ **Dangerous**: Risk of matching wrong card numbers (e.g. GTX 1060 -> GTX 1050). |

---

## 4. Two-Stage Matching Architecture Defense

* **Stage 1 (Probabilistic)**: `token_sort_ratio` retrieves the closest string match using the calibrated threshold (85 / 82).
* **Stage 2 (Deterministic)**: `check_keyword_guard` strictly verifies critical modifiers (`Ti`, `Super`, `XT`). Even if similarity is high, variant mismatch is immediately rejected.
