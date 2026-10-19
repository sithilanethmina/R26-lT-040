# Experiment: Empirical Proof of Warranty & Condition Pricing

## 1. Research Question
Does verified warranty duration have a statistically significant positive effect on pre-owned GPU pricing in Sri Lanka, and how much premium does the secondary market actually pay per month of warranty?

## 2. Methodology: Two-Stage Hedonic Residual Regression
Following Rosen's (1974) Two-Stage Hedonic pricing framework:
1. **Stage 1 (Baseline Prediction):** The base LightGBM/Random Forest model predicts log price using purely continuous physical hardware specifications (PassMark 3DMark, VRAM, TDP, Age).
2. **Stage 2 (Residual Regression):** We compute the price residual:
   $$\text{Residual} = \ln(\text{Actual Price}) - \ln(\text{Predicted Hardware Price})$$
3. We fit a Ridge regression ($\alpha=1.0$) on these residuals across verified marketplace listings against extracted condition tags (warranty months, shop status, repairs, urgency).

## 3. Empirical Results

| Condition Feature | Beta Coefficient ($\beta$) | Standard Error | t-statistic | Effect on Price |
| :--- | :---: | :---: | :---: | :---: |
| **Warranty (Months)** | **+0.0084** | 0.0018 | **+4.62** ($p < 0.001$) | **+0.84% per month** |
| **Verified Shop (`is_shop`)** | **+0.0758** | 0.0165 | **+4.59** ($p < 0.001$) | **+7.9% shop premium** |
| **Needs Repair / Faulty** | **-0.3792** | 0.0312 | **-12.15** ($p < 0.001$) | **-31.6% fault penalty** |
| **Urgent Sale** | **-0.0277** | 0.0124 | **-2.23** ($p = 0.026$) | **-2.7% discount** |
| *Subjective ("Mint Condition")* | *+0.0167* | *0.0189* | *+0.88 (Not Significant)* | *Discarded / Blocked* |

## 4. Key Scientific Proofs
1. **Statistically Significant:** The warranty beta ($t = 4.62, p < 0.001$) confirms that warranty is not random noise; the market genuinely pays extra for verified warranty protection.
2. **Strict Anti-Marketing Enforcement:** Subjective claims ("mint", "like new") had non-significant t-statistics ($t=0.88$) and are mathematically blocked from increasing consumer prices in production.
3. **Bounded Valuation:** Warranty is capped at 24 months and +15% total upside to prevent speculative over-valuation.
