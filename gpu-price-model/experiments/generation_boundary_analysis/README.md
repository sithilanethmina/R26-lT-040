# Generation Boundary & New Architecture Restriction Analysis

This experiment documents the academic, statistical, and market justification for restricting price valuation on the **NVIDIA RTX 50-Series (Blackwell Architecture)** and newly released hardware generations in FairPriceLK.

---

## 1. Executive Summary & Research Motivation

Predicting secondary market prices on newly released hardware architectures during their initial launch window introduces severe statistical and economic distortions:
1. **Absence of Mature Depreciation Curves**: Machine learning regression algorithms rely on empirical depreciation rates over hardware operational lifetimes ($1 \dots 7$ years). Launch-window listings have $0$ years of secondary market stabilization.
2. **Initial Distribution & Scalper Pollution**: Listings for newly released series during early launch months represent importer markups, pre-orders, and scalper premiums rather than fair secondary market equilibrium.
3. **Defensive Model Boundary Enforcement**: In safety-critical valuation systems, explicitly restricting immature product domains preserves model credibility and protects buyers against speculative prices.

---

## 2. Affected Hardware Generation (NVIDIA RTX 50-Series / Blackwell)

| Model Name | Release Date | Architecture | Sri Lankan Sample Count ($N$) | Restriction Status |
| :--- | :---: | :---: | :---: | :---: |
| **GeForce RTX 5090** | 2025 / 2026 | Blackwell | $0$ | 🛑 **Restricted** |
| **GeForce RTX 5080** | 2025 / 2026 | Blackwell | $2$ | 🛑 **Restricted** |
| **GeForce RTX 5070 Ti** | 2025 / 2026 | Blackwell | $3$ | 🛑 **Restricted** |
| **GeForce RTX 5070** | 2025 / 2026 | Blackwell | $3$ | 🛑 **Restricted** |
| **GeForce RTX 5060 Ti** | 2025 / 2026 | Blackwell | $4$ | 🛑 **Restricted** |
| **GeForce RTX 5060** | 2025 / 2026 | Blackwell | $8$ | 🛑 **Restricted** |
| **GeForce RTX 5050** | 2025 / 2026 | Blackwell | $3$ | 🛑 **Restricted** |

---

## 3. Comparison of Market Maturity by GPU Generation

| Generation | Era | Sample Density | Market Maturity | Valuation Support |
| :--- | :---: | :---: | :---: | :---: |
| **GTX 900 / 10-Series (Pascal)** | 2014 – 2016 | $> 1,800$ listings | Fully Mature | ✅ **Full Support** |
| **GTX 16 / RTX 20-Series (Turing)** | 2018 – 2019 | $> 1,500$ listings | Fully Mature | ✅ **Full Support** |
| **RTX 30-Series / RX 6000 (Ampere / RDNA2)** | 2020 – 2022 | $> 1,900$ listings | Established Secondary Market | ✅ **Full Support** |
| **RTX 40-Series / RX 7000 (Ada / RDNA3)** | 2022 – 2024 | $> 500$ listings | Active Secondary Market | ✅ **Full Support** |
| **RTX 50-Series (Blackwell)** | 2025 – 2026 | $< 25$ listings | **Immature Launch Window** | 🛑 **Restricted** |

---

## 4. User-Facing Implementation & Architectural Boundary

When a user attempts to evaluate an RTX 50-series card, the backend API interceptor prevents model execution and returns a dedicated boundary notification:

### API Response (`/api/gpu/predict`):
```json
{
  "status": "generation_restricted",
  "can_predict": false,
  "predicted_price": null,
  "fair_market_range": null,
  "evaluation": {
    "verdict": "Newly Released Generation",
    "badge_text": "New Architecture",
    "badge_class": "warning",
    "message": "The RTX 5070 Ti belongs to a newly released hardware generation. Secondary market pricing has not yet stabilized in Sri Lanka, so automatic price valuation is restricted to ensure accuracy."
  }
}
```

---

## 5. Viva Panel Defense Points

* **Q: Why exclude RTX 50-series if the model can theoretically predict based on hardware specs?**  
  * **Answer:** An ML model can calculate a mathematical number from VRAM and CUDA cores, but without an established secondary market, that number has no empirical market validity in Sri Lanka. It risks confusing launch-window dealer markups with true used market value.
* **Q: When should the restriction be lifted?**  
  * **Answer:** Once the generation reaches at least 6–12 months of secondary market trading volume and passes the minimum sample threshold ($N \ge 30$) across multiple retail and consumer channels.
