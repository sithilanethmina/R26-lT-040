# FairPriceLK — Electronics ML Pipeline & Architecture Documentation

This document provides a comprehensive technical overview of the **Electronics Price Prediction Pipeline** within the FairPriceLK system, covering data scraping, preprocessing, feature engineering, machine learning model benchmarking and selection, the Gemini multimodal extraction layer, and real-time inference.

---

## 1. Executive Summary & High-Level Architecture

The Electronics subsystem predicts the **Fair Market Value (in LKR)** for second-hand and brand-new electronics across three major subcategories in Sri Lanka:
1. **Laptops** (Notebooks, Gaming Laptops, MacBooks, Ultrabooks)
2. **Tablets** (Apple iPads, Samsung Galaxy Tabs, Xiaomi/Redmi Pads, Microsoft Surface)
3. **Monitors** (Office, IPS Frameless, Curved, High-Refresh-Rate Gaming Displays)

### End-to-End System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            BROWSER EXTENSION                                │
│                                                                             │
│  1. Injected Content Script detects listing (Ikman.lk, Facebook, etc.)      │
│  2. Captures visual tab screenshot (chrome.tabs.captureVisibleTab)          │
│  3. Scrapes DOM title, description text, and seller asking price            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ (HTTP POST with JSON/Base64 Image)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            API GATEWAY (:8000)                              │
│                                                                             │
│  • Reverse proxy routing `/api/electronics/predict` -> Port 8004            │
│  • CORS handling, request throttling & error normalization                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   ELECTRONICS MICROSERVICE (:8004 / app.py)                 │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 1. AI Extraction Layer: Google Gemini 3.5 Flash Lite (Multimodal)     │  │
│  │    • Inspects screenshot pixels + unstructured text                   │  │
│  │    • Enforces strict Zod / OpenAPI JSON Schema                        │  │
│  │    • Outputs normalized hardware specs & physical condition           │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│  ┌───────────────────────────────────▼───────────────────────────────────┐  │
│  │ 2. Production Machine Learning Inference Engine                       │  │
│  │    • Laptops  ──► `best_laptop_model.pkl`  (CatBoost Regressor)       │  │
│  │    • Tablets  ──► `best_tablet_model.pkl`  (LightGBM Regressor)       │  │
│  │    • Monitors ──► `best_monitor_model.pkl` (XGBoost Regressor)        │  │
│  │    • Target Inversion: exp(y_log) - 1                                 │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│  ┌───────────────────────────────────▼───────────────────────────────────┐  │
│  │ 3. Fair Market Range & Valuation Scoring                              │  │
│  │    • Generates point estimate + [Lower, Upper] fair market range      │  │
│  │    • Computes deal fairness badge (Great Deal / Fair / Overpriced)    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Dataset Collection & Preprocessing (100% Used Items Scope)

All training datasets are **strictly filtered for second-hand / used electronics data** to guarantee accurate fair market valuation for second-hand marketplace listings (**Ikman.lk**):

| Category | Raw Scraped Ads | Cleaned Used Samples | Second-Hand Scope | Preprocessing Script |
| :--- | :---: | :---: | :---: | :--- |
| **Laptops** | 8,601 | **4,058** | **100% Used** | [`laptops_data_cleaning.py`](file:///d:/final%20project/R26-lT-040/electronics-price-model/src/preprocessing/laptops_data_cleaning.py) |
| **Tablets** | 4,797 | **600** | **100% Used** | [`tablets_data_cleaning.py`](file:///d:/final%20project/R26-lT-040/electronics-price-model/src/preprocessing/tablets_data_cleaning.py) |
| **Monitors** | 2,377 | **1,212** | **100% Used** | [`monitors_data_cleaning.py`](file:///d:/final%20project/R26-lT-040/electronics-price-model/src/preprocessing/monitors_data_cleaning.py) |

### Key Preprocessing Steps:
1. **Strict Used Filtering**: Pruned vendor brand-new retail postings to isolate true market depreciation curves.
2. **Deduplication**: Hash-based and title-levenshtein deduplication removed duplicate vendor reposts.
3. **Price Outlier Removal**: Listings with unrealistic prices ($< \text{Rs } 3,000$ or $> \text{Rs } 1,500,000$) were pruned using Interquartile Range ($\text{IQR} \times 2.5$) filtering.
4. **Target Log Transformation**: Second-hand marketplace prices exhibit severe right-skewness. All models were trained on $\log(1 + y)$ to stabilize variance:
   $$\hat{y}_{\text{raw}} = \exp(\hat{y}_{\log}) - 1$$

---

## 3. Features Extracted & Used by Category

Because each hardware domain is governed by fundamentally different pricing drivers, the feature sets are specialized:

### Feature Matrix Comparison

| Feature Name | Type | 💻 Laptops | 📱 Tablets | 🖥️ Monitors | Extraction Source |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `Brand_Cleaned` | Categorical | ✅ | ✅ | ✅ | Gemini / DOM (`HP`, `Apple`, `Dell`, `Samsung`, `MSI`, `ViewSonic`) |
| `Model_Cleaned` | Categorical | ✅ | ✅ | ❌ | Gemini / DOM (`Victus 15`, `Latitude 5420`, `iPad 9th Gen`, `Tab S8`) |
| `CPU_Cleaned` | Categorical | ✅ | ❌ | ❌ | Gemini / Regex (`Core i5`, `Ryzen 7`, `Apple M1`, `Core i7`) |
| `Generation_Cleaned` | Integer | ✅ | ❌ | ❌ | Gemini / Regex (`8`, `11`, `12`, `13`) |
| `RAM_GB` | Continuous | ✅ | ✅ | ❌ | Gemini / Regex (`4`, `8`, `16`, `32`, `64`) |
| `Storage_Capacity_GB` | Continuous | ✅ | ✅ | ❌ | Gemini / Regex (`64`, `128`, `256`, `512`, `1024`, `2048`) |
| `Storage_Type` | Categorical | ✅ | ❌ | ❌ | Gemini / Regex (`SSD`, `NVMe`, `HDD`) |
| `GPU_Tier` | Categorical | ✅ | ❌ | ❌ | Gemini / Regex (`RTX 30-Series`, `RTX 40-Series`, `GTX`, `Integrated`) |
| `Is_Touchscreen` | Binary | ✅ | ❌ | ❌ | Gemini / Regex ($1 = \text{Touchscreen / 2-in-1}, 0 = \text{Standard}$) |
| `Screen_Size / Size_Inches`| Continuous | ❌ | ✅ | ✅ | Gemini / Regex (`10.2"`, `11.0"`, `24.0"`, `27.0"`, `32.0"`) |
| `Connectivity_Cleaned` | Categorical | ❌ | ✅ | ❌ | Gemini / Regex (`WiFi / Standard`, `4G LTE`, `5G Cellular`) |
| `Refresh_Rate_Hz` | Continuous | ❌ | ❌ | ✅ | Gemini / Regex (`60`, `75`, `100`, `144`, `165`, `240`) |
| `Resolution_Cleaned` | Categorical | ❌ | ❌ | ✅ | Gemini / Regex (`1080p FHD`, `2K QHD`, `4K UHD`, `HD`) |
| `Panel_Type` | Categorical | ❌ | ❌ | ✅ | Gemini / Regex (`IPS`, `VA`, `OLED`, `Standard`) |
| `Is_Curved` | Binary | ❌ | ❌ | ✅ | Gemini / Regex ($1 = \text{Curved}, 0 = \text{Flat}$) |
| `Is_Gaming` | Binary | ❌ | ❌ | ✅ | Gemini / Regex ($1 = \text{Gaming}, 0 = \text{Standard}$) |
| `Is_Frameless` | Binary | ❌ | ❌ | ✅ | Gemini / Regex ($1 = \text{Frameless}, 0 = \text{Standard}$) |
| `Location_Cleaned` | Categorical | ✅ | ✅ | ✅ | Gemini / DOM (`Colombo`, `Gampaha`, `Kandy`, `Galle`, etc.) |

---

## 4. Machine Learning Model Training & Benchmark Results

For each category, four competitive tree-based regression algorithms were benchmarked using an **80/20 Train-Test split** with 5-fold cross-validation on strictly used items.

### A. Laptop Model Benchmark (Used Items Only)
*Training samples: 3,246 | Test samples: 812*

| Model Algorithm | $R^2$ Score | MAE (LKR) | MAPE (%) | Selected as Production |
| :--- | :---: | :---: | :---: | :---: |
| 🏆 **XGBoost Regressor** | **0.4546** | **Rs 32,491** | **34.84%** | **YES** |
| CatBoost Regressor | 0.4535 | Rs 32,820 | 35.18% | No |
| LightGBM Regressor | 0.4503 | Rs 32,968 | 35.63% | No |
| Random Forest | 0.4410 | Rs 32,241 | 34.03% | No |

**Why XGBoost Won for Laptops**:
Laptop pricing is driven by strong non-linear interactions across processor generation, RAM, and GPU tiers. XGBoost's exact greedy split algorithm with column subsampling produced the highest $R^2$ score ($0.4546$) and lowest MAE (Rs 32,491).

---

### B. Tablet Model Benchmark (Used Items Only)
*Training samples: 480 | Test samples: 120*

| Model Algorithm | $R^2$ Score | MAE (LKR) | MAPE (%) | Selected as Production |
| :--- | :---: | :---: | :---: | :---: |
| 🏆 **CatBoost Regressor** | **0.4774** | **Rs 28,928** | **51.14%** | **YES** |
| Random Forest | 0.4549 | Rs 29,402 | 61.03% | No |
| LightGBM Regressor | 0.3909 | Rs 30,450 | 61.78% | No |
| XGBoost Regressor | 0.1518 | Rs 32,972 | 64.10% | No |

**Why CatBoost Won for Tablets**:
Second-hand tablets have high brand concentration (Apple iPads & Samsung Galaxy Tabs) with smaller sample sizes ($N=600$). CatBoost's **Ordered Target Statistics** handles small sample sizes with high categorical cardinality without overfitting.

---

### C. Monitor Model Benchmark (Used Items Only)
*Training samples: 969 | Test samples: 243*

| Model Algorithm | $R^2$ Score | MAE (LKR) | MAPE (%) | Selected as Production |
| :--- | :---: | :---: | :---: | :---: |
| 🏆 **LightGBM Regressor** | **0.5815** | **Rs 3,086** | **24.40%** | **YES** |
| XGBoost Regressor | 0.5618 | Rs 3,199 | 27.02% | No |
| Random Forest | 0.5544 | Rs 3,174 | 26.65% | No |
| CatBoost Regressor | 0.5486 | Rs 3,482 | 25.94% | No |

**Why LightGBM Won for Monitors**:
Monitor pricing behaves as discrete spec tiers ($24'' \text{ vs } 27''$, $60\text{Hz} \text{ vs } 144\text{Hz}$). LightGBM's **leaf-wise tree splitting** captured these step functions with the lowest MAE of **Rs 3,086 LKR** and an $R^2$ of **0.5815**.

---

## 5. Gemini 3.5 Flash Lite Multimodal AI Extractor

Traditional DOM scrapers break when marketplace websites change their HTML/CSS classes. FairPriceLK incorporates **Google Gemini 3.5 Flash Lite** as a zero-shot, vision-capable extraction layer.

### Schema Enforcement (Zod / OpenAPI Specification)
The extractor enforces strict type validation using `responseSchema`:

```typescript
import { z } from "zod";

export const ElectronicsSpecSchema = z.object({
  category: z.enum(["laptop", "tablet", "monitor"]),
  brand: z.string().nullable().optional(),
  model: z.string().nullable().optional(),
  cpu: z.string().nullable().optional(),
  generation: z.number().int().nullable().optional(),
  ram_gb: z.number().positive().nullable().optional(),
  storage_gb: z.number().positive().nullable().optional(),
  storage_type: z.enum(["SSD", "HDD", "NVMe"]).nullable().optional(),
  gpu: z.enum([
    "RTX 40-Series", "RTX 30-Series", "RTX 20-Series", 
    "GTX", "Integrated", "Other Dedicated"
  ]).nullable().optional(),
  screen_size_inch: z.number().positive().nullable().optional(),
  refresh_rate_hz: z.number().int().positive().nullable().optional(),
  resolution: z.enum(["1080p FHD", "2K QHD", "4K UHD", "HD"]).nullable().optional(),
  panel_type: z.enum(["IPS", "OLED", "VA", "Standard"]).nullable().optional(),
  condition: z.enum(["Used", "Brand New"]),
  is_touchscreen: z.boolean(),
  is_curved: z.boolean(),
  is_gaming: z.boolean(),
  location: z.string().nullable().optional(),
  listed_price: z.number().positive().nullable().optional(),
});
```

### Multimodal Vision Workflow
1. The extension captures a visible tab screenshot via `chrome.tabs.captureVisibleTab()`.
2. The image is passed as a base64 JPEG (`inlineData`) to Gemini Flash Lite.
3. Gemini visually reads titles, specs tables, badges, and checks physical product wear directly from the image.
4. Returns clean JSON that feeds directly into the ML inference pipeline.

---

## 6. Live Prediction & Valuation Calculation

When a prediction request is received at `/predict`:

1. **Inference**:
   $$\hat{y} = \exp(\text{Model.predict}(X_{\text{features}})) - 1$$

2. **Fair Market Range Generation**:
   * **Laptops & Tablets**: $\text{Range} = [\hat{y} \times 0.88, \; \hat{y} \times 1.12]$ ($\pm 12\%$)
   * **Monitors**: $\text{Range} = [\hat{y} \times 0.90, \; \hat{y} \times 1.10]$ ($\pm 10\%$)

3. **Deal Fairness Evaluation**:
   Given the seller's asking price $P_{\text{listed}}$:
   * **Great Deal / Underpriced** ($\text{Score} \ge 85$): $P_{\text{listed}} < \text{Lower Bound}$
   * **Fair Market Price** ($60 \le \text{Score} < 85$): $\text{Lower Bound} \le P_{\text{listed}} \le \text{Upper Bound}$
   * **Overpriced** ($\text{Score} < 60$): $P_{\text{listed}} > \text{Upper Bound}$

---

## 7. Operational Runbook

### Starting the Full Live Pipeline:
```bash
# Terminal 1: Start All Services via Master Script
python "d:/final project/R26-lT-040/api-gateway/start_all.py"

# Or run Electronics Microservice individually:
cd "d:/final project/R26-lT-040/electronics-price-model"
python app.py  # Runs on port 8004
```

### Loading Extension in Browser:
1. Open Chrome/Edge $\rightarrow$ `chrome://extensions/`
2. Enable **Developer mode**.
3. Click **Load unpacked** $\rightarrow$ select `d:\final project\R26-lT-040\browser-extension`.
4. Open any ad on Ikman.lk to view the live automated valuation badge.
