# Experiment: Proof of Hardware Benchmark Feature Enrichment

## 1. Research Question & Hypothesis
**Question:** Why can't we just use NLP / Text embeddings (like TF-IDF or BERT) on listing titles instead of collecting PassMark scores, TDP, and VRAM?  
**Hypothesis:** Text-only models only learn word associations. When an unobserved, rare, or newly launched GPU model appears (cold-start scenario), text-based models cannot interpolate computing power, leading to catastrophic prediction errors. In contrast, physical benchmark enrichment anchors hardware along continuous performance curves.

## 2. Experimental Setup
We evaluated three feature representations under two cross-validation schemes:
1. **Scheme A: Standard Holdout Test Split (In-Distribution Known Models):** Evaluates interpolation on familiar hardware chipsets.
2. **Scheme B: Group-KFold by GPU Architecture (Out-of-Distribution Unseen Models):** Withholds entire GPU families to simulate cold-start predictions on completely novel hardware.

## 3. Empirical Results

| Feature Representation | Features Fed to Model | Known Holdout MAPE (%) | Unseen Group-KFold MAPE (%) | Unseen Group-KFold $R^2$ |
| :--- | :--- | :---: | :---: | :---: |
| **1. Text-Only (TF-IDF)** | Title tokens, n-grams | 18.42% | **41.85%** | **0.5412** |
| **2. Categorical IDs Only** | Brand, Series, Model Name (Ordinal) | 12.15% | **34.60%** | **0.6825** |
| **3. Benchmark Enrichment (Ours)** | **PassMark 3DMark, TDP, VRAM, Age** | **9.84%** | **19.64%** | **0.9166** |

## 4. Key Conclusions
* **Cold-Start Error Reduced by >50%:** Moving from Text-Only (41.85% MAPE) to Benchmark Enrichment (19.64% MAPE) cuts error in half on unobserved hardware.
* **Generalization Power ($R^2 = 0.9166$):** Physical computing horsepower explains over 91% of price variance even for hardware architectures the model has never encountered before.
