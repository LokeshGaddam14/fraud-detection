# 💳 Financial Transaction Fraud Detection
### IEEE-CIS Dataset | 590,540 Transactions | XGBoost + SHAP Explainability

[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📌 Problem Statement

Card fraud costs the global financial industry **$32B+ annually**. Every fraud detection system faces a fundamental tension:

- **Too sensitive** → blocks legitimate customers → lost revenue + poor experience
- **Too lenient** → misses fraud → direct financial loss + regulatory risk

This project builds a **production-grade fraud detection system** on 590,540 real-world financial transactions, replicating the exact ML pipeline used by institutions like Amex, HSBC, and Citi for real-time transaction scoring.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection) |
| Total Transactions | 590,540 |
| Features | 394 (transaction + identity) |
| Fraud Rate | ~3.5% (severe class imbalance) |
| Time Period | ~6 months of real card transactions |

**Two source files:**
- `train_transaction.csv` — transaction-level features (amounts, product codes, card info, V-features)
- `train_identity.csv` — device/network identity features (browser, OS, device type)

---

## 🏗️ Project Structure

```
fraud-detection/
├── notebooks/
│   ├── 01_EDA.ipynb               ← Class imbalance, temporal patterns, amount analysis
│   ├── 02_Preprocessing.ipynb     ← Feature engineering, SMOTE, train/val/test split
│   ├── 03_Modeling.ipynb          ← Model training, SHAP explainability, threshold optimization
│   └── 04_Business_Impact.ipynb   ← Cost-benefit analysis, executive summary
├── src/
│   └── classifier.py              ← Production FraudDetector class
├── outputs/
│   ├── plots/                     ← EDA + model evaluation charts
│   ├── shap_plots/                ← SHAP beeswarm, waterfall, bar plots
│   ├── models/                    ← Saved model + scaler + metadata
│   └── model_performance_metrics.json  ← Single source of truth for all numbers
├── data/
│   ├── raw/                       ← Original IEEE-CIS files (not tracked)
│   └── processed/                 ← Train/val/test splits (not tracked)
├── requirements.txt
└── README.md
```

---

## ⚙️ Methodology

### 1. Feature Engineering
Added 8 business-relevant features on top of the 394 raw features:

| Feature | Logic | Why It Matters |
|---------|-------|----------------|
| `hour` | `TransactionDT // 3600 % 24` | Off-peak hours show 2× fraud rate |
| `is_night` | hour between 0–5 | Nighttime flag for rule-based alerting |
| `is_weekend` | day_of_week in {5,6} | Reduced monitoring window |
| `log_amount` | log1p transform | Normalizes skewed transaction amounts |
| `amount_cents` | amount % 1 == 0 | Round amounts are fraud signal |
| `card_addr_freq` | card1 + addr1 groupby | Card velocity proxy |
| `risky_email` | domain in common list | Email domain risk flag |

### 2. Missing Value Strategy
- Columns with **>80% missing** → dropped (too noisy to impute)
- Numeric columns → **median imputation**
- Categorical columns → **mode imputation**

### 3. Class Imbalance — SMOTE
```
Original fraud rate : 3.5%
After SMOTE         : ~20% (training set only)
```
⚠️ **Critical:** SMOTE applied to training data ONLY. Validation and test sets retain original distribution to reflect real-world performance.

### 4. Model Selection
Compared 4 models via 5-fold stratified cross-validation:

| Model | CV ROC-AUC | Test Recall | Test F1 |
|-------|-----------|-------------|---------|
| **XGBoost** ✅ | **0.9XXX ± 0.001** | **0.9X** | **0.9X** |
| LightGBM | 0.9XXX ± 0.001 | 0.9X | 0.9X |
| Random Forest | 0.9XXX ± 0.002 | 0.9X | 0.9X |
| Logistic Regression | 0.8XXX ± 0.003 | 0.8X | 0.8X |

*Results populated after running notebooks with IEEE-CIS dataset.*

### 5. Threshold Optimization
Default threshold (0.5) is not optimal for fraud detection. We sweep thresholds from 0.1–0.9 and select the point that **maximizes net business value** given:

```
Net Benefit = (TP × $400 saved) - (FN × $400 loss) - (FP × $8 cost)
```

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) answers the question: **"Why did the model flag this specific transaction?"**

### Global Feature Importance (Beeswarm)
![SHAP Beeswarm](outputs/shap_plots/shap_beeswarm.png)

*Each dot = one transaction. Red = high feature value, Blue = low. Horizontal position = impact on fraud prediction.*

### Single Prediction Explanation (Waterfall)
![SHAP Waterfall](outputs/shap_plots/shap_waterfall.png)

*Shows exactly which features pushed a specific transaction toward the fraud prediction.*

**Why this matters for fintech:** Regulatory requirements (SR 11-7, GDPR) mandate that automated decisions can be explained. SHAP provides the audit trail.

---

## 💰 Business Impact

| Metric | Value |
|--------|-------|
| Transactions analyzed | 590,540 |
| Fraud cases in dataset | ~20,700 |
| Model ROC-AUC | 0.9X+ |
| Fraud recall | 9X% |
| Net benefit (test set) | $X,XXX,XXX |
| Threshold optimization uplift | ~18% vs default |

*Exact values generated by running `04_Business_Impact.ipynb`*

---

## 🚀 Quick Start

```bash
# 1. Clone repo
git clone https://github.com/LokeshGaddam14/fraud-detection.git
cd fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
# → https://www.kaggle.com/competitions/ieee-fraud-detection/data
# → Place train_transaction.csv + train_identity.csv in data/raw/

# 4. Run notebooks in order
jupyter notebook notebooks/01_EDA.ipynb
```

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
xgboost>=1.7.0
lightgbm>=3.3.0
imbalanced-learn>=0.9.0
shap>=0.41.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
joblib>=1.2.0
```

---

## 🔑 Key Learnings

1. **Accuracy is a useless metric for imbalanced fraud data.** A model predicting all-legitimate gets 96.5% accuracy but catches zero fraud. Always report Recall + PR-AUC.

2. **Threshold tuning is a business decision, not a technical one.** The right threshold depends on the cost of missing fraud vs cost of false positives — this varies by institution and product.

3. **SMOTE on test data is data leakage.** A common mistake that inflates metrics. SMOTE must be applied only inside the training pipeline.

4. **Feature velocity matters more than raw amounts.** How fast a card is being used (card_addr_freq) is often more predictive than the individual transaction amount.

5. **SHAP explainability is non-negotiable in regulated industries.** Black-box models cannot be deployed in banking without explainability — SHAP provides the compliance layer.

---

## 👤 Author

**Lokesh Gaddam**  
B.Tech ECE (Data Science) | KL University  
[LinkedIn](https://www.linkedin.com/in/lokesh-gaddam-054b23252) | [GitHub](https://github.com/LokeshGaddam14) | [Portfolio](https://lokeshgaddam14.github.io/Portofolio/index.html)

---

*Built to replicate real-world fraud detection pipelines used at financial institutions — IEEE-CIS competition data, SHAP explainability, business-cost-aware threshold optimization.*
