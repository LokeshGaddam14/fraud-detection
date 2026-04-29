💳 Financial Transaction Fraud Detection
IEEE-CIS Dataset · 590,540 Transactions · XGBoost + LightGBM + SHAP Explainability
Show Image
Show Image
Show Image
Show Image
Show Image

📌 Problem Statement
Card fraud costs the global financial industry $32B+ annually. Every fraud detection system faces a fundamental tension:

Too sensitive → blocks legitimate customers → lost revenue + poor experience
Too lenient → misses fraud → direct financial loss + regulatory risk

This project builds a production-grade fraud detection pipeline on 590,540 real-world financial transactions — replicating the ML workflows used by institutions like Amex, HSBC, and Citi for real-time transaction scoring. The system goes beyond model training to include business-cost-aware threshold optimization and SHAP-based explainability for regulatory compliance.

📊 Dataset
PropertyValueSourceIEEE-CIS Fraud Detection — KaggleTotal Transactions590,540Features394 (transaction + identity)Fraud Rate~3.5% (severe class imbalance)Time Period~6 months of real card transactions
Two source files merged at the pipeline entry point:

train_transaction.csv — amounts, product codes, card info, 300+ Vesta V-features
train_identity.csv — device type, browser, OS, network fingerprint


⚠️ Raw data not tracked in this repo (Kaggle terms). Download instructions below.


🏗️ Project Structure
fraud-detection/
├── notebooks/
│   ├── 01_EDA.ipynb                ← Class imbalance, temporal patterns, amount distributions
│   ├── 02_Preprocessing.ipynb      ← Feature engineering, SMOTE, stratified splits
│   ├── 03_Modeling.ipynb           ← XGBoost/LightGBM training, SHAP, threshold sweep
│   └── 04_Business_Impact.ipynb    ← Cost-benefit analysis, executive summary
├── src/
│   ├── classifier.py               ← Production FraudDetector class (predict + explain)
│   └── threshold_optimizer.py      ← Business-cost-aware threshold selection
├── outputs/
│   ├── plots/                      ← EDA + model evaluation charts (PNG)
│   ├── shap_plots/                 ← SHAP beeswarm, waterfall, bar plots (PNG)
│   ├── models/                     ← Saved model + scaler + metadata (.pkl / .json)
│   └── model_performance_metrics.json  ← Single source of truth for all numbers
├── data/
│   ├── raw/                        ← IEEE-CIS files (not tracked — see Quick Start)
│   └── processed/                  ← Train/val/test splits (not tracked)
├── .gitignore
├── requirements.txt
└── README.md

⚙️ Methodology
Phase 1 — EDA (01_EDA.ipynb)
Key findings that drove modeling decisions:

Fraud rate: 3.5% → accuracy is a useless metric; optimized for PR-AUC + Recall
Off-peak hours (12am–5am) show 2.1× higher fraud rate → is_night feature added
Transaction amounts are heavily right-skewed → log_amount transform applied
~45% of identity features have >80% missing → dropped before modeling

Phase 2 — Feature Engineering (02_Preprocessing.ipynb)
8 business-relevant features added on top of 394 raw features:
FeatureLogicSignalhourTransactionDT // 3600 % 24Off-peak fraud spikeis_nighthour in [0, 5]Rule-based alert flagis_weekendday_of_week in {5, 6}Reduced monitoring windowlog_amountlog1p(TransactionAmt)Normalizes skewed amountsamount_centsamount % 1 == 0Round amounts are fraud signalcard_addr_freqgroupby(card1 + addr1)Card velocity proxyrisky_emaildomain in low-trust listEmail domain risk flagday_of_weekTransactionDT // 86400 % 7Weekly fraud patterns
Missing value strategy:

Columns with >80% missing → dropped
Numeric → median imputation
Categorical → mode imputation

Class imbalance — SMOTE:
Original fraud rate : 3.5%
After SMOTE         : ~20% (training set only — never applied to val/test)
Phase 3 — Modeling (03_Modeling.ipynb)
4 models evaluated via 5-fold stratified cross-validation on ROC-AUC:
ModelCV ROC-AUCNotesXGBoost ✅Bestscale_pos_weight tuned, early stoppingLightGBM~0.001 lowerFaster training, comparable AUCRandom ForestModerateNo boosting, weaker on V-featuresLogistic RegressionLowestLinear model struggles with interactions
Why XGBoost won: Better handling of the sparse V-features and interaction terms that dominate the IEEE-CIS feature space.
Threshold optimization — default 0.5 is suboptimal for fraud. Sweep from 0.1–0.9 and select threshold maximizing net business value:
Net Benefit = (TP × $400 saved) − (FN × $400 loss) − (FP × $8 review cost)
Phase 4 — Business Impact (04_Business_Impact.ipynb)
Translates model metrics into dollar terms for stakeholder communication — the exact framing used in fintech analyst presentations. Includes threshold sensitivity analysis and annual benefit projection.

🔍 SHAP Explainability
SHAP (SHapley Additive exPlanations) answers: "Why did the model flag this specific transaction?"
Global Feature Importance (Beeswarm)
Show Image
Each dot = one transaction. Red = high feature value, Blue = low. Horizontal position = impact on fraud probability.
Single Prediction Waterfall
Show Image
Shows exactly which features pushed a specific transaction toward the fraud label — essential for fraud analyst review queues.
Why this matters: Regulatory frameworks (SR 11-7, GDPR Article 22) require that automated financial decisions be explainable. SHAP provides the audit trail. Black-box deployment in banking is not compliant.

💰 Business Impact Summary
MetricValueTransactions analyzed590,540Fraud cases in dataset~20,700Model ROC-AUCSee model_performance_metrics.jsonFraud recall (optimized threshold)See model_performance_metrics.jsonThreshold optimization uplift~18% net benefit vs default 0.5Cost of false negative$400 (avg fraud transaction)Cost of false positive$8 (manual review)
Run 04_Business_Impact.ipynb to generate exact dollar figures for your test split.

🚀 Quick Start
bash# 1. Clone
git clone https://github.com/LokeshGaddam14/fraud-detection.git
cd fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data from Kaggle
#    → https://www.kaggle.com/competitions/ieee-fraud-detection/data
#    → Place train_transaction.csv + train_identity.csv in data/raw/

# 4. Run notebooks in order
jupyter notebook notebooks/01_EDA.ipynb
Using the Production Classifier
pythonfrom src.classifier import FraudDetector

detector = FraudDetector.load('outputs/models/')
score, explanation = detector.predict_explain(transaction_dict)

print(f"Fraud probability: {score:.3f}")
print(f"Top risk factors: {explanation}")

📦 Requirements
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

🔑 Key Learnings

Accuracy is a trap on imbalanced data. A model predicting all-legitimate gets 96.5% accuracy but catches zero fraud. Always optimize and report Recall + PR-AUC for fraud problems.
Threshold tuning is a business decision. The optimal threshold is not 0.5 — it's wherever (TP × fraud_value) − (FP × review_cost) is maximized. This number differs by institution and product line.
SMOTE on test data is data leakage. The most common mistake in Kaggle fraud tutorials. Synthetic samples must never touch validation or test sets.
Card velocity beats transaction amount. card_addr_freq (how fast a card is being used) typically outranks the raw transaction amount in SHAP importance. Pattern matters more than magnitude.
SHAP is compliance infrastructure. It's not optional in regulated industries. SR 11-7 (Fed model risk guidance) and GDPR require explainability for automated credit/fraud decisions.


👤 Author
Lokesh Gaddam
B.Tech ECE (Data Science Specialization) | KL University
LinkedIn · GitHub · Portfolio

Built to replicate production fraud detection pipelines — IEEE-CIS data, business-cost threshold optimization, SHAP regulatory explainability.
