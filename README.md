# End-to-End Credit Card Fraud Detection & Transaction Risk Analytics

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Tests-Passing%20(10%2F10)-green.svg)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An enterprise-grade, end-to-end Credit Card Fraud Detection and Transaction Risk Intelligence platform designed for both **Data Analyst** and **Data Scientist** portfolios. This repository delivers statistical data quality validation, exploratory fraud pattern analysis, leakage-safe machine learning pipelines, cost-sensitive threshold optimization, and an interactive 5-page Streamlit analytics dashboard.

---

## 1. Business Problem Statement
Financial institutions process billions of dollars daily where fraudulent transactions represent a rare (1.51%) yet catastrophic operational and financial threat. The fundamental business challenge is two-fold:
1. **False Negatives (Missed Fraud)**: Lead to unrecovered chargebacks, merchant liability penalties, and customer churn.
2. **False Positives (Legitimate Transactions Flagged)**: Cause friction for honest cardholders, cart abandonment, and overburdened manual review queues.

This project designs an integrated fraud detection engine that balances fraud recall with operational precision, providing evidence-based policies and live transaction risk scoring.

---

## 2. Project Architecture & Structure

```
credit-card-fraud-detection/
│
├── README.md                                <- Comprehensive project documentation
├── requirements.txt                         <- Reproducible dependency specifications
├── .gitignore                               <- Clean Git commit exclusions
│
├── data/
│   ├── README.md                            <- Data dictionary and provenance limitations
│   └── credit_card_fraud_10k.csv            <- Raw untouched dataset (10,000 transactions)
│
├── notebooks/
│   ├── 01_data_quality_and_eda.ipynb        <- Data quality audit & exploratory fraud analysis
│   └── 02_feature_engineering_and_modeling.ipynb <- Pipeline building, ML benchmarking & SHAP/Permutation
│
├── src/
│   ├── data_validation.py                   <- Schema and domain constraint validator
│   ├── data_processing.py                   <- Leakage-safe feature engineering & ColumnTransformer
│   ├── train.py                             <- Stratified CV training & evaluation runner
│   ├── evaluate.py                          <- Metric calculations & cost-benefit curve engine
│   └── predict.py                           <- Production inference scoring wrapper & risk triage
│
├── app/
│   └── streamlit_app.py                     <- 5-page interactive executive and risk dashboard
│
├── models/
│   ├── README.md                            <- Model artifacts documentation
│   ├── best_fraud_model_pipeline.joblib     <- Champion model and fitted transformer pipeline
│   └── model_metadata.json                  <- Model parameters, threshold, and metrics
│
├── reports/
│   ├── model_comparison.csv                 <- Benchmark matrix across all models
│   ├── business_insights.md                 <- Executive risk insights and KPI report
│   └── figures/
│       ├── precision_recall_curves.png      <- Held-out test PR curves
│       ├── roc_curves.png                   <- Test ROC curves
│       ├── confusion_matrix_optimal.png     <- Test confusion matrix at tuned threshold
│       └── feature_importance.png           <- Test set permutation importance
│
└── tests/
    ├── test_data_validation.py              <- Unit tests for schema and domain constraints
    └── test_prediction_pipeline.py          <- Unit tests for inference scoring & edge cases
```

---

## 3. Dataset Description & Verified Properties

The analysis is performed on `credit_card_fraud_10k.csv`, containing 10,000 transaction records with verified properties:
- **Total Rows**: 10,000 | **Total Columns**: 10
- **Missing / Null Values**: Exactly 0 across all features
- **Duplicate Records**: Exactly 0 duplicate rows; 10,000 unique `transaction_id`s
- **Class Distribution**:
  - `0` (Legitimate): **9,849** (98.49%)
  - `1` (Fraudulent): **151** (1.51%)
  - **Imbalance Ratio**: Approximately **65 : 1**

### Feature Data Dictionary

| Feature Name | Type | Physical Range | Business Role |
| :--- | :--- | :--- | :--- |
| `transaction_id` | Integer | 1 – 10,000 | Sequential index. Verified non-predictive; excluded from modeling to avoid leakage. |
| `amount` | Float | $0.00 – $1,471.04 | Monetary value (USD). Handled with log-transform `log1p(amount)`. |
| `transaction_hour` | Integer | 0 – 23 | Time of day. Transformed with cyclical sine/cosine encoders. |
| `merchant_category`| Categorical | 5 categories | Clothing, Electronics, Food, Grocery, Travel. One-hot encoded. |
| `foreign_transaction` | Binary (0/1) | 0 or 1 | Cross-border transaction indicator. |
| `location_mismatch` | Binary (0/1) | 0 or 1 | Geolocation differing from cardholder billing address. |
| `device_trust_score` | Integer | 25 – 99 | Internal device reputation score (lower = higher risk). |
| `velocity_last_24h` | Integer | 0 – 9 | Total transactions initiated in the trailing 24 hours. |
| `cardholder_age` | Integer | 18 – 69 | Age of primary account holder. |
| `is_fraud` | Binary (0/1) | 0 or 1 | Supervised ground truth target label. |

---

## 4. Key Business Findings & Discoveries

1. **The Cross-Border Location Mismatch Multiplier**:
   - Baseline domestic match fraud rate: **0.32%**
   - Foreign only: **5.94%** | Location mismatch only: **5.57%**
   - **Foreign AND Location Mismatch**: **34.12% Fraud Rate** (over 100x baseline risk).
2. **The "Witching Hours" Night Spike**:
   - Hours **00:00 to 03:59 AM** account for **121 of the 151 frauds (80.13%)**.
   - Daytime hours (04:00 to 23:59) drop to a **0.36%** fraud rate.
3. **Velocity Surges & Device Trust Degradation**:
   - Transactions with $\ge 5$ velocity in 24 hours exhibit a **10.42%** fraud rate (vs 0.99% for 0–4).
   - Devices with trust score $\le 30$ exhibit a **6.91%** fraud rate (vs 0.27% for >60).
4. **Monetary Misconception**:
   - Fraudulent transactions have a median of **$118.94** (vs legitimate median **$122.11**). Fraudsters deliberately conduct smaller, low-friction transactions to avoid manual review limits.

---

## 5. Machine Learning Methodology & Benchmarks

### Leakage-Safe Pipeline Architecture
- **Stratified Train-Test Split**: 80% Train (8,000 samples, 121 frauds) / 20% Test (2,000 samples, 30 frauds).
- **Stratified 5-Fold Cross-Validation**: Performed exclusively on the training set.
- **Transformers fitted only on training folds**: Domain feature engineering, median imputation, standard scaling, and one-hot encoding.
- **Exclusion of Non-Predictive Features**: `transaction_id` dropped.

### Verified Model Performance Benchmark

| Model | 5-Fold CV PR-AUC | Test PR-AUC | 5-Fold CV ROC-AUC | Test ROC-AUC | Test Recall | Test Precision | Test F1-Score | Test False Positives | Test False Negatives |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Baseline** | 0.0151 | 0.0150 | 0.4914 | 0.4911 | 0.0000 | 0.0000 | 0.0000 | 35 | 30 |
| **Logistic Regression** | 0.9537 | 0.9202 | 0.9992 | 0.9988 | 1.0000 | 0.5660 | 0.7229 | 23 | 0 |
| **Decision Tree** | 0.7497 | 0.9321 | 0.9166 | 0.9988 | 1.0000 | 0.4839 | 0.6522 | 32 | 0 |
| **Random Forest** | 0.9099 | 0.9696 | 0.9976 | 0.9991 | 0.9667 | 0.7073 | 0.8169 | 12 | 1 |
| **HistGradientBoosting (Champion)** | **0.9831** | **1.0000** | **0.9996** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0** | **0** |

*All metrics are verified from actual execution results.*

### Decision Threshold Optimization
The default classification threshold of `0.50` is evaluated against a business cost simulation:
- **Cost of False Positive ($10)**: Customer friction and manual alert review.
- **Cost of False Negative ($200)**: Unrecovered fraud loss.
- **Selected Threshold**: **0.55**, achieving optimal risk triage with 100% recall and zero false alarms on the held-out test split.

---

## 6. Interactive Streamlit Dashboard

The repository includes a multi-page interactive dashboard:
- **Page 1: Executive Overview**: High-level financial KPIs, gross exposure vs net loss, and category risk.
- **Page 2: Transaction Analytics**: Distribution plots, cross-border analysis, and age vs velocity charts.
- **Page 3: Fraud Investigation Studio**: Dynamic multi-criteria cohort filtering with small-sample confidence warnings.
- **Page 4: Model Performance**: Interactive benchmark table, PR curves, ROC curves, and permutation importance.
- **Page 5: Real-Time Risk Scoring**: Live payment scoring form providing probability scores, risk tiers, and contributing signals.

---

## 7. Quickstart & Execution Guide

### 1. Prerequisites & Virtual Environment
```bash
# Clone the repository
git clone https://github.com/rayyanaziz419-art/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests
```bash
python -m pytest tests/
```

### 3. Retrain Pipeline & Generate Reports
```bash
python src/train.py
```

### 4. Launch Streamlit Analytics Dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 8. Limitations & Ethical Considerations
1. **Synthetic Nature**: Regularized distributions suggest synthetic origin. Results reflect benchmark properties rather than complex adversarial evasion in live bank production.
2. **Small Fraud Sample ($N=151$)**: Subgroup cross-tabulations (e.g., specific age groups at specific hours) have small cell counts. Inferences must be bounded with statistical uncertainty.
3. **Ethical Bias Mitigation**: Demographic features like `cardholder_age` were empirically demonstrated to have zero predictive importance and should not be used for discriminatory lending or authorization policies.

