# Machine Learning Models & Serialized Artifacts

## Directory Contents
- `best_fraud_model_pipeline.joblib`: Serialized scikit-learn Pipeline containing the fitted preprocessing transformers (imputation, domain feature engineering, one-hot encoding, standard scaling) and the champion `HistGradientBoostingClassifier`.
- `model_metadata.json`: Machine-readable metadata capturing the model type, training parameters, optimal decision threshold, cross-validation metrics, and feature names.

## Champion Model Specification
- **Algorithm**: `HistGradientBoostingClassifier`
- **Class Balancing**: Balanced class weight configuration
- **Primary Metric for Selection**: PR-AUC (Average Precision) across 5-fold Stratified Cross-Validation
- **CV PR-AUC**: `0.9831` (Baseline: `0.0151`)
- **CV ROC-AUC**: `0.9996` (Baseline: `0.4914`)
- **Held-Out Test Set PR-AUC**: `1.0000`
- **Held-Out Test Set ROC-AUC**: `1.0000`

## Decision Threshold Optimization
- **Default Threshold**: 0.50
- **Optimized Threshold**: 0.55
- **Tuned Test Performance**:
  - Precision: `1.0000`
  - Recall: `1.0000`
  - False Positives: `0`
  - False Negatives: `0`
  - Actual Frauds in Test Set: `30`
  - Actual Legitimate in Test Set: `1,970`

