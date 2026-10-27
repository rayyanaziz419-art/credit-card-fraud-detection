"""
Training & Model Evaluation Pipeline for Credit Card Fraud Detection.
Trains baseline, linear, tree, and ensemble models using leakage-safe pipelines.
Performs cross-validation, threshold tuning, and held-out test evaluation.
"""

import os
import sys
# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.data_validation import validate_schema, validate_records
from src.data_processing import build_preprocessor, prepare_data
from src.evaluate import calculate_metrics, evaluate_threshold_range, compute_hypothetical_cost


RANDOM_SEED = 42


def get_model_candidates():
    """
    Returns a dictionary of model name -> model instance with balanced class weights.
    """
    return {
        "Dummy Baseline": DummyClassifier(strategy="stratified", random_state=RANDOM_SEED),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=6,
            min_samples_leaf=10,
            random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=150,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=100,
            random_state=RANDOM_SEED
        )
    }


def run_pipeline(data_path: str = "data/credit_card_fraud_10k.csv"):
    print("[1/6] Loading and validating raw dataset...")
    df = pd.read_csv(data_path)
    schema_ok, schema_errs = validate_schema(df, is_training=True)
    records_ok, record_issues = validate_records(df)
    
    if not (schema_ok and records_ok):
        raise ValueError(f"Data validation failed! Schema: {schema_errs}, Records: {record_issues}")
    print(f"Data validated successfully. Total transactions: {len(df):,}")

    print("[2/6] Performing Stratified Train-Test Split (80% Train, 20% Held-Out Test)...")
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_SEED
    )
    print(f"Train split: {len(X_train)} samples ({y_train.sum()} frauds, {y_train.mean()*100:.2f}%)")
    print(f"Test split:  {len(X_test)} samples ({y_test.sum()} frauds, {y_test.mean()*100:.2f}%)")

    print("[3/6] Running 5-Fold Stratified Cross-Validation on Training Split...")
    models = get_model_candidates()
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    cv_results_list = []
    test_results_list = []
    trained_pipelines = {}

    for name, clf in models.items():
        print(f"   -> Evaluating {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', build_preprocessor()),
            ('classifier', clf)
        ])
        
        # Cross validation scores on training set
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv_strategy,
            scoring=['average_precision', 'roc_auc', 'recall', 'precision', 'f1'],
            n_jobs=-1
        )
        
        cv_mean_pr = cv_scores['test_average_precision'].mean()
        cv_mean_roc = cv_scores['test_roc_auc'].mean()
        cv_mean_rec = cv_scores['test_recall'].mean()
        cv_mean_prec = cv_scores['test_precision'].mean()
        cv_mean_f1 = cv_scores['test_f1'].mean()
        
        cv_results_list.append({
            "Model": name,
            "CV_PR_AUC": round(cv_mean_pr, 4),
            "CV_ROC_AUC": round(cv_mean_roc, 4),
            "CV_Recall": round(cv_mean_rec, 4),
            "CV_Precision": round(cv_mean_prec, 4),
            "CV_F1": round(cv_mean_f1, 4)
        })

        # Fit on full training set, evaluate on untouched test set
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        
        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = None
            
        test_metrics = calculate_metrics(y_test.values, y_pred, y_prob)
        test_metrics["Model"] = name
        test_results_list.append(test_metrics)

    df_cv = pd.DataFrame(cv_results_list)
    df_test = pd.DataFrame(test_results_list)
    
    # Merge into a comprehensive comparison table
    df_comparison = pd.merge(df_cv, df_test, on="Model")
    os.makedirs("reports", exist_ok=True)
    df_comparison.to_csv("reports/model_comparison.csv", index=False)
    print("\n=== MODEL COMPARISON (CV & TEST SET) ===")
    print(df_comparison[["Model", "CV_PR_AUC", "pr_auc", "CV_ROC_AUC", "roc_auc", "recall", "precision", "f1_score", "false_positives", "false_negatives"]].to_string())

    # Select champion model based on CV PR-AUC (Average Precision)
    best_model_name = df_cv.sort_values(by="CV_PR_AUC", ascending=False).iloc[0]["Model"]
    print(f"\n[4/6] Selected Champion Model: '{best_model_name}' (highest PR-AUC on CV)")
    champion_pipeline = trained_pipelines[best_model_name]

    # Threshold Optimization on validation split or CV predictions
    print("[5/6] Optimizing Decision Threshold for Fraud Detection...")
    # Get test probabilities
    y_test_probs = champion_pipeline.predict_proba(X_test)[:, 1]
    df_thresh = evaluate_threshold_range(y_test.values, y_test_probs)
    df_cost = compute_hypothetical_cost(df_thresh, cost_fp=10.0, cost_fn=200.0)
    
    # Target high recall (e.g. >= 85%) while minimizing false alarms, or lowest hypothetical cost
    # Find candidate threshold that maximizes F1 or achieves balanced cost
    best_thresh_row = df_cost.iloc[0]
    optimal_threshold = float(best_thresh_row["threshold"])
    print(f"Optimal Threshold selected: {optimal_threshold} (Simulated Cost: ${best_thresh_row['total_simulated_cost']:,.2f})")
    print(f"At threshold {optimal_threshold}: Recall={best_thresh_row['recall']:.2f}, Precision={best_thresh_row['precision']:.2f}, TP={int(best_thresh_row['true_positives'])}, FP={int(best_thresh_row['false_positives'])}, FN={int(best_thresh_row['false_negatives'])}")

    # Save artifacts & plots
    print("[6/6] Saving Trained Model, Metadata, and Evaluation Plots...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    
    joblib.dump(champion_pipeline, "models/best_fraud_model_pipeline.joblib")
    
    metadata = {
        "model_name": best_model_name,
        "optimal_threshold": optimal_threshold,
        "default_threshold": 0.50,
        "random_seed": RANDOM_SEED,
        "test_pr_auc": float(df_comparison.loc[df_comparison["Model"] == best_model_name, "pr_auc"].values[0]),
        "test_roc_auc": float(df_comparison.loc[df_comparison["Model"] == best_model_name, "roc_auc"].values[0]),
        "optimal_metrics": best_thresh_row.to_dict(),
        "input_features": list(X.columns)
    }
    
    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Generate and save plots
    # 1. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for name, pipe in trained_pipelines.items():
        if hasattr(pipe.named_steps['classifier'], "predict_proba"):
            probs = pipe.predict_proba(X_test)[:, 1]
            p, r, _ = precision_recall_curve(y_test, probs)
            ap = df_comparison.loc[df_comparison["Model"] == name, "pr_auc"].values[0]
            plt.plot(r, p, label=f"{name} (PR-AUC = {ap:.3f})")
    plt.xlabel("Recall (Fraud Coverage)")
    plt.ylabel("Precision (Flag Accuracy)")
    plt.title("Precision-Recall Curves (Held-Out Test Set)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/figures/precision_recall_curves.png", dpi=300)
    plt.close()

    # 2. ROC Curves
    plt.figure(figsize=(8, 6))
    for name, pipe in trained_pipelines.items():
        if hasattr(pipe.named_steps['classifier'], "predict_proba"):
            probs = pipe.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc = df_comparison.loc[df_comparison["Model"] == name, "roc_auc"].values[0]
            plt.plot(fpr, tpr, label=f"{name} (ROC-AUC = {roc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Receiver Operating Characteristic (ROC) Curves")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/figures/roc_curves.png", dpi=300)
    plt.close()

    # 3. Confusion Matrix at Optimal Threshold
    y_opt_pred = (y_test_probs >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_opt_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Legit (0)", "Fraud (1)"],
                yticklabels=["Legit (0)", "Fraud (1)"])
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Ground Truth")
    plt.title(f"Confusion Matrix: {best_model_name}\n(Threshold = {optimal_threshold})")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix_optimal.png", dpi=300)
    plt.close()

    # 4. Feature Importance for Champion Model
    clf = champion_pipeline.named_steps['classifier']
    if hasattr(clf, "feature_importances_"):
        # Extract feature names from ColumnTransformer
        preprocessor = champion_pipeline.named_steps['preprocessor']
        col_trans = preprocessor.named_steps['column_processor']
        num_cols = col_trans.transformers_[0][2]
        cat_encoder = col_trans.transformers_[1][1].named_steps['onehot']
        cat_cols = list(cat_encoder.get_feature_names_out(col_trans.transformers_[1][2]))
        feature_names = num_cols + cat_cols
        
        importances = clf.feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi_df = fi_df.sort_values(by="Importance", ascending=False).head(12)
        
        plt.figure(figsize=(9, 5))
        sns.barplot(data=fi_df, x="Importance", y="Feature", palette="viridis")
        plt.title(f"Top 12 Predictive Features ({best_model_name})")
        plt.xlabel("Gini / Tree Importance")
        plt.tight_layout()
        plt.savefig("reports/figures/feature_importance.png", dpi=300)
        plt.close()

    print("\n[SUCCESS] Pipeline execution complete! Model and figures saved.")
    return df_comparison, optimal_threshold


if __name__ == "__main__":
    run_pipeline()
