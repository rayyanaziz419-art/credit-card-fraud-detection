"""
Evaluation module for imbalanced credit card fraud detection.
Computes PR-AUC, ROC-AUC, Precision, Recall, F1, and cost-benefit trade-offs.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve
)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
    """
    Computes all essential evaluation metrics for binary classification.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    total_positives = int(tp + fn)
    total_negatives = int(tn + fp)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "actual_fraud_support": total_positives,
        "actual_legit_support": total_negatives
    }
    
    if y_prob is not None:
        try:
            metrics["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 4)
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        except Exception:
            metrics["pr_auc"] = np.nan
            metrics["roc_auc"] = np.nan
    else:
        metrics["pr_auc"] = np.nan
        metrics["roc_auc"] = np.nan
        
    return metrics


def evaluate_threshold_range(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray = np.linspace(0.05, 0.95, 19)
) -> pd.DataFrame:
    """
    Evaluates model performance across a range of decision thresholds.
    """
    results = []
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        m = calculate_metrics(y_true, y_pred, y_prob)
        m["threshold"] = round(float(th), 2)
        results.append(m)
        
    df_results = pd.DataFrame(results)
    # Reorder columns
    cols = ["threshold", "precision", "recall", "f1_score", "true_positives", "false_positives", "false_negatives", "true_negatives"]
    return df_results[cols]


def compute_hypothetical_cost(
    df_thresholds: pd.DataFrame,
    cost_fp: float = 10.0,
    cost_fn: float = 200.0
) -> pd.DataFrame:
    """
    Hypothetical cost simulation:
    - cost_fp: Cost of customer friction / manual alert review ($10)
    - cost_fn: Average unrecovered loss from a missed fraud transaction ($200)
    Total Cost = (False Positives * cost_fp) + (False Negatives * cost_fn)
    """
    df_cost = df_thresholds.copy()
    df_cost["total_simulated_cost"] = (df_cost["false_positives"] * cost_fp) + (df_cost["false_negatives"] * cost_fn)
    return df_cost.sort_values(by="total_simulated_cost")

