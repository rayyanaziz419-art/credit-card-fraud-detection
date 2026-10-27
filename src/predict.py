"""
Inference & Risk Scoring Engine for Credit Card Fraud Detection.
Loads champion pipeline, evaluates incoming transactions, and provides risk flags.
"""

import os
import sys
# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import joblib
from typing import Dict, Any, Union, List
import pandas as pd
import numpy as np
from src.data_validation import validate_schema, validate_records

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_fraud_model_pipeline.joblib')
DEFAULT_META_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_metadata.json')


class FraudRiskScorer:
    """
    Production-style inference scoring wrapper with risk categorization and explanatory signals.
    """
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, meta_path: str = DEFAULT_META_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Run 'python src/train.py' first.")
            
        self.pipeline = joblib.load(model_path)
        
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
                self.threshold = float(self.metadata.get("optimal_threshold", 0.50))
        else:
            self.metadata = {}
            self.threshold = 0.50

    def score_single(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scores a single transaction dictionary.
        """
        df_single = pd.DataFrame([transaction])
        
        # Schema validation
        schema_ok, schema_errs = validate_schema(df_single, is_training=False)
        if not schema_ok:
            raise ValueError(f"Invalid transaction payload: {schema_errs}")
            
        # Domain range check
        records_ok, record_issues = validate_records(df_single)
        if not records_ok:
            raise ValueError(f"Transaction values violate domain constraints: {record_issues}")

        # Compute probability
        prob = float(self.pipeline.predict_proba(df_single)[:, 1][0])
        is_flagged = bool(prob >= self.threshold)
        
        # Categorize risk tier
        if prob >= 0.70:
            risk_tier = "CRITICAL RISK"
            action = "Decline / Immediate Security Escalation"
        elif prob >= self.threshold:
            risk_tier = "HIGH RISK"
            action = "Step-Up Authentication (2FA / Push Challenge)"
        elif prob >= 0.20:
            risk_tier = "MEDIUM RISK"
            action = "Passive Monitoring / Velocity Tracking"
        else:
            risk_tier = "LOW RISK"
            action = "Standard Approval"

        # Signal breakdown
        signals = []
        if transaction.get('foreign_transaction') == 1 and transaction.get('location_mismatch') == 1:
            signals.append("Cross-Border Location Mismatch (Historical Fraud Rate: 34.1%)")
        elif transaction.get('foreign_transaction') == 1:
            signals.append("Foreign / International Transaction (Historical Fraud Rate: 8.4%)")
        elif transaction.get('location_mismatch') == 1:
            signals.append("Location Mismatch from Billing Address (Historical Fraud Rate: 8.4%)")
            
        if transaction.get('transaction_hour', 12) in [0, 1, 2, 3]:
            signals.append("High-Risk Night Hours Window 00:00-03:59 (Accounts for 80% of Frauds)")
            
        if transaction.get('device_trust_score', 100) <= 30:
            signals.append(f"Untrusted Device Score ({transaction.get('device_trust_score')}/100 <= 30)")
            
        if transaction.get('velocity_last_24h', 0) >= 5:
            signals.append(f"High Velocity Surge ({transaction.get('velocity_last_24h')} txns in 24h)")
            
        if not signals:
            signals.append("Transaction characteristics conform to baseline legitimate profile.")

        return {
            "fraud_probability": round(prob, 4),
            "threshold_applied": self.threshold,
            "is_flagged_fraud": is_flagged,
            "risk_tier": risk_tier,
            "recommended_action": action,
            "contributing_signals": signals,
            "disclaimer": "Experimental risk score for fraud analytics demonstration. Not an automated banking authorization decision."
        }

