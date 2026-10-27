"""
Data Validation Module for Credit Card Fraud Detection Pipeline.
Ensures incoming tabular transactions conform to strict schema and domain business rules.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    'transaction_id',
    'amount',
    'transaction_hour',
    'merchant_category',
    'foreign_transaction',
    'location_mismatch',
    'device_trust_score',
    'velocity_last_24h',
    'cardholder_age',
]

VALID_CATEGORIES = {'Clothing', 'Electronics', 'Food', 'Grocery', 'Travel'}


def validate_schema(df: pd.DataFrame, is_training: bool = True) -> Tuple[bool, list]:
    """
    Validates presence of required columns.
    
    Args:
        df: Input DataFrame to check.
        is_training: If True, checks for presence of 'is_fraud' target.
        
    Returns:
        Tuple of (is_valid: bool, errors: list[str])
    """
    errors = []
    expected_cols = REQUIRED_COLUMNS + (['is_fraud'] if is_training else [])
    missing_cols = [c for c in expected_cols if c not in df.columns]
    
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        
    return len(errors) == 0, errors


def validate_records(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validates value ranges and domain validity rules:
    - amount >= 0
    - transaction_hour in [0, 23]
    - merchant_category in VALID_CATEGORIES
    - binary flags in {0, 1}
    - device_trust_score in [0, 100]
    - velocity_last_24h >= 0
    - cardholder_age between 18 and 120
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Tuple of (is_valid: bool, issues: list[str])
    """
    issues = []
    
    # 1. Null check
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        issues.append(f"Found null values in columns: {cols_with_nulls.to_dict()}")

    # 2. Amount validity
    if (df['amount'] < 0).any():
        num_neg = (df['amount'] < 0).sum()
        issues.append(f"Found {num_neg} negative transaction amounts.")
        
    # 3. Transaction hour validity
    invalid_hours = ~df['transaction_hour'].between(0, 23)
    if invalid_hours.any():
        issues.append(f"Found {invalid_hours.sum()} records with transaction_hour outside [0, 23].")
        
    # 4. Merchant category validity
    invalid_cats = ~df['merchant_category'].isin(VALID_CATEGORIES)
    if invalid_cats.any():
        bad_values = df.loc[invalid_cats, 'merchant_category'].unique().tolist()
        issues.append(f"Found invalid merchant categories: {bad_values}")

    # 5. Binary flags validity
    for flag in ['foreign_transaction', 'location_mismatch']:
        invalid_flag = ~df[flag].isin([0, 1])
        if invalid_flag.any():
            issues.append(f"Found non-binary values in '{flag}': {df.loc[invalid_flag, flag].unique().tolist()}")
            
    # 6. Device trust score range [0, 100]
    invalid_scores = ~df['device_trust_score'].between(0, 100)
    if invalid_scores.any():
        issues.append(f"Found {invalid_scores.sum()} device_trust_score outside [0, 100].")

    # 7. Velocity validity
    if (df['velocity_last_24h'] < 0).any():
        issues.append(f"Found negative transaction velocity values.")

    # 8. Cardholder age validity [18, 120]
    invalid_ages = ~df['cardholder_age'].between(18, 120)
    if invalid_ages.any():
        issues.append(f"Found {invalid_ages.sum()} records with cardholder_age outside [18, 120].")
        
    return len(issues) == 0, issues


def generate_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a structured data quality summary dictionary.
    """
    schema_ok, schema_errors = validate_schema(df, is_training=('is_fraud' in df.columns))
    records_ok, record_issues = validate_records(df)
    
    report = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "is_schema_valid": schema_ok,
        "schema_errors": schema_errors,
        "is_data_valid": records_ok,
        "data_issues": record_issues,
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_transaction_ids": int(df['transaction_id'].duplicated().sum()) if 'transaction_id' in df.columns else None,
        "null_counts": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe().round(2).to_dict()
    }
    
    if 'is_fraud' in df.columns:
        report["target_distribution"] = {
            "fraud_count": int((df['is_fraud'] == 1).sum()),
            "legit_count": int((df['is_fraud'] == 0).sum()),
            "fraud_percentage": round(float((df['is_fraud'] == 1).mean() * 100), 2)
        }
        
    return report

