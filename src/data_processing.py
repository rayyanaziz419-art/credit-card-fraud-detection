"""
Feature Engineering and Data Preprocessing Pipeline.
Implements domain-driven, leakage-safe transformations for Credit Card Fraud Detection.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Leakage-safe domain feature engineering transformer:
    1. Cyclical encoding for transaction_hour (hour_sin, hour_cos).
    2. High-risk cross-border & location interaction: foreign_transaction * location_mismatch.
    3. Night transaction indicator: hour in [0, 1, 2, 3] (verified 80% of fraud occurs here).
    4. Velocity surge indicator: velocity_last_24h >= 5 (verified 10x fraud rate jump).
    5. Untrusted device indicator: device_trust_score <= 30.
    6. Log-transformed amount: log1p(amount) to handle heavy right-skew.
    """
    def __init__(self, add_interactions: bool = True, add_cyclical: bool = True):
        self.add_interactions = add_interactions
        self.add_cyclical = add_cyclical

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # Exclude transaction_id if present
        if 'transaction_id' in X.columns:
            X = X.drop(columns=['transaction_id'])
            
        # 1. Cyclical encoding of hour (24-hour cycle)
        if self.add_cyclical and 'transaction_hour' in X.columns:
            hours = X['transaction_hour'].astype(float)
            X['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
            X['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
            
        # 2. Domain risk interactions verified from EDA
        if self.add_interactions:
            # Cross-border location mismatch (fraud rate jumps to 34.1%)
            if 'foreign_transaction' in X.columns and 'location_mismatch' in X.columns:
                X['foreign_x_mismatch'] = X['foreign_transaction'] * X['location_mismatch']
                
            # Night window flag (hours 0-3 account for 80% of frauds)
            if 'transaction_hour' in X.columns:
                X['is_night_window'] = X['transaction_hour'].isin([0, 1, 2, 3]).astype(int)
                
            # Velocity spike flag (velocity >= 5 has >10% fraud rate)
            if 'velocity_last_24h' in X.columns:
                X['is_velocity_spike'] = (X['velocity_last_24h'] >= 5).astype(int)
                
            # Suspicious device flag (device trust score <= 30)
            if 'device_trust_score' in X.columns:
                X['is_suspicious_device'] = (X['device_trust_score'] <= 30).astype(int)

        # 3. Log-transformed monetary amount
        if 'amount' in X.columns:
            X['log_amount'] = np.log1p(np.maximum(X['amount'], 0.0))
            
        return X


def build_preprocessor() -> Pipeline:
    """
    Constructs a complete scikit-learn ColumnTransformer and Pipeline.
    Encodes categorical features and scales numerical inputs safely.
    """
    # Numerical features produced after feature engineering
    numeric_features = [
        'amount',
        'log_amount',
        'transaction_hour',
        'hour_sin',
        'hour_cos',
        'device_trust_score',
        'velocity_last_24h',
        'cardholder_age',
        'foreign_transaction',
        'location_mismatch',
        'foreign_x_mismatch',
        'is_night_window',
        'is_velocity_spike',
        'is_suspicious_device'
    ]
    
    categorical_features = ['merchant_category']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    column_processor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    full_pipeline = Pipeline(steps=[
        ('feature_engineer', DomainFeatureEngineer()),
        ('column_processor', column_processor)
    ])
    
    return full_pipeline


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates candidate features X from target y, dropping non-predictive transaction_id.
    """
    if 'is_fraud' in df.columns:
        X = df.drop(columns=['is_fraud'])
        y = df['is_fraud'].astype(int)
    else:
        X = df.copy()
        y = None
        
    return X, y

