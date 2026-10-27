"""
Unit tests for data validation module.
"""

import pytest
import pandas as pd
from src.data_validation import validate_schema, validate_records, generate_quality_report


@pytest.fixture
def sample_valid_df():
    return pd.DataFrame({
        'transaction_id': [1, 2],
        'amount': [50.0, 120.5],
        'transaction_hour': [14, 2],
        'merchant_category': ['Grocery', 'Electronics'],
        'foreign_transaction': [0, 1],
        'location_mismatch': [0, 1],
        'device_trust_score': [85, 30],
        'velocity_last_24h': [2, 5],
        'cardholder_age': [35, 62],
        'is_fraud': [0, 1]
    })


def test_validate_schema_success(sample_valid_df):
    is_valid, errors = validate_schema(sample_valid_df, is_training=True)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_schema_missing_column(sample_valid_df):
    bad_df = sample_valid_df.drop(columns=['amount'])
    is_valid, errors = validate_schema(bad_df, is_training=True)
    assert is_valid is False
    assert any('amount' in err for err in errors)


def test_validate_records_negative_amount(sample_valid_df):
    bad_df = sample_valid_df.copy()
    bad_df.loc[0, 'amount'] = -15.0
    is_valid, issues = validate_records(bad_df)
    assert is_valid is False
    assert any('negative transaction amount' in issue for issue in issues)


def test_validate_records_invalid_hour(sample_valid_df):
    bad_df = sample_valid_df.copy()
    bad_df.loc[0, 'transaction_hour'] = 25
    is_valid, issues = validate_records(bad_df)
    assert is_valid is False
    assert any('transaction_hour' in issue for issue in issues)


def test_validate_records_invalid_merchant(sample_valid_df):
    bad_df = sample_valid_df.copy()
    bad_df.loc[0, 'merchant_category'] = 'Cryptocurrency'
    is_valid, issues = validate_records(bad_df)
    assert is_valid is False
    assert any('invalid merchant categories' in issue for issue in issues)


def test_validate_records_invalid_binary(sample_valid_df):
    bad_df = sample_valid_df.copy()
    bad_df.loc[0, 'foreign_transaction'] = 3
    is_valid, issues = validate_records(bad_df)
    assert is_valid is False
    assert any('non-binary values' in issue for issue in issues)


def test_generate_quality_report(sample_valid_df):
    report = generate_quality_report(sample_valid_df)
    assert report['total_rows'] == 2
    assert report['is_schema_valid'] is True
    assert report['is_data_valid'] is True
    assert report['target_distribution']['fraud_count'] == 1
    assert report['target_distribution']['legit_count'] == 1

