"""
Unit tests for inference and prediction pipeline.
"""

import pytest
from src.predict import FraudRiskScorer


@pytest.fixture
def scorer():
    return FraudRiskScorer()


def test_scorer_low_risk(scorer):
    tx = {
        'transaction_id': 99999,
        'amount': 45.50,
        'transaction_hour': 14,
        'merchant_category': 'Grocery',
        'foreign_transaction': 0,
        'location_mismatch': 0,
        'device_trust_score': 90,
        'velocity_last_24h': 1,
        'cardholder_age': 40
    }
    res = scorer.score_single(tx)
    assert res['is_flagged_fraud'] is False
    assert res['risk_tier'] == 'LOW RISK'
    assert res['fraud_probability'] < 0.20


def test_scorer_high_risk(scorer):
    tx = {
        'transaction_id': 88888,
        'amount': 550.00,
        'transaction_hour': 2,
        'merchant_category': 'Travel',
        'foreign_transaction': 1,
        'location_mismatch': 1,
        'device_trust_score': 25,
        'velocity_last_24h': 6,
        'cardholder_age': 29
    }
    res = scorer.score_single(tx)
    assert res['is_flagged_fraud'] is True
    assert res['risk_tier'] in ['HIGH RISK', 'CRITICAL RISK']
    assert len(res['contributing_signals']) >= 3


def test_scorer_invalid_input(scorer):
    bad_tx = {
        'transaction_id': 100,
        'amount': -50.00,  # invalid negative
        'transaction_hour': 12,
        'merchant_category': 'Grocery',
        'foreign_transaction': 0,
        'location_mismatch': 0,
        'device_trust_score': 80,
        'velocity_last_24h': 1,
        'cardholder_age': 30
    }
    with pytest.raises(ValueError):
        scorer.score_single(bad_tx)

