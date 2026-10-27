# Dataset Documentation & Provenance

## Overview
- **Filename**: `credit_card_fraud_10k.csv`
- **Total Records**: 10,000 transactions
- **Total Features**: 10 columns (9 candidate features + 1 target variable)
- **Target Variable**: `is_fraud` (Binary: `0` = Legitimate transaction, `1` = Fraudulent transaction)

## Class Imbalance
- **Legitimate Transactions (`0`)**: 9,849 (98.49%)
- **Fraudulent Transactions (`1`)**: 151 (1.51%)
- **Imbalance Ratio**: Approximately 65 : 1

## Feature Dictionary

| Column Name | Data Type | Physical Range | Business Description |
| :--- | :--- | :--- | :--- |
| `transaction_id` | Integer | 1 to 10,000 | Unique transaction identifier. Non-predictive index; excluded from modeling to avoid leakage. |
| `amount` | Float | $0.00 to $1,471.04 | Monetary value of transaction in currency units (USD). Zero amount represents an authorization probe. |
| `transaction_hour`| Integer | 0 to 23 | Hour of day (24-hour clock) when transaction was initiated. |
| `merchant_category`| String | 5 categories | Category of merchant: Clothing, Electronics, Food, Grocery, Travel. |
| `foreign_transaction`| Binary (0/1)| 0 or 1 | 1 indicates cross-border/foreign transaction; 0 indicates domestic. |
| `location_mismatch`| Binary (0/1)| 0 or 1 | 1 indicates transaction geolocation differed from cardholder's usual home/billing location. |
| `device_trust_score`| Integer | 25 to 99 | Internal risk engine score assessing device reputation (lower = suspicious/new, higher = trusted). |
| `velocity_last_24h`| Integer | 0 to 9 | Count of distinct transactions initiated by the cardholder in the preceding 24-hour window. |
| `cardholder_age` | Integer | 18 to 69 | Age of the primary cardholder in years. |
| `is_fraud` | Binary (0/1)| 0 or 1 | Supervised ground truth target label. |

## Data Provenance & Limitations
1. **Synthetic / Curated Nature**: The dataset displays structured distributions characteristic of synthetic or semi-synthetic benchmark fraud datasets (clean uniform/truncated ranges, zero missing values, zero corrupted strings).
2. **Missing Real-World Elements**: Does not include temporal timestamps (dates/days of week), merchant ID granularity, terminal types (card-present vs. card-not-present / e-commerce), chargeback disposition dates, or investigation costs.
3. **Small Sample of Fraud (`N = 151`)**: While 10,000 records provide adequate sample sizes for standard analysis, 151 positive fraud instances mean any multi-way subgroup split (e.g., specific age x hour x category) quickly encounters small-sample variance. Subgroup conclusions must be qualified with sample size indicators.

