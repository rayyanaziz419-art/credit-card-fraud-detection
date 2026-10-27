# Executive Business Insights & Fraud Analytics Report

## 1. Executive Summary & Verified KPI Scorecard

This report delivers a rigorous analysis of 10,000 credit card transactions to diagnose fraud patterns, measure financial exposure, and establish evidence-based risk mitigation strategies.

| Key Performance Indicator (KPI) | Verified Value | Benchmark / Context |
| :--- | :--- | :--- |
| **Total Transaction Volume (Count)** | **10,000** | 100% of analyzed portfolio |
| **Legitimate Transactions** | **9,849 (98.49%)** | Core operational baseline |
| **Fraud-Labeled Transactions** | **151 (1.51%)** | Highly imbalanced target class (~65:1 ratio) |
| **Total Gross Monetary Volume** | **$1,759,498.49** | Aggregate processed payments |
| **Legitimate Processed Volume** | **$1,726,854.86 (98.14%)** | Legitimate customer commerce |
| **Fraud-Labeled Transaction Volume** | **$32,643.63 (1.86%)** | Gross monetary exposure labeled as fraud |
| **Average Amount — Fraudulent** | **$216.18** | Median: $118.94 (Min: $0.11, Max: $1,185.07) |
| **Average Amount — Legitimate** | **$175.33** | Median: $122.11 (Min: $0.00, Max: $1,471.04) |

> [!IMPORTANT]
> **Crucial Financial Distinction — Exposure vs. Confirmed Loss:**
> In professional banking risk management, **Gross Fraud-Labeled Volume ($32,643.63)** must **never** be cited as confirmed net financial loss. In real banking workflows:
> 1. Transactions declined pre-authorization incur **zero loss**.
> 2. Card-present and 3DS-authenticated transactions frequently shift liability to the merchant or issuing network.
> 3. Recoveries through chargeback represent substantial clawbacks.
> Because this dataset lacks recovery and pre/post-authorization status fields, $32,643.63 represents **potential gross monetary exposure**, not realized bank write-offs.

---

## 2. In-Depth Fraud Pattern Discovery

### A. Geolocation & Cross-Border Indicators (Primary Signal)
- **Domestic Transactions (`foreign_transaction = 0`)**:
  - Count: 9,022 transactions | Fraud Count: 69 | **Fraud Rate: 0.76%**
- **Foreign Transactions (`foreign_transaction = 1`)**:
  - Count: 978 transactions | Fraud Count: 82 | **Fraud Rate: 8.38%** (11x baseline)
- **Billing Geolocation Match (`location_mismatch = 0`)**:
  - Count: 9,143 transactions | Fraud Count: 79 | **Fraud Rate: 0.86%**
- **Billing Geolocation Mismatch (`location_mismatch = 1`)**:
  - Count: 857 transactions | Fraud Count: 72 | **Fraud Rate: 8.40%** (almost 10x baseline)

#### The High-Risk Intersection Multiplier:
| Geolocation Status | Legitimate Count | Fraud Count | Total Count | Fraud Rate (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Domestic + Location Match** | 8,224 | 26 | 8,250 | **0.32%** (Ultra-low risk) |
| **Domestic + Location Mismatch** | 729 | 43 | 772 | **5.57%** |
| **Foreign + Location Match** | 840 | 53 | 893 | **5.94%** |
| **Foreign + Location Mismatch** | **56** | **29** | **85** | **34.12%** |

> **Key Finding**: When a transaction is both foreign AND originates from a location mismatched with the cardholder's billing record, the fraud probability jumps to **34.12%** (1 in every 3 transactions).

---

### B. Temporal Concentration (The "Witching Hours" Anomaly)
- Transactions initiated between **00:00 (Midnight) and 03:59 AM** account for **121 out of the 151 total fraud cases (80.13% of all fraud)**:
  - Hour 0 (12 AM - 1 AM): 417 total transactions, 37 fraud -> **8.87% fraud rate**
  - Hour 1 (1 AM - 2 AM): 425 total transactions, 34 fraud -> **8.00% fraud rate**
  - Hour 2 (2 AM - 3 AM): 385 total transactions, 23 fraud -> **5.97% fraud rate**
  - Hour 3 (3 AM - 4 AM): 425 total transactions, 27 fraud -> **6.35% fraud rate**
- Daytime Hours (04:00 to 23:59):
  - 8,348 total transactions, 30 fraud -> **0.36% fraud rate**

---

### C. Device Trust Score & Velocity Surges
1. **Device Trust Score (Engine Reputation Score: 25 to 99)**:
   - Score $\le 30$: 810 transactions, 56 fraud -> **6.91% fraud rate**
   - Score $31 - 60$: 4,041 transactions, 81 fraud -> **2.00% fraud rate**
   - Score $61 - 100$: 5,149 transactions, 14 fraud -> **0.27% fraud rate**
2. **24-Hour Transaction Velocity**:
   - Velocity 0 to 4: 9,453 transactions, 94 fraud -> **0.99% fraud rate**
   - Velocity $\ge 5$: 547 transactions, 57 fraud -> **10.42% fraud rate** (over 10x risk)

---

### D. Merchant Category Breakdown
| Merchant Category | Total Volume ($) | Total Count | Fraud Count | Fraud Rate by Count (%) | Fraud Share of Category ($) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Grocery** | $333,701.32 | 1,944 | 39 | **2.01%** | 2.51% |
| **Food** | $370,559.85 | 2,093 | 35 | **1.67%** | 2.11% |
| **Travel** | $354,801.44 | 1,990 | 29 | **1.46%** | 1.83% |
| **Electronics** | $341,114.73 | 1,923 | 24 | **1.25%** | 1.54% |
| **Clothing** | $359,321.15 | 2,050 | 24 | **1.17%** | 1.34% |

---

## 3. Actionable Business Recommendations & Policy Framework

To balance fraud prevention with customer experience and merchant conversion rates:

1. **Automated Step-Up Authentication (3D Secure / OTP)**:
   - **Trigger**: Any transaction with `foreign_transaction = 1 AND location_mismatch = 1`, OR transactions during the night window (00:00 - 04:00) with `velocity >= 5`.
   - **Rationale**: Replaces outright declines with multi-factor verification, protecting 99.68% of genuine cardholders from wrongful friction while intercepting high-probability fraud.

2. **Dynamic Velocity Caps & Device Fingerprint Cooling**:
   - **Policy**: When a cardholder hits $\ge 5$ transactions in a 24-hour window from an unrecognized device (`device_trust_score <= 30`), temporarily hold automated processing and route to interactive push notification.

3. **Analyst Review Queue Optimization**:
   - Prioritize manual investigator queues by composite risk score rather than transaction amount alone, as 50% of fraudulent transactions are below $120.

