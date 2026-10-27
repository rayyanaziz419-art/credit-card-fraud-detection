"""
Interactive Credit Card Fraud Detection & Transaction Risk Analytics Dashboard.
Built with Streamlit and Plotly for Executive, Business Analyst, and Data Science stakeholders.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import FraudRiskScorer

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Risk Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .metric-badge-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dataset():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'credit_card_fraud_10k.csv')
    return pd.read_csv(data_path)


@st.cache_data
def load_comparison_data():
    comp_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'model_comparison.csv')
    if os.path.exists(comp_path):
        return pd.read_csv(comp_path)
    return None


@st.cache_resource
def load_scorer():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_fraud_model_pipeline.joblib')
    meta_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_metadata.json')
    if os.path.exists(model_path):
        return FraudRiskScorer(model_path=model_path, meta_path=meta_path)
    return None


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/shield.png", width=64)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Portal View:",
    [
        "1. Executive Overview",
        "2. Transaction Analytics",
        "3. Fraud Investigation",
        "4. Model Performance",
        "5. Real-Time Risk Scoring"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("🛡️ **Enterprise Risk Intelligence Engine**")
st.sidebar.caption("Portfolio Size: **10,000 Transactions**")
st.sidebar.caption("Supervised Target: `is_fraud` (1.51% Imbalance)")

df = load_dataset()
scorer = load_scorer()

# ==============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==============================================================================
if page == "1. Executive Overview":
    st.markdown('<div class="main-header">Executive Risk & Fraud Analytics Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Executive KPI scorecard, financial risk exposure, and portfolio telemetry.</div>', unsafe_allow_html=True)

    # Core Financial & Volume KPIs
    total_txns = len(df)
    fraud_txns = int((df['is_fraud'] == 1).sum())
    fraud_pct = (fraud_txns / total_txns) * 100
    
    total_vol = df['amount'].sum()
    fraud_vol = df[df['is_fraud'] == 1]['amount'].sum()
    fraud_vol_share = (fraud_vol / total_vol) * 100
    avg_fraud_amt = df[df['is_fraud'] == 1]['amount'].mean()
    avg_legit_amt = df[df['is_fraud'] == 0]['amount'].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Volume Processed</div>
            <div class="kpi-value">${total_vol:,.2f}</div>
            <span style="color:#64748B; font-size:0.85rem;">10,000 Total Transactions</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Fraud-Labeled Transactions</div>
            <div class="kpi-value">{fraud_txns:,} <span style="font-size:1.1rem; color:#DC2626;">({fraud_pct:.2f}%)</span></div>
            <span style="color:#64748B; font-size:0.85rem;">9,849 Legitimate (98.49%)</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Fraud Exposure Volume</div>
            <div class="kpi-value">${fraud_vol:,.2f} <span style="font-size:1.1rem; color:#DC2626;">({fraud_vol_share:.2f}%)</span></div>
            <span style="color:#64748B; font-size:0.85rem;">Gross Exposed Payments</span>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg Transaction Size</div>
            <div class="kpi-value">${avg_fraud_amt:.2f} <span style="font-size:0.9rem; color:#64748B;">vs ${avg_legit_amt:.2f}</span></div>
            <span style="color:#64748B; font-size:0.85rem;">Fraud Avg vs Legit Avg</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Executive charts
    row_c1, row_c2 = st.columns([1, 1])
    with row_c1:
        # Category volume vs fraud share
        cat_df = df.groupby('merchant_category').agg(
            total_count=('amount', 'count'),
            fraud_count=('is_fraud', 'sum'),
            total_amt=('amount', 'sum'),
            fraud_amt=('amount', lambda s: s[df.loc[s.index, 'is_fraud'] == 1].sum())
        ).reset_index()
        cat_df['fraud_rate_%'] = (cat_df['fraud_count'] / cat_df['total_count']) * 100

        fig_cat = px.bar(
            cat_df,
            x='merchant_category',
            y='total_count',
            color='fraud_rate_%',
            color_continuous_scale='Reds',
            title='Transaction Distribution by Merchant Category (Colored by Fraud Rate %)',
            labels={'merchant_category': 'Category', 'total_count': 'Total Transactions', 'fraud_rate_%': 'Fraud Rate %'}
        )
        fig_cat.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cat, use_container_width=True)

    with row_c2:
        # Hour of day fraud trend
        hr_df = df.groupby('transaction_hour').agg(
            total=('is_fraud', 'count'),
            fraud=('is_fraud', 'sum')
        ).reset_index()
        hr_df['fraud_rate_%'] = (hr_df['fraud_count'] if 'fraud_count' in hr_df else hr_df['fraud']) / hr_df['total'] * 100

        fig_hr = px.line(
            hr_df,
            x='transaction_hour',
            y='fraud_rate_%',
            markers=True,
            title='Fraud Rate % by Hour of Day (Notice Sharp Spike Between 00:00 - 04:00)',
            labels={'transaction_hour': 'Hour of Day (0-23)', 'fraud_rate_%': 'Fraud Rate (%)'}
        )
        fig_hr.add_vrect(x0=-0.5, x1=3.5, fillcolor="red", opacity=0.15, line_width=0, annotation_text="Night Window (80% of Frauds)")
        fig_hr.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_hr, use_container_width=True)

    st.info("""
    **Executive Risk Summary:**
    1. **Extreme Nighttime Concentration**: Transactions between 00:00 and 03:59 represent **80.1%** of all fraud occurrences in the portfolio.
    2. **Cross-Border Multiplier**: Foreign transactions with a billing location mismatch display an alarming **34.12%** fraud rate (over 20x average).
    3. **Regulatory Distinction**: Labeled fraud volume ($32,643.63) represents gross monetary exposure, not confirmed net bank loss, as chargeback clawbacks and pre-auth declines prevent total loss.
    """)

# ==============================================================================
# PAGE 2: TRANSACTION ANALYTICS
# ==============================================================================
elif page == "2. Transaction Analytics":
    st.markdown('<div class="main-header">Exploratory Transaction Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep-dive into distributions, merchant trends, and customer transaction habits.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Monetary Distributions", "Merchant & Channel Analysis", "Temporal Behaviors"])

    with tab1:
        st.subheader("Transaction Amount Distributions")
        c1, c2 = st.columns(2)
        with c1:
            fig_box = px.box(
                df,
                x='is_fraud',
                y='amount',
                color='is_fraud',
                title='Transaction Amount Boxplot: Legit (0) vs Fraud (1)',
                labels={'is_fraud': 'Fraud Label (0 = Legit, 1 = Fraud)', 'amount': 'Amount ($)'},
                color_discrete_map={0: '#3B82F6', 1: '#EF4444'}
            )
            fig_box.update_layout(height=380)
            st.plotly_chart(fig_box, use_container_width=True)

        with c2:
            fig_hist = px.histogram(
                df[df['amount'] < 600],
                x='amount',
                color='is_fraud',
                barmode='overlay',
                nbins=50,
                title='Amount Distribution (Zoomed < $600)',
                labels={'amount': 'Amount ($)', 'is_fraud': 'Fraud'},
                color_discrete_map={0: '#3B82F6', 1: '#EF4444'}
            )
            fig_hist.update_layout(height=380)
            st.plotly_chart(fig_hist, use_container_width=True)

        st.caption("Observation: Both legitimate and fraudulent payments are heavily right-skewed with medians around $120. Monetary amount alone is not a sufficient differentiator.")

    with tab2:
        st.subheader("Cross-Border & Location Anomaly Breakdown")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            for_df = df.groupby(['foreign_transaction', 'is_fraud']).size().reset_index(name='count')
            fig_for = px.bar(
                for_df,
                x='foreign_transaction',
                y='count',
                color='is_fraud',
                barmode='group',
                title='Domestic (0) vs Foreign (1) Transaction Counts',
                labels={'foreign_transaction': 'Foreign Transaction Flag', 'count': 'Transaction Count', 'is_fraud': 'Fraud'},
                color_discrete_map={0: '#3B82F6', 1: '#EF4444'}
            )
            st.plotly_chart(fig_for, use_container_width=True)

        with col_m2:
            loc_df = df.groupby(['location_mismatch', 'is_fraud']).size().reset_index(name='count')
            fig_loc = px.bar(
                loc_df,
                x='location_mismatch',
                y='count',
                color='is_fraud',
                barmode='group',
                title='Location Match (0) vs Mismatch (1) Transaction Counts',
                labels={'location_mismatch': 'Location Mismatch Flag', 'count': 'Transaction Count', 'is_fraud': 'Fraud'},
                color_discrete_map={0: '#3B82F6', 1: '#EF4444'}
            )
            st.plotly_chart(fig_loc, use_container_width=True)

    with tab3:
        st.subheader("Cardholder Age vs Transaction Velocity")
        fig_scatter = px.scatter(
            df.sample(2000, random_state=42),
            x='cardholder_age',
            y='velocity_last_24h',
            color='is_fraud',
            title='Cardholder Age vs 24-Hour Velocity (2,000 Sample Points)',
            labels={'cardholder_age': 'Cardholder Age (Years)', 'velocity_last_24h': 'Transactions in Last 24h', 'is_fraud': 'Fraud'},
            color_discrete_map={0: '#94A3B8', 1: '#DC2626'},
            opacity=0.7
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption(r"Observation: High velocity ($\ge 5$) shows notable clusters of fraud irrespective of cardholder age.")

# ==============================================================================
# PAGE 3: FRAUD INVESTIGATION
# ==============================================================================
elif page == "3. Fraud Investigation":
    st.markdown('<div class="main-header">Interactive Fraud Investigation Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Filter transaction cohorts, inspect micro-segments, and evaluate small-sample signals.</div>', unsafe_allow_html=True)

    st.sidebar.subheader("Filter Portfolio Cohort:")
    selected_cats = st.sidebar.multiselect(
        "Merchant Category:",
        options=sorted(df['merchant_category'].unique()),
        default=sorted(df['merchant_category'].unique())
    )
    
    hour_range = st.sidebar.slider("Transaction Hour Range:", 0, 23, (0, 23))
    
    foreign_filter = st.sidebar.selectbox("Foreign Transaction Status:", ["All", "Domestic Only (0)", "Foreign Only (1)"])
    mismatch_filter = st.sidebar.selectbox("Location Mismatch Status:", ["All", "Match (0)", "Mismatch (1)"])

    # Apply filters
    filtered_df = df[
        (df['merchant_category'].isin(selected_cats)) &
        (df['transaction_hour'].between(hour_range[0], hour_range[1]))
    ]
    if foreign_filter == "Domestic Only (0)":
        filtered_df = filtered_df[filtered_df['foreign_transaction'] == 0]
    elif foreign_filter == "Foreign Only (1)":
        filtered_df = filtered_df[filtered_df['foreign_transaction'] == 1]

    if mismatch_filter == "Match (0)":
        filtered_df = filtered_df[filtered_df['location_mismatch'] == 0]
    elif mismatch_filter == "Mismatch (1)":
        filtered_df = filtered_df[filtered_df['location_mismatch'] == 1]

    # Cohort KPIs
    cohort_total = len(filtered_df)
    cohort_fraud = int((filtered_df['is_fraud'] == 1).sum()) if cohort_total > 0 else 0
    cohort_rate = (cohort_fraud / cohort_total * 100) if cohort_total > 0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Filtered Transactions", f"{cohort_total:,}")
    c2.metric("Filtered Fraud Cases", f"{cohort_fraud:,}")
    c3.metric("Cohort Fraud Rate", f"{cohort_rate:.2f}%")

    if cohort_total < 50:
        st.warning(f"⚠️ Small Sample Warning: Only {cohort_total} transactions match this filter. Inferences should be treated with caution.")

    st.markdown("### Cohort Breakdown by Device Trust Score")
    if cohort_total > 0:
        filtered_df['trust_bracket'] = pd.cut(
            filtered_df['device_trust_score'],
            bins=[0, 30, 60, 80, 100],
            labels=['Critical (0-30)', 'Low (31-60)', 'Medium (61-80)', 'High Trust (81-100)']
        )
        trust_agg = filtered_df.groupby('trust_bracket', observed=False).agg(
            total=('is_fraud', 'count'),
            fraud=('is_fraud', 'sum')
        ).reset_index()
        trust_agg['fraud_rate_%'] = np.where(trust_agg['total'] > 0, (trust_agg['fraud'] / trust_agg['total']) * 100, 0.0)

        fig_trust = px.bar(
            trust_agg,
            x='trust_bracket',
            y='total',
            color='fraud_rate_%',
            text='fraud',
            color_continuous_scale='OrRd',
            title='Transaction Volume and Fraud Rate Across Device Trust Brackets',
            labels={'trust_bracket': 'Device Trust Score Bracket', 'total': 'Total Transactions', 'fraud_rate_%': 'Fraud Rate (%)'}
        )
        st.plotly_chart(fig_trust, use_container_width=True)

        st.markdown("### Sample Suspicious Records")
        suspicious_sample = filtered_df[filtered_df['is_fraud'] == 1].head(10)
        st.dataframe(suspicious_sample, use_container_width=True)
    else:
        st.info("No transactions match the selected criteria.")

# ==============================================================================
# PAGE 4: MODEL PERFORMANCE
# ==============================================================================
elif page == "4. Model Performance":
    st.markdown('<div class="main-header">Machine Learning Model Evaluation & Benchmark</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rigorous, leakage-safe model comparison, ROC/PR curves, and threshold tuning.</div>', unsafe_allow_html=True)

    df_comp = load_comparison_data()
    
    if df_comp is not None:
        st.subheader("1. Cross-Validation & Held-Out Test Evaluation Matrix")
        st.dataframe(
            df_comp.style.format({
                "CV_PR_AUC": "{:.4f}",
                "pr_auc": "{:.4f}",
                "CV_ROC_AUC": "{:.4f}",
                "roc_auc": "{:.4f}",
                "recall": "{:.4f}",
                "precision": "{:.4f}",
                "f1_score": "{:.4f}"
            }),
            use_container_width=True
        )

    st.subheader("2. Diagnostic Model Curves & Explanations")
    col1, col2 = st.columns(2)
    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
    
    pr_fig_path = os.path.join(fig_dir, 'precision_recall_curves.png')
    roc_fig_path = os.path.join(fig_dir, 'roc_curves.png')
    cm_fig_path = os.path.join(fig_dir, 'confusion_matrix_optimal.png')
    fi_fig_path = os.path.join(fig_dir, 'feature_importance.png')

    with col1:
        if os.path.exists(pr_fig_path):
            st.image(pr_fig_path, caption="Precision-Recall Curves (Primary Imbalanced Metric)")
        if os.path.exists(cm_fig_path):
            st.image(cm_fig_path, caption="Confusion Matrix at Tuned Threshold")

    with col2:
        if os.path.exists(roc_fig_path):
            st.image(roc_fig_path, caption="Receiver Operating Characteristic (ROC) Curves")
        if os.path.exists(fi_fig_path):
            st.image(fi_fig_path, caption="Permutation Feature Importance (Test Set)")

    st.markdown("""
    ### Why Accuracy is a Dangerous Metric in Fraud Detection
    - In our dataset, predicting `0` (legitimate) for 100% of transactions yields **98.49% Accuracy**, but **catches zero fraud (0% Recall)**.
    - **PR-AUC (Average Precision)** is the Gold Standard: It evaluates the precision-recall trade-off specifically focusing on the rare positive class, completely unpolluted by the overwhelming count of true negatives.
    """)

# ==============================================================================
# PAGE 5: REAL-TIME RISK SCORING
# ==============================================================================
elif page == "5. Real-Time Risk Scoring":
    st.markdown('<div class="main-header">Transaction Risk Scoring & Triage Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Simulate live payments, score incoming transaction payloads, and inspect risk drivers.</div>', unsafe_allow_html=True)

    if scorer is None:
        st.error("Model pipeline artifact not found! Please run `python src/train.py` to train and save the champion model.")
    else:
        st.write("Enter transaction attributes below to evaluate immediate fraud risk:")
        
        with st.form("risk_scoring_form"):
            r1, r2, r3 = st.columns(3)
            with r1:
                tx_id = st.number_input("Transaction ID", min_value=1, value=10001, step=1)
                amount = st.number_input("Amount ($ USD)", min_value=0.0, value=250.00, step=10.0)
                hour = st.slider("Transaction Hour (24h Clock)", min_value=0, max_value=23, value=2)
            with r2:
                merchant = st.selectbox("Merchant Category", ["Clothing", "Electronics", "Food", "Grocery", "Travel"])
                foreign = st.selectbox("Foreign / Cross-Border Transaction?", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
                location_mismatch = st.selectbox("Billing Location Mismatch?", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
            with r3:
                device_score = st.slider("Device Trust Score", min_value=0, max_value=100, value=28)
                velocity = st.number_input("24h Transaction Velocity", min_value=0, max_value=50, value=6, step=1)
                age = st.number_input("Cardholder Age", min_value=18, max_value=100, value=35, step=1)

            submit_btn = st.form_submit_button("⚡ Evaluate Transaction Risk")

        if submit_btn:
            payload = {
                "transaction_id": int(tx_id),
                "amount": float(amount),
                "transaction_hour": int(hour),
                "merchant_category": str(merchant),
                "foreign_transaction": int(foreign),
                "location_mismatch": int(location_mismatch),
                "device_trust_score": int(device_score),
                "velocity_last_24h": int(velocity),
                "cardholder_age": int(age)
            }
            
            result = scorer.score_single(payload)
            prob = result['fraud_probability']
            tier = result['risk_tier']
            action = result['recommended_action']
            signals = result['contributing_signals']

            st.markdown("---")
            st.subheader("Inference Result & Decision Recommendation")
            
            res_c1, res_c2, res_c3 = st.columns(3)
            with res_c1:
                st.metric("Fraud Risk Probability", f"{prob*100:.2f}%")
            with res_c2:
                if "CRITICAL" in tier or "HIGH" in tier:
                    st.markdown(f'<div class="metric-badge-high" style="font-size:1.2rem; text-align:center; padding:10px;">{tier}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-badge-low" style="font-size:1.2rem; text-align:center; padding:10px;">{tier}</div>', unsafe_allow_html=True)
            with res_c3:
                st.metric("Applied Threshold", f"{result['threshold_applied']:.2f}")

            st.markdown(f"**Recommended Workflow Action:** `{action}`")

            st.markdown("#### Primary Model Signals Detected:")
            for s in signals:
                st.markdown(f"- ⚠️ {s}")

            st.caption(f"**Audit Disclaimer**: {result['disclaimer']}")
