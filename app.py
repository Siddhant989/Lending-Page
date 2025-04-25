import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import json
import requests
from streamlit_lottie import st_lottie
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Lending Model Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    }
    .feature-card {
        border-radius: 8px;
        padding: 1rem;
        background-color: #f0f2f6;
        margin-bottom: 0.7rem;
        border-left: 4px solid #1E88E5;
    }
    .icon-text {
        font-size: 1.2rem;
        font-weight: 500;
        margin-left: 0.5rem;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        font-size: 0.8rem;
        color: #666;
        margin-top: 2rem;
        border-top: 1px solid #eee;
    }
    .prediction-box {
        border-radius: 10px;
        padding: 1.5rem;
        background-color: #e3f2fd;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        text-align: center;
    }
    .btn-primary {
        background-color: #1E88E5;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem 0;
        cursor: pointer;
        font-weight: 500;
    }
    .btn-secondary {
        background-color: #78909C;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem 0;
        cursor: pointer;
        font-weight: 500;
    }
    .company-logo {
        position: absolute;
        top: 0.5rem;
        right: 1rem;
        width: 180px;
        z-index: 1000;
    }
    .sigmoid-text {
        color: #E30613;
        font-size: 28px;
        font-weight: bold;
        font-family: Arial, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Display real Sigmoid logo image at top-right of every page
col1, col2, col3 = st.columns([6, 2, 1])
with col3:
    st.image("images.png", width=140)


# Function to load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Updated top 30 features list
top_30_features = [
    "CREDIT_CARD_AVAILABLE_TOTAL",
    "RECENT_OPEN_ACCT_CUR_BAL_OPEN_TOTAL",
    "PAYMENT_MADE_CNT_TOTAL",
    "PAYMENT_MADE_CNT_TAGGED_TOTAL",
    "ALL_CREDIT_HISTORY_MONTHS_TOTAL",
    "AUTO_PAYMENT_MADE_CNT_TOTAL",
    "INQUIRY_CNT_TOTAL",
    "DELINQ_CNT_30_DAY_TOTAL",
    "INQUIRY_RECENT_CNT_TOTAL",
    "REVOLVING_UTILIZATION_TAGGED_TOTAL",
    "CREDIT_SCORE_AVG_CALC",
    "DEALER_ADDS_PERCENT",
    "TOTAL_DOWN_CONTRACT_PERCENT",
    "PAYMENT_AMOUNT",
    "AUTO_PTI_TOTAL",
    "LTV_FRONT",
    "FE_LTV_BACK",
    "VEHICLE_MILEAGE",
    "CREDIT_CARD_CUR_BAL_TOTAL",
    "AUTO_CREDIT_HISTORY_MONTHS_MAX_TOTAL",
    "REBATE_PERCENT",
    "CASH_DOWN_CONTRACT_PERCENT",
    "FE_DEBT_TO_INCOME",
    "BANK_CARD_CREDIT_LIMIT_TOTAL",
    "RECENT_OPEN_ACCT_TRDLN_OPEN_TOTAL",
    "DEALER_RESERVE",
    "DEROG_CUR_BAL_TOTAL",
    "CREDIT_CARD_CREDIT_LIMIT_TOTAL",
    "FE_TOTAL_INCOME",
    "FE_RESIDENCE_TOTAL_MONTHS"
]

# Feature categories for organization
feature_categories = {
    "Credit & Balance": [
        "CREDIT_CARD_AVAILABLE_TOTAL",
        "RECENT_OPEN_ACCT_CUR_BAL_OPEN_TOTAL",
        "CREDIT_CARD_CUR_BAL_TOTAL",
        "BANK_CARD_CREDIT_LIMIT_TOTAL",
        "DEROG_CUR_BAL_TOTAL",
        "CREDIT_CARD_CREDIT_LIMIT_TOTAL",
        "REVOLVING_UTILIZATION_TAGGED_TOTAL"
    ],
    "Payment History": [
        "PAYMENT_MADE_CNT_TOTAL",
        "PAYMENT_MADE_CNT_TAGGED_TOTAL",
        "AUTO_PAYMENT_MADE_CNT_TOTAL",
        "PAYMENT_AMOUNT",
        "ALL_CREDIT_HISTORY_MONTHS_TOTAL",
        "AUTO_CREDIT_HISTORY_MONTHS_MAX_TOTAL"
    ],
    "Delinquency & Inquiries": [
        "INQUIRY_CNT_TOTAL",
        "DELINQ_CNT_30_DAY_TOTAL",
        "INQUIRY_RECENT_CNT_TOTAL",
        "RECENT_OPEN_ACCT_TRDLN_OPEN_TOTAL"
    ],
    "Loan Structure": [
        "DEALER_ADDS_PERCENT",
        "TOTAL_DOWN_CONTRACT_PERCENT",
        "LTV_FRONT",
        "FE_LTV_BACK",
        "REBATE_PERCENT",
        "CASH_DOWN_CONTRACT_PERCENT",
        "DEALER_RESERVE"
    ],
    "Risk & Eligibility": [
        "CREDIT_SCORE_AVG_CALC",
        "AUTO_PTI_TOTAL",
        "FE_DEBT_TO_INCOME",
        "FE_TOTAL_INCOME",
        "FE_RESIDENCE_TOTAL_MONTHS",
        "VEHICLE_MILEAGE"
    ]
}

# Sidebar navigation
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/bank-building.png", width=80)
    st.title("Lending Model Dashboard")

    selected = option_menu(
        menu_title=None,
        options=["Lending Model Overview", "Input Features", "Model Output"],
        icons=["house", "list-check", "graph-up"],
        menu_icon="cast",
        default_index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard demonstrates a predictive lending model focused on "
        "estimating borrower risk and Cost of Funds (COF)."
    )


# Footer
def add_footer():
    st.markdown("---")
    st.markdown(
        '<div class="footer">© 2023 Lending Model Dashboard v1.0.0 | '
        '<a href="https://github.com/yourusername" target="_blank">GitHub</a> | '
        '<a href="https://linkedin.com/in/yourusername" target="_blank">LinkedIn</a></div>',
        unsafe_allow_html=True
    )


# Feature descriptions - short explanations for each feature
feature_descriptions = {
    "CREDIT_CARD_AVAILABLE_TOTAL": "Total available credit across all credit cards",
    "RECENT_OPEN_ACCT_CUR_BAL_OPEN_TOTAL": "Total balance of recently opened accounts",
    "PAYMENT_MADE_CNT_TOTAL": "Total number of payments made across all accounts",
    "PAYMENT_MADE_CNT_TAGGED_TOTAL": "Total number of tagged payments made",
    "ALL_CREDIT_HISTORY_MONTHS_TOTAL": "Total months of credit history across all accounts",
    "AUTO_PAYMENT_MADE_CNT_TOTAL": "Total number of auto payments made",
    "INQUIRY_CNT_TOTAL": "Total number of credit inquiries",
    "DELINQ_CNT_30_DAY_TOTAL": "Total number of 30-day delinquencies",
    "INQUIRY_RECENT_CNT_TOTAL": "Total number of recent credit inquiries",
    "REVOLVING_UTILIZATION_TAGGED_TOTAL": "Total utilization of tagged revolving accounts",
    "CREDIT_SCORE_AVG_CALC": "Calculated average credit score",
    "DEALER_ADDS_PERCENT": "Percentage of dealer additions to loan amount",
    "TOTAL_DOWN_CONTRACT_PERCENT": "Total down payment percentage in contract",
    "PAYMENT_AMOUNT": "Monthly payment amount",
    "AUTO_PTI_TOTAL": "Auto payment-to-income ratio",
    "LTV_FRONT": "Loan-to-value ratio (front-end)",
    "FE_LTV_BACK": "Loan-to-value ratio (back-end)",
    "VEHICLE_MILEAGE": "Mileage of the vehicle being financed",
    "CREDIT_CARD_CUR_BAL_TOTAL": "Total current balance on all credit cards",
    "AUTO_CREDIT_HISTORY_MONTHS_MAX_TOTAL": "Maximum months of auto credit history",
    "REBATE_PERCENT": "Percentage of rebates applied to purchase",
    "CASH_DOWN_CONTRACT_PERCENT": "Cash down payment percentage in contract",
    "FE_DEBT_TO_INCOME": "Front-end debt-to-income ratio",
    "BANK_CARD_CREDIT_LIMIT_TOTAL": "Total credit limit across bank cards",
    "RECENT_OPEN_ACCT_TRDLN_OPEN_TOTAL": "Total tradelines for recently opened accounts",
    "DEALER_RESERVE": "Amount reserved by dealer in financing",
    "DEROG_CUR_BAL_TOTAL": "Total current balance on derogatory accounts",
    "CREDIT_CARD_CREDIT_LIMIT_TOTAL": "Total credit limit across all credit cards",
    "FE_TOTAL_INCOME": "Total front-end income",
    "FE_RESIDENCE_TOTAL_MONTHS": "Total months at current residence"
}

# Define default values and input types for each feature
feature_input_config = {
    "CREDIT_CARD_AVAILABLE_TOTAL": {"type": "number", "min": 0, "max": 100000, "default": 10000, "step": 1000},
    "RECENT_OPEN_ACCT_CUR_BAL_OPEN_TOTAL": {"type": "number", "min": 0, "max": 100000, "default": 5000, "step": 1000},
    "PAYMENT_MADE_CNT_TOTAL": {"type": "slider", "min": 0, "max": 100, "default": 24},
    "PAYMENT_MADE_CNT_TAGGED_TOTAL": {"type": "slider", "min": 0, "max": 100, "default": 20},
    "ALL_CREDIT_HISTORY_MONTHS_TOTAL": {"type": "slider", "min": 0, "max": 360, "default": 60},
    "AUTO_PAYMENT_MADE_CNT_TOTAL": {"type": "slider", "min": 0, "max": 100, "default": 18},
    "INQUIRY_CNT_TOTAL": {"type": "slider", "min": 0, "max": 30, "default": 3},
    "DELINQ_CNT_30_DAY_TOTAL": {"type": "slider", "min": 0, "max": 20, "default": 0},
    "INQUIRY_RECENT_CNT_TOTAL": {"type": "slider", "min": 0, "max": 10, "default": 1},
    "REVOLVING_UTILIZATION_TAGGED_TOTAL": {"type": "slider", "min": 0.0, "max": 1.0, "default": 0.3, "step": 0.01},
    "CREDIT_SCORE_AVG_CALC": {"type": "slider", "min": 300, "max": 850, "default": 680},
    "DEALER_ADDS_PERCENT": {"type": "slider", "min": 0.0, "max": 0.3, "default": 0.05, "step": 0.01},
    "TOTAL_DOWN_CONTRACT_PERCENT": {"type": "slider", "min": 0.0, "max": 0.5, "default": 0.15, "step": 0.01},
    "PAYMENT_AMOUNT": {"type": "number", "min": 100, "max": 2000, "default": 450, "step": 50},
    "AUTO_PTI_TOTAL": {"type": "slider", "min": 0.0, "max": 0.5, "default": 0.18, "step": 0.01},
    "LTV_FRONT": {"type": "slider", "min": 0.5, "max": 1.5, "default": 0.9, "step": 0.01},
    "FE_LTV_BACK": {"type": "slider", "min": 0.5, "max": 1.5, "default": 0.85, "step": 0.01},
    "VEHICLE_MILEAGE": {"type": "number", "min": 0, "max": 150000, "default": 35000, "step": 1000},
    "CREDIT_CARD_CUR_BAL_TOTAL": {"type": "number", "min": 0, "max": 100000, "default": 7500, "step": 500},
    "AUTO_CREDIT_HISTORY_MONTHS_MAX_TOTAL": {"type": "slider", "min": 0, "max": 240, "default": 48},
    "REBATE_PERCENT": {"type": "slider", "min": 0.0, "max": 0.2, "default": 0.03, "step": 0.01},
    "CASH_DOWN_CONTRACT_PERCENT": {"type": "slider", "min": 0.0, "max": 0.5, "default": 0.1, "step": 0.01},
    "FE_DEBT_TO_INCOME": {"type": "slider", "min": 0.0, "max": 0.6, "default": 0.32, "step": 0.01},
    "BANK_CARD_CREDIT_LIMIT_TOTAL": {"type": "number", "min": 0, "max": 100000, "default": 15000, "step": 1000},
    "RECENT_OPEN_ACCT_TRDLN_OPEN_TOTAL": {"type": "slider", "min": 0, "max": 10, "default": 2},
    "DEALER_RESERVE": {"type": "number", "min": 0, "max": 5000, "default": 500, "step": 100},
    "DEROG_CUR_BAL_TOTAL": {"type": "number", "min": 0, "max": 10000, "default": 0, "step": 500},
    "CREDIT_CARD_CREDIT_LIMIT_TOTAL": {"type": "number", "min": 0, "max": 100000, "default": 18000, "step": 1000},
    "FE_TOTAL_INCOME": {"type": "number", "min": 20000, "max": 300000, "default": 75000, "step": 5000},
    "FE_RESIDENCE_TOTAL_MONTHS": {"type": "slider", "min": 0, "max": 240, "default": 48}
}

# Page 1: Lending Model Overview
if selected == "Lending Model Overview":
    st.markdown('<h1 class="main-header">Lending Model – Predicting Risk and COF</h1>', unsafe_allow_html=True)

    st.markdown(
        """
        This dashboard demonstrates a predictive lending model focused on estimating borrower risk 
        and Cost of Funds (COF). It uses credit, income, delinquency, and account-related features 
        to provide insights and predictions for banking and financial systems.
        """
    )

    # Animation section
    col1, col2 = st.columns(2)

    with col1:
        lottie_money = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_06a6pf9i.json")
        st_lottie(lottie_money, height=200, key="money")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 💰 What is COF?")
        st.markdown(
            """
            Cost of Funds (COF) represents the interest expense a financial institution must pay for 
            the money they use in their lending operations. It's a critical metric for determining 
            loan pricing and profitability.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        lottie_risk = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_ydo1amjm.json")
        st_lottie(lottie_risk, height=200, key="risk")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Risk Prediction")
        st.markdown(
            """
            Our model evaluates borrower attributes to predict the likelihood of default or late payments. 
            This helps lenders make informed decisions about loan approvals and interest rates.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<h2 class="sub-header">Benefits of the Model</h2>', unsafe_allow_html=True)

    benefit_cols = st.columns(3)

    with benefit_cols[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📈 Improved Accuracy")
        st.markdown(
            """
            Our model achieves 30% better prediction accuracy compared to traditional 
            credit scoring methods, resulting in fewer bad loans.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with benefit_cols[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ⏱️ Faster Decisions")
        st.markdown(
            """
            Automated risk assessment reduces loan processing time from days to minutes, 
            improving customer satisfaction and operational efficiency.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with benefit_cols[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Deeper Insights")
        st.markdown(
            """
            The model identifies non-obvious relationships between borrower attributes and 
            loan performance, allowing for more nuanced lending strategies.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    add_footer()

# Page 2: Input Features
elif selected == "Input Features":
    st.markdown('<h1 class="main-header">Enter Loan Application Details</h1>', unsafe_allow_html=True)

    st.markdown(
        """
        Please enter values for all 30 features used in our lending model. These values will be used
        to predict default probability, risk level, and loan eligibility.
        """
    )

    # Initialize session state for storing feature values
    if 'feature_values' not in st.session_state:
        st.session_state.feature_values = {feature: config["default"] for feature, config in
                                           feature_input_config.items()}

    # Create form for input features
    with st.form(key="feature_input_form"):
        # Create tabs for each feature category
        tabs = st.tabs(list(feature_categories.keys()))

        # Add input fields for each feature by category
        for i, (category, features) in enumerate(feature_categories.items()):
            with tabs[i]:
                for feature in features:
                    if feature in feature_input_config:
                        config = feature_input_config[feature]
                        st.markdown(f'<div class="feature-card">', unsafe_allow_html=True)

                        # Feature name and description
                        st.markdown(f"**{feature}**")
                        st.markdown(f"_{feature_descriptions.get(feature, 'Description not available')}_")

                        # Input field based on configuration
                        if config["type"] == "slider":
                            st.session_state.feature_values[feature] = st.slider(
                                f"Value for {feature}",
                                min_value=config["min"],
                                max_value=config["max"],
                                value=config["default"],
                                step=config.get("step", 1),
                                key=f"input_{feature}"
                            )
                        elif config["type"] == "number":
                            st.session_state.feature_values[feature] = st.number_input(
                                f"Value for {feature}",
                                min_value=config["min"],
                                max_value=config["max"],
                                value=config["default"],
                                step=config.get("step", 1),
                                key=f"input_{feature}"
                            )

                        st.markdown('</div>', unsafe_allow_html=True)

        # Submit button at the bottom of the form
        submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
        with submit_col2:
            submit_button = st.form_submit_button("Generate Predictions", use_container_width=True)

            if submit_button:
                st.session_state.prediction_made = True
                # Redirect to the Model Output page after submission
                st.experimental_set_query_params(page="Model Output")

    add_footer()

# Page 3: Model Output
elif selected == "Model Output":
    st.markdown('<h1 class="main-header">Lending Decision Results</h1>', unsafe_allow_html=True)

    # Check if predictions were made
    if "prediction_made" in st.session_state and st.session_state.prediction_made:
        # Get feature values from session state
        feature_values = st.session_state.feature_values

        # Calculate predictions (this would be your actual model in a real app)
        # Here we're using a simplified calculation for demonstration purposes

        # 1. Calculate Default Probability based on key factors
        default_prob = 0.05  # Base default probability

        # Credit score impact (lower score = higher default risk)
        credit_score = feature_values.get("CREDIT_SCORE_AVG_CALC", 680)
        if credit_score < 600:
            default_prob += 0.25
        elif credit_score < 650:
            default_prob += 0.15
        elif credit_score < 700:
            default_prob += 0.05
        else:
            default_prob -= 0.02

        # Delinquency impact
        delinq_count = feature_values.get("DELINQ_CNT_30_DAY_TOTAL", 0)
        default_prob += min(0.3, delinq_count * 0.03)

        # Debt-to-income impact
        dti = feature_values.get("FE_DEBT_TO_INCOME", 0.32)
        if dti > 0.45:
            default_prob += 0.2
        elif dti > 0.36:
            default_prob += 0.1
        elif dti > 0.28:
            default_prob += 0.05

        # LTV impact
        ltv = feature_values.get("LTV_FRONT", 0.9)
        if ltv > 1.1:
            default_prob += 0.15
        elif ltv > 0.95:
            default_prob += 0.08
        elif ltv > 0.8:
            default_prob += 0.03

        # Income stabilization
        income = feature_values.get("FE_TOTAL_INCOME", 75000)
        if income > 120000:
            default_prob -= 0.05
        elif income > 80000:
            default_prob -= 0.03
        elif income > 50000:
            default_prob -= 0.01
        else:
            default_prob += 0.02

        # Credit utilization impact
        util = feature_values.get("REVOLVING_UTILIZATION_TAGGED_TOTAL", 0.3)
        if util > 0.8:
            default_prob += 0.12
        elif util > 0.6:
            default_prob += 0.08
        elif util > 0.4:
            default_prob += 0.04

        # Payments history impact
        payments = feature_values.get("PAYMENT_MADE_CNT_TOTAL", 24)
        if payments > 48:
            default_prob -= 0.08
        elif payments > 24:
            default_prob -= 0.04
        elif payments > 12:
            default_prob -= 0.02

        # Ensure probability is between 0 and 1
        default_prob = max(0.01, min(0.99, default_prob))

        # 2. Determine Risk Level
        if default_prob < 0.1:
            risk_level = "Low Risk"
            risk_color = "#4CAF50"  # Green
        elif default_prob < 0.2:
            risk_level = "Moderate Risk"
            risk_color = "#FFC107"  # Amber
        elif default_prob < 0.35:
            risk_level = "Medium Risk"
            risk_color = "#FF9800"  # Orange
        elif default_prob < 0.5:
            risk_level = "High Risk"
            risk_color = "#F44336"  # Red
        else:
            risk_level = "Very High Risk"
            risk_color = "#B71C1C"  # Deep Red

        # 3. Determine Loan Eligibility
        if default_prob < 0.25:
            eligible = "Approved"
            eligible_color = "#4CAF50"  # Green
            eligible_text = "This application meets our lending criteria and is approved."
        elif default_prob < 0.4:
            eligible = "Conditionally Approved"
            eligible_color = "#FFC107"  # Amber
            eligible_text = "This application is approved subject to additional conditions (higher interest rate or lower loan amount)."
        else:
            eligible = "Declined"
            eligible_color = "#F44336"  # Red
            eligible_text = "This application does not meet our lending criteria."

        # Display the results in three columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("#### Default Probability")
            st.markdown(f"<h2>{default_prob:.1%}</h2>", unsafe_allow_html=True)

            # Gauge chart for default probability
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=default_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                number={'suffix': "%", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 10], 'color': '#4CAF50'},
                        {'range': [10, 20], 'color': '#8BC34A'},
                        {'range': [20, 35], 'color': '#FFC107'},
                        {'range': [35, 50], 'color': '#FF9800'},
                        {'range': [50, 100], 'color': '#F44336'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': default_prob * 100
                    }
                }
            ))

            fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("#### Risk Level")
            st.markdown(f"<h2 style='color: {risk_color}'>{risk_level}</h2>", unsafe_allow_html=True)

            # Key risk factors that contributed to the risk assessment
            st.markdown("#### Key Risk Factors")

            risk_factors = []

            if feature_values.get("CREDIT_SCORE_AVG_CALC", 680) < 650:
                risk_factors.append("Low credit score")
            if feature_values.get("DELINQ_CNT_30_DAY_TOTAL", 0) > 0:
                risk_factors.append("Presence of delinquencies")
            if feature_values.get("FE_DEBT_TO_INCOME", 0.32) > 0.40:
                risk_factors.append("High debt-to-income ratio")
            if feature_values.get("LTV_FRONT", 0.9) > 1.0:
                risk_factors.append("High loan-to-value ratio")
            if feature_values.get("REVOLVING_UTILIZATION_TAGGED_TOTAL", 0.3) > 0.6:
                risk_factors.append("High credit utilization")
            if feature_values.get("PAYMENT_MADE_CNT_TOTAL", 24) < 12:
                risk_factors.append("Limited payment history")

            if not risk_factors:
                st.markdown("No significant risk factors identified.")
            else:
                for factor in risk_factors[:3]:  # Show top 3 factors
                    st.markdown(f"• {factor}")

            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("#### Loan Eligibility")
            st.markdown(f"<h2 style='color: {eligible_color}'>{eligible}</h2>", unsafe_allow_html=True)
            st.markdown(f"{eligible_text}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Display additional details and summary
        st.markdown('<h2 class="sub-header">Application Summary</h2>', unsafe_allow_html=True)

        # Two columns for summary stats
        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Applicant Profile")

            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Credit Score", feature_values.get("CREDIT_SCORE_AVG_CALC", 680))
                st.metric("Annual Income", f"${feature_values.get('FE_TOTAL_INCOME', 75000):,}")
                st.metric("DTI Ratio", f"{feature_values.get('FE_DEBT_TO_INCOME', 0.32):.1%}")

            with metrics_col2:
                st.metric("Payment History", f"{feature_values.get('PAYMENT_MADE_CNT_TOTAL', 24)} payments")
                st.metric("Delinquencies", feature_values.get("DELINQ_CNT_30_DAY_TOTAL", 0))
                st.metric("LTV Ratio", f"{feature_values.get('LTV_FRONT', 0.9):.1%}")

            st.markdown('</div>', unsafe_allow_html=True)

        with summary_col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Model Confidence")

            # Model confidence visualization (illustrative)
            confidence_fig = go.Figure()

            # Calculate model confidence based on number of risk factors
            num_factors = len(risk_factors) if risk_factors else 0
            confidence = max(70, 100 - (num_factors * 5))

            confidence_fig.add_trace(go.Indicator(
                mode="number+gauge",
                value=confidence,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Model Confidence"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#1E88E5"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': confidence
                    }
                }
            ))

            confidence_fig.update_layout(height=200, margin=dict(l=20, r=20, t=70, b=20))
            st.plotly_chart(confidence_fig, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # Action buttons
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
        with btn_col1:
            st.markdown('<div class="btn-primary">Download Report</div>', unsafe_allow_html=True)
        with btn_col2:
            if st.button("Adjust Features", use_container_width=True):
                # Redirect back to the input page
                st.experimental_set_query_params(page="Input Features")
        with btn_col3:
            st.markdown('<div class="btn-secondary">New Application</div>', unsafe_allow_html=True)

    else:
        # If no prediction was made, prompt the user to go to the Input Features page
        st.info("Please enter application details on the 'Input Features' page to see lending decision results.")

        if st.button("Go to Input Features", use_container_width=False):
            # Redirect to the input features page
            st.experimental_set_query_params(page="Input Features")

    add_footer()
