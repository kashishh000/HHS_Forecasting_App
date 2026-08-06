"""
HHS UAC Predictive Forecasting & Operational Intelligence Streamlit Application.
Main Entry Point File (app.py)
"""

import streamlit as st
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import APP_TITLE, APP_ICON, STYLE_PATH

# 1. Streamlit Page Configuration
st.set_page_config(
    page_title="HHS UAC Predictive Forecasting Dashboard",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Custom CSS
def inject_custom_css():
    if os.path.exists(STYLE_PATH):
        with open(STYLE_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

inject_custom_css()

# 3. Define Multipage Structure using st.Page and st.navigation
pages = {
    "Overview & Info": [
        st.Page("pages/home.py", title="Home", icon="🏠"),
        st.Page("pages/about.py", title="About Project", icon="ℹ️"),
        st.Page("pages/dataset_explorer.py", title="Dataset Explorer", icon="📋"),
        st.Page("pages/eda.py", title="EDA & System Dynamics", icon="📊"),
    ],
    "Predictive Analytics": [
        st.Page("pages/predictive_forecasting.py", title="Predictive Forecasting", icon="🎯"),
        st.Page("pages/future_forecast.py", title="Future Horizon Forecast", icon="📈"),
        st.Page("pages/discharge_forecast.py", title="Discharge Demand Modeling", icon="🏥"),
        st.Page("pages/confidence_intervals.py", title="Confidence Intervals", icon="🛡️"),
        st.Page("pages/scenario_analysis.py", title="What-If Scenario Simulator", icon="🎛️"),
        st.Page("pages/model_comparison.py", title="Model Benchmark Leaderboard", icon="🏆"),
    ],
    "Executive & Intelligence": [
        st.Page("pages/bi_dashboard.py", title="Power BI Executive Dashboard", icon="📊"),
        st.Page("pages/research_summary.py", title="Research & Policy Summary", icon="📝"),
    ]
}

# 4. Streamlit Navigation Execution
pg = st.navigation(pages)
pg.run()
