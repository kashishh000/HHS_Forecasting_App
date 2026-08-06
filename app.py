"""
app.py — Main entry point for the HHS UAC Predictive Forecasting Application.
Author: Kashish Thakare | Research Analyst Intern | Unified Mentor
"""

import streamlit as st
import os

# Import all page modules at top level
import importlib

home = importlib.import_module("pages.home")
dataset_explorer = importlib.import_module("pages.dataset_explorer")
eda = importlib.import_module("pages.eda")
predictive_forecasting = importlib.import_module("pages.predictive_forecasting")
future_forecast = importlib.import_module("pages.future_forecast")
discharge_forecast = importlib.import_module("pages.discharge_forecast")
model_comparison = importlib.import_module("pages.model_comparison")
confidence_intervals = importlib.import_module("pages.confidence_intervals")
scenario_analysis = importlib.import_module("pages.scenario_analysis")
bi_dashboard = importlib.import_module("pages.bi_dashboard")
research_summary = importlib.import_module("pages.research_summary")
about = importlib.import_module("pages.about")

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HHS UAC Forecasting | Unified Mentor",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "HHS UAC Predictive Forecasting App — Unified Mentor | Kashish Thakare",
    },
)

# ─── Load Custom CSS ──────────────────────────────────────────────────────────
def load_css(filepath: str):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("assets/style.css")

# Page mapping table using clean ASCII keys to prevent Windows cp1252 UnicodeEncodeError
PAGES = {
    "Home": (home, "🏠"),
    "Dataset Explorer": (dataset_explorer, "📊"),
    "Exploratory Data Analysis": (eda, "🔍"),
    "Predictive Forecasting": (predictive_forecasting, "🤖"),
    "Future Care Load Forecast": (future_forecast, "🔮"),
    "Discharge Demand Forecast": (discharge_forecast, "📉"),
    "Model Comparison": (model_comparison, "⚖️"),
    "Confidence Intervals": (confidence_intervals, "📐"),
    "Scenario Analysis": (scenario_analysis, "🧪"),
    "BI Dashboard": (bi_dashboard, "📈"),
    "Research Summary": (research_summary, "📋"),
    "About": (about, "👤"),
}

# ─── Sidebar Branding & Navigation ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem;">
        <div style="font-size:3rem; line-height:1;">🏛️</div>
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:700;
                    background:linear-gradient(135deg,#6366F1,#EC4899);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text; margin-top:0.5rem;">
            HHS UAC Forecasting
        </div>
        <div style="font-size:0.72rem; color:#94A3B8; margin-top:0.3rem; line-height:1.4;">
            Predictive Intelligence for<br>Proactive Resource Planning
        </div>
    </div>
    <hr style="border:none;border-top:1px solid rgba(99,102,241,0.2);margin:0 0 1rem;">
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown("### Navigation")
    page_options = list(PAGES.keys())
    selected_page_name = st.selectbox(
        "Select Page",
        options=page_options,
        index=0,
        format_func=lambda x: f"{PAGES[x][1]} {x}",
        label_visibility="collapsed",
    )

    st.markdown("""
    <hr style="border:none;border-top:1px solid rgba(99,102,241,0.2);margin:1rem 0;">
    <div style="font-size:0.72rem; color:#64748B; text-align:center; line-height:1.6;">
        <b style="color:#94A3B8;">Author:</b> Kashish Thakare<br>
        <b style="color:#94A3B8;">Role:</b> Research Analyst Intern<br>
        <b style="color:#94A3B8;">Org:</b> Unified Mentor<br>
        <span style="color:#4338CA;">━━━━━━━━━━━━━━━━</span><br>
        Dataset: HHS UAC Program
    </div>
    """, unsafe_allow_html=True)

# ─── Robust Page Routing ───────────────────────────────────────────────────────
page_entry = PAGES.get(selected_page_name)
target_module = page_entry[0] if page_entry else home

# Final render with exception handling (never leaves page blank)
if target_module and hasattr(target_module, "render"):
    try:
        target_module.render()
    except Exception as e:
        st.error(f"Error rendering page '{selected_page_name}': {e}")
        st.exception(e)
else:
    home.render()

# ─── Global Footer ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    🏛️ HHS UAC Predictive Forecasting &nbsp;|&nbsp;
    Author: <b>Kashish Thakare</b> &nbsp;|&nbsp;
    <b>Unified Mentor</b> &nbsp;|&nbsp;
    Built with ❤️ using Streamlit &amp; Plotly
</div>
""", unsafe_allow_html=True)
