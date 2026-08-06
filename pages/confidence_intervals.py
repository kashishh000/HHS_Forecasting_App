"""
Confidence Intervals Page Component for HHS UAC Dashboard.
"""

import streamlit as st
import pandas as pd
from utils.data_loader import load_raw_data
from utils.preprocessing import create_engineered_features
from utils.ml_models import generate_future_forecast, compute_confidence_intervals
from utils.helpers import render_header, format_number
from utils.charts import plot_confidence_intervals

def render():
    render_header(
        title="Statistical Prediction Intervals & Risk Bands",
        subtitle="Quantifying Forecast Uncertainty Across Short and Long Operational Horizons"
    )

    df_raw = load_raw_data()
    if df_raw.empty:
        st.error("Data unavailable.")
        return

    df_feat = create_engineered_features(df_raw)

    c1, c2, c3 = st.columns(3)
    with c1:
        horizon = st.selectbox("Forecast Horizon", options=[30, 60, 90, 180, 365], index=2)
    with c2:
        model_name = st.selectbox("Forecasting Algorithm", options=["Random Forest", "Gradient Boosting", "ARIMA", "SARIMA"], index=0)
    with c3:
        conf_level = st.select_slider("Confidence Level", options=[0.80, 0.90, 0.95, 0.99], value=0.95, format_func=lambda x: f"{int(x*100)}%")

    # Generate Forecast & Confidence Intervals
    raw_forecast = generate_future_forecast(df_feat, model_name=model_name, days=horizon)
    conf_df = compute_confidence_intervals(raw_forecast, confidence_level=conf_level)

    st.markdown("<br>", unsafe_allow_html=True)

    # Uncertainty Statistics
    avg_margin = int(conf_df["Margin_Error"].mean())
    max_margin = int(conf_df["Margin_Error"].max())

    st.columns(4)[0].metric("Selected Confidence", f"{int(conf_level*100)}%")
    st.columns(4)[1].metric("Avg Error Margin", f"±{avg_margin:,}")
    st.columns(4)[2].metric("Max Error Margin (End Date)", f"±{max_margin:,}")
    st.columns(4)[3].metric("Projected Median Care Load", format_number(conf_df["Forecast_Care_Load"].mean()))

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualizing Shaded Prediction Bands
    st.markdown(f"### 🛡️ {int(conf_level*100)}% Prediction Interval Fan Chart ({model_name})")
    fig_ci = plot_confidence_intervals(df_feat, conf_df, target_col="Children in HHS Care")
    st.plotly_chart(fig_ci, use_container_width=True)

    # Interval Table View
    st.markdown("### 📋 Prediction Intervals Schedule Table")
    st.dataframe(
        conf_df.style.format({
            "Forecast_Care_Load": "{:,}",
            "Lower_Bound": "{:,}",
            "Upper_Bound": "{:,}",
            "Margin_Error": "±{:,}"
        }),
        use_container_width=True
    )

if __name__ == "__main__" or True:
    render()

