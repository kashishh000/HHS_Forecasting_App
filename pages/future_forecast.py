"""
Future Forecast Page Component for HHS UAC Dashboard.
"""

import streamlit as st
import pandas as pd
from utils.data_loader import load_raw_data
from utils.preprocessing import create_engineered_features
from utils.ml_models import generate_future_forecast, compute_confidence_intervals
from utils.helpers import render_header, render_alert, create_download_button, format_number
from utils.charts import plot_future_forecast, plot_confidence_intervals

def render():
    render_header(
        title="Multi-Horizon Future Forecast Generator",
        subtitle="Project Children in HHS Care up to 365 Days Ahead with Statistical Uncertainty Bounds"
    )

    df_raw = load_raw_data()
    if df_raw.empty:
        st.error("Dataset missing.")
        return

    df_feat = create_engineered_features(df_raw)

    # Control Parameters
    c_horizon, c_model, c_conf = st.columns([1.5, 1.5, 1])

    with c_horizon:
        horizon_days = st.selectbox(
            "Select Forecast Horizon",
            options=[30, 60, 90, 180, 365],
            index=2,
            format_func=lambda x: f"{x} Days Ahead ({x//30} Months)" if x < 365 else "365 Days Ahead (1 Year)"
        )

    with c_model:
        model_name = st.selectbox(
            "Forecast Model Engine",
            options=["Random Forest", "Gradient Boosting", "ARIMA", "SARIMA"],
            index=0
        )

    with c_conf:
        conf_level = st.selectbox(
            "Confidence Band",
            options=[0.80, 0.90, 0.95],
            index=2,
            format_func=lambda x: f"{int(x*100)}% CI"
        )

    # Run Future Forecast Generation
    with st.spinner(f"Generating {horizon_days}-day future projection using {model_name}..."):
        raw_forecast = generate_future_forecast(df_feat, model_name=model_name, days=horizon_days)
        conf_forecast = compute_confidence_intervals(raw_forecast, confidence_level=conf_level)

    st.markdown("<br>", unsafe_allow_html=True)

    # Summary Metrics of Future Projection
    last_care = df_feat["Children in HHS Care"].iloc[-1]
    peak_care = conf_forecast["Forecast_Care_Load"].max()
    end_care = conf_forecast["Forecast_Care_Load"].iloc[-1]
    delta_care = end_care - last_care

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Current Care Census", format_number(last_care))
    f2.metric("Projected Peak Load", format_number(peak_care))
    f3.metric(f"Projected Load at Day {horizon_days}", format_number(end_care), delta=f"{delta_care:+} children")
    f4.metric("Avg Prediction Uncertainty", f"±{int(conf_forecast['Margin_Error'].mean()):,} children")

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Chart View
    st.markdown(f"### 📈 Projected Care Load ({horizon_days}-Day Horizon)")
    fig_ci = plot_confidence_intervals(df_feat, conf_forecast, target_col="Children in HHS Care")
    st.plotly_chart(fig_ci, use_container_width=True)

    # Detailed Forecast Table and Export
    st.markdown("### 📋 Detailed Multi-Horizon Forecast Schedule")
    create_download_button(
        conf_forecast,
        filename=f"hhs_uac_{horizon_days}d_future_forecast.csv",
        button_text=f"📥 Download {horizon_days}-Day Forecast Schedule (CSV)"
    )
    st.dataframe(
        conf_forecast.style.format({
            "Forecast_Care_Load": "{:,}",
            "Lower_Bound": "{:,}",
            "Upper_Bound": "{:,}",
            "Margin_Error": "±{:,}"
        }),
        use_container_width=True
    )

if __name__ == "__main__" or True:
    render()

