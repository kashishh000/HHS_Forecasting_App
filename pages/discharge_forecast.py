"""
Discharge Forecast Page Component for HHS UAC Dashboard.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.data_loader import load_raw_data
from utils.preprocessing import create_engineered_features
from utils.ml_models import generate_future_forecast
from utils.helpers import render_header, render_alert, format_number
from utils.charts import apply_dark_layout

def render():
    render_header(
        title="Discharge Velocity & Placement Demand Forecasting",
        subtitle="Sponsor Placement Throughput Modeling, Length of Stay (LOS) Pressure, and Capacity Bottlenecks"
    )

    df_raw = load_raw_data()
    if df_raw.empty:
        st.error("Data unavailable.")
        return

    df_feat = create_engineered_features(df_raw, target_col="Children discharged from HHS Care")

    col_h, col_cap = st.columns([2, 1])
    with col_h:
        horizon = st.selectbox("Select Discharge Projection Window", options=[30, 60, 90, 180], index=1)
    with col_cap:
        capacity_threshold = st.number_input("HHS Shelter Capacity Limit", value=12000, step=500)

    # Generate Discharge Forecast
    disc_forecast = generate_future_forecast(df_feat, model_name="Random Forest", days=horizon, target_col="Children discharged from HHS Care")

    # Generate Care Load Forecast
    df_care_feat = create_engineered_features(df_raw, target_col="Children in HHS Care")
    care_forecast = generate_future_forecast(df_care_feat, model_name="Random Forest", days=horizon, target_col="Children in HHS Care")

    combined_df = pd.DataFrame({
        "Date": disc_forecast["Date"],
        "Projected_Discharges": disc_forecast["Forecast_Care_Load"],
        "Projected_HHS_Care": care_forecast["Forecast_Care_Load"]
    })

    combined_df["Capacity_Utilization_%"] = np.round((combined_df["Projected_HHS_Care"] / capacity_threshold) * 100, 1)

    st.markdown("<br>", unsafe_allow_html=True)

    # Discharge Metrics
    avg_daily_discharge = combined_df["Projected_Discharges"].mean()
    total_projected_discharges = combined_df["Projected_Discharges"].sum()
    max_utilization = combined_df["Capacity_Utilization_%"].max()

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Projected Total Discharges", format_number(total_projected_discharges))
    d2.metric("Daily Placement Pace", f"{int(avg_daily_discharge):,} children/day")
    d3.metric("Max Shelter Utilization", f"{max_utilization:.1f}%")
    if max_utilization > 100:
        d4.metric("Capacity Status", "CRITICAL OVERFLOW", delta="-Capacity Breach", delta_color="inverse")
    elif max_utilization > 85:
        d4.metric("Capacity Status", "HIGH PRESSURE", delta="Warning", delta_color="inverse")
    else:
        d4.metric("Capacity Status", "OPTIMAL", delta="Normal", delta_color="normal")

    st.markdown("<br>", unsafe_allow_html=True)

    if max_utilization > 90:
        render_alert(f"⚠️ <b>Capacity Warning:</b> Projected HHS Care census reaches {max_utilization}% of configured shelter threshold ({capacity_threshold:,} beds). Sponsor vetting throughput must increase to avoid overcrowding.", "warning")

    # Dual Axis Forecast Visual
    st.markdown("### 📊 Care Census vs Discharge Throughput Trajectory")
    fig_dual = go.Figure()

    fig_dual.add_trace(go.Scatter(
        x=combined_df["Date"],
        y=combined_df["Projected_HHS_Care"],
        mode='lines',
        name='Projected Care Census',
        line=dict(color='#3b82f6', width=3)
    ))

    fig_dual.add_trace(go.Bar(
        x=combined_df["Date"],
        y=combined_df["Projected_Discharges"],
        name='Daily Discharges',
        marker_color='#10b981',
        opacity=0.7
    ))

    fig_dual.add_hline(
        y=capacity_threshold,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Max Bed Limit ({capacity_threshold:,})",
        annotation_position="top right"
    )

    apply_dark_layout(fig_dual, title="HHS Care Load vs Placement Outflow Projection", x_title="Date", y_title="Children Count")
    st.plotly_chart(fig_dual, use_container_width=True)

    st.dataframe(combined_df, use_container_width=True)

if __name__ == "__main__" or True:
    render()

