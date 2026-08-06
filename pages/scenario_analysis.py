"""
Scenario Analysis Page Component for HHS UAC Dashboard.
"""

import streamlit as st
import pandas as pd
from utils.data_loader import load_raw_data
from utils.preprocessing import create_engineered_features
from utils.ml_models import simulate_scenario
from utils.helpers import render_header, render_alert, format_number
from utils.charts import plot_scenario_comparison

def render():
    render_header(
        title="What-If Operational Scenario Simulator",
        subtitle="Simulate Border Surge Spikes, Discharge Acceleration, and Transfer Bottlenecks"
    )

    df_raw = load_raw_data()
    if df_raw.empty:
        st.error("Data unavailable.")
        return

    df_feat = create_engineered_features(df_raw)

    st.markdown("### 🎛️ Configure What-If Policy & Surge Sliders")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        apprehend_shift = st.slider(
            "CBP Apprehensions Surge",
            min_value=-50, max_value=100, value=20, step=5,
            format="%d%%"
        )

    with col_s2:
        transfer_shift = st.slider(
            "CBP Transfer Velocity",
            min_value=-50, max_value=50, value=10, step=5,
            format="%d%%"
        )

    with col_s3:
        discharge_shift = st.slider(
            "HHS Discharge Velocity",
            min_value=-50, max_value=50, value=-10, step=5,
            format="%d%%"
        )

    with col_s4:
        horizon_sim = st.selectbox("Simulation Window", options=[30, 60, 90, 180], index=1)

    # Run Simulation Engine
    sim_df = simulate_scenario(
        df_feat,
        days=horizon_sim,
        apprehension_pct_change=apprehend_shift,
        discharge_pct_change=discharge_shift,
        transfer_pct_change=transfer_shift
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Calculate Changes vs Current Baseline
    latest_care = df_feat["Children in HHS Care"].iloc[-1]
    final_sim_care = sim_df["Simulated_HHS_Care"].iloc[-1]
    pct_change = ((final_sim_care - latest_care) / latest_care) * 100.0 if latest_care > 0 else 0.0

    daily_net = sim_df["Daily_Net_Change"].iloc[0]

    # Metrics Display
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Care Census", format_number(latest_care))
    m2.metric(f"Simulated Care Census (Day {horizon_sim})", format_number(final_sim_care), delta=f"{pct_change:+.1f}% Care Load Change", delta_color="inverse")
    m3.metric("Simulated Net Daily Inflow", f"{daily_net:+} children/day")

    if daily_net > 50:
        m4.metric("Risk Status", "SEVERE SURGE", delta="High Risk", delta_color="inverse")
    elif daily_net > 0:
        m4.metric("Risk Status", "MODERATE ACCUMULATION", delta="Warning", delta_color="inverse")
    else:
        m4.metric("Risk Status", "STABLE / DECLINING", delta="Normal", delta_color="normal")

    st.markdown("<br>", unsafe_allow_html=True)

    if pct_change > 25:
        render_alert(f"🚨 <b>Surge Alert:</b> Under this scenario, HHS Care Census increases by <b>{pct_change:+.1f}%</b> to <b>{final_sim_care:,} children</b>. Emergency shelter activation or sponsor vetting acceleration required.", "warning")
    elif pct_change < -10:
        render_alert(f"✅ <b>Capacity Relief:</b> Care census drops by <b>{pct_change:.1f}%</b>. Shelter bed utilization eases.", "success")

    # Simulation Chart
    st.markdown("### 📉 Simulated Care Load vs Baseline Trajectory")
    fig_sim = plot_scenario_comparison(df_feat, sim_df, target_col="Children in HHS Care")
    st.plotly_chart(fig_sim, use_container_width=True)

    st.dataframe(sim_df, use_container_width=True)

if __name__ == "__main__" or True:
    render()

