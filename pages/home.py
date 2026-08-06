"""
Home Page Component for HHS UAC Forecasting Application.
"""

import streamlit as st
from utils.data_loader import load_raw_data, get_dataset_summary
from utils.helpers import render_header, render_kpi_card, render_alert, format_number
from utils.charts import plot_eda_timelines

def render():
    render_header(
        title="Predictive Forecasting of Care Load & Placement Demand",
        subtitle="Operational Intelligence & Machine Learning Support System for HHS Unaccompanied Alien Children Program"
    )

    df = load_raw_data()
    if df.empty:
        st.error("Failed to load dataset.")
        return

    summary = get_dataset_summary(df)

    # 1. Live KPI Summary Cards Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            title="Children in HHS Care",
            value=format_number(summary["latest_care"]),
            subtitle=f"As of {summary['latest_date']}",
            delta_badge="Live Load"
        )
    with c2:
        render_kpi_card(
            title="Daily CBP Apprehensions",
            value=format_number(summary["latest_apprehended"]),
            subtitle="Border Inflow Rate",
            delta_badge="Intake"
        )
    with c3:
        render_kpi_card(
            title="Daily HHS Discharges",
            value=format_number(summary["latest_discharged"]),
            subtitle="Sponsor Placement Outflow",
            delta_badge="Outflow"
        )
    with c4:
        render_kpi_card(
            title="7-Day Avg HHS Care",
            value=format_number(summary["avg_7d_care"]),
            subtitle="Rolling Operational Trend",
            delta_badge="Baseline"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Executive Overview Section
    col_left, col_right = st.columns([1.6, 1.0])

    with col_left:
        st.markdown("""
        <div class="section-box">
            <h3>Executive Summary & Program Objectives</h3>
            <p style='line-height: 1.6; color: #d1d5db;'>
                The <b>HHS Unaccompanied Alien Children (UAC) Program</b> provides shelter, care, and placement for vulnerable minors transferred from Department of Homeland Security (CBP) custody to the Department of Health and Human Services (HHS) Office of Refugee Resettlement (ORR).
            </p>
            <p style='line-height: 1.6; color: #d1d5db;'>
                This Decision Support System leverages machine learning models (Random Forest, Gradient Boosting) and time-series forecasting algorithms (ARIMA, SARIMA) to predict placement demand, anticipate capacity bottlenecks, model discharge velocities, and support strategic resource allocation.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Quick Highlights
        render_alert("💡 <b>Key Capability:</b> Forecast care load up to 365 days into the future with 95% confidence bounds.", "info")
        render_alert("⚡ <b>What-If Simulator:</b> Model policy changes in apprehensions and discharge velocity dynamically.", "success")

    with col_right:
        st.markdown("""
        <div class="section-box">
            <h3>Dataset Quick Spec</h3>
            <ul style='line-height: 1.8; color: #d1d5db; padding-left: 1.2rem;'>
                <li><b>Total Observations:</b> """ + str(summary['total_days']) + """ daily records</li>
                <li><b>Date Range:</b> """ + summary['start_date'] + """ to """ + summary['end_date'] + """</li>
                <li><b>Primary Features:</b> Apprehensions, CBP Custody, Transfers, HHS Care, Discharges</li>
                <li><b>Frequency:</b> Daily granularity</li>
                <li><b>Data Quality:</b> 100% verified & imputed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # 3. Macro Time Series Snapshot
    st.markdown("### 📊 Historical System Dynamics")
    fig = plot_eda_timelines(df)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__" or True:
    render()

