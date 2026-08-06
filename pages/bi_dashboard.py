"""
BI Dashboard Page Component for HHS UAC Application (Power BI Aesthetic).
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils.data_loader import load_raw_data, get_dataset_summary
from utils.helpers import render_header, render_kpi_card, format_number
from utils.charts import plot_bi_kpi_gauge, apply_dark_layout

def render():
    render_header(
        title="Power BI Style Executive Intelligence Dashboard",
        subtitle="Real-Time Operational Cockpit, Facility Occupancy Gauges, and System Balance Metrics"
    )

    df_raw = load_raw_data()
    if df_raw.empty:
        st.error("Data missing.")
        return

    summary = get_dataset_summary(df_raw)

    # 1. Top Executive Control Toolbar
    st.markdown("### 🎛️ Executive Filter Bar")
    c_f1, c_f2 = st.columns([2, 1])

    with c_f1:
        date_sel = st.slider(
            "Historical Lookback Window (Days)",
            min_value=30, max_value=len(df_raw), value=180, step=30
        )
    with c_f2:
        capacity_target = st.number_input("Configured Bed Capacity Target", value=11000, step=500)

    df_view = df_raw.tail(date_sel).copy()

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Executive KPI Summary Strip
    k1, k2, k3, k4 = st.columns(4)
    latest_care = summary["latest_care"]
    latest_app = summary["latest_apprehended"]
    latest_disc = summary["latest_discharged"]
    latest_trans = summary["latest_transferred"]

    net_flow = latest_trans - latest_disc

    k1.metric("Active HHS Care Census", format_number(latest_care), delta=f"{(latest_care/capacity_target)*100:.1f}% Capacity")
    k2.metric("Daily Border Intake (CBP)", format_number(latest_app), delta="Apprehensions")
    k3.metric("Daily HHS Discharges", format_number(latest_disc), delta="Sponsor Releases")
    k4.metric("Net Daily Care Flow", f"{net_flow:+} children/day", delta="Intake - Outflow", delta_color="inverse" if net_flow > 0 else "normal")

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Middle BI Visual Row: Gauge + Donut + Bar
    col_g, col_pie = st.columns([1.2, 1.0])

    with col_g:
        st.markdown("#### Facility Capacity Occupancy Gauge")
        fig_gauge = plot_bi_kpi_gauge(latest_care, target_val=capacity_target, title_text="Care Census vs Max Beds")
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_pie:
        st.markdown("#### System Velocity Distribution (Recent 30 Days)")
        last_30 = df_raw.tail(30)
        tot_app = last_30["Children apprehended and placed in CBP custody"].sum()
        tot_trans = last_30["Children transferred out of CBP custody"].sum()
        tot_disc = last_30["Children discharged from HHS Care"].sum()

        pie_df = pd.DataFrame({
            "Stage": ["Border Apprehensions", "CBP Transfers to HHS", "Discharges to Sponsors"],
            "Count": [tot_app, tot_trans, tot_disc]
        })

        fig_pie = px.pie(
            pie_df,
            values="Count",
            names="Stage",
            hole=0.5,
            color_discrete_sequence=["#ef4444", "#3b82f6", "#10b981"]
        )
        apply_dark_layout(fig_pie, title="")
        fig_pie.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=250)
        st.plotly_chart(fig_pie, use_container_width=True)

    # 4. Operational Trend Bar Chart
    st.markdown("### 📊 Intake vs Discharge Balance Trend")
    fig_bar = go.Figure()

    fig_bar.add_trace(go.Bar(
        x=df_view["Date"],
        y=df_view["Children transferred out of CBP custody"],
        name='Transfers into HHS Shelter',
        marker_color='#3b82f6'
    ))

    fig_bar.add_trace(go.Bar(
        x=df_view["Date"],
        y=df_view["Children discharged from HHS Care"],
        name='Discharges out of HHS Shelter',
        marker_color='#10b981'
    ))

    apply_dark_layout(fig_bar, title="Daily Intake vs Placement Outflow", x_title="Date", y_title="Children Count")
    fig_bar.update_layout(barmode='group')
    st.plotly_chart(fig_bar, use_container_width=True)

if __name__ == "__main__" or True:
    render()

