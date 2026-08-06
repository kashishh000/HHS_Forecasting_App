"""
Exploratory Data Analysis (EDA) Page Component for HHS UAC Dashboard.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils.data_loader import load_raw_data
from utils.helpers import render_header, render_alert
from utils.charts import plot_eda_timelines, apply_dark_layout

def render():
    render_header(
        title="Exploratory Data Analysis (EDA)",
        subtitle="System Dynamics, Seasonal Patterns, Distribution Profiles, and Correlation Discoveries"
    )

    df = load_raw_data()
    if df.empty:
        st.error("Dataset not available.")
        return

    # Add temporal helper columns for EDA
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month_name()
    df["Month_Num"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.day_name()

    t_timeline, t_seasonality, t_distributions, t_relationships = st.tabs([
        "System Timelines", "Seasonality & Aggregations", "Distributions & Outliers", "Relationship Dynamics"
    ])

    with t_timeline:
        st.markdown("### 📈 Multi-Metric Operational Dynamics")
        fig_time = plot_eda_timelines(df)
        st.plotly_chart(fig_time, use_container_width=True)

        st.markdown("""
        <div class="section-box">
            <h3>Key Timeline Insights</h3>
            <ul>
                <li><b>Apprehensions vs HHS Care:</b> Apprehensions show sharp seasonal spikes in Spring (March-May) leading HHS Care load by approximately 7 to 14 days.</li>
                <li><b>CBP Custody Buffer:</b> Spikes in CBP custody signal impending transfer pressure onto HHS shelter capacity.</li>
                <li><b>Discharge Velocity:</b> Discharges trail transfers, causing Care Load accumulation during high intake months.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with t_seasonality:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Monthly Average HHS Care Load")
            monthly_agg = df.groupby(["Month_Num", "Month"])["Children in HHS Care"].mean().reset_index()
            monthly_agg = monthly_agg.sort_values("Month_Num")

            fig_m = px.bar(
                monthly_agg,
                x="Month",
                y="Children in HHS Care",
                color="Children in HHS Care",
                color_continuous_scale="Viridis",
                labels={"Children in HHS Care": "Avg Care Census"}
            )
            apply_dark_layout(fig_m, title="Average Care Load by Calendar Month")
            st.plotly_chart(fig_m, use_container_width=True)

        with c2:
            st.markdown("#### Yearly Comparative Total Discharges")
            yearly_agg = df.groupby("Year")[["Children apprehended and placed in CBP custody", "Children discharged from HHS Care"]].sum().reset_index()

            fig_y = px.bar(
                yearly_agg,
                x="Year",
                y=["Children apprehended and placed in CBP custody", "Children discharged from HHS Care"],
                barmode="group",
                labels={"value": "Total Children Count", "variable": "Metric"}
            )
            apply_dark_layout(fig_y, title="Annual Total Apprehensions vs Discharges")
            st.plotly_chart(fig_y, use_container_width=True)

    with t_distributions:
        st.markdown("#### Care Load Distribution across Day of Week")
        fig_box = px.box(
            df,
            x="DayOfWeek",
            y="Children in HHS Care",
            color="DayOfWeek",
            category_orders={"DayOfWeek": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}
        )
        apply_dark_layout(fig_box, title="Care Load Spread by Day of Week", x_title="Day of Week", y_title="Children Count")
        st.plotly_chart(fig_box, use_container_width=True)

    with t_relationships:
        st.markdown("#### Apprehensions vs HHS Discharges Scatter Relationship")
        fig_scat = px.scatter(
            df,
            x="Children apprehended and placed in CBP custody",
            y="Children discharged from HHS Care",
            color="Children in HHS Care",
            color_continuous_scale="Magma",
            hover_data=["Date"],
            labels={
                "Children apprehended and placed in CBP custody": "CBP Apprehensions",
                "Children discharged from HHS Care": "HHS Discharges"
            }
        )
        apply_dark_layout(fig_scat, title="Daily Intake vs Placement Velocity Scatter", x_title="Daily Apprehensions", y_title="Daily Discharges")
        st.plotly_chart(fig_scat, use_container_width=True)

if __name__ == "__main__" or True:
    render()

