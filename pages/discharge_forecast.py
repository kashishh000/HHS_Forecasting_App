"""
pages/discharge_forecast.py — Discharge Demand Forecast page.
Forecasts future discharge demand using historical data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.data_loader import load_data, prepare_time_series
from utils.ml_models import train_random_forest, rf_forecast_future, train_arima, arima_forecast_future


def render():
    """Render the Discharge Demand Forecast page."""

    st.markdown("# 📉 Discharge Demand Forecast")
    st.markdown("<p style='color:#94A3B8; margin-top:-0.5rem;'>Predict future discharge demand from HHS care facilities using historical discharge patterns.</p>", unsafe_allow_html=True)
    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Load Data ──────────────────────────────────────────────────────────────
    with st.spinner("Loading data..."):
        try:
            df = load_data("data/dataset.csv")
        except FileNotFoundError as e:
            st.error(str(e))
            return

    # ── Controls ───────────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Configuration")

    col1, col2, col3 = st.columns(3)
    with col1:
        horizon_label = st.selectbox(
            "📅 Forecast Horizon",
            options=["7 Days", "14 Days", "30 Days", "60 Days", "90 Days"],
            index=2,
            key="disc_horizon",
        )
        horizon_map = {"7 Days": 7, "14 Days": 14, "30 Days": 30, "60 Days": 60, "90 Days": 90}
        horizon = horizon_map[horizon_label]

    with col2:
        smoothing = st.slider("Rolling smoothing (months)", 1, 6, 3, key="disc_smooth")

    with col3:
        show_trend = st.checkbox("Show Trend Line", value=True, key="disc_trend")
        show_ci = st.checkbox("Show Confidence Band", value=True, key="disc_ci")

    run_btn = st.button("🚀 Generate Discharge Forecast", use_container_width=True)

    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Historical Discharge Analysis ──────────────────────────────────────────
    st.markdown("### 📊 Historical Discharge Analysis")

    dc1, dc2, dc3, dc4 = st.columns(4)
    with dc1:
        st.metric("Total Discharged", f"{df['Discharged'].sum():,}", help="Sum over all months")
    with dc2:
        st.metric("Monthly Average", f"{int(df['Discharged'].mean()):,}")
    with dc3:
        st.metric("Peak Month", f"{int(df['Discharged'].max()):,}", delta=df.loc[df['Discharged'].idxmax(), 'Date'].strftime("%b %Y"))
    with dc4:
        st.metric("Current (Last Month)", f"{int(df['Discharged'].iloc[-1]):,}")

    # Historical chart
    ts_discharge = prepare_time_series(df, "Discharged")
    rolling_avg = ts_discharge.rolling(smoothing).mean()

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(
        x=ts_discharge.index, y=ts_discharge.values,
        name="Monthly Discharged",
        marker=dict(color=ts_discharge.values, colorscale=[[0,"#1E293B"],[0.5,"#10B981"],[1,"#6EE7B7"]], showscale=False),
        hovertemplate="<b>%{x|%b %Y}</b><br>Discharged: <b>%{y:,}</b><extra></extra>",
    ))
    if show_trend:
        fig_hist.add_trace(go.Scatter(
            x=rolling_avg.index, y=rolling_avg.values,
            mode="lines", name=f"{smoothing}-Month Rolling Avg",
            line=dict(color="#EC4899", width=2.5),
            hovertemplate="Rolling Avg: <b>%{y:,.0f}</b><extra></extra>",
        ))

        # Linear trend
        x_num = np.arange(len(ts_discharge))
        z = np.polyfit(x_num, ts_discharge.values, 1)
        p = np.poly1d(z)
        fig_hist.add_trace(go.Scatter(
            x=ts_discharge.index, y=p(x_num),
            mode="lines", name="Long-term Trend",
            line=dict(color="#F59E0B", width=1.5, dash="dot"),
        ))

    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,sans-serif", color="#E2E8F0"),
        height=420,
        title=dict(text="Children Discharged from HHS Care — Historical", font=dict(size=16)),
        legend=dict(bgcolor="rgba(30,41,59,0.8)", bordercolor="rgba(16,185,129,0.3)", borderwidth=1),
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)", title="Children Discharged"),
        barmode="overlay",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Generate Forecast ──────────────────────────────────────────────────────
    if run_btn or "discharge_rf_forecast" not in st.session_state:
        with st.spinner("🔄 Training models on discharge data..."):
            # Random Forest
            rf_result = train_random_forest(df, target_col="Discharged", n_estimators=200)
            rf_fc = rf_forecast_future(rf_result, df, horizon=horizon)
            st.session_state["discharge_rf_result"] = rf_result
            st.session_state["discharge_rf_forecast"] = rf_fc

            # SARIMA
            arima_result = train_arima(ts_discharge, test_size=0.2)
            if arima_result.get("success"):
                ar_fc = arima_forecast_future(arima_result, horizon=horizon)
                st.session_state["discharge_arima_forecast"] = ar_fc
                st.session_state["discharge_arima_result"] = arima_result
            else:
                st.session_state["discharge_arima_forecast"] = None
                st.warning(f"SARIMA fitting issue: {arima_result.get('error','')}")

    rf_fc = st.session_state.get("discharge_rf_forecast")
    ar_fc = st.session_state.get("discharge_arima_forecast")
    rf_result = st.session_state.get("discharge_rf_result")

    # ── Forecast Metrics ───────────────────────────────────────────────────────
    if rf_result:
        st.markdown("### 🎯 Model Performance on Discharge Data")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("MAE", f"{rf_result['mae']:,.1f}", help="Mean Absolute Error")
        with mc2:
            st.metric("RMSE", f"{rf_result['rmse']:,.1f}")
        with mc3:
            st.metric("R² Score", f"{rf_result['r2']:.4f}")

    # ── Discharge Forecast Chart ───────────────────────────────────────────────
    st.markdown(f"### 🔮 {horizon_label} Discharge Demand Forecast")

    historical_slice = ts_discharge.iloc[-36:]
    fig_fc = go.Figure()

    # Historical
    fig_fc.add_trace(go.Scatter(
        x=historical_slice.index, y=historical_slice.values,
        mode="lines", name="Historical",
        line=dict(color="#10B981", width=2),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.06)",
        hovertemplate="<b>%{x|%b %Y}</b><br>Actual: <b>%{y:,}</b><extra></extra>",
    ))

    fig_fc.add_vline(
        x=historical_slice.index[-1],
        line_dash="dash", line_color="rgba(148,163,184,0.4)",
        annotation_text="Forecast Start", annotation_font_color="#94A3B8",
    )

    if rf_fc is not None:
        if show_ci:
            fig_fc.add_trace(go.Scatter(
                x=list(rf_fc["dates"]) + list(rf_fc["dates"][::-1]),
                y=list(rf_fc["upper"]) + list(rf_fc["lower"][::-1]),
                fill="toself", fillcolor="rgba(99,102,241,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="RF Confidence Band", hoverinfo="skip",
            ))
        fig_fc.add_trace(go.Scatter(
            x=rf_fc["dates"], y=rf_fc["values"],
            mode="lines+markers", name="RF Discharge Forecast",
            line=dict(color="#6366F1", width=3),
            marker=dict(size=9, color="#A5B4FC", symbol="circle", line=dict(color="#6366F1", width=2)),
            hovertemplate="<b>%{x|%b %Y}</b><br>RF Forecast: <b>%{y:,.0f}</b><extra></extra>",
        ))

    if ar_fc and "values" in ar_fc:
        if show_ci:
            fig_fc.add_trace(go.Scatter(
                x=list(ar_fc["dates"]) + list(ar_fc["dates"][::-1]),
                y=list(ar_fc["upper"]) + list(ar_fc["lower"][::-1]),
                fill="toself", fillcolor="rgba(236,72,153,0.1)",
                line=dict(color="rgba(0,0,0,0)"),
                name="SARIMA Confidence Band", hoverinfo="skip",
            ))
        fig_fc.add_trace(go.Scatter(
            x=ar_fc["dates"], y=ar_fc["values"],
            mode="lines+markers", name="SARIMA Forecast",
            line=dict(color="#EC4899", width=3),
            marker=dict(size=9, color="#F9A8D4", symbol="diamond", line=dict(color="#EC4899", width=2)),
            hovertemplate="<b>%{x|%b %Y}</b><br>SARIMA Forecast: <b>%{y:,.0f}</b><extra></extra>",
        ))

    fig_fc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,sans-serif", color="#E2E8F0"),
        height=480,
        title=dict(text=f"Discharge Demand Forecast — {horizon_label}", font=dict(size=17)),
        legend=dict(bgcolor="rgba(30,41,59,0.8)", bordercolor="rgba(99,102,241,0.3)", borderwidth=1),
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)", title="Children Discharged"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # ── Discharge Forecast Table ───────────────────────────────────────────────
    if rf_fc is not None:
        st.markdown("### 📋 Discharge Forecast Table")
        fc_df = pd.DataFrame({
            "Forecast Month": [d.strftime("%B %Y") for d in rf_fc["dates"]],
            "RF Discharge Forecast": [f"{v:,.0f}" for v in rf_fc["values"]],
            "Lower Bound": [f"{v:,.0f}" for v in rf_fc["lower"]],
            "Upper Bound": [f"{v:,.0f}" for v in rf_fc["upper"]],
        })
        if ar_fc and "values" in ar_fc:
            min_len = min(len(fc_df), len(ar_fc["values"]))
            fc_df.loc[:min_len-1, "SARIMA Forecast"] = [f"{v:,.0f}" for v in ar_fc["values"][:min_len]]
        st.dataframe(fc_df, use_container_width=True)

        csv_bytes = fc_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Discharge Forecast CSV", csv_bytes, "discharge_forecast.csv", "text/csv")

    # ── Seasonal Discharge Patterns ────────────────────────────────────────────
    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)
    st.markdown("### 📅 Seasonal Discharge Patterns")

    col_a, col_b = st.columns(2)
    with col_a:
        month_avg = df.groupby("Month_Name")["Discharged"].mean()
        month_order = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        month_avg = month_avg.reindex(month_order)
        fig_ma = go.Figure(go.Bar(
            x=month_avg.index, y=month_avg.values,
            marker=dict(color=month_avg.values, colorscale=[[0,"#1E293B"],[0.5,"#10B981"],[1,"#6EE7B7"]], showscale=False),
            hovertemplate="<b>%{x}</b><br>Avg Discharged: <b>%{y:,.0f}</b><extra></extra>",
        ))
        fig_ma.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter,sans-serif", color="#E2E8F0"),
            height=340, title=dict(text="Average Discharged by Month", font=dict(size=14)),
            xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        )
        st.plotly_chart(fig_ma, use_container_width=True)

    with col_b:
        yearly_avg = df.groupby("Year")["Discharged"].mean()
        fig_ya = go.Figure(go.Scatter(
            x=yearly_avg.index, y=yearly_avg.values,
            mode="lines+markers",
            line=dict(color="#10B981", width=3),
            marker=dict(size=10, color="#6EE7B7", symbol="circle", line=dict(color="#10B981", width=2)),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.1)",
            hovertemplate="<b>%{x}</b><br>Avg Discharged: <b>%{y:,.0f}</b><extra></extra>",
        ))
        fig_ya.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter,sans-serif", color="#E2E8F0"),
            height=340, title=dict(text="Average Discharged by Year", font=dict(size=14)),
            xaxis=dict(gridcolor="rgba(148,163,184,0.1)", dtick=1),
            yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        )
        st.plotly_chart(fig_ya, use_container_width=True)
