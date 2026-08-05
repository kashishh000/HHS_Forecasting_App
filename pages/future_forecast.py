"""
pages/future_forecast.py — Future Care Load Forecast page.
7, 14, 30, 60-day horizon forecasts using Random Forest and SARIMA.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.data_loader import load_data, prepare_time_series
from utils.ml_models import train_random_forest, rf_forecast_future, train_arima, arima_forecast_future
from utils.charts import plot_future_forecast


def render():
    """Render the Future Care Load Forecast page."""

    st.markdown("# 🔮 Future Care Load Forecast")
    st.markdown("<p style='color:#94A3B8; margin-top:-0.5rem;'>Generate multi-horizon forecasts for Children in HHS Care using trained ML models.</p>", unsafe_allow_html=True)
    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Load Data ──────────────────────────────────────────────────────────────
    with st.spinner("Loading data..."):
        try:
            df = load_data("data/dataset.csv")
        except FileNotFoundError as e:
            st.error(str(e))
            return

    # ── Controls ───────────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Forecast Configuration")

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)

    with ctrl_col1:
        horizon_label = st.selectbox(
            "📅 Forecast Horizon",
            options=["7 Days", "14 Days", "30 Days", "60 Days", "90 Days"],
            index=2,
            help="Select the number of days to forecast into the future.",
        )
        horizon_map = {"7 Days": 7, "14 Days": 14, "30 Days": 30, "60 Days": 60, "90 Days": 90}
        horizon = horizon_map[horizon_label]

    with ctrl_col2:
        model_choice = st.selectbox(
            "🤖 Forecasting Model",
            options=["Random Forest", "SARIMA", "Both (Ensemble)"],
            index=0,
        )

    with ctrl_col3:
        show_ci = st.checkbox("Show Confidence Bands", value=True)
        show_history = st.slider("Historical months to display", 12, len(df), 36, 6)

    run_btn = st.button("🚀 Generate Forecast", use_container_width=True)

    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Train & Forecast ───────────────────────────────────────────────────────
    cache_key = f"future_fc_{horizon}_{model_choice}"
    if run_btn or cache_key not in st.session_state:
        ts = prepare_time_series(df, "In_HHS")

        rf_forecast = None
        arima_forecast = None

        if model_choice in ("Random Forest", "Both (Ensemble)"):
            with st.spinner("🌲 Training Random Forest..."):
                rf_result = train_random_forest(df, target_col="In_HHS", n_estimators=200)
                rf_forecast = rf_forecast_future(rf_result, df, horizon=horizon)
                st.session_state["ff_rf_result"] = rf_result
                st.session_state["ff_rf_forecast"] = rf_forecast

        if model_choice in ("SARIMA", "Both (Ensemble)"):
            with st.spinner("📈 Fitting SARIMA model..."):
                arima_result = train_arima(ts, test_size=0.2)
                if arima_result.get("success"):
                    arima_forecast = arima_forecast_future(arima_result, horizon=horizon)
                    st.session_state["ff_arima_forecast"] = arima_forecast
                else:
                    st.warning(f"SARIMA failed: {arima_result.get('error')}")

        st.session_state[cache_key] = True
        st.session_state["ff_ts"] = ts
        st.session_state["ff_horizon"] = horizon
        st.session_state["ff_model"] = model_choice

    rf_forecast = st.session_state.get("ff_rf_forecast")
    arima_forecast = st.session_state.get("ff_arima_forecast")
    ts = st.session_state.get("ff_ts")

    if ts is None:
        st.info("👆 Click **Generate Forecast** to begin.")
        return

    # ── Forecast KPI Cards ─────────────────────────────────────────────────────
    st.markdown("### 📊 Forecast Summary")

    current_val = int(ts.values[-1])
    forecasts_to_show = []

    if rf_forecast is not None:
        forecasts_to_show.append(("🌲 RF Forecast", rf_forecast, "#6366F1"))
    if arima_forecast and "values" in arima_forecast:
        forecasts_to_show.append(("📈 SARIMA Forecast", arima_forecast, "#EC4899"))

    kpi_cols = st.columns(2 + len(forecasts_to_show))
    with kpi_cols[0]:
        st.metric("📍 Current HHS Care", f"{current_val:,}", help="Last known value in dataset")
    with kpi_cols[1]:
        st.metric("📅 Forecast Horizon", horizon_label, help="Days into the future")

    for i, (label, fc, color) in enumerate(forecasts_to_show):
        with kpi_cols[2 + i]:
            fc_last = int(fc["values"][-1])
            delta = fc_last - current_val
            delta_pct = (delta / current_val * 100) if current_val != 0 else 0
            st.metric(
                label,
                f"{fc_last:,}",
                delta=f"{delta:+,} ({delta_pct:+.1f}%)",
                delta_color="inverse" if delta > 0 else "normal",
            )

    # ── Main Forecast Chart ────────────────────────────────────────────────────
    st.markdown("### 📈 Interactive Forecast Chart")

    historical_slice = ts.iloc[-show_history:]

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=historical_slice.index, y=historical_slice.values,
        mode="lines", name="Historical Data",
        line=dict(color="#3B82F6", width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.06)",
        hovertemplate="<b>%{x|%b %Y}</b><br>Actual: <b>%{y:,}</b><extra></extra>",
    ))

    # Vertical "now" line
    now_date = historical_slice.index[-1]
    fig.add_vline(
        x=now_date,
        line_dash="dash",
        line_color="rgba(148,163,184,0.5)",
        annotation_text="Latest Data",
        annotation_font_color="#94A3B8",
        annotation_position="top left",
    )

    if rf_forecast is not None:
        if show_ci:
            fig.add_trace(go.Scatter(
                x=list(rf_forecast["dates"]) + list(rf_forecast["dates"][::-1]),
                y=list(rf_forecast["upper"]) + list(rf_forecast["lower"][::-1]),
                fill="toself", fillcolor="rgba(99,102,241,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="RF Confidence Band", hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=rf_forecast["dates"], y=rf_forecast["values"],
            mode="lines+markers", name="Random Forest Forecast",
            line=dict(color="#6366F1", width=3),
            marker=dict(size=9, color="#A5B4FC", symbol="circle", line=dict(color="#6366F1", width=2)),
            hovertemplate="<b>%{x|%b %Y}</b><br>RF Forecast: <b>%{y:,.0f}</b><extra></extra>",
        ))

    if arima_forecast and "values" in arima_forecast:
        if show_ci:
            fig.add_trace(go.Scatter(
                x=list(arima_forecast["dates"]) + list(arima_forecast["dates"][::-1]),
                y=list(arima_forecast["upper"]) + list(arima_forecast["lower"][::-1]),
                fill="toself", fillcolor="rgba(236,72,153,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="SARIMA Confidence Band", hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=arima_forecast["dates"], y=arima_forecast["values"],
            mode="lines+markers", name="SARIMA Forecast",
            line=dict(color="#EC4899", width=3),
            marker=dict(size=9, color="#F9A8D4", symbol="diamond", line=dict(color="#EC4899", width=2)),
            hovertemplate="<b>%{x|%b %Y}</b><br>SARIMA Forecast: <b>%{y:,.0f}</b><extra></extra>",
        ))

    # Ensemble average
    if model_choice == "Both (Ensemble)" and rf_forecast is not None and arima_forecast and "values" in arima_forecast:
        rf_vals = rf_forecast["values"]
        ar_vals = arima_forecast["values"]
        min_len = min(len(rf_vals), len(ar_vals))
        ensemble = (rf_vals[:min_len] + ar_vals[:min_len]) / 2
        ensemble_dates = rf_forecast["dates"][:min_len]
        fig.add_trace(go.Scatter(
            x=ensemble_dates, y=ensemble,
            mode="lines", name="Ensemble Average",
            line=dict(color="#F59E0B", width=2.5, dash="dot"),
            hovertemplate="<b>%{x|%b %Y}</b><br>Ensemble: <b>%{y:,.0f}</b><extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#E2E8F0"),
        height=500,
        title=dict(text=f"Children in HHS Care — {horizon_label} Forecast", font=dict(size=17, color="#E2E8F0")),
        legend=dict(bgcolor="rgba(30,41,59,0.8)", bordercolor="rgba(99,102,241,0.3)", borderwidth=1, font=dict(size=11)),
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)", linecolor="rgba(148,163,184,0.2)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)", linecolor="rgba(148,163,184,0.2)", title="Children Count"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Forecast Table ─────────────────────────────────────────────────────────
    st.markdown("### 📋 Forecast Values Table")
    if rf_forecast is not None:
        fc_table = pd.DataFrame({
            "Forecast Date": [d.strftime("%B %Y") for d in rf_forecast["dates"]],
            "RF Forecast": [f"{v:,.0f}" for v in rf_forecast["values"]],
            "Lower Bound": [f"{v:,.0f}" for v in rf_forecast["lower"]],
            "Upper Bound": [f"{v:,.0f}" for v in rf_forecast["upper"]],
        })
        if arima_forecast and "values" in arima_forecast:
            min_len = min(len(fc_table), len(arima_forecast["values"]))
            fc_table.loc[:min_len - 1, "SARIMA Forecast"] = [f"{v:,.0f}" for v in arima_forecast["values"][:min_len]]

        st.dataframe(fc_table, use_container_width=True)

        csv = fc_table.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Forecast as CSV", csv, "hhs_future_forecast.csv", "text/csv")

    # ── Interpretation ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:rgba(99,102,241,0.05); border:1px solid rgba(99,102,241,0.2);
                border-radius:12px; padding:1.2rem 1.5rem; margin-top:1rem;">
        <h4 style="color:#A5B4FC; margin:0 0 0.5rem;">📌 Forecast Interpretation Guide</h4>
        <ul style="color:#94A3B8; font-size:0.88rem; line-height:1.8; margin:0; padding-left:1.2rem;">
            <li><b style="color:#6366F1;">Random Forest</b> — Uses lag features, rolling statistics, and other metrics as predictors. Best for capturing complex nonlinear patterns.</li>
            <li><b style="color:#EC4899;">SARIMA</b> — Captures seasonal and temporal autocorrelation patterns inherent to time series.</li>
            <li><b style="color:#F59E0B;">Ensemble</b> — Simple average of RF and SARIMA. Often more robust than either model alone.</li>
            <li><b style="color:#94A3B8;">Confidence Bands</b> — Represent the range of likely outcomes (±8% for RF; 95% CI for SARIMA).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
