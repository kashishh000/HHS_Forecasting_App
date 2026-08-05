"""
pages/predictive_forecasting.py — Predictive Forecasting page.
Trains Random Forest and ARIMA models, displays metrics and prediction charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.data_loader import load_data, prepare_time_series, get_column_display_names
from utils.ml_models import train_random_forest, train_arima
from utils.charts import plot_rf_predictions, plot_arima_predictions, plot_feature_importance


def render():
    """Render the Predictive Forecasting page."""

    st.markdown("# 🤖 Predictive Forecasting")
    st.markdown("<p style='color:#94A3B8; margin-top:-0.5rem;'>Train and evaluate Random Forest Regression and SARIMA models on the HHS UAC dataset.</p>", unsafe_allow_html=True)
    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Load Data ──────────────────────────────────────────────────────────────
    with st.spinner("Loading data..."):
        try:
            df = load_data("data/dataset.csv")
        except FileNotFoundError as e:
            st.error(str(e))
            return

    display_names = get_column_display_names()

    # ── Model Config Sidebar ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Model Configuration")
        target_col = st.selectbox(
            "Prediction Target",
            options=["In_HHS", "Discharged", "Apprehended"],
            format_func=lambda x: display_names.get(x, x),
            key="pred_target",
        )
        test_size = st.slider("Test Set Size (%)", 10, 35, 20, 5) / 100
        n_estimators = st.slider("RF Trees (n_estimators)", 50, 500, 200, 50)
        run_arima = st.checkbox("Also run SARIMA", value=True)
        train_btn = st.button("🚀 Train Models", use_container_width=True)

    # ── Train Models ───────────────────────────────────────────────────────────
    if "rf_result" not in st.session_state or train_btn:
        with st.spinner("🔄 Training Random Forest model..."):
            rf_result = train_random_forest(
                df, target_col=target_col, test_size=test_size, n_estimators=n_estimators
            )
            st.session_state["rf_result"] = rf_result
            st.session_state["pred_target"] = target_col
            st.session_state["pred_df"] = df

        if run_arima:
            with st.spinner("🔄 Fitting SARIMA model (this may take ~30s)..."):
                ts = prepare_time_series(df, target_col)
                arima_result = train_arima(ts, test_size=test_size)
                st.session_state["arima_result"] = arima_result
                st.session_state["ts"] = ts
        else:
            st.session_state["arima_result"] = {"success": False, "error": "SARIMA not requested."}

    rf_result = st.session_state.get("rf_result")
    arima_result = st.session_state.get("arima_result", {"success": False})

    if rf_result is None:
        st.info("👈 Click **Train Models** in the sidebar to begin.")
        return

    # ── Model Metrics Banner ───────────────────────────────────────────────────
    st.markdown("## 📊 Model Performance Metrics")
    st.markdown(f"<p style='color:#94A3B8;'>Target: <b style='color:#A5B4FC;'>{display_names.get(target_col, target_col)}</b> | Test size: <b style='color:#A5B4FC;'>{int(test_size*100)}%</b></p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background:linear-gradient(145deg,#1E293B,#263348); border:1px solid rgba(99,102,241,0.3);
                    border-radius:14px; padding:1.4rem; margin-bottom:1rem;">
            <div style="font-size:0.75rem;font-weight:600;color:#94A3B8;text-transform:uppercase;
                        letter-spacing:0.06em;margin-bottom:1rem;">
                🌲 RANDOM FOREST REGRESSION
            </div>
        """, unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("MAE", f"{rf_result['mae']:,.1f}", help="Mean Absolute Error — lower is better")
        with m2:
            st.metric("RMSE", f"{rf_result['rmse']:,.1f}", help="Root Mean Squared Error — lower is better")
        with m3:
            st.metric("R² Score", f"{rf_result['r2']:.4f}", help="R-squared — closer to 1.0 is better")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if arima_result.get("success"):
            st.markdown("""
            <div style="background:linear-gradient(145deg,#1E293B,#263348); border:1px solid rgba(236,72,153,0.3);
                        border-radius:14px; padding:1.4rem; margin-bottom:1rem;">
                <div style="font-size:0.75rem;font-weight:600;color:#94A3B8;text-transform:uppercase;
                            letter-spacing:0.06em;margin-bottom:1rem;">
                    📈 SARIMA TIME-SERIES MODEL
                </div>
            """, unsafe_allow_html=True)
            m4, m5, m6 = st.columns(3)
            with m4:
                st.metric("MAE", f"{arima_result['mae']:,.1f}")
            with m5:
                st.metric("RMSE", f"{arima_result['rmse']:,.1f}")
            with m6:
                st.metric("R² Score", f"{arima_result['r2']:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info(f"ℹ️ SARIMA: {arima_result.get('error', 'Not trained')}")

    # ── Comparison insight ─────────────────────────────────────────────────────
    if arima_result.get("success"):
        better = "Random Forest" if rf_result["r2"] >= arima_result["r2"] else "SARIMA"
        diff = abs(rf_result["r2"] - arima_result["r2"])
        st.success(f"✅ **{better}** performs better on this target (R² advantage: **{diff:.4f}**).")

    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Prediction Charts ──────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🌲 Random Forest", "📈 SARIMA", "🎯 Feature Importance"])

    with tab1:
        st.markdown("### 🌲 Random Forest — Actual vs Predicted")
        st.markdown("<p style='color:#94A3B8;'>Blue = Actual data, Green = Train predictions (fitted), Pink = Test predictions (unseen data).</p>", unsafe_allow_html=True)
        fig_rf = plot_rf_predictions(rf_result, df)
        st.plotly_chart(fig_rf, use_container_width=True)

        # Residuals
        y_test = rf_result["y_test"].values
        y_pred = rf_result["y_pred"]
        residuals = y_test - y_pred

        # Get test dates safely
        x_test_indices = list(rf_result["X_test"].index)
        dates_test = df["Date"].iloc[x_test_indices].reset_index(drop=True)

        fig_resid = go.Figure()
        fig_resid.add_trace(go.Bar(
            x=dates_test, y=residuals,
            marker_color=["#10B981" if r >= 0 else "#EF4444" for r in residuals],
            hovertemplate="<b>%{x|%b %Y}</b><br>Residual: <b>%{y:,.0f}</b><extra></extra>",
        ))
        fig_resid.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
        fig_resid.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter,sans-serif", color="#E2E8F0"),
            height=320,
            title=dict(text="Residuals (Actual − Predicted)", font=dict(size=15)),
            xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            showlegend=False,
        )
        st.plotly_chart(fig_resid, use_container_width=True)

        # Residual stats
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            st.metric("Mean Residual", f"{np.mean(residuals):,.1f}")
        with rc2:
            st.metric("Std Residual", f"{np.std(residuals):,.1f}")
        with rc3:
            st.metric("Max Overprediction", f"{residuals.min():,.1f}")
        with rc4:
            st.metric("Max Underprediction", f"{residuals.max():,.1f}")

    with tab2:
        if arima_result.get("success"):
            st.markdown("### 📈 SARIMA — Actual vs Predicted")
            st.markdown(f"<p style='color:#94A3B8;'>SARIMA order: <b style='color:#A5B4FC;'>{arima_result['order']}</b> | Seasonal: <b style='color:#A5B4FC;'>{arima_result['seasonal_order']}</b></p>", unsafe_allow_html=True)
            fig_arima = plot_arima_predictions(arima_result)
            if fig_arima:
                st.plotly_chart(fig_arima, use_container_width=True)
            else:
                st.warning("Chart unavailable.")
        else:
            st.info(f"SARIMA not available: {arima_result.get('error', 'Not trained')}")

    with tab3:
        st.markdown("### 🎯 Feature Importance — Random Forest")
        st.markdown("<p style='color:#94A3B8;'>Top features ranked by their contribution to the Random Forest's prediction accuracy.</p>", unsafe_allow_html=True)
        fig_fi = plot_feature_importance(rf_result["feature_importance"])
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("#### 📋 Feature Importance Table")
        fi_df = rf_result["feature_importance"].copy()
        fi_df["Importance %"] = (fi_df["Importance"] / fi_df["Importance"].sum() * 100).round(2)
        st.dataframe(fi_df, use_container_width=True)
