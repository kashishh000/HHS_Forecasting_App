"""
pages/model_comparison.py — Model Comparison page.
Side-by-side comparison of Random Forest and SARIMA with metrics and charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_loader import load_data, prepare_time_series, get_column_display_names
from utils.ml_models import train_random_forest, train_arima, compare_models
from utils.charts import plot_model_comparison


def render():
    """Render the Model Comparison page."""

    st.markdown("# ⚖️ Model Comparison")
    st.markdown("<p style='color:#94A3B8; margin-top:-0.5rem;'>Side-by-side comparison of Random Forest Regression vs. SARIMA Time-Series Forecasting.</p>", unsafe_allow_html=True)
    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Load Data ──────────────────────────────────────────────────────────────
    with st.spinner("Loading data..."):
        try:
            df = load_data("data/dataset.csv")
        except FileNotFoundError as e:
            st.error(str(e))
            return

    display_names = get_column_display_names()

    # ── Config ─────────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        target = st.selectbox(
            "🎯 Prediction Target",
            ["In_HHS", "Discharged", "Apprehended"],
            format_func=lambda x: display_names.get(x, x),
            key="mc_target",
        )
    with col2:
        test_size = st.slider("Test Set Size (%)", 10, 35, 20, 5, key="mc_test") / 100
    with col3:
        run_btn = st.button("🚀 Run Full Comparison", use_container_width=True)

    if run_btn or "mc_rf_result" not in st.session_state:
        with st.spinner("🌲 Training Random Forest..."):
            rf_result = train_random_forest(df, target_col=target, n_estimators=200, test_size=test_size)
            st.session_state["mc_rf_result"] = rf_result
            st.session_state["mc_target"] = target

        with st.spinner("📈 Fitting SARIMA (may take ~30s)..."):
            ts = prepare_time_series(df, target)
            arima_result = train_arima(ts, test_size=test_size)
            st.session_state["mc_arima_result"] = arima_result
            st.session_state["mc_ts"] = ts

    rf_result = st.session_state.get("mc_rf_result")
    arima_result = st.session_state.get("mc_arima_result", {"success": False})
    ts = st.session_state.get("mc_ts")
    target = st.session_state.get("mc_target", target)

    if rf_result is None:
        st.info("👆 Click **Run Full Comparison** to train and compare models.")
        return

    # ── Metrics Side-by-Side ───────────────────────────────────────────────────
    st.markdown("## 📊 Performance Metrics")
    st.markdown(f"<p style='color:#94A3B8;'>Target: <b style='color:#A5B4FC;'>{display_names.get(target, target)}</b></p>", unsafe_allow_html=True)

    col_rf, col_vs, col_ar = st.columns([5, 1, 5])

    with col_rf:
        st.markdown("""
        <div style="background:linear-gradient(145deg,#1a2235,#1E293B); border:1px solid rgba(99,102,241,0.35);
                    border-radius:16px; padding:1.5rem; text-align:center;">
            <div style="font-size:2rem; margin-bottom:0.3rem;">🌲</div>
            <div style="font-size:1.1rem; font-weight:700; color:#A5B4FC; margin-bottom:1.2rem;">Random Forest</div>
        """, unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("MAE", f"{rf_result['mae']:,.1f}")
        with m2:
            st.metric("RMSE", f"{rf_result['rmse']:,.1f}")
        with m3:
            st.metric("R²", f"{rf_result['r2']:.4f}")
        st.markdown("""
            <div style="margin-top:1rem; font-size:0.8rem; color:#64748B; line-height:1.6;">
                Type: <b style="color:#94A3B8;">Ensemble ML</b><br>
                Seasonality: <b style="color:#94A3B8;">Via features</b><br>
                Nonlinearity: <b style="color:#10B981;">✅ Handles well</b><br>
                Speed: <b style="color:#94A3B8;">Fast</b>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_vs:
        st.markdown("""
        <div style="display:flex; align-items:center; justify-content:center; height:200px;">
            <div style="font-size:1.5rem; font-weight:700; color:#475569; text-align:center;">VS</div>
        </div>""", unsafe_allow_html=True)

    with col_ar:
        if arima_result.get("success"):
            st.markdown("""
            <div style="background:linear-gradient(145deg,#1a2235,#1E293B); border:1px solid rgba(236,72,153,0.35);
                        border-radius:16px; padding:1.5rem; text-align:center;">
                <div style="font-size:2rem; margin-bottom:0.3rem;">📈</div>
                <div style="font-size:1.1rem; font-weight:700; color:#F9A8D4; margin-bottom:1.2rem;">SARIMA</div>
            """, unsafe_allow_html=True)
            m4, m5, m6 = st.columns(3)
            with m4:
                st.metric("MAE", f"{arima_result['mae']:,.1f}")
            with m5:
                st.metric("RMSE", f"{arima_result['rmse']:,.1f}")
            with m6:
                st.metric("R²", f"{arima_result['r2']:.4f}")
            st.markdown(f"""
                <div style="margin-top:1rem; font-size:0.8rem; color:#64748B; line-height:1.6;">
                    Type: <b style="color:#94A3B8;">Statistical TS</b><br>
                    Order: <b style="color:#94A3B8;">{arima_result['order']}</b><br>
                    Seasonal: <b style="color:#94A3B8;">{arima_result['seasonal_order']}</b><br>
                    Speed: <b style="color:#94A3B8;">Moderate</b>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info(f"SARIMA not available: {arima_result.get('error','')}")

    # ── Winner Banner ──────────────────────────────────────────────────────────
    if arima_result.get("success"):
        if rf_result["r2"] > arima_result["r2"]:
            winner, win_r2, loser_r2 = "Random Forest 🌲", rf_result["r2"], arima_result["r2"]
            winner_color = "#6366F1"
        else:
            winner, win_r2, loser_r2 = "SARIMA 📈", arima_result["r2"], rf_result["r2"]
            winner_color = "#EC4899"

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(99,102,241,0.08),rgba(236,72,153,0.08));
                    border:1px solid {winner_color}40; border-radius:12px; padding:1.2rem;
                    text-align:center; margin:1.2rem 0;">
            <div style="font-size:0.85rem; color:#94A3B8; font-weight:600; text-transform:uppercase; letter-spacing:0.06em;">
                🏆 Best Performing Model
            </div>
            <div style="font-size:1.6rem; font-weight:700; color:{winner_color}; margin-top:0.3rem;">
                {winner}
            </div>
            <div style="font-size:0.85rem; color:#94A3B8; margin-top:0.3rem;">
                R² = <b style="color:{winner_color};">{win_r2:.4f}</b> vs {loser_r2:.4f} (advantage: {abs(win_r2-loser_r2):.4f})
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Comparison Chart ───────────────────────────────────────────────────────
    st.markdown("### 📊 Metric Comparison Chart")
    comp_df = compare_models(rf_result, arima_result)
    fig_comp = plot_model_comparison(comp_df)
    st.plotly_chart(fig_comp, use_container_width=True)

    # ── Side-by-Side Prediction Chart ─────────────────────────────────────────
    st.markdown("### 📈 Predictions Overlay")
    st.markdown("<p style='color:#94A3B8;'>Both models' test-set predictions plotted against actual values for direct comparison.</p>", unsafe_allow_html=True)

    # Build overlay chart
    X_test = rf_result["X_test"]
    y_test = rf_result["y_test"].values
    y_pred_rf = rf_result["y_pred"]

    # Safely get test dates using positional indexing
    x_test_indices = list(X_test.index)
    dates_test = df["Date"].iloc[x_test_indices].reset_index(drop=True)

    fig_ov = go.Figure()
    fig_ov.add_trace(go.Scatter(
        x=dates_test, y=y_test,
        mode="lines+markers", name="Actual",
        line=dict(color="#3B82F6", width=2),
        marker=dict(size=5),
        hovertemplate="<b>%{x|%b %Y}</b><br>Actual: <b>%{y:,}</b><extra></extra>",
    ))
    fig_ov.add_trace(go.Scatter(
        x=dates_test, y=y_pred_rf,
        mode="lines+markers", name="Random Forest",
        line=dict(color="#6366F1", width=2.5, dash="dot"),
        marker=dict(size=7, symbol="diamond"),
        hovertemplate="<b>%{x|%b %Y}</b><br>RF: <b>%{y:,}</b><extra></extra>",
    ))
    if arima_result.get("success"):
        arima_pred = arima_result["y_pred"]
        arima_dates = arima_result["test_ts"].index
        min_len = min(len(dates_test), len(arima_pred))
        fig_ov.add_trace(go.Scatter(
            x=arima_dates[:min_len], y=arima_pred[:min_len],
            mode="lines+markers", name="SARIMA",
            line=dict(color="#EC4899", width=2.5, dash="dash"),
            marker=dict(size=7, symbol="square"),
            hovertemplate="<b>%{x|%b %Y}</b><br>SARIMA: <b>%{y:,}</b><extra></extra>",
        ))

    fig_ov.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,sans-serif", color="#E2E8F0"),
        height=440,
        title=dict(text="Test Set: Actual vs Predictions Comparison", font=dict(size=16)),
        legend=dict(bgcolor="rgba(30,41,59,0.8)", bordercolor="rgba(99,102,241,0.3)", borderwidth=1),
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)", title=display_names.get(target, target)),
        hovermode="x unified",
    )
    st.plotly_chart(fig_ov, use_container_width=True)

    # ── Comparison Table ───────────────────────────────────────────────────────
    st.markdown("### 📋 Detailed Comparison Table")
    st.dataframe(comp_df.set_index("Model"), use_container_width=True)

    # ── Radar Chart ───────────────────────────────────────────────────────────
    st.markdown("### 🕸️ Model Capability Radar")
    categories = ["Accuracy (R²)", "Speed", "Interpretability", "Nonlinearity", "Uncertainty", "Seasonality"]

    # Score each model on 1-10 scale
    rf_r2_score = min(10, max(0, rf_result["r2"] * 10))
    ar_r2_score = min(10, max(0, arima_result["r2"] * 10)) if arima_result.get("success") else 0

    rf_scores = [rf_r2_score, 9, 6, 9, 7, 5]
    ar_scores = [ar_r2_score, 5, 9, 3, 8, 9]

    fig_radar = go.Figure()
    for name, scores, color, fill_color in [
        ("Random Forest", rf_scores, "#6366F1", "rgba(99,102,241,0.12)"),
        ("SARIMA", ar_scores, "#EC4899", "rgba(236,72,153,0.12)")
    ]:
        fig_radar.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=name,
            line_color=color,
            fillcolor=fill_color,
        ))

    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(30,41,59,0.5)",
            radialaxis=dict(visible=True, range=[0, 10], gridcolor="rgba(148,163,184,0.1)", color="#94A3B8"),
            angularaxis=dict(gridcolor="rgba(148,163,184,0.1)", color="#94A3B8"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,sans-serif", color="#E2E8F0"),
        height=450,
        title=dict(text="Model Capability Radar Chart (Score /10)", font=dict(size=16)),
        legend=dict(bgcolor="rgba(30,41,59,0.8)", bordercolor="rgba(99,102,241,0.3)", borderwidth=1),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
