"""
pages/scenario_analysis.py — Scenario Analysis page.
Dynamically update predictions by changing input parameters.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils.data_loader import load_data, get_column_display_names
from utils.ml_models import train_random_forest, engineer_features


def render():
    """Render the Scenario Analysis page."""

    st.markdown("# 🧪 Scenario Analysis")
    st.markdown("<p style='color:#94A3B8; margin-top:-0.5rem;'>Adjust input parameters to dynamically simulate and update HHS care load predictions.</p>", unsafe_allow_html=True)
    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Load & train ───────────────────────────────────────────────────────────
    with st.spinner("Loading data..."):
        try:
            df = load_data("data/dataset.csv")
        except FileNotFoundError as e:
            st.error(str(e))
            return

    display_names = get_column_display_names()

    # Train RF once and cache
    if "scenario_rf_result" not in st.session_state:
        with st.spinner("🌲 Training model for scenario analysis..."):
            rf_result = train_random_forest(df, target_col="In_HHS", n_estimators=200)
            st.session_state["scenario_rf_result"] = rf_result
            st.session_state["scenario_df"] = df

    rf_result = st.session_state["scenario_rf_result"]
    model = rf_result["model"]
    feature_cols = rf_result["feature_columns"]

    # Get baseline (last row of features)
    X_full = rf_result["X_full"]
    baseline_features = X_full.iloc[-1:].copy()
    baseline_pred = float(model.predict(baseline_features)[0])

    last_row = df.iloc[-1]
    current_apprehended = int(last_row["Apprehended"])
    current_transferred = int(last_row["Transferred_Out"])
    current_discharged = int(last_row["Discharged"])
    current_in_hhs = int(last_row["In_HHS"])

    # ── Explainer ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:rgba(245,158,11,0.05); border:1px solid rgba(245,158,11,0.2);
                border-radius:12px; padding:1rem 1.4rem; margin-bottom:1.2rem;">
        <h4 style="color:#FCD34D; margin:0 0 0.5rem;">🧪 How Scenario Analysis Works</h4>
        <p style="color:#94A3B8; font-size:0.88rem; margin:0; line-height:1.7;">
            Adjust the sliders below to simulate changes in key input variables.
            The trained <b>Random Forest model</b> re-runs predictions in real-time based on your scenario,
            allowing planners to assess the impact of policy changes, seasonal surges, or resource interventions
            on <b>Children in HHS Care</b> before they occur.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Scenario Presets ───────────────────────────────────────────────────────
    st.markdown("### 🎭 Quick Scenario Presets")
    preset_cols = st.columns(4)
    preset = None
    with preset_cols[0]:
        if st.button("📈 Surge (+30%)", use_container_width=True, help="Simulates a 30% increase in all inputs"):
            preset = "surge"
    with preset_cols[1]:
        if st.button("📉 Reduction (−20%)", use_container_width=True, help="Simulates a 20% drop in all inputs"):
            preset = "reduce"
    with preset_cols[2]:
        if st.button("✅ High Discharge (+40%)", use_container_width=True, help="Simulates high discharge scenario"):
            preset = "high_discharge"
    with preset_cols[3]:
        if st.button("🔄 Baseline", use_container_width=True, help="Reset to current actual values"):
            preset = "baseline"

    # Set session state for sliders
    if "sc_apprehended" not in st.session_state:
        st.session_state["sc_apprehended"] = current_apprehended
        st.session_state["sc_transferred"] = current_transferred
        st.session_state["sc_discharged"] = current_discharged
        st.session_state["sc_in_cbp"] = int(last_row["In_CBP"])

    if preset == "surge":
        st.session_state["sc_apprehended"] = int(current_apprehended * 1.3)
        st.session_state["sc_transferred"] = int(current_transferred * 1.3)
        st.session_state["sc_discharged"] = int(current_discharged * 1.1)
    elif preset == "reduce":
        st.session_state["sc_apprehended"] = int(current_apprehended * 0.8)
        st.session_state["sc_transferred"] = int(current_transferred * 0.8)
        st.session_state["sc_discharged"] = int(current_discharged * 0.85)
    elif preset == "high_discharge":
        st.session_state["sc_apprehended"] = current_apprehended
        st.session_state["sc_transferred"] = current_transferred
        st.session_state["sc_discharged"] = int(current_discharged * 1.4)
    elif preset == "baseline":
        st.session_state["sc_apprehended"] = current_apprehended
        st.session_state["sc_transferred"] = current_transferred
        st.session_state["sc_discharged"] = current_discharged

    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Sliders ────────────────────────────────────────────────────────────────
    st.markdown("### 🎛️ Scenario Input Parameters")
    st.markdown("<p style='color:#94A3B8;'>Adjust sliders — predictions update automatically.</p>", unsafe_allow_html=True)

    sl_col1, sl_col2 = st.columns(2)

    with sl_col1:
        sc_apprehended = st.slider(
            "🚔 Children Apprehended (Monthly)",
            min_value=500, max_value=int(df["Apprehended"].max() * 2),
            value=st.session_state["sc_apprehended"],
            step=100,
            help="Adjust the monthly number of children apprehended and placed in CBP custody.",
            key="sc_apprehended_slider",
        )
        sc_transferred = st.slider(
            "🔄 Children Transferred Out of CBP",
            min_value=100, max_value=int(df["Transferred_Out"].max() * 2),
            value=st.session_state["sc_transferred"],
            step=100,
            key="sc_transferred_slider",
        )

    with sl_col2:
        sc_discharged = st.slider(
            "✅ Children Discharged from HHS",
            min_value=500, max_value=int(df["Discharged"].max() * 2),
            value=st.session_state["sc_discharged"],
            step=100,
            key="sc_discharged_slider",
        )
        sc_cbp = st.slider(
            "🏛️ Children in CBP Custody",
            min_value=100, max_value=int(df["In_CBP"].max() * 2),
            value=st.session_state["sc_in_cbp"],
            step=100,
            key="sc_cbp_slider",
        )

    # ── Compute Scenario Prediction ────────────────────────────────────────────
    scenario_features = baseline_features.copy()

    # Update feature values that correspond to our input variables
    for col in scenario_features.columns:
        col_lower = col.lower()
        if "apprehended" in col_lower and "lag" not in col_lower and "roll" not in col_lower:
            scenario_features[col] = sc_apprehended
        elif "transferred" in col_lower and "lag" not in col_lower and "roll" not in col_lower:
            scenario_features[col] = sc_transferred
        elif "discharged" in col_lower and "lag" not in col_lower and "roll" not in col_lower:
            scenario_features[col] = sc_discharged
        elif "in_cbp" in col_lower and "lag" not in col_lower and "roll" not in col_lower:
            scenario_features[col] = sc_cbp

    scenario_pred = float(model.predict(scenario_features)[0])
    delta = scenario_pred - baseline_pred
    delta_pct = (delta / baseline_pred * 100) if baseline_pred != 0 else 0

    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Results ────────────────────────────────────────────────────────────────
    st.markdown("### 📊 Scenario Prediction Results")

    res1, res2, res3, res4 = st.columns(4)
    with res1:
        st.metric("📍 Current Actual", f"{current_in_hhs:,}", help="Last known value in dataset")
    with res2:
        st.metric("📐 Baseline Forecast", f"{int(baseline_pred):,}", help="Model prediction using current input values")
    with res3:
        st.metric(
            "🧪 Scenario Forecast",
            f"{int(scenario_pred):,}",
            delta=f"{delta:+,.0f} ({delta_pct:+.1f}%)",
            delta_color="inverse" if delta > 0 else "normal",
        )
    with res4:
        urgency = "🔴 High" if abs(delta_pct) > 15 else "🟡 Moderate" if abs(delta_pct) > 5 else "🟢 Low"
        st.metric("⚠️ Impact Level", urgency)

    # ── Bar Chart Comparison ───────────────────────────────────────────────────
    labels = ["Current Actual", "Baseline Forecast", "Scenario Forecast"]
    values = [current_in_hhs, int(baseline_pred), int(scenario_pred)]
    colors = ["#3B82F6", "#6366F1", "#EC4899" if delta > 0 else "#10B981"]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:,}" for v in values],
        textposition="outside",
        textfont=dict(size=14, color="#E2E8F0"),
        hovertemplate="<b>%{x}</b><br>Value: <b>%{y:,}</b><extra></extra>",
        width=[0.4, 0.4, 0.4],
    ))

    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,sans-serif", color="#E2E8F0"),
        height=380,
        title=dict(text=f"Scenario Impact: {delta:+,.0f} children ({delta_pct:+.1f}%)", font=dict(size=16)),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)", title="Children in HHS Care"),
        showlegend=False,
    )
    if delta != 0:
        fig_bar.add_hline(
            y=baseline_pred,
            line_dash="dash",
            line_color="rgba(148,163,184,0.4)",
            annotation_text="Baseline",
            annotation_font_color="#94A3B8",
        )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Input vs Baseline delta table ─────────────────────────────────────────
    st.markdown("### 📋 Scenario vs Baseline Input Comparison")
    comp_df = pd.DataFrame({
        "Parameter": ["Children Apprehended", "Children Transferred Out", "Children Discharged", "Children in CBP"],
        "Baseline (Actual)": [current_apprehended, current_transferred, current_discharged, int(last_row["In_CBP"])],
        "Scenario Value": [sc_apprehended, sc_transferred, sc_discharged, sc_cbp],
        "Change": [
            f"{sc_apprehended - current_apprehended:+,}",
            f"{sc_transferred - current_transferred:+,}",
            f"{sc_discharged - current_discharged:+,}",
            f"{sc_cbp - int(last_row['In_CBP']):+,}",
        ],
        "Change %": [
            f"{(sc_apprehended - current_apprehended)/current_apprehended*100:+.1f}%",
            f"{(sc_transferred - current_transferred)/current_transferred*100:+.1f}%",
            f"{(sc_discharged - current_discharged)/current_discharged*100:+.1f}%",
            f"{(sc_cbp - int(last_row['In_CBP']))/max(1,int(last_row['In_CBP']))*100:+.1f}%",
        ],
    })
    st.dataframe(comp_df, use_container_width=True)

    # ── Impact interpretation ──────────────────────────────────────────────────
    if abs(delta_pct) > 15:
        msg_color, msg_icon = "#EF4444", "🚨"
        msg = f"**Critical Impact**: This scenario projects a **{abs(delta_pct):.1f}%** {'increase' if delta > 0 else 'decrease'} in HHS care load. Immediate resource planning action is recommended."
    elif abs(delta_pct) > 5:
        msg_color, msg_icon = "#F59E0B", "⚠️"
        msg = f"**Moderate Impact**: A **{abs(delta_pct):.1f}%** {'increase' if delta > 0 else 'decrease'} is projected. Monitor closely and prepare contingency plans."
    else:
        msg_color, msg_icon = "#10B981", "✅"
        msg = f"**Minimal Impact**: The scenario projects only a **{abs(delta_pct):.1f}%** change. Current capacity plans remain adequate."

    st.markdown(f"""
    <div style="background:rgba(0,0,0,0.2); border:1px solid {msg_color}40; border-radius:12px;
                padding:1rem 1.4rem; margin-top:0.5rem; border-left:4px solid {msg_color};">
        <p style="color:{msg_color}; font-weight:600; margin:0 0 0.3rem;">{msg_icon} Policy Recommendation</p>
        <p style="color:#CBD5E1; font-size:0.9rem; margin:0; line-height:1.6;">{msg}</p>
    </div>
    """, unsafe_allow_html=True)
