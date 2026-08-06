"""
Predictive Forecasting Page Component for HHS UAC Dashboard.
"""

import streamlit as st
import pandas as pd
from utils.data_loader import load_raw_data
from utils.preprocessing import create_engineered_features, prepare_train_test_split
from utils.ml_models import train_eval_model
from utils.helpers import render_header, render_kpi_card, render_alert
from utils.charts import plot_actual_vs_predicted, plot_residual_diagnostics, plot_feature_importance

def render():
    render_header(
        title="Predictive Machine Learning Forecasting Engine",
        subtitle="Train, Evaluate, and Diagnostically Inspect Supervised ML & Time-Series Algorithms"
    )

    df_raw = load_raw_data()
    if df_raw.empty:
        st.error("Data unavailable.")
        return

    # Engineering features
    df_feat = create_engineered_features(df_raw, target_col="Children in HHS Care")

    # Controls Row
    c_mod, c_split = st.columns([2, 1])
    with c_mod:
        selected_model = st.selectbox(
            "Select Forecasting Model Algorithm",
            options=["Random Forest", "Gradient Boosting", "ARIMA", "SARIMA", "Baseline Persistence", "Moving Average"],
            index=0
        )
    with c_split:
        test_pct = st.slider("Holdout Test Set Size (%)", min_value=10, max_value=30, value=15, step=5) / 100.0

    # Train / Test split
    train_df, test_df, feature_cols = prepare_train_test_split(df_feat, target_col="Children in HHS Care", test_size=test_pct)

    # Model Execution with spinner
    with st.spinner(f"Training and Evaluating {selected_model}..."):
        res = train_eval_model(
            model_name=selected_model,
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target_col="Children in HHS Care"
        )

    metrics = res["metrics"]
    res_df = res["results_df"]
    importances = res["feature_importances"]

    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics Display Bar
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("MAE (Mean Abs Error)", f"{metrics['MAE']:.1f}")
    m2.metric("RMSE (Root Mean Sq Error)", f"{metrics['RMSE']:.1f}")
    m3.metric("MAPE (Error %)", f"{metrics['MAPE']:.2f}%")
    m4.metric("R² Score", f"{metrics['R2']:.4f}")
    m5.metric("Forecast Accuracy", f"{metrics['Accuracy']:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Actual vs Predicted Visual
    st.markdown(f"### 🎯 Actual vs Predicted Performance ({selected_model})")
    fig_pred = plot_actual_vs_predicted(res_df, target_name="Children in HHS Care")
    st.plotly_chart(fig_pred, use_container_width=True)

    # Diagnostics & Feature Importance Tabs
    t_res, t_imp, t_table = st.tabs(["Residual Diagnostics Error Analysis", "Feature Importance Rankings", "Prediction Data Table"])

    with t_res:
        fig_res = plot_residual_diagnostics(res_df)
        st.plotly_chart(fig_res, use_container_width=True)

    with t_imp:
        if importances is not None and not importances.empty:
            fig_imp = plot_feature_importance(importances)
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            render_alert("Feature importance is available for tree-based ensemble models (Random Forest, Gradient Boosting).", "info")

    with t_table:
        st.dataframe(res_df, use_container_width=True)

if __name__ == "__main__" or True:
    render()

