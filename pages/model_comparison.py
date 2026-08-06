"""
Model Comparison Page Component for HHS UAC Dashboard.
"""

import streamlit as st
import pandas as pd
from utils.data_loader import load_raw_data
from utils.preprocessing import create_engineered_features, prepare_train_test_split
from utils.ml_models import train_eval_model
from utils.helpers import render_header, render_alert
from utils.charts import plot_model_leaderboard
from utils.config import MODELS_LIST

def render():
    render_header(
        title="Comprehensive Machine Learning & Time-Series Leaderboard",
        subtitle="Benchmarking Random Forest, Gradient Boosting, ARIMA, SARIMA, Baseline Persistence, and Moving Average"
    )

    df_raw = load_raw_data()
    if df_raw.empty:
        st.error("Data missing.")
        return

    df_feat = create_engineered_features(df_raw)
    train_df, test_df, feature_cols = prepare_train_test_split(df_feat, test_size=0.15)

    # Evaluate all 6 models
    with st.spinner("Benchmarking all 6 ML & Time-Series Models on Holdout Test Set..."):
        results_list = []
        for model_name in MODELS_LIST:
            eval_res = train_eval_model(
                model_name=model_name,
                train_df=train_df,
                test_df=test_df,
                feature_cols=feature_cols,
                target_col="Children in HHS Care"
            )
            metrics = eval_res["metrics"]
            results_list.append({
                "Model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE (%)": metrics["MAPE"],
                "R² Score": metrics["R2"],
                "Accuracy (%)": metrics["Accuracy"]
            })

    comp_df = pd.DataFrame(results_list)

    # Automatically determine Best Champion Model (lowest MAE)
    best_model_row = comp_df.sort_values("MAE").iloc[0]
    best_name = best_model_row["Model"]

    st.markdown("<br>", unsafe_allow_html=True)

    # Champion Banner
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #065f46 0%, #047857 100%); border-radius: 14px; padding: 1.5rem 2rem; color: white; margin-bottom: 2rem; box-shadow: 0 10px 25px rgba(5, 150, 105, 0.3);'>
        <span class="champion-badge">🏆 CHAMPION MODEL WINNER</span>
        <h2 style='margin-top: 0.5rem; margin-bottom: 0.3rem; color: #ffffff;'>{best_name}</h2>
        <p style='color: #a7f3d0; margin-bottom: 0;'>
            Achieved lowest Mean Absolute Error of <b>{best_model_row['MAE']:.1f} children</b> and <b>{best_model_row['Accuracy (%)']:.1f}% Forecast Accuracy</b> on holdout evaluation data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metric Selection for Chart
    col_sel, col_blank = st.columns([1.5, 1])
    with col_sel:
        metric_choice = st.selectbox("Rank Leaderboard By Metric", options=["MAE", "RMSE", "MAPE (%)", "R² Score", "Accuracy (%)"], index=0)

    # Leaderboard Bar Chart
    chart_metric = "MAPE (%)" if metric_choice == "MAPE (%)" else ("R2" if metric_choice == "R² Score" else ("Accuracy" if metric_choice == "Accuracy (%)" else metric_choice))
    
    # Standardize column naming for chart
    chart_df = comp_df.rename(columns={"MAPE (%)": "MAPE", "R² Score": "R2", "Accuracy (%)": "Accuracy"})
    fig_lead = plot_model_leaderboard(chart_df, metric_col=chart_metric)
    st.plotly_chart(fig_lead, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Full Leaderboard Table
    st.markdown("### 📊 Full Model Comparison Table")
    st.dataframe(
        comp_df.sort_values("MAE").style.highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], color="#065f46")
        .highlight_max(subset=["R² Score", "Accuracy (%)"], color="#065f46"),
        use_container_width=True
    )

if __name__ == "__main__" or True:
    render()

