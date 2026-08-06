"""
Plotly Chart Generators with Modern Dark Theme Aesthetic for HHS UAC Dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.config import PLOTLY_TEMPLATE, PLOTLY_BG_COLOR, PLOTLY_PAPER_COLOR, DARK_THEME_COLORWAY

def apply_dark_layout(fig, title: str = "", x_title: str = "", y_title: str = ""):
    """
    Applies unified dark theme styling to Plotly figures.
    """
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.0,
            'xanchor': 'left',
            'yanchor': 'top',
            'font': {'size': 18, 'color': '#ffffff', 'family': 'Inter'}
        },
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PLOTLY_PAPER_COLOR,
        plot_bgcolor=PLOTLY_BG_COLOR,
        font={'family': 'Inter', 'color': '#9ca3af'},
        xaxis=dict(
            title=x_title,
            gridcolor='#1e293b',
            showline=True,
            linecolor='#334155',
            zeroline=False
        ),
        yaxis=dict(
            title=y_title,
            gridcolor='#1e293b',
            showline=True,
            linecolor='#334155',
            zeroline=False
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            bgcolor='rgba(19, 27, 46, 0.8)',
            bordercolor='#233154',
            borderwidth=1,
            font=dict(color='#f3f4f6')
        ),
        colorway=DARK_THEME_COLORWAY
    )
    return fig

def plot_actual_vs_predicted(df_res: pd.DataFrame, target_name: str = "Children in HHS Care") -> go.Figure:
    """
    Line chart comparing actual target vs model prediction.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_res["Date"],
        y=df_res[target_name],
        mode='lines',
        name='Actual Target',
        line=dict(color='#3b82f6', width=2.5)
    ))

    fig.add_trace(go.Scatter(
        x=df_res["Date"],
        y=df_res["Predicted"],
        mode='lines',
        name='Predicted Value',
        line=dict(color='#10b981', width=2, dash='dot')
    ))

    apply_dark_layout(fig, title="Actual vs. Predicted Target", x_title="Date", y_title="Count")
    return fig

def plot_residual_diagnostics(df_res: pd.DataFrame) -> go.Figure:
    """
    Line chart of model residual errors over time.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_res["Date"],
        y=df_res["Residual"],
        mode='lines',
        name='Residual (Actual - Pred)',
        line=dict(color='#ef4444', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.15)'
    ))

    apply_dark_layout(fig, title="Model Residual Diagnostics Over Time", x_title="Date", y_title="Residual Error")
    return fig

def plot_feature_importance(imp_series: pd.Series) -> go.Figure:
    """
    Horizontal bar chart for model feature importance.
    """
    fig = go.Figure()

    top_imp = imp_series.head(10).sort_values(ascending=True)

    fig.add_trace(go.Bar(
        x=top_imp.values,
        y=top_imp.index,
        orientation='h',
        marker=dict(
            color=top_imp.values,
            colorscale='Viridis',
            showscale=False
        )
    ))

    apply_dark_layout(fig, title="Top 10 Feature Importances", x_title="Relative Importance Score", y_title="Feature")
    return fig

def plot_future_forecast(hist_df: pd.DataFrame, forecast_df: pd.DataFrame, target_col: str = "Children in HHS Care") -> go.Figure:
    """
    Combines recent historical time series with multi-horizon future forecast curve.
    """
    fig = go.Figure()

    # Show last 120 days of historical data for context
    recent_hist = hist_df.tail(120)

    fig.add_trace(go.Scatter(
        x=recent_hist["Date"],
        y=recent_hist[target_col],
        mode='lines',
        name='Historical Care Load',
        line=dict(color='#3b82f6', width=2.5)
    ))

    # Forecast trace
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast_Care_Load"],
        mode='lines+markers',
        name='Future Forecast',
        line=dict(color='#f59e0b', width=3),
        marker=dict(size=4)
    ))

    apply_dark_layout(fig, title="Future Projection Curve", x_title="Date", y_title="Children in HHS Care")
    return fig

def plot_confidence_intervals(hist_df: pd.DataFrame, conf_df: pd.DataFrame, target_col: str = "Children in HHS Care") -> go.Figure:
    """
    Renders prediction band confidence interval chart with shaded area.
    """
    fig = go.Figure()

    recent_hist = hist_df.tail(90)

    # Historical
    fig.add_trace(go.Scatter(
        x=recent_hist["Date"],
        y=recent_hist[target_col],
        mode='lines',
        name='Historical Actual',
        line=dict(color='#3b82f6', width=2)
    ))

    # Upper Bound
    fig.add_trace(go.Scatter(
        x=conf_df["Date"],
        y=conf_df["Upper_Bound"],
        mode='lines',
        name='Upper Confidence Bound',
        line=dict(color='rgba(245, 158, 11, 0.4)', width=1, dash='dash')
    ))

    # Lower Bound with fill
    fig.add_trace(go.Scatter(
        x=conf_df["Date"],
        y=conf_df["Lower_Bound"],
        mode='lines',
        name='Lower Confidence Bound',
        line=dict(color='rgba(245, 158, 11, 0.4)', width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(245, 158, 11, 0.15)'
    ))

    # Point Forecast
    fig.add_trace(go.Scatter(
        x=conf_df["Date"],
        y=conf_df["Forecast_Care_Load"],
        mode='lines',
        name='Point Forecast',
        line=dict(color='#f59e0b', width=3)
    ))

    apply_dark_layout(fig, title="Predictive Confidence Interval Bands", x_title="Date", y_title="HHS Care Demand")
    return fig

def plot_scenario_comparison(hist_df: pd.DataFrame, sim_df: pd.DataFrame, target_col: str = "Children in HHS Care") -> go.Figure:
    """
    Visualizes baseline vs simulated What-If scenario trajectory.
    """
    fig = go.Figure()

    recent_hist = hist_df.tail(90)

    fig.add_trace(go.Scatter(
        x=recent_hist["Date"],
        y=recent_hist[target_col],
        mode='lines',
        name='Historical Baseline',
        line=dict(color='#64748b', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=sim_df["Date"],
        y=sim_df["Simulated_HHS_Care"],
        mode='lines',
        name='Simulated Scenario Care Load',
        line=dict(color='#ec4899', width=3)
    ))

    apply_dark_layout(fig, title="What-If Scenario Projection vs Baseline", x_title="Date", y_title="Projected Care Load")
    return fig

def plot_model_leaderboard(comparison_df: pd.DataFrame, metric_col: str = "MAE") -> go.Figure:
    """
    Bar chart comparing models across performance metric.
    """
    fig = go.Figure()

    sorted_df = comparison_df.sort_values(metric_col, ascending=(metric_col != "R2" and metric_col != "Accuracy"))

    colors = ['#10b981' if i == 0 else '#3b82f6' for i in range(len(sorted_df))]

    fig.add_trace(go.Bar(
        x=sorted_df["Model"],
        y=sorted_df[metric_col],
        marker_color=colors,
        text=sorted_df[metric_col],
        textposition='auto'
    ))

    apply_dark_layout(fig, title=f"Model Performance Leaderboard ({metric_col})", x_title="Model Algorithm", y_title=metric_col)
    return fig

def plot_eda_timelines(df: pd.DataFrame) -> go.Figure:
    """
    Multi-line time series timeline of all key HHS UAC metrics.
    """
    fig = go.Figure()

    metrics = [
        ("Children apprehended and placed in CBP custody", "Apprehended (CBP)", "#ef4444"),
        ("Children in CBP custody", "In CBP Custody", "#f59e0b"),
        ("Children transferred out of CBP custody", "Transferred from CBP", "#06b6d4"),
        ("Children in HHS Care", "In HHS Care Load", "#3b82f6"),
        ("Children discharged from HHS Care", "Discharged from HHS", "#10b981")
    ]

    for col, label, color in metrics:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["Date"],
                y=df[col],
                mode='lines',
                name=label,
                line=dict(color=color, width=1.8)
            ))

    apply_dark_layout(fig, title="Comprehensive Time Series Dynamics", x_title="Date", y_title="Children Count")
    return fig

def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """
    Heatmap of numeric column correlations.
    """
    numeric_cols = [c for c in df.columns if df[c].dtype in [np.int64, np.float64, int, float]]
    corr = df[numeric_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='Viridis',
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 11, "color": "#ffffff"}
    ))

    apply_dark_layout(fig, title="Feature Correlation Heatmap Matrix", x_title="", y_title="")
    return fig

def plot_bi_kpi_gauge(current_val: float, target_val: float, title_text: str = "HHS Occupancy Rate") -> go.Figure:
    """
    Gauge indicator chart for operational capacity monitoring.
    """
    pct = min(100.0, (current_val / target_val) * 100.0) if target_val > 0 else 0.0

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_val,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title_text, 'font': {'color': '#ffffff', 'size': 16}},
        delta={'reference': target_val, 'increasing': {'color': "#ef4444"}, 'decreasing': {'color': "#10b981"}},
        gauge={
            'axis': {'range': [None, target_val * 1.3], 'tickcolor': "#9ca3af"},
            'bar': {'color': "#3b82f6"},
            'bgcolor': "#131b2e",
            'borderwidth': 2,
            'bordercolor': "#233154",
            'steps': [
                {'range': [0, target_val * 0.7], 'color': 'rgba(16, 185, 129, 0.3)'},
                {'range': [target_val * 0.7, target_val], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [target_val, target_val * 1.3], 'color': 'rgba(239, 68, 68, 0.4)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': target_val
            }
        }
    ))

    apply_dark_layout(fig, title="")
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=250)
    return fig
