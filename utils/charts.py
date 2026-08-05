"""
charts.py - Reusable Plotly chart functions for HHS UAC Forecasting App.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ─── Color Palette ────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#6366F1",      # Indigo
    "secondary": "#8B5CF6",    # Violet
    "accent": "#EC4899",       # Pink
    "success": "#10B981",      # Emerald
    "warning": "#F59E0B",      # Amber
    "danger": "#EF4444",       # Red
    "info": "#3B82F6",         # Blue
    "teal": "#14B8A6",
    "orange": "#F97316",
    "bg": "#0F172A",
    "card": "#1E293B",
    "text": "#E2E8F0",
    "muted": "#94A3B8",
}

CHART_COLORS = [
    COLORS["primary"], COLORS["accent"], COLORS["success"],
    COLORS["warning"], COLORS["info"], COLORS["teal"], COLORS["orange"]
]

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS["text"], size=12),
    legend=dict(
        bgcolor="rgba(30,41,59,0.8)",
        bordercolor="rgba(99,102,241,0.3)",
        borderwidth=1,
        font=dict(size=11),
    ),
    margin=dict(l=50, r=30, t=60, b=50),
    xaxis=dict(
        gridcolor="rgba(148,163,184,0.1)",
        linecolor="rgba(148,163,184,0.2)",
        showgrid=True,
    ),
    yaxis=dict(
        gridcolor="rgba(148,163,184,0.1)",
        linecolor="rgba(148,163,184,0.2)",
        showgrid=True,
    ),
)


def _apply_layout(fig, title: str = "", height: int = 420, **kwargs):
    """Apply common layout settings to a figure."""
    layout = {**LAYOUT_DEFAULTS, "title": dict(text=title, font=dict(size=16, color=COLORS["text"])), "height": height}
    layout.update(kwargs)
    fig.update_layout(**layout)
    return fig


def _get_dates_for_index(df: pd.DataFrame, index_values) -> pd.Series:
    """
    Safely retrieve Date values from df matching given integer index positions.
    Works correctly regardless of whether df uses a default RangeIndex or custom index.
    """
    # index_values comes from X.index which refers to positions in the original df
    # We need to map these back to dates safely
    try:
        # Try positional iloc first (works when index values are valid positions)
        valid_positions = [i for i in index_values if 0 <= i < len(df)]
        return df["Date"].iloc[valid_positions]
    except Exception:
        # Fall back to loc if iloc fails
        return df.loc[df.index.isin(index_values), "Date"]


# ─── EDA Charts ───────────────────────────────────────────────────────────────

def plot_hhs_care_timeline(df: pd.DataFrame) -> go.Figure:
    """Interactive time series of Children in HHS Care."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["In_HHS"],
        mode="lines",
        name="Children in HHS Care",
        line=dict(color=COLORS["primary"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.15)",
        hovertemplate="<b>%{x|%b %Y}</b><br>Children in HHS Care: <b>%{y:,}</b><extra></extra>",
    ))

    # Add trend line
    z = np.polyfit(range(len(df)), df["In_HHS"], 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=p(range(len(df))),
        mode="lines",
        name="Trend",
        line=dict(color=COLORS["accent"], width=1.5, dash="dot"),
        hovertemplate="Trend: <b>%{y:,.0f}</b><extra></extra>",
    ))

    _apply_layout(fig, "Children in HHS Care Over Time", height=440)
    return fig


def plot_apprehended_vs_discharged(df: pd.DataFrame) -> go.Figure:
    """Bar + line combo chart comparing apprehensions and discharges."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Apprehended"],
        name="Apprehended",
        marker_color=COLORS["warning"],
        opacity=0.8,
        hovertemplate="<b>%{x|%b %Y}</b><br>Apprehended: <b>%{y:,}</b><extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Discharged"],
        mode="lines+markers",
        name="Discharged",
        line=dict(color=COLORS["success"], width=2.5),
        marker=dict(size=5),
        hovertemplate="<b>%{x|%b %Y}</b><br>Discharged: <b>%{y:,}</b><extra></extra>",
    ), secondary_y=True)

    _apply_layout(fig, "Children Apprehended vs. Discharged", height=420)
    fig.update_yaxes(title_text="Apprehended", secondary_y=False, gridcolor="rgba(148,163,184,0.1)")
    fig.update_yaxes(title_text="Discharged", secondary_y=True, showgrid=False)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list) -> go.Figure:
    """Correlation heatmap for numeric columns."""
    corr = df[numeric_cols].corr()

    display_names = {
        "Apprehended": "Apprehended",
        "In_CBP": "In CBP",
        "Transferred_Out": "Transferred Out",
        "In_HHS": "In HHS",
        "Discharged": "Discharged"
    }
    labels = [display_names.get(c, c) for c in numeric_cols]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, "#1E293B"],
            [0.3, "#4338CA"],
            [0.6, "#6366F1"],
            [0.8, "#A78BFA"],
            [1.0, "#EC4899"],
        ],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        hoverongaps=False,
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>Correlation: <b>%{z:.3f}</b><extra></extra>",
    ))

    _apply_layout(fig, "Correlation Heatmap", height=420)
    fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    return fig


def plot_distribution_boxplot(df: pd.DataFrame, numeric_cols: list) -> go.Figure:
    """Box plots for all numeric columns."""
    display_names = {
        "Apprehended": "Apprehended",
        "In_CBP": "In CBP",
        "Transferred_Out": "Transferred Out",
        "In_HHS": "In HHS",
        "Discharged": "Discharged"
    }
    fig = go.Figure()

    for i, col in enumerate(numeric_cols):
        fig.add_trace(go.Box(
            y=df[col],
            name=display_names.get(col, col),
            marker_color=CHART_COLORS[i % len(CHART_COLORS)],
            boxmean="sd",
            hovertemplate="<b>%{x}</b><br>Value: <b>%{y:,}</b><extra></extra>",
        ))

    _apply_layout(fig, "Distribution of All Metrics (Box Plot)", height=420)
    return fig


def plot_monthly_analysis(df: pd.DataFrame, col: str = "In_HHS") -> go.Figure:
    """Monthly aggregation bar chart."""
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    monthly = df.groupby("Month_Name")[col].mean().reindex(month_order)

    display = {
        "Apprehended": "Apprehended", "In_CBP": "In CBP",
        "Transferred_Out": "Transferred Out", "In_HHS": "In HHS", "Discharged": "Discharged"
    }

    fig = go.Figure(go.Bar(
        x=monthly.index,
        y=monthly.values,
        marker=dict(
            color=monthly.values,
            colorscale=[[0, "#4338CA"], [0.5, "#6366F1"], [1, "#EC4899"]],
            showscale=False,
        ),
        hovertemplate="<b>%{x}</b><br>Avg: <b>%{y:,.0f}</b><extra></extra>",
    ))

    _apply_layout(fig, f"Average Monthly Pattern — {display.get(col, col)}", height=400)
    return fig


def plot_yearly_analysis(df: pd.DataFrame, col: str = "In_HHS") -> go.Figure:
    """Yearly trend line chart."""
    yearly = df.groupby("Year")[col].mean().reset_index()
    display = {
        "Apprehended": "Apprehended", "In_CBP": "In CBP",
        "Transferred_Out": "Transferred Out", "In_HHS": "In HHS", "Discharged": "Discharged"
    }

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly["Year"], y=yearly[col],
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=3),
        marker=dict(size=10, color=COLORS["accent"], symbol="circle",
                    line=dict(color=COLORS["primary"], width=2)),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.1)",
        hovertemplate="<b>%{x}</b><br>Avg: <b>%{y:,.0f}</b><extra></extra>",
    ))

    _apply_layout(fig, f"Yearly Average — {display.get(col, col)}", height=380)
    fig.update_xaxes(dtick=1)
    return fig


# ─── Forecast Charts ──────────────────────────────────────────────────────────

def plot_rf_predictions(rf_result: dict, df: pd.DataFrame) -> go.Figure:
    """Random Forest actual vs predicted chart."""
    X_full = rf_result["X_full"]
    y_full = rf_result["y_full"]
    y_full_pred = rf_result["y_full_pred"]

    # Align dates with the cleaned X index - use positional indexing
    x_indices = list(X_full.index)
    dates = df["Date"].iloc[x_indices].reset_index(drop=True)

    split_idx = len(rf_result["X_train"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=y_full.values,
        mode="lines",
        name="Actual",
        line=dict(color=COLORS["info"], width=2),
        hovertemplate="<b>%{x|%b %Y}</b><br>Actual: <b>%{y:,}</b><extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=dates.iloc[:split_idx], y=y_full_pred[:split_idx],
        mode="lines",
        name="Train Prediction",
        line=dict(color=COLORS["success"], width=2, dash="dot"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Train Pred: <b>%{y:,}</b><extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=dates.iloc[split_idx:], y=y_full_pred[split_idx:],
        mode="lines+markers",
        name="Test Prediction",
        line=dict(color=COLORS["accent"], width=2.5),
        marker=dict(size=6),
        hovertemplate="<b>%{x|%b %Y}</b><br>Test Pred: <b>%{y:,}</b><extra></extra>",
    ))

    _apply_layout(fig, "Random Forest — Actual vs Predicted", height=450)
    return fig


def plot_arima_predictions(arima_result: dict) -> go.Figure:
    """ARIMA/SARIMA actual vs predicted chart with confidence intervals."""
    if not arima_result.get("success"):
        return None

    train_ts = arima_result["train_ts"]
    test_ts = arima_result["test_ts"]
    y_pred = arima_result["y_pred"]
    conf_int = arima_result["conf_int"]
    in_sample = arima_result["in_sample"]

    fig = go.Figure()

    # Actual train
    fig.add_trace(go.Scatter(
        x=train_ts.index, y=train_ts.values,
        mode="lines", name="Training Data",
        line=dict(color=COLORS["muted"], width=1.5),
        hovertemplate="<b>%{x|%b %Y}</b><br>Actual: <b>%{y:,}</b><extra></extra>",
    ))

    # Actual test
    fig.add_trace(go.Scatter(
        x=test_ts.index, y=test_ts.values,
        mode="lines+markers", name="Test Actual",
        line=dict(color=COLORS["info"], width=2),
        marker=dict(size=6),
        hovertemplate="<b>%{x|%b %Y}</b><br>Actual: <b>%{y:,}</b><extra></extra>",
    ))

    # Predictions
    pred_dates = test_ts.index
    fig.add_trace(go.Scatter(
        x=pred_dates, y=y_pred,
        mode="lines+markers", name="SARIMA Forecast",
        line=dict(color=COLORS["accent"], width=2.5),
        marker=dict(size=7, symbol="diamond"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: <b>%{y:,}</b><extra></extra>",
    ))

    # Confidence interval
    if conf_int is not None and len(conf_int) > 0:
        fig.add_trace(go.Scatter(
            x=list(pred_dates) + list(pred_dates[::-1]),
            y=list(conf_int.iloc[:, 1].values) + list(conf_int.iloc[:, 0].values[::-1]),
            fill="toself",
            fillcolor="rgba(236,72,153,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Confidence Interval",
            hoverinfo="skip",
        ))

    _apply_layout(fig, "SARIMA — Actual vs Predicted", height=450)
    return fig


def plot_future_forecast(forecast: dict, historical_ts: pd.Series, model_name: str = "Forecast") -> go.Figure:
    """Plot future forecast with confidence bands."""
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=historical_ts.index, y=historical_ts.values,
        mode="lines", name="Historical Data",
        line=dict(color=COLORS["info"], width=2),
        hovertemplate="<b>%{x|%b %Y}</b><br>Actual: <b>%{y:,}</b><extra></extra>",
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(forecast["dates"]) + list(forecast["dates"][::-1]),
        y=list(forecast["upper"]) + list(forecast["lower"][::-1]),
        fill="toself",
        fillcolor="rgba(99,102,241,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Band",
        hoverinfo="skip",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast["dates"], y=forecast["values"],
        mode="lines+markers",
        name=f"{model_name} Forecast",
        line=dict(color=COLORS["primary"], width=3),
        marker=dict(size=8, color=COLORS["accent"], symbol="circle",
                    line=dict(color="white", width=1)),
        hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: <b>%{y:,.0f}</b><extra></extra>",
    ))

    _apply_layout(fig, f"Future Forecast — {forecast['horizon_days']} Day Horizon", height=460)
    return fig


def plot_feature_importance(feature_importance: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    top = feature_importance.head(12)

    fig = go.Figure(go.Bar(
        x=top["Importance"],
        y=top["Feature"],
        orientation="h",
        marker=dict(
            color=top["Importance"],
            colorscale=[[0, "#4338CA"], [0.5, "#6366F1"], [1, "#EC4899"]],
            showscale=False,
        ),
        hovertemplate="<b>%{y}</b><br>Importance: <b>%{x:.4f}</b><extra></extra>",
    ))

    _apply_layout(fig, "Feature Importance — Random Forest", height=400)
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing model metrics."""
    metrics = ["MAE", "RMSE", "R² Score"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=metrics)

    for i, metric in enumerate(metrics, 1):
        for j, (_, row) in enumerate(comparison_df.iterrows()):
            fig.add_trace(go.Bar(
                x=[row["Model"]],
                y=[row[metric]],
                name=row["Model"],
                marker_color=CHART_COLORS[j],
                showlegend=(i == 1),
                hovertemplate=f"<b>{metric}</b>: <b>%{{y:,.4f}}</b><extra></extra>",
            ), row=1, col=i)

    fig.update_layout(
        title=dict(text="Model Performance Comparison", font=dict(size=16, color=COLORS["text"])),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        height=380,
        showlegend=True,
        barmode="group",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.1)")
    return fig


def plot_confidence_intervals(rf_result: dict, df: pd.DataFrame) -> go.Figure:
    """Prediction with confidence intervals from Random Forest."""
    X_test = rf_result["X_test"]
    y_test = rf_result["y_test"]
    y_pred = rf_result["y_pred"]

    # Simulate confidence intervals from tree variance
    model = rf_result["model"]
    individual_preds = np.array([tree.predict(X_test) for tree in model.estimators_])
    lower = np.percentile(individual_preds, 5, axis=0)
    upper = np.percentile(individual_preds, 95, axis=0)

    # Use positional indexing to get dates
    x_indices = list(X_test.index)
    dates = df["Date"].iloc[x_indices].reset_index(drop=True)

    fig = go.Figure()

    # CI band
    fig.add_trace(go.Scatter(
        x=list(dates) + list(dates[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself",
        fillcolor="rgba(99,102,241,0.2)",
        line=dict(color="rgba(0,0,0,0)"),
        name="90% Confidence Band",
        hoverinfo="skip",
    ))

    # Actual
    fig.add_trace(go.Scatter(
        x=dates, y=y_test.values,
        mode="lines+markers",
        name="Actual",
        line=dict(color=COLORS["info"], width=2),
        marker=dict(size=5),
        hovertemplate="<b>%{x|%b %Y}</b><br>Actual: <b>%{y:,}</b><extra></extra>",
    ))

    # Predicted
    fig.add_trace(go.Scatter(
        x=dates, y=y_pred,
        mode="lines+markers",
        name="Predicted",
        line=dict(color=COLORS["accent"], width=2.5),
        marker=dict(size=7, symbol="diamond"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Predicted: <b>%{y:,}</b><extra></extra>",
    ))

    _apply_layout(fig, "Prediction with 90% Confidence Intervals (Random Forest)", height=450)
    return fig


def plot_scenario_analysis(baseline: float, scenario_pred: float, scenario_params: dict) -> go.Figure:
    """Gauge + bar for scenario analysis output."""
    labels = ["Baseline Forecast", "Scenario Forecast"]
    values = [baseline, scenario_pred]
    colors = [COLORS["info"], COLORS["accent"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:,.0f}" for v in values],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Value: <b>%{y:,}</b><extra></extra>",
    ))

    delta_pct = ((scenario_pred - baseline) / baseline) * 100 if baseline != 0 else 0
    delta_label = f"{'▲' if delta_pct >= 0 else '▼'} {abs(delta_pct):.1f}% change from baseline"

    _apply_layout(fig, f"Scenario Analysis — {delta_label}", height=380)
    return fig


def plot_bi_dashboard(df: pd.DataFrame) -> go.Figure:
    """Power BI-style multi-panel dashboard."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Children in HHS Care (Timeline)",
            "Monthly Average Pattern",
            "Apprehended vs Discharged",
            "Yearly Trend",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )

    # Panel 1: HHS Care timeline
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["In_HHS"],
        mode="lines", name="In HHS",
        line=dict(color=COLORS["primary"], width=2),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
    ), row=1, col=1)

    # Panel 2: Monthly averages
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly = df.groupby(df["Date"].dt.strftime("%b"))["In_HHS"].mean().reindex(month_order)
    fig.add_trace(go.Bar(
        x=monthly.index, y=monthly.values,
        name="Monthly Avg",
        marker=dict(color=monthly.values,
                    colorscale=[[0,"#4338CA"],[1,"#EC4899"]],
                    showscale=False),
    ), row=1, col=2)

    # Panel 3: Apprehended vs Discharged
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Apprehended"],
        name="Apprehended", marker_color=COLORS["warning"], opacity=0.7,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Discharged"],
        mode="lines", name="Discharged",
        line=dict(color=COLORS["success"], width=2),
    ), row=2, col=1)

    # Panel 4: Yearly trend
    yearly = df.groupby("Year")["In_HHS"].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=yearly["Year"], y=yearly["In_HHS"],
        mode="lines+markers", name="Yearly Avg",
        line=dict(color=COLORS["accent"], width=3),
        marker=dict(size=10, color=COLORS["primary"]),
    ), row=2, col=2)

    fig.update_layout(
        title=dict(text="Business Intelligence Dashboard", font=dict(size=18, color=COLORS["text"])),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        height=700,
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.1)", linecolor="rgba(148,163,184,0.2)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.1)", linecolor="rgba(148,163,184,0.2)")
    return fig
