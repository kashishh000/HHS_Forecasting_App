"""
pages/eda.py — Exploratory Data Analysis page.
Correlation heatmap, time series, box plots, monthly/yearly analysis.
"""

import streamlit as st
from utils.data_loader import load_data, get_numeric_columns, get_column_display_names
from utils.charts import (
    plot_hhs_care_timeline,
    plot_apprehended_vs_discharged,
    plot_correlation_heatmap,
    plot_distribution_boxplot,
    plot_monthly_analysis,
    plot_yearly_analysis,
)
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def render():
    """Render the EDA page."""

    st.markdown("# 🔍 Exploratory Data Analysis")
    st.markdown("<p style='color:#94A3B8; margin-top:-0.5rem;'>Deep-dive into patterns, trends, correlations, and distributions in the HHS UAC dataset.</p>", unsafe_allow_html=True)
    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Load Data ──────────────────────────────────────────────────────────────
    with st.spinner("Loading data..."):
        try:
            df = load_data("data/dataset.csv")
        except FileNotFoundError as e:
            st.error(str(e))
            return

    numeric_cols = get_numeric_columns()
    display_names = get_column_display_names()

    # ── Sidebar Controls ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### EDA Controls")
        year_filter = st.multiselect(
            "Filter by Year",
            options=sorted(df["Year"].unique()),
            default=sorted(df["Year"].unique()),
            key="eda_year_filter",
        )

    df_filtered = df[df["Year"].isin(year_filter)] if year_filter else df

    # ── Tab Layout ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Timeline",
        "🔥 Correlation",
        "📦 Distributions",
        "📅 Monthly Pattern",
        "📆 Yearly Trend",
        "🔄 Comparisons",
    ])

    # ── Tab 1: Timeline ────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### 👶 Children in HHS Care — Full Timeline")
        st.markdown("<p style='color:#94A3B8;'>Tracks the total number of unaccompanied children in HHS care over the study period.</p>", unsafe_allow_html=True)
        fig1 = plot_hhs_care_timeline(df_filtered)
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)
        st.markdown("### 🚔 Apprehended vs. Discharged")
        st.markdown("<p style='color:#94A3B8;'>Comparison of monthly apprehensions (bar) and discharges (line) over time.</p>", unsafe_allow_html=True)
        fig2 = plot_apprehended_vs_discharged(df_filtered)
        st.plotly_chart(fig2, use_container_width=True)

        # Multi-line chart for all metrics
        st.markdown("### 📊 All Metrics Overlay")
        selected_cols = st.multiselect(
            "Select metrics to overlay",
            options=numeric_cols,
            default=["In_HHS", "Apprehended", "Discharged"],
            format_func=lambda x: display_names.get(x, x),
            key="eda_overlay",
        )
        if selected_cols:
            colors = ["#6366F1", "#EC4899", "#10B981", "#F59E0B", "#3B82F6"]
            fig_overlay = go.Figure()
            for i, col in enumerate(selected_cols):
                fig_overlay.add_trace(go.Scatter(
                    x=df_filtered["Date"], y=df_filtered[col],
                    mode="lines", name=display_names.get(col, col),
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{display_names.get(col,col)}: <b>%{{y:,}}</b><extra></extra>",
                ))
            fig_overlay.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#E2E8F0"),
                height=400,
                legend=dict(bgcolor="rgba(30,41,59,0.8)", bordercolor="rgba(99,102,241,0.3)", borderwidth=1),
                xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
                yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
                title=dict(text="All Metrics Overlay", font=dict(size=16)),
            )
            st.plotly_chart(fig_overlay, use_container_width=True)

    # ── Tab 2: Correlation ─────────────────────────────────────────────────────
    with tab2:
        st.markdown("### 🔥 Correlation Heatmap")
        st.markdown("<p style='color:#94A3B8;'>Pearson correlation coefficients between all numeric metrics. Values close to ±1 indicate strong relationships.</p>", unsafe_allow_html=True)
        fig_corr = plot_correlation_heatmap(df_filtered, numeric_cols)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Correlation insights
        corr = df_filtered[numeric_cols].corr()
        st.markdown("#### 🔎 Strongest Correlations")
        pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                c1, c2 = numeric_cols[i], numeric_cols[j]
                pairs.append({
                    "Pair": f"{display_names.get(c1, c1)} ↔ {display_names.get(c2, c2)}",
                    "Correlation": round(corr.loc[c1, c2], 4),
                    "Strength": (
                        "🔴 Very Strong" if abs(corr.loc[c1, c2]) >= 0.85
                        else "🟠 Strong" if abs(corr.loc[c1, c2]) >= 0.6
                        else "🟡 Moderate" if abs(corr.loc[c1, c2]) >= 0.4
                        else "🔵 Weak"
                    ),
                })
        pairs_df = pd.DataFrame(pairs).sort_values("Correlation", ascending=False, key=abs)
        st.dataframe(pairs_df, use_container_width=True)

    # ── Tab 3: Distributions ───────────────────────────────────────────────────
    with tab3:
        st.markdown("### 📦 Distribution Box Plots")
        st.markdown("<p style='color:#94A3B8;'>Box plots showing median, IQR, and outliers for each metric. The X marks the mean.</p>", unsafe_allow_html=True)
        fig_box = plot_distribution_boxplot(df_filtered, numeric_cols)
        st.plotly_chart(fig_box, use_container_width=True)

        # Histogram
        st.markdown("### 📊 Distribution Histogram")
        hist_col = st.selectbox(
            "Select column for histogram",
            options=numeric_cols,
            format_func=lambda x: display_names.get(x, x),
            key="hist_col",
        )
        fig_hist = px.histogram(
            df_filtered, x=hist_col, nbins=30,
            title=f"Distribution — {display_names.get(hist_col, hist_col)}",
            color_discrete_sequence=["#6366F1"],
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#E2E8F0"),
            height=380, bargap=0.05,
            xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.1)", title="Count"),
        )
        fig_hist.update_traces(
            hovertemplate="Value: <b>%{x:,}</b><br>Count: <b>%{y}</b><extra></extra>"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Tab 4: Monthly Pattern ─────────────────────────────────────────────────
    with tab4:
        st.markdown("### 📅 Average Monthly Pattern")
        monthly_col = st.selectbox(
            "Select metric for monthly analysis",
            options=numeric_cols,
            format_func=lambda x: display_names.get(x, x),
            key="monthly_col",
        )
        fig_monthly = plot_monthly_analysis(df_filtered, monthly_col)
        st.plotly_chart(fig_monthly, use_container_width=True)

        # Monthly pivot heatmap
        st.markdown("### 🗓️ Monthly Heatmap (Year × Month)")
        pivot = df_filtered.pivot_table(index="Year", columns="Month", values=monthly_col, aggfunc="mean").round(0)
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot.columns = [month_names[m-1] for m in pivot.columns]

        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale=[[0,"#1E293B"],[0.4,"#4338CA"],[0.7,"#6366F1"],[1,"#EC4899"]],
            text=[[f"{v:,.0f}" if not pd.isna(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont=dict(size=11, color="white"),
            hovertemplate="<b>%{y} | %{x}</b><br>Avg: <b>%{z:,.0f}</b><extra></extra>",
        ))
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter,sans-serif", color="#E2E8F0"),
            height=380,
            title=dict(text=f"Monthly Heatmap — {display_names.get(monthly_col, monthly_col)}", font=dict(size=16)),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Tab 5: Yearly Trend ────────────────────────────────────────────────────
    with tab5:
        st.markdown("### 📆 Yearly Average Trends")
        yearly_col = st.selectbox(
            "Select metric for yearly analysis",
            options=numeric_cols,
            format_func=lambda x: display_names.get(x, x),
            key="yearly_col",
        )
        fig_yearly = plot_yearly_analysis(df_filtered, yearly_col)
        st.plotly_chart(fig_yearly, use_container_width=True)

        # Yearly stats table
        yearly_stats = df_filtered.groupby("Year")[numeric_cols].agg(["mean","min","max"]).round(0).astype(int)
        yearly_stats.columns = [f"{display_names.get(c,c)} ({s.title()})" for c, s in yearly_stats.columns]
        st.markdown("#### 📋 Yearly Statistics Table")
        st.dataframe(yearly_stats, use_container_width=True)

    # ── Tab 6: Comparisons ─────────────────────────────────────────────────────
    with tab6:
        st.markdown("### 🔄 Scatter Plot — Metric vs Metric")
        col_x = st.selectbox("X-Axis", numeric_cols, index=0, format_func=lambda x: display_names.get(x, x), key="scatter_x")
        col_y = st.selectbox("Y-Axis", numeric_cols, index=3, format_func=lambda x: display_names.get(x, x), key="scatter_y")

        df_scatter = df_filtered.copy()
        df_scatter["Date_Formatted"] = df_scatter["Date"].dt.strftime("%b %Y")
        fig_scatter = px.scatter(
            df_scatter, x=col_x, y=col_y, color="Year",
            trendline="ols",
            labels={col_x: display_names.get(col_x, col_x), col_y: display_names.get(col_y, col_y)},
            title=f"{display_names.get(col_x,col_x)} vs {display_names.get(col_y,col_y)}",
            color_continuous_scale="Plasma",
            hover_data=["Date_Formatted"],
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#E2E8F0"),
            height=430,
            xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Rolling average
        st.markdown("### 📉 Rolling Average Analysis")
        roll_col = st.selectbox("Select metric", numeric_cols, index=3, format_func=lambda x: display_names.get(x,x), key="roll_col")
        window = st.slider("Rolling window (months)", 2, 12, 3)

        df_roll = df_filtered.copy()
        df_roll[f"Rolling_{window}M"] = df_roll[roll_col].rolling(window).mean()

        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            x=df_roll["Date"], y=df_roll[roll_col],
            mode="lines", name="Raw Data",
            line=dict(color="#6366F1", width=1.5, dash="dot"), opacity=0.6,
        ))
        fig_roll.add_trace(go.Scatter(
            x=df_roll["Date"], y=df_roll[f"Rolling_{window}M"],
            mode="lines", name=f"{window}-Month Rolling Avg",
            line=dict(color="#EC4899", width=2.5),
        ))
        fig_roll.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter,sans-serif", color="#E2E8F0"),
            height=380,
            title=dict(text=f"Rolling Average — {display_names.get(roll_col,roll_col)}", font=dict(size=16)),
            legend=dict(bgcolor="rgba(30,41,59,0.8)", bordercolor="rgba(99,102,241,0.3)", borderwidth=1),
            xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        )
        st.plotly_chart(fig_roll, use_container_width=True)
