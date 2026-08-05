"""
pages/bi_dashboard.py — Business Intelligence Dashboard page.
Power BI-style dashboard with KPIs, trend charts, monthly/yearly analysis, and filters.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_loader import load_data, compute_kpis, get_numeric_columns, get_column_display_names


def render():
    """Render the BI Dashboard page."""

    st.markdown("# 📈 Business Intelligence Dashboard")
    st.markdown("<p style='color:#94A3B8; margin-top:-0.5rem;'>Power BI-style operational intelligence dashboard for the HHS UAC Program.</p>", unsafe_allow_html=True)
    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Load ───────────────────────────────────────────────────────────────────
    with st.spinner("Loading dashboard data..."):
        try:
            df = load_data("data/dataset.csv")
        except FileNotFoundError as e:
            st.error(str(e))
            return

    numeric_cols = get_numeric_columns()
    display_names = get_column_display_names()
    kpis = compute_kpis(df)

    # ── Dashboard Filters ──────────────────────────────────────────────────────
    st.markdown("### 🔧 Dashboard Filters")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        years = sorted(df["Year"].unique().tolist())
        sel_years = st.multiselect("📅 Year", years, default=years, key="bi_years")
    with f2:
        quarters = sorted(df["Quarter"].unique().tolist())
        sel_quarters = st.multiselect("🗓️ Quarter", quarters, default=quarters, key="bi_quarters")
    with f3:
        primary_metric = st.selectbox(
            "📊 Primary Metric",
            numeric_cols,
            index=3,
            format_func=lambda x: display_names.get(x, x),
            key="bi_primary",
        )
    with f4:
        chart_type = st.selectbox("📈 Chart Style", ["Area", "Line", "Bar"], key="bi_chart_type")

    # Apply filters
    df_f = df[df["Year"].isin(sel_years) & df["Quarter"].isin(sel_quarters)].copy() if sel_years and sel_quarters else df.copy()

    if df_f.empty:
        st.warning("No data matches the selected filters.")
        return

    # ── KPI Row ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><span class="section-icon">📊</span><h2>Key Performance Indicators</h2></div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpi_configs = [
        (k1, "👶 In HHS Care", f"{int(df_f['In_HHS'].iloc[-1]):,}", f"{df_f['In_HHS'].mean():,.0f} avg", "#6366F1"),
        (k2, "🚔 Apprehended", f"{int(df_f['Apprehended'].iloc[-1]):,}", f"{df_f['Apprehended'].mean():,.0f} avg", "#F59E0B"),
        (k3, "✅ Discharged", f"{int(df_f['Discharged'].iloc[-1]):,}", f"{df_f['Discharged'].mean():,.0f} avg", "#10B981"),
        (k4, "🏛️ In CBP", f"{int(df_f['In_CBP'].iloc[-1]):,}", f"{df_f['In_CBP'].mean():,.0f} avg", "#3B82F6"),
        (k5, "🔄 Transferred", f"{int(df_f['Transferred_Out'].iloc[-1]):,}", f"{df_f['Transferred_Out'].mean():,.0f} avg", "#EC4899"),
        (k6, "📅 Months", f"{len(df_f):,}", f"{df_f['Year'].min()}–{df_f['Year'].max()}", "#8B5CF6"),
    ]
    for col, label, val, delta_text, color in kpi_configs:
        with col:
            st.markdown(f"""
            <div style="background:linear-gradient(145deg,#1E293B,#263348); border:1px solid {color}40;
                        border-radius:12px; padding:1rem; text-align:center; margin-bottom:0.5rem;
                        border-top:3px solid {color};">
                <div style="font-size:0.7rem; color:#94A3B8; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">{label}</div>
                <div style="font-size:1.5rem; font-weight:700; color:#E2E8F0; font-family:'Space Grotesk',sans-serif; margin:0.3rem 0;">{val}</div>
                <div style="font-size:0.72rem; color:{color};">{delta_text}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Main Timeline Chart ────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header"><span class="section-icon">📈</span><h2>{display_names.get(primary_metric, primary_metric)} — Full Timeline</h2></div>', unsafe_allow_html=True)

    colors_map = {
        "In_HHS": "#6366F1", "Apprehended": "#F59E0B",
        "Discharged": "#10B981", "In_CBP": "#3B82F6",
        "Transferred_Out": "#EC4899",
    }
    color = colors_map.get(primary_metric, "#6366F1")

    rgba_color = "rgba(99, 102, 241, 0.2)"
    if primary_metric == "Apprehended":
        rgba_color = "rgba(245, 158, 11, 0.2)"
    elif primary_metric == "Discharged":
        rgba_color = "rgba(16, 185, 129, 0.2)"
    elif primary_metric == "In_CBP":
        rgba_color = "rgba(59, 130, 246, 0.2)"
    elif primary_metric == "Transferred_Out":
        rgba_color = "rgba(236, 72, 153, 0.2)"

    fig_main = go.Figure()
    if chart_type == "Area":
        fig_main.add_trace(go.Scatter(
            x=df_f["Date"], y=df_f[primary_metric],
            mode="lines", name=display_names.get(primary_metric, primary_metric),
            line=dict(color=color, width=2.5),
            fill="tozeroy", fillcolor=rgba_color,
            hovertemplate="<b>%{x|%b %Y}</b><br>Value: <b>%{y:,}</b><extra></extra>",
        ))
    elif chart_type == "Line":
        fig_main.add_trace(go.Scatter(
            x=df_f["Date"], y=df_f[primary_metric],
            mode="lines+markers", name=display_names.get(primary_metric, primary_metric),
            line=dict(color=color, width=2.5),
            marker=dict(size=5),
            hovertemplate="<b>%{x|%b %Y}</b><br>Value: <b>%{y:,}</b><extra></extra>",
        ))
    else:
        fig_main.add_trace(go.Bar(
            x=df_f["Date"], y=df_f[primary_metric],
            name=display_names.get(primary_metric, primary_metric),
            marker=dict(color=df_f[primary_metric], colorscale=[[0, rgba_color], [1, color]], showscale=False),
            hovertemplate="<b>%{x|%b %Y}</b><br>Value: <b>%{y:,}</b><extra></extra>",
        ))

    # Add 6-month rolling average
    roll = df_f[primary_metric].rolling(6).mean()
    fig_main.add_trace(go.Scatter(
        x=df_f["Date"], y=roll,
        mode="lines", name="6-Month Avg",
        line=dict(color="#EC4899", width=2, dash="dot"),
        hovertemplate="6M Avg: <b>%{y:,.0f}</b><extra></extra>",
    ))

    fig_main.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,sans-serif", color="#E2E8F0"),
        height=420,
        legend=dict(bgcolor="rgba(30,41,59,0.8)", bordercolor="rgba(99,102,241,0.3)", borderwidth=1),
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)", title="Count"),
        hovermode="x unified",
        title=dict(text=f"{display_names.get(primary_metric, primary_metric)} — {sel_years[0] if sel_years else ''} to {sel_years[-1] if sel_years else ''}", font=dict(size=16)),
    )
    st.plotly_chart(fig_main, use_container_width=True)

    # ── 4-Panel Dashboard ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><span class="section-icon">🗂️</span><h2>Multi-Panel Analytics</h2></div>', unsafe_allow_html=True)

    fig_multi = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Monthly Average Pattern",
            "Quarterly Breakdown",
            "Year-over-Year Comparison",
            "All Metrics — Latest Month",
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.1,
    )

    # Panel 1: Monthly average
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly_avg = df_f.groupby(df_f["Date"].dt.strftime("%b"))[primary_metric].mean().reindex(month_order)
    fig_multi.add_trace(go.Bar(
        x=monthly_avg.index, y=monthly_avg.values,
        marker=dict(color=monthly_avg.values,
                    colorscale=[[0,"#1E293B"],[0.5,color],[1,"#EC4899"]], showscale=False),
        hovertemplate="<b>%{x}</b><br>Avg: <b>%{y:,.0f}</b><extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # Panel 2: Quarterly boxplot
    for q, q_color in zip([1,2,3,4], ["#6366F1","#EC4899","#10B981","#F59E0B"]):
        q_data = df_f[df_f["Quarter"] == q][primary_metric]
        fig_multi.add_trace(go.Box(
            y=q_data, name=f"Q{q}",
            marker_color=q_color, boxmean="sd", showlegend=False,
        ), row=1, col=2)

    # Panel 3: Yearly trend
    yearly = df_f.groupby("Year")[primary_metric].mean().reset_index()
    fig_multi.add_trace(go.Scatter(
        x=yearly["Year"], y=yearly[primary_metric],
        mode="lines+markers", name="Yearly",
        line=dict(color=color, width=2.5),
        marker=dict(size=9, color="#EC4899", symbol="circle"),
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>Avg: <b>%{y:,.0f}</b><extra></extra>",
    ), row=2, col=1)

    # Panel 4: Latest month horizontal bar
    latest = df_f.iloc[-1][numeric_cols]
    bar_colors = [colors_map.get(c, "#6366F1") for c in numeric_cols]
    bar_labels = [display_names.get(c, c) for c in numeric_cols]
    fig_multi.add_trace(go.Bar(
        x=latest.values, y=bar_labels,
        orientation="h",
        marker_color=bar_colors,
        showlegend=False,
        hovertemplate="<b>%{y}</b>: <b>%{x:,}</b><extra></extra>",
    ), row=2, col=2)

    fig_multi.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,sans-serif", color="#E2E8F0"),
        height=680,
        title=dict(text="Multi-Panel Business Intelligence View", font=dict(size=18)),
        showlegend=False,
    )
    fig_multi.update_xaxes(gridcolor="rgba(148,163,184,0.1)", linecolor="rgba(148,163,184,0.2)")
    fig_multi.update_yaxes(gridcolor="rgba(148,163,184,0.1)", linecolor="rgba(148,163,184,0.2)")
    st.plotly_chart(fig_multi, use_container_width=True)

    # ── Stacked Area Chart — All Metrics ───────────────────────────────────────
    st.markdown('<div class="section-header"><span class="section-icon">📊</span><h2>All Metrics Stacked Overview</h2></div>', unsafe_allow_html=True)

    fig_stack = go.Figure()
    for col, c, rgba_c in zip(
        ["Apprehended", "Transferred_Out", "Discharged"],
        ["#F59E0B", "#EC4899", "#10B981"],
        ["rgba(245,158,11,0.25)", "rgba(236,72,153,0.25)", "rgba(16,185,129,0.25)"]
    ):
        fig_stack.add_trace(go.Scatter(
            x=df_f["Date"], y=df_f[col],
            mode="lines", name=display_names.get(col, col),
            line=dict(color=c, width=1.5),
            stackgroup="one",
            fillcolor=rgba_c,
            hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{display_names.get(col,col)}: <b>%{{y:,}}</b><extra></extra>",
        ))

    fig_stack.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,sans-serif", color="#E2E8F0"),
        height=380,
        title=dict(text="Stacked Area — Apprehended, Transferred, Discharged", font=dict(size=16)),
        legend=dict(bgcolor="rgba(30,41,59,0.8)", bordercolor="rgba(99,102,241,0.3)", borderwidth=1),
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)", title="Count (Stacked)"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    # ── Treemap ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><span class="section-icon">🗺️</span><h2>Yearly Volume Treemap</h2></div>', unsafe_allow_html=True)

    tree_data = df_f.groupby("Year")[primary_metric].sum().reset_index()
    fig_tree = go.Figure(go.Treemap(
        labels=[str(y) for y in tree_data["Year"]],
        parents=["" for _ in tree_data["Year"]],
        values=tree_data[primary_metric],
        textinfo="label+value+percent root",
        marker=dict(
            colorscale=[[0,"#1E293B"],[0.4,"#4338CA"],[0.7,"#6366F1"],[1,"#EC4899"]],
            showscale=True,
        ),
        hovertemplate="<b>%{label}</b><br>Total: <b>%{value:,}</b><extra></extra>",
    ))
    fig_tree.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,sans-serif", color="#E2E8F0"),
        height=380,
        title=dict(text=f"Yearly Volume Treemap — {display_names.get(primary_metric, primary_metric)}", font=dict(size=16)),
        margin=dict(t=60, l=10, r=10, b=10),
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    # ── Download ───────────────────────────────────────────────────────────────
    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)
    dl_df = df_f[["Date","Year","Quarter","Month_Name"] + numeric_cols].copy()
    dl_df["Date"] = dl_df["Date"].dt.strftime("%Y-%m-%d")
    csv = dl_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export Dashboard Data (CSV)", csv, "bi_dashboard_export.csv", "text/csv")
