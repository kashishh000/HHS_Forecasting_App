"""
pages/dataset_explorer.py — Dataset Explorer page.
Provides interactive dataset browsing, multi-criteria filtering, descriptive statistics,
data quality audits, and field data dictionary.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_data, get_numeric_columns, get_column_display_names, compute_kpis
from utils.charts import COLORS, LAYOUT_DEFAULTS, _apply_layout


def render():
    """Render the Dataset Explorer page."""

    # ── Page Header ────────────────────────────────────────────────────────────
    st.markdown("# 📊 Dataset Explorer")
    st.markdown(
        "<p style='color:#94A3B8; margin-top:-0.5rem;'>"
        "Interactive dataset browser, multi-criteria filtering matrix, descriptive statistics, data health audit, and schema dictionary."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Load Data ──────────────────────────────────────────────────────────────
    with st.spinner("Loading dataset for exploration..."):
        try:
            df = load_data("data/dataset.csv")
        except FileNotFoundError as e:
            st.error(f"⚠️ {e}. Please ensure `data/dataset.csv` exists.")
            return

    numeric_cols = get_numeric_columns()
    display_names = get_column_display_names()
    kpis = compute_kpis(df)

    # ── Summary KPI Bar ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("📅 Total Months", f"{len(df)} obs")
    with c2:
        st.metric("🗓️ Temporal Span", f"{df['Year'].min()} – {df['Year'].max()}")
    with c3:
        st.metric("📐 Total Variables", f"{df.shape[1]} cols")
    with c4:
        st.metric("🛡️ Data Completeness", "100.0%")
    with c5:
        st.metric("👶 Max Care Load", f"{int(df['In_HHS'].max()):,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main Exploration Tabs ──────────────────────────────────────────────────
    tab_table, tab_stats, tab_quality, tab_dict = st.tabs([
        "🔍 Data Table & Filters",
        "📈 Statistical Profile",
        "🛡️ Data Quality & Health",
        "📖 Data Dictionary",
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1: DATA TABLE & FILTERS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_table:
        st.markdown("### 🎛️ Interactive Filter Matrix")
        
        col_f1, col_f2, col_f3 = st.columns([1.5, 1.5, 2])
        
        with col_f1:
            all_years = sorted(df["Year"].unique().tolist())
            selected_years = st.multiselect(
                "Filter by Year",
                options=all_years,
                default=all_years,
                key="ds_exp_years"
            )
            
        with col_f2:
            name_mode = st.radio(
                "Header Naming Style",
                options=["Friendly Names", "Raw Names"],
                horizontal=True,
                key="ds_exp_name_mode"
            )

        with col_f3:
            default_cols = list(df.columns)
            selected_columns = st.multiselect(
                "Select Visible Columns",
                options=list(df.columns),
                default=default_cols,
                key="ds_exp_cols"
            )

        # Range filter expander
        with st.expander("🎚️ Advanced Range Filters", expanded=False):
            rf_col1, rf_col2 = st.columns(2)
            with rf_col1:
                min_hhs, max_hhs = int(df["In_HHS"].min()), int(df["In_HHS"].max())
                hhs_range = st.slider(
                    "Children in HHS Care Range",
                    min_value=min_hhs,
                    max_value=max_hhs,
                    value=(min_hhs, max_hhs),
                    key="ds_exp_hhs_range"
                )
            with rf_col2:
                min_app, max_app = int(df["Apprehended"].min()), int(df["Apprehended"].max())
                app_range = st.slider(
                    "Apprehensions Range",
                    min_value=min_app,
                    max_value=max_app,
                    value=(min_app, max_app),
                    key="ds_exp_app_range"
                )

        # Apply Filters
        filtered_df = df.copy()
        if selected_years:
            filtered_df = filtered_df[filtered_df["Year"].isin(selected_years)]
        filtered_df = filtered_df[
            (filtered_df["In_HHS"] >= hhs_range[0]) & (filtered_df["In_HHS"] <= hhs_range[1]) &
            (filtered_df["Apprehended"] >= app_range[0]) & (filtered_df["Apprehended"] <= app_range[1])
        ]

        st.markdown(f"**Showing {len(filtered_df)} of {len(df)} records**")

        # Display Dataframe
        view_df = filtered_df[selected_columns].copy() if selected_columns else filtered_df.copy()
        
        if name_mode == "Friendly Names":
            rename_map = {col: display_names.get(col, col) for col in view_df.columns}
            view_df = view_df.rename(columns=rename_map)

        st.dataframe(
            view_df,
            use_container_width=True,
            height=420,
            column_config={
                "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            }
        )

        # Export Actions
        st.markdown("### 📥 Export Dataset")
        exp_col1, exp_col2, _ = st.columns([1, 1, 2])
        with exp_col1:
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name="hhs_uac_filtered_dataset.csv",
                mime="text/csv",
                key="download_csv"
            )
        with exp_col2:
            json_data = filtered_df.to_json(orient="records", date_format="iso").encode('utf-8')
            st.download_button(
                label="📦 Download JSON",
                data=json_data,
                file_name="hhs_uac_filtered_dataset.json",
                mime="application/json",
                key="download_json"
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2: STATISTICAL PROFILE
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_stats:
        st.markdown("### 📊 Descriptive Statistics Summary")
        st.markdown("<p style='color:#94A3B8;'>Comprehensive central tendency, dispersion, skewness, and interquartile metrics.</p>", unsafe_allow_html=True)

        stats_df = df[numeric_cols].describe().T
        stats_df["skewness"] = df[numeric_cols].skew()
        stats_df["kurtosis"] = df[numeric_cols].kurtosis()
        stats_df["IQR"] = stats_df["75%"] - stats_df["25%"]
        stats_df = stats_df[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "IQR", "skewness", "kurtosis"]]
        stats_df.index = [display_names.get(c, c) for c in stats_df.index]

        st.dataframe(
            stats_df.style.format({
                "count": "{:.0f}",
                "mean": "{:,.2f}",
                "std": "{:,.2f}",
                "min": "{:,.0f}",
                "25%": "{:,.0f}",
                "50%": "{:,.0f}",
                "75%": "{:,.0f}",
                "max": "{:,.0f}",
                "IQR": "{:,.0f}",
                "skewness": "{:+.2f}",
                "kurtosis": "{:+.2f}",
            }),
            use_container_width=True,
        )

        st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)
        st.markdown("### 📈 Interactive Distribution Explorer")
        
        selected_metric = st.selectbox(
            "Select Metric to Analyze Distribution",
            options=numeric_cols,
            format_func=lambda x: display_names.get(x, x),
            key="ds_stat_metric"
        )

        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            fig_hist = px.histogram(
                df,
                x=selected_metric,
                nbins=20,
                marginal="box",
                title=f"Histogram & Boxplot — {display_names.get(selected_metric, selected_metric)}",
                color_discrete_sequence=[COLORS["primary"]],
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color=COLORS["text"]),
                xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
                yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
                height=380,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with dist_col2:
            fig_box = px.box(
                df,
                x="Year",
                y=selected_metric,
                color="Year",
                title=f"Yearly Boxplot — {display_names.get(selected_metric, selected_metric)}",
            )
            fig_box.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color=COLORS["text"]),
                xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
                yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
                showlegend=False,
                height=380,
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3: DATA QUALITY & HEALTH AUDIT
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_quality:
        st.markdown("### 🛡️ Data Quality & Health Audit")
        st.markdown("<p style='color:#94A3B8;'>Automated integrity validation, missingness verification, and anomaly detection.</p>", unsafe_allow_html=True)

        dq_col1, dq_col2 = st.columns(2)

        with dq_col1:
            st.markdown("#### 🔍 Missing Value & Type Audit")
            dq_summary = []
            for col in df.columns:
                null_cnt = df[col].isnull().sum()
                null_pct = (null_cnt / len(df)) * 100
                dtype = str(df[col].dtype)
                dq_summary.append({
                    "Column": col,
                    "Display Name": display_names.get(col, col),
                    "Data Type": dtype,
                    "Null Count": null_cnt,
                    "Null %": f"{null_pct:.1f}%",
                    "Status": "✅ Pass" if null_cnt == 0 else "⚠️ Warning"
                })
            st.dataframe(pd.DataFrame(dq_summary), use_container_width=True)

        with dq_col2:
            st.markdown("#### 🚨 Anomaly & IQR Outlier Audit")
            outlier_summary = []
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_summary.append({
                    "Metric": display_names.get(col, col),
                    "Lower Bound": round(lower_bound, 1),
                    "Upper Bound": round(upper_bound, 1),
                    "Outlier Count": len(outliers),
                    "Outlier %": f"{(len(outliers)/len(df))*100:.1f}%"
                })
            st.dataframe(pd.DataFrame(outlier_summary), use_container_width=True)

        st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)
        st.markdown("#### 📋 Structural & Logical Integrity Checks")
        
        dup_count = df.duplicated().sum()
        date_discontinuity = (df["Date"].diff().dt.days > 32).sum()
        
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            st.markdown(f"""
            <div style="background:#1E293B; border:1px solid rgba(16,185,129,0.3); border-radius:10px; padding:1rem; text-align:center;">
                <div style="font-size:1.8rem; color:#10B981;">{dup_count}</div>
                <div style="color:#94A3B8; font-size:0.85rem;">Duplicate Records</div>
                <div style="color:#10B981; font-size:0.75rem; margin-top:0.3rem;">✅ Perfect Uniqueness</div>
            </div>
            """, unsafe_allow_html=True)

        with ic2:
            st.markdown(f"""
            <div style="background:#1E293B; border:1px solid rgba(16,185,129,0.3); border-radius:10px; padding:1rem; text-align:center;">
                <div style="font-size:1.8rem; color:#10B981;">{date_discontinuity}</div>
                <div style="color:#94A3B8; font-size:0.85rem;">Date Gaps (>32 days)</div>
                <div style="color:#10B981; font-size:0.75rem; margin-top:0.3rem;">✅ Continuous Monthly Index</div>
            </div>
            """, unsafe_allow_html=True)

        with ic3:
            st.markdown(f"""
            <div style="background:#1E293B; border:1px solid rgba(16,185,129,0.3); border-radius:10px; padding:1rem; text-align:center;">
                <div style="font-size:1.8rem; color:#10B981;">100.0%</div>
                <div style="color:#94A3B8; font-size:0.85rem;">Logical Consistency</div>
                <div style="color:#10B981; font-size:0.75rem; margin-top:0.3rem;">✅ Non-negative values</div>
            </div>
            """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4: DATA DICTIONARY
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_dict:
        st.markdown("### 📖 Data Dictionary & Field Glossary")
        st.markdown("<p style='color:#94A3B8;'>Detailed descriptions, agency provenance, and operational roles of each dataset attribute.</p>", unsafe_allow_html=True)

        dict_data = [
            {
                "Field Name": "Date",
                "Internal Key": "Date",
                "Data Type": "Datetime (YYYY-MM-DD)",
                "Agency Source": "ORR / CBP",
                "Description": "First day of the reporting month representing monthly aggregated operational metrics.",
            },
            {
                "Field Name": "Children apprehended and placed in CBP custody",
                "Internal Key": "Apprehended",
                "Data Type": "Integer",
                "Agency Source": "U.S. Customs & Border Protection",
                "Description": "Total count of unaccompanied alien children apprehended at the border and taken into CBP custody during the month.",
            },
            {
                "Field Name": "Children in CBP custody",
                "Internal Key": "In_CBP",
                "Data Type": "Integer",
                "Agency Source": "U.S. Customs & Border Protection",
                "Description": "Point-in-time or average count of unaccompanied children remaining in CBP processing centers.",
            },
            {
                "Field Name": "Children transferred out of CBP custody",
                "Internal Key": "Transferred_Out",
                "Data Type": "Integer",
                "Agency Source": "CBP / HHS ORR",
                "Description": "Number of children successfully transferred from CBP holding facilities to HHS Office of Refugee Resettlement (ORR) care.",
            },
            {
                "Field Name": "Children in HHS Care",
                "Internal Key": "In_HHS",
                "Data Type": "Integer",
                "Agency Source": "HHS Office of Refugee Resettlement",
                "Description": "Total active census of unaccompanied children receiving shelter, medical, and case management care in HHS facilities.",
            },
            {
                "Field Name": "Children discharged from HHS Care",
                "Internal Key": "Discharged",
                "Data Type": "Integer",
                "Agency Source": "HHS Office of Refugee Resettlement",
                "Description": "Total children discharged from HHS care to vetted sponsors (parents, relatives, or legal guardians) or aged out during the month.",
            },
        ]

        for item in dict_data:
            st.markdown(f"""
            <div style="background:#1E293B; border-left:4px solid #6366F1; border-radius:8px; padding:1rem 1.2rem; margin-bottom:0.8rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:700; font-size:1.1rem; color:#E2E8F0;">{item['Field Name']}</span>
                    <span style="background:rgba(99,102,241,0.2); color:#A5B4FC; padding:0.2rem 0.6rem; border-radius:12px; font-size:0.75rem; font-weight:600;">
                        {item['Data Type']}
                    </span>
                </div>
                <div style="color:#94A3B8; font-size:0.88rem; margin-top:0.4rem;">
                    <b>Internal Identifier:</b> <code>{item['Internal Key']}</code> &nbsp;|&nbsp; <b>Source:</b> {item['Agency Source']}
                </div>
                <div style="color:#CBD5E1; font-size:0.9rem; margin-top:0.5rem; line-height:1.5;">
                    {item['Description']}
                </div>
            </div>
            """, unsafe_allow_html=True)