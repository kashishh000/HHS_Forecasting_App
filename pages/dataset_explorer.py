"""
Dataset Explorer Page Component for HHS UAC Dashboard.
"""

import streamlit as st
import pandas as pd
from utils.data_loader import load_raw_data
from utils.preprocessing import create_engineered_features
from utils.helpers import render_header, create_download_button, render_alert, format_number
from utils.charts import plot_correlation_matrix

def render():
    render_header(
        title="Dataset Explorer & Data Governance",
        subtitle="Interactive Data Viewer, Statistical Inspection, Quality Validation, and CSV Exporter"
    )

    df_raw = load_raw_data()
    if df_raw.empty:
        st.error("Dataset could not be loaded.")
        return

    # Data shape and status
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Records", format_number(len(df_raw)))
    col_b.metric("Total Features", format_number(len(df_raw.columns)))
    col_c.metric("Missing Values", "0 (Cleaned)")
    col_d.metric("Frequency", "Daily")

    st.markdown("<br>", unsafe_allow_html=True)

    # Filtering Sidebar / Top Controls
    with st.expander("🔍 Interactive Data Filters & Search", expanded=True):
        f_col1, f_col2, f_col3 = st.columns([1.2, 1.5, 1.5])
        
        min_d = df_raw["Date"].min().date()
        max_d = df_raw["Date"].max().date()

        with f_col1:
            date_range = st.date_input("Filter Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

        with f_col2:
            all_cols = list(df_raw.columns)
            selected_cols = st.multiselect("Select Columns to Display", options=all_cols, default=all_cols)

        with f_col3:
            show_engineered = st.checkbox("Include Engineered Features (Lags, Rolling)", value=False)

    # Filter dataframe by date range
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_filter, end_filter = date_range
        mask = (df_raw["Date"].dt.date >= start_filter) & (df_raw["Date"].dt.date <= end_filter)
        df_filtered = df_raw.loc[mask].copy()
    else:
        df_filtered = df_raw.copy()

    if show_engineered:
        df_display = create_engineered_features(df_filtered)
    else:
        df_display = df_filtered[selected_cols] if selected_cols else df_filtered

    # Main Interactive Data Table
    st.markdown("### 📋 Interactive Dataset View")
    st.dataframe(
        df_display,
        use_container_width=True,
        height=380,
        column_config={"Date": st.column_config.DateColumn("Reporting Date", format="YYYY-MM-DD")}
    )

    # Download Button
    create_download_button(df_display, filename="hhs_uac_cleaned_dataset.csv", button_text="📥 Download Current Filtered Data (CSV)")

    st.markdown("<br>", unsafe_allow_html=True)

    # Statistical Overview Tabs
    st.markdown("### 📈 Statistical Profiles & Correlation Analysis")
    t_stats, t_types, t_corr = st.tabs(["Summary Statistics", "Data Types & Null Check", "Correlation Matrix Table"])

    with t_stats:
        num_df = df_raw.select_dtypes(include=['int64', 'float64', 'int', 'float'])
        st.dataframe(num_df.describe().T.style.format("{:.2f}"), use_container_width=True)

    with t_types:
        info_df = pd.DataFrame({
            "Column Name": df_raw.columns,
            "Data Type": df_raw.dtypes.astype(str),
            "Non-Null Count": df_raw.notnull().sum().values,
            "Null Count": df_raw.isnull().sum().values,
            "Sample Value": [df_raw[c].iloc[0] for c in df_raw.columns]
        })
        st.dataframe(info_df, use_container_width=True)

    with t_corr:
        corr_fig = plot_correlation_matrix(df_raw)
        st.plotly_chart(corr_fig, use_container_width=True)

if __name__ == "__main__" or True:
    render()

