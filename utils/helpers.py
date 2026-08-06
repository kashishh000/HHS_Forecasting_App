"""
UI Helper and HTML Component Rendering Utilities for HHS UAC Application.
"""

import streamlit as st
import pandas as pd
import base64

def render_header(title: str, subtitle: str):
    """
    Renders custom header banner with glassmorphism styling.
    """
    html = f"""
    <div class="main-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_kpi_card(title: str, value: str, subtitle: str = "", delta_badge: str = ""):
    """
    Renders styled metric card.
    """
    badge_html = ""
    if delta_badge:
        if delta_badge.startswith("+"):
            badge_html = f'<span class="badge-positive">{delta_badge}</span>'
        elif delta_badge.startswith("-"):
            badge_html = f'<span class="badge-negative">{delta_badge}</span>'
        else:
            badge_html = f'<span class="badge-positive">{delta_badge}</span>'

    html = f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value} {badge_html}</div>
        <div class="kpi-sub">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_alert(message: str, type_: str = "info"):
    """
    Renders custom colored alert box (info, warning, success).
    """
    html = f"""
    <div class="custom-alert alert-{type_}">
        {message}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def format_number(val: float or int) -> str:
    """
    Formats numbers with comma separators.
    """
    try:
        return f"{int(val):,}"
    except (ValueError, TypeError):
        return str(val)

def create_download_button(df: pd.DataFrame, filename: str = "uac_dataset.csv", button_text: str = "📥 Download CSV Data"):
    """
    Generates download button for dataframes.
    """
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=button_text,
        data=csv,
        file_name=filename,
        mime='text/csv'
    )
