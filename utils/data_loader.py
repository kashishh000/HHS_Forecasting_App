"""
data_loader.py - Data loading and preprocessing utilities for HHS UAC Forecasting App
"""

import pandas as pd
import numpy as np
import os
import streamlit as st


@st.cache_data(show_spinner=False)
def load_data(filepath: str = "data/dataset.csv") -> pd.DataFrame:
    """
    Load and preprocess the HHS UAC dataset.
    
    Args:
        filepath: Path to the CSV dataset file.
    
    Returns:
        Preprocessed DataFrame with datetime index.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)

    # Parse the Date column
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Derive extra time features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%B")
    df["Quarter"] = df["Date"].dt.quarter
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)

    # Rename columns for convenience (short names for internal use)
    df.rename(columns={
        "Children apprehended and placed in CBP custody": "Apprehended",
        "Children in CBP custody": "In_CBP",
        "Children transferred out of CBP custody": "Transferred_Out",
        "Children in HHS Care": "In_HHS",
        "Children discharged from HHS Care": "Discharged"
    }, inplace=True)

    return df


def get_column_display_names() -> dict:
    """Return mapping of internal column names to display-friendly names."""
    return {
        "Apprehended": "Children Apprehended & Placed in CBP Custody",
        "In_CBP": "Children in CBP Custody",
        "Transferred_Out": "Children Transferred Out of CBP Custody",
        "In_HHS": "Children in HHS Care",
        "Discharged": "Children Discharged from HHS Care"
    }


def get_numeric_columns() -> list:
    """Return list of numeric metric columns."""
    return ["Apprehended", "In_CBP", "Transferred_Out", "In_HHS", "Discharged"]


def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Compute key performance indicators from the dataset.
    
    Args:
        df: Preprocessed DataFrame.
    
    Returns:
        Dictionary of KPI name -> value.
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    def pct_change(curr, prev_val):
        if prev_val == 0:
            return 0.0
        return round(((curr - prev_val) / prev_val) * 100, 2)

    return {
        "Total Records": len(df),
        "Date Range": f"{df['Date'].min().strftime('%b %Y')} – {df['Date'].max().strftime('%b %Y')}",
        "Current HHS Care": int(latest["In_HHS"]),
        "HHS Care Change %": pct_change(latest["In_HHS"], prev["In_HHS"]),
        "Current Apprehended": int(latest["Apprehended"]),
        "Apprehended Change %": pct_change(latest["Apprehended"], prev["Apprehended"]),
        "Current Discharged": int(latest["Discharged"]),
        "Discharged Change %": pct_change(latest["Discharged"], prev["Discharged"]),
        "Peak HHS Care": int(df["In_HHS"].max()),
        "Peak HHS Date": df.loc[df["In_HHS"].idxmax(), "Date"].strftime("%b %Y"),
        "Avg Monthly Apprehended": int(df["Apprehended"].mean()),
        "Avg Monthly Discharged": int(df["Discharged"].mean()),
        "Total Apprehended": int(df["Apprehended"].sum()),
        "Total Discharged": int(df["Discharged"].sum()),
    }


def prepare_time_series(df: pd.DataFrame, target_col: str = "In_HHS") -> pd.Series:
    """
    Prepare a time series with DatetimeIndex.
    
    Args:
        df: Preprocessed DataFrame.
        target_col: Column to use as the time series values.
    
    Returns:
        pd.Series with DatetimeIndex at monthly frequency.
    """
    ts = df.set_index("Date")[target_col].copy()
    ts.index = pd.DatetimeIndex(ts.index)
    # Set monthly start frequency explicitly — required for SARIMAX
    try:
        ts.index.freq = pd.tseries.frequencies.to_offset("MS")
    except Exception:
        ts = ts.asfreq("MS")
    return ts
