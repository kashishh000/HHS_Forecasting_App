"""
Data Loading & Cleaning Utilities for HHS UAC Forecasting App.
"""

import pandas as pd
import numpy as np
import streamlit as st
import os
from utils.config import DATA_PATH, RAW_NUMERIC_COLS

@st.cache_data(show_spinner=False)
def load_raw_data(data_path=DATA_PATH) -> pd.DataFrame:
    """
    Loads raw CSV data, parses dates, strips commas, converts types, and cleans dataset.
    """
    if not os.path.exists(data_path):
        st.error(f"Dataset file not found at: {data_path}")
        return pd.DataFrame()

    df = pd.read_csv(data_path)

    # Standardize column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Convert Date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    # Drop rows where Date is NaT
    df = df.dropna(subset=["Date"]).copy()

    # Clean numeric columns (remove commas if string, handle nulls)
    for col in RAW_NUMERIC_COLS:
        if col in df.columns:
            if df[col].dtype == object or isinstance(df[col].iloc[0], str):
                df[col] = df[col].astype(str).str.replace(",", "").str.strip()
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop duplicate dates and sort chronologically
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Forward fill / backward fill any missing values in numeric metrics
    numeric_cols = [c for c in RAW_NUMERIC_COLS if c in df.columns]
    df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0)

    # Ensure integer types for count columns
    for col in numeric_cols:
        df[col] = df[col].astype(int)

    return df

@st.cache_data(show_spinner=False)
def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Computes key statistical metrics and dataset info.
    """
    if df.empty:
        return {}

    start_date = df["Date"].min().strftime("%Y-%m-%d")
    end_date = df["Date"].max().strftime("%Y-%m-%d")
    total_days = len(df)

    latest = df.iloc[-1]
    prev_7d_mean = df.tail(7)["Children in HHS Care"].mean()

    return {
        "start_date": start_date,
        "end_date": end_date,
        "total_days": total_days,
        "latest_date": latest["Date"].strftime("%Y-%m-%d"),
        "latest_care": latest["Children in HHS Care"],
        "latest_apprehended": latest["Children apprehended and placed in CBP custody"],
        "latest_discharged": latest["Children discharged from HHS Care"],
        "latest_transferred": latest["Children transferred out of CBP custody"],
        "avg_7d_care": round(prev_7d_mean, 1),
        "columns_count": len(df.columns)
    }
