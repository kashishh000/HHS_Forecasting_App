"""
Feature Engineering and Preprocessing Pipeline for HHS UAC Forecasting.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

def create_engineered_features(df: pd.DataFrame, target_col: str = "Children in HHS Care") -> pd.DataFrame:
    """
    Applies comprehensive time-series feature engineering.
    """
    data = df.copy()
    data = data.sort_values("Date").reset_index(drop=True)

    # 1. Calendar & Cyclical Features
    data["Month"] = data["Date"].dt.month
    data["Quarter"] = data["Date"].dt.quarter
    data["Week"] = data["Date"].dt.isocalendar().week.astype(int)
    data["Year"] = data["Date"].dt.year
    data["DayOfWeek"] = data["Date"].dt.dayofweek

    # Sine/Cosine cyclical transforms
    data["Month_Sin"] = np.sin(2 * np.pi * data["Month"] / 12.0)
    data["Month_Cos"] = np.cos(2 * np.pi * data["Month"] / 12.0)
    data["Quarter_Sin"] = np.sin(2 * np.pi * data["Quarter"] / 4.0)
    data["Quarter_Cos"] = np.cos(2 * np.pi * data["Quarter"] / 4.0)
    data["DayOfWeek_Sin"] = np.sin(2 * np.pi * data["DayOfWeek"] / 7.0)
    data["DayOfWeek_Cos"] = np.cos(2 * np.pi * data["DayOfWeek"] / 7.0)

    # 2. Net Operational Flow
    if "Children transferred out of CBP custody" in data.columns and "Children discharged from HHS Care" in data.columns:
        data["Net_Flow"] = data["Children transferred out of CBP custody"] - data["Children discharged from HHS Care"]
    else:
        data["Net_Flow"] = 0

    # 3. Lag Features for target column
    data["Lag1"] = data[target_col].shift(1)
    data["Lag7"] = data[target_col].shift(7)
    data["Lag14"] = data[target_col].shift(14)

    # Lags for related intake/discharge indicators
    if "Children apprehended and placed in CBP custody" in data.columns:
        data["Apprehended_Lag1"] = data["Children apprehended and placed in CBP custody"].shift(1)
        data["Apprehended_Lag7"] = data["Children apprehended and placed in CBP custody"].shift(7)

    if "Children discharged from HHS Care" in data.columns:
        data["Discharged_Lag1"] = data["Children discharged from HHS Care"].shift(1)
        data["Discharged_Lag7"] = data["Children discharged from HHS Care"].shift(7)

    # 4. Rolling Statistics
    data["Rolling_Mean_7"] = data[target_col].shift(1).rolling(window=7).mean()
    data["Rolling_Mean_14"] = data[target_col].shift(1).rolling(window=14).mean()
    data["Rolling_Std_7"] = data[target_col].shift(1).rolling(window=7).std()
    data["Rolling_Std_14"] = data[target_col].shift(1).rolling(window=14).std()

    # 5. Holiday Proxy (Flagging major US holidays/periods)
    is_jan1 = (data["Month"] == 1) & (data["Date"].dt.day == 1)
    is_july4 = (data["Month"] == 7) & (data["Date"].dt.day == 4)
    is_dec = (data["Month"] == 12) & (data["Date"].dt.day >= 24)
    is_thanksgiving = (data["Month"] == 11) & (data["Date"].dt.day >= 22) & (data["Date"].dt.day <= 28) & (data["DayOfWeek"] == 3)

    data["Is_Holiday"] = (is_jan1 | is_july4 | is_dec | is_thanksgiving).astype(int)

    # Drop early rows with NaN from lags/rolling windows
    data_clean = data.dropna().reset_index(drop=True)
    return data_clean

def prepare_train_test_split(
    df_features: pd.DataFrame,
    target_col: str = "Children in HHS Care",
    test_size: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Splits features and target chronologically.
    """
    feature_cols = [
        "Lag1", "Lag7", "Lag14",
        "Rolling_Mean_7", "Rolling_Mean_14", "Rolling_Std_7", "Rolling_Std_14",
        "Net_Flow", "Month", "Quarter", "Week", "Year", "DayOfWeek",
        "Month_Sin", "Month_Cos", "Quarter_Sin", "Quarter_Cos",
        "DayOfWeek_Sin", "DayOfWeek_Cos", "Is_Holiday"
    ]

    # Include additional available lag columns
    optional_cols = ["Apprehended_Lag1", "Apprehended_Lag7", "Discharged_Lag1", "Discharged_Lag7"]
    for c in optional_cols:
        if c in df_features.columns:
            feature_cols.append(c)

    n_test = int(len(df_features) * test_size)
    train_df = df_features.iloc[:-n_test].copy()
    test_df = df_features.iloc[-n_test:].copy()

    return train_df, test_df, feature_cols
