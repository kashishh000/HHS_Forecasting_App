"""
Machine Learning and Time Series Forecasting Models for HHS UAC Dashboard.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes MAE, RMSE, MAPE, R2, and Forecast Accuracy.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Avoid division by zero in MAPE
    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0

    r2 = r2_score(y_true, y_pred)
    accuracy = max(0.0, 100.0 - mape)

    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "MAPE": round(float(mape), 2),
        "R2": round(float(r2), 4),
        "Accuracy": round(float(accuracy), 2)
    }

def train_eval_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "Children in HHS Care"
) -> Dict[str, Any]:
    """
    Fits specified model on train_df and evaluates on test_df.
    """
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model_obj = None
    feature_importances = None

    if model_name == "Random Forest":
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        model_obj = rf
        feature_importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    elif model_name == "Gradient Boosting":
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        gb.fit(X_train, y_train)
        y_pred = gb.predict(X_test)
        model_obj = gb
        feature_importances = pd.Series(gb.feature_importances_, index=feature_cols).sort_values(ascending=False)

    elif model_name == "ARIMA":
        try:
            # Fit ARIMA(2,1,2) on target time series
            history = y_train.tolist()
            preds = []
            # Fit once for fast test evaluation
            arima_model = ARIMA(history, order=(2, 1, 2)).fit()
            y_pred = arima_model.forecast(steps=len(y_test))
            model_obj = arima_model
        except Exception:
            # Fallback to simple AR
            arima_model = ARIMA(y_train, order=(1, 1, 0)).fit()
            y_pred = arima_model.forecast(steps=len(y_test))
            model_obj = arima_model

    elif model_name == "SARIMA":
        try:
            sarima_model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)).fit(disp=False)
            y_pred = sarima_model.forecast(steps=len(y_test))
            model_obj = sarima_model
        except Exception:
            sarima_model = SARIMAX(y_train, order=(1, 1, 0)).fit(disp=False)
            y_pred = sarima_model.forecast(steps=len(y_test))
            model_obj = sarima_model

    elif model_name == "Baseline Persistence":
        # Naive forecast: predict previous day's actual
        if "Lag1" in test_df.columns:
            y_pred = test_df["Lag1"].values
        else:
            y_pred = np.roll(y_test.values, 1)
            y_pred[0] = y_train.iloc[-1]

    elif model_name == "Moving Average":
        # 7-day moving average forecast
        if "Rolling_Mean_7" in test_df.columns:
            y_pred = test_df["Rolling_Mean_7"].values
        else:
            y_pred = y_test.rolling(window=7, min_periods=1).mean().values

    else:
        # Default fallback: Random Forest
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        model_obj = rf

    metrics = compute_metrics(y_test.values, y_pred)

    results_df = test_df[["Date", target_col]].copy()
    results_df["Predicted"] = y_pred
    results_df["Residual"] = results_df[target_col] - results_df["Predicted"]

    return {
        "model_name": model_name,
        "metrics": metrics,
        "results_df": results_df,
        "model_obj": model_obj,
        "feature_importances": feature_importances
    }

def generate_future_forecast(
    df_clean: pd.DataFrame,
    model_name: str,
    days: int = 30,
    target_col: str = "Children in HHS Care"
) -> pd.DataFrame:
    """
    Generates multi-step future forecasts for N days ahead.
    """
    last_date = df_clean["Date"].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, days + 1)]

    # Prepare latest series
    history_series = df_clean[target_col].tolist()
    last_val = history_series[-1]

    preds = []

    if model_name in ["Random Forest", "Gradient Boosting"]:
        # Recursive multi-step ML forecast
        feature_cols = [
            "Lag1", "Lag7", "Lag14",
            "Rolling_Mean_7", "Rolling_Mean_14", "Rolling_Std_7", "Rolling_Std_14",
            "Net_Flow", "Month", "Quarter", "Week", "Year", "DayOfWeek",
            "Month_Sin", "Month_Cos", "Quarter_Sin", "Quarter_Cos",
            "DayOfWeek_Sin", "DayOfWeek_Cos", "Is_Holiday"
        ]
        
        # Fit model on all available engineered data
        X_all = df_clean[feature_cols]
        y_all = df_clean[target_col]
        
        if model_name == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        else:
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        
        model.fit(X_all, y_all)

        current_series = history_series.copy()
        for fdate in future_dates:
            # Build feature row for fdate
            lag1 = current_series[-1]
            lag7 = current_series[-7] if len(current_series) >= 7 else current_series[-1]
            lag14 = current_series[-14] if len(current_series) >= 14 else current_series[-1]
            
            rm7 = np.mean(current_series[-7:])
            rm14 = np.mean(current_series[-14:])
            rstd7 = np.std(current_series[-7:]) if len(current_series) >= 7 else 0.0
            rstd14 = np.std(current_series[-14:]) if len(current_series) >= 14 else 0.0

            m = fdate.month
            q = fdate.quarter
            w = fdate.isocalendar().week
            y = fdate.year
            dow = fdate.dayofweek

            row = {
                "Lag1": lag1, "Lag7": lag7, "Lag14": lag14,
                "Rolling_Mean_7": rm7, "Rolling_Mean_14": rm14,
                "Rolling_Std_7": rstd7, "Rolling_Std_14": rstd14,
                "Net_Flow": df_clean["Net_Flow"].iloc[-1] if "Net_Flow" in df_clean.columns else 0,
                "Month": m, "Quarter": q, "Week": w, "Year": y, "DayOfWeek": dow,
                "Month_Sin": np.sin(2 * np.pi * m / 12.0), "Month_Cos": np.cos(2 * np.pi * m / 12.0),
                "Quarter_Sin": np.sin(2 * np.pi * q / 4.0), "Quarter_Cos": np.cos(2 * np.pi * q / 4.0),
                "DayOfWeek_Sin": np.sin(2 * np.pi * dow / 7.0), "DayOfWeek_Cos": np.cos(2 * np.pi * dow / 7.0),
                "Is_Holiday": 0
            }

            X_pred = pd.DataFrame([row])[feature_cols]
            next_val = float(model.predict(X_pred)[0])
            preds.append(next_val)
            current_series.append(next_val)

    elif model_name in ["ARIMA", "SARIMA"]:
        try:
            if model_name == "ARIMA":
                mod = ARIMA(history_series, order=(2, 1, 2)).fit()
            else:
                mod = SARIMAX(history_series, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)).fit(disp=False)
            preds = mod.forecast(steps=days).tolist()
        except Exception:
            # Fallback exponential decay / persistence trend
            recent_trend = np.mean(np.diff(history_series[-14:]))
            for i in range(1, days + 1):
                preds.append(last_val + recent_trend * i * 0.5)

    elif model_name == "Moving Average":
        ma_val = np.mean(history_series[-7:])
        preds = [ma_val] * days

    else: # Baseline Persistence
        preds = [last_val] * days

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast_Care_Load": np.round(preds).astype(int)
    })

    return forecast_df

def compute_confidence_intervals(
    forecast_df: pd.DataFrame,
    confidence_level: float = 0.95,
    hist_std: float = 250.0
) -> pd.DataFrame:
    """
    Computes upper and lower prediction bounds for forecast dataframe.
    """
    df = forecast_df.copy()

    # Z-scores for confidence levels
    z_map = {0.80: 1.282, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z_score = z_map.get(confidence_level, 1.96)

    # Uncertainty expands with time step (horizon uncertainty)
    steps = np.arange(1, len(df) + 1)
    uncertainty_margin = z_score * hist_std * np.sqrt(1 + (steps / 30.0) * 0.3)

    df["Lower_Bound"] = np.maximum(0, np.round(df["Forecast_Care_Load"] - uncertainty_margin)).astype(int)
    df["Upper_Bound"] = np.round(df["Forecast_Care_Load"] + uncertainty_margin).astype(int)
    df["Margin_Error"] = np.round(uncertainty_margin).astype(int)

    return df

def simulate_scenario(
    df_clean: pd.DataFrame,
    days: int = 60,
    apprehension_pct_change: float = 0.0,
    discharge_pct_change: float = 0.0,
    transfer_pct_change: float = 0.0
) -> pd.DataFrame:
    """
    Simulates operational scenario by scaling inflows and outflows.
    """
    last_date = df_clean["Date"].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, days + 1)]

    # Baseline daily rates from recent 30-day window
    recent_30 = df_clean.tail(30)
    base_apprehended = recent_30["Children apprehended and placed in CBP custody"].mean()
    base_transferred = recent_30["Children transferred out of CBP custody"].mean()
    base_discharged = recent_30["Children discharged from HHS Care"].mean()

    # Apply user scenario percentage adjustments
    sim_apprehended = base_apprehended * (1.0 + apprehension_pct_change / 100.0)
    sim_transferred = base_transferred * (1.0 + transfer_pct_change / 100.0)
    sim_discharged = base_discharged * (1.0 + discharge_pct_change / 100.0)

    # Daily net change in HHS Care = Transferred into HHS - Discharged from HHS
    daily_net_change = sim_transferred - sim_discharged

    current_care = df_clean["Children in HHS Care"].iloc[-1]
    sim_care_list = []

    for _ in range(days):
        current_care = max(0, current_care + daily_net_change)
        sim_care_list.append(current_care)

    sim_df = pd.DataFrame({
        "Date": future_dates,
        "Simulated_HHS_Care": np.round(sim_care_list).astype(int),
        "Simulated_Daily_Transfers": np.round([sim_transferred] * days).astype(int),
        "Simulated_Daily_Discharges": np.round([sim_discharged] * days).astype(int),
        "Daily_Net_Change": np.round([daily_net_change] * days).astype(int)
    })

    return sim_df
