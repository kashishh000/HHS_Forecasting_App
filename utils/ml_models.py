"""
ml_models.py - Machine Learning model training, evaluation and forecasting utilities.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame, target_col: str = "In_HHS") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create ML features from the time series dataframe.

    Args:
        df: Preprocessed DataFrame with time features.
        target_col: Target column to predict.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    feature_df = df.copy()

    # Lag features
    for lag in [1, 2, 3, 6, 12]:
        feature_df[f"{target_col}_lag{lag}"] = feature_df[target_col].shift(lag)

    # Rolling statistics
    for window in [3, 6, 12]:
        feature_df[f"{target_col}_roll_mean_{window}"] = (
            feature_df[target_col].shift(1).rolling(window).mean()
        )
        feature_df[f"{target_col}_roll_std_{window}"] = (
            feature_df[target_col].shift(1).rolling(window).std()
        )

    # Time features
    feature_df["month_sin"] = np.sin(2 * np.pi * feature_df["Month"] / 12)
    feature_df["month_cos"] = np.cos(2 * np.pi * feature_df["Month"] / 12)
    feature_df["quarter_sin"] = np.sin(2 * np.pi * feature_df["Quarter"] / 4)
    feature_df["quarter_cos"] = np.cos(2 * np.pi * feature_df["Quarter"] / 4)

    # Other numeric columns as features (excluding target)
    numeric_cols = ["Apprehended", "In_CBP", "Transferred_Out", "Discharged"]
    feature_cols = [c for c in numeric_cols if c != target_col]

    feature_df = feature_df.dropna()

    lag_cols = [c for c in feature_df.columns if "_lag" in c or "_roll_" in c]
    time_cols = ["month_sin", "month_cos", "quarter_sin", "quarter_cos", "Year"]
    all_feature_cols = feature_cols + lag_cols + time_cols

    X = feature_df[all_feature_cols]
    y = feature_df[target_col]

    return X, y


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(
    df: pd.DataFrame,
    target_col: str = "In_HHS",
    test_size: float = 0.2,
    n_estimators: int = 200,
    random_state: int = 42
) -> dict:
    """
    Train a Random Forest Regressor and return results.

    Returns:
        Dictionary with model, metrics, predictions, actual values, and feature importances.
    """
    X, y = engineer_features(df, target_col)

    # Chronological split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # In-sample full prediction
    y_full_pred = model.predict(X)

    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(15)

    return {
        "model": model,
        "feature_columns": list(X.columns),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_pred": y_pred,
        "y_full_pred": y_full_pred,
        "X_full": X,
        "y_full": y,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "feature_importance": feature_importance,
        "test_size": test_size,
        "target_col": target_col,
    }


def rf_forecast_future(rf_result: dict, df: pd.DataFrame, horizon: int = 30) -> dict:
    """
    Generate future forecasts using the trained Random Forest model.
    Uses recursive forecasting over a specified number of months.

    Args:
        rf_result: Result from train_random_forest().
        df: Original preprocessed dataframe.
        horizon: Number of days to forecast ahead (converted to months).

    Returns:
        Dictionary with forecast dates and values.
    """
    model = rf_result["model"]
    target_col = rf_result["target_col"]
    X_full = rf_result["X_full"]
    y_full = rf_result["y_full"]

    # Determine number of months to forecast
    months = max(1, round(horizon / 30))

    # Use last known row features as base for prediction
    last_features = X_full.iloc[-1:].copy()

    forecasts = []
    last_date = df["Date"].max()

    for i in range(months):
        pred = model.predict(last_features)[0]
        forecasts.append(pred)

        # Update lag features for next step (simplified recursive)
        new_features = last_features.copy()
        for col in new_features.columns:
            if "_lag1" in col:
                new_features[col] = pred
        last_features = new_features

    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=months,
        freq="MS"
    )

    return {
        "dates": forecast_dates,
        "values": np.array(forecasts),
        "lower": np.array(forecasts) * 0.92,
        "upper": np.array(forecasts) * 1.08,
        "horizon_days": horizon,
        "months": months,
    }


# ---------------------------------------------------------------------------
# ARIMA / SARIMA
# ---------------------------------------------------------------------------

def train_arima(
    ts: pd.Series,
    order: tuple = (2, 1, 2),
    seasonal_order: tuple = (1, 1, 1, 12),
    test_size: float = 0.2
) -> dict:
    """
    Train a SARIMA model and return results.

    Args:
        ts: Monthly time series with DatetimeIndex.
        order: ARIMA (p, d, q) order.
        seasonal_order: Seasonal (P, D, Q, s) order.
        test_size: Fraction of data to use for testing.

    Returns:
        Dictionary with fitted model, metrics, and predictions.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        split_idx = int(len(ts) * (1 - test_size))
        train_ts = ts.iloc[:split_idx]
        test_ts = ts.iloc[split_idx:]

        # Ensure the training series has a proper frequency
        try:
            train_ts = train_ts.asfreq("MS")
        except Exception:
            pass

        model = SARIMAX(
            train_ts,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted = model.fit(disp=False)

        # Forecast on test period
        forecast_result = fitted.get_forecast(steps=len(test_ts))
        y_pred = forecast_result.predicted_mean.values
        conf_int = forecast_result.conf_int()

        # Clip negative predictions (care load can't be negative)
        y_pred = np.clip(y_pred, 0, None)

        mae = mean_absolute_error(test_ts.values, y_pred)
        rmse = np.sqrt(mean_squared_error(test_ts.values, y_pred))
        r2 = r2_score(test_ts.values, y_pred)

        # In-sample fit
        in_sample = fitted.fittedvalues

        return {
            "model": fitted,
            "train_ts": train_ts,
            "test_ts": test_ts,
            "y_pred": y_pred,
            "in_sample": in_sample,
            "conf_int": conf_int,
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "order": order,
            "seasonal_order": seasonal_order,
            "success": True,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def arima_forecast_future(arima_result: dict, horizon: int = 30) -> dict:
    """
    Generate future forecasts from the fitted SARIMA model.

    Args:
        arima_result: Result from train_arima().
        horizon: Number of days to forecast (converted to months).

    Returns:
        Dictionary with forecast dates and values.
    """
    if not arima_result.get("success", False):
        return None

    months = max(1, round(horizon / 30))

    # Refit on all data for forecasting
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        full_ts = pd.concat([arima_result["train_ts"], arima_result["test_ts"]])

        try:
            full_ts = full_ts.asfreq("MS")
        except Exception:
            pass

        model = SARIMAX(
            full_ts,
            order=arima_result["order"],
            seasonal_order=arima_result["seasonal_order"],
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted_full = model.fit(disp=False)
        fc = fitted_full.get_forecast(steps=months)
        forecast_mean = np.clip(fc.predicted_mean.values, 0, None)
        ci = fc.conf_int()

        return {
            "dates": fc.predicted_mean.index,
            "values": forecast_mean,
            "lower": np.clip(ci.iloc[:, 0].values, 0, None),
            "upper": ci.iloc[:, 1].values,
            "horizon_days": horizon,
            "months": months,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Metrics Comparison
# ---------------------------------------------------------------------------

def compare_models(rf_result: dict, arima_result: dict) -> pd.DataFrame:
    """
    Create a side-by-side model comparison dataframe.

    Args:
        rf_result: Random Forest results dict.
        arima_result: ARIMA results dict.

    Returns:
        Comparison DataFrame.
    """
    records = []

    records.append({
        "Model": "Random Forest",
        "MAE": rf_result["mae"],
        "RMSE": rf_result["rmse"],
        "R² Score": rf_result["r2"],
        "Type": "ML (Ensemble)",
        "Handles Nonlinearity": "✅ Yes",
        "Interpretability": "Medium",
        "Training Speed": "Fast",
    })

    if arima_result and arima_result.get("success"):
        records.append({
            "Model": "SARIMA",
            "MAE": arima_result["mae"],
            "RMSE": arima_result["rmse"],
            "R² Score": arima_result["r2"],
            "Type": "Statistical (Time Series)",
            "Handles Nonlinearity": "⚠️ Limited",
            "Interpretability": "High",
            "Training Speed": "Moderate",
        })

    return pd.DataFrame(records)
