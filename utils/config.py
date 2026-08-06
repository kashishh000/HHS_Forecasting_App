"""
Application Configuration and Constants for HHS UAC Forecasting Application.
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "uac_daily_data.csv")
STYLE_PATH = os.path.join(BASE_DIR, "assets", "style.css")

# App Metadata
APP_TITLE = "HHS UAC Predictive Forecasting & Operational Intelligence"
APP_ICON = "📈"

# Target & Feature Columns
DATE_COL = "Date"
TARGET_CARE = "Children in HHS Care"
TARGET_DISCHARGE = "Children discharged from HHS Care"

RAW_NUMERIC_COLS = [
    "Children apprehended and placed in CBP custody",
    "Children in CBP custody",
    "Children transferred out of CBP custody",
    "Children in HHS Care",
    "Children discharged from HHS Care"
]

COLUMN_RENAMES = {
    "Children apprehended and placed in CBP custody": "Apprehended_CBP",
    "Children in CBP custody": "In_CBP_Custody",
    "Children transferred out of CBP custody": "Transferred_out_CBP",
    "Children in HHS Care": "In_HHS_Care",
    "Children discharged from HHS Care": "Discharged_HHS"
}

# Theme Color Palette for Plotly Charts
DARK_THEME_COLORWAY = [
    "#3b82f6",  # Primary Blue
    "#10b981",  # Emerald Green
    "#f59e0b",  # Amber / Gold
    "#ef4444",  # Coral Red
    "#8b5cf6",  # Purple
    "#06b6d4",  # Cyan
    "#ec4899",  # Pink
    "#64748b"   # Slate Grey
]

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_BG_COLOR = "#131b2e"
PLOTLY_PAPER_COLOR = "rgba(0,0,0,0)"

# Available ML & Time Series Models
MODELS_LIST = [
    "Random Forest",
    "Gradient Boosting",
    "ARIMA",
    "SARIMA",
    "Baseline Persistence",
    "Moving Average"
]

PAGE_LIST = [
    "Home",
    "About",
    "Dataset Explorer",
    "EDA",
    "Predictive Forecasting",
    "Future Forecast",
    "Discharge Forecast",
    "Confidence Intervals",
    "Scenario Analysis",
    "Model Comparison",
    "BI Dashboard",
    "Research Summary"
]
