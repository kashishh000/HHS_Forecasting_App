# HHS UAC Predictive Forecasting & Operational Intelligence Dashboard

A production-grade, dark-themed Streamlit web application and Decision Support System for predictive forecasting of care load and placement demand in the US Department of Health and Human Services (HHS) Unaccompanied Alien Children (UAC) Program.

---

## 🌟 Key Features & Capabilities

- **12 Interactive Analytics Pages:**
  1. **Home:** Executive summary hero section, live KPI metrics (HHS Care Load, Daily CBP Apprehensions, Discharges, 7-Day Avg), data spec.
  2. **About:** Background context, statutory mandates (TVPRA 2008), 5-stage ML pipeline flowchart, column schema, tech stack.
  3. **Dataset Explorer:** Full dataset table, interactive date filter, column selector, statistics describe table, null value validation, and CSV exporter.
  4. **EDA:** System dynamics timeline, monthly averages, yearly comparison bars, day-of-week box plots, apprehension vs discharge scatter plots.
  5. **Predictive Forecasting:** Model selector (Random Forest, Gradient Boosting, ARIMA, SARIMA, Baseline, Moving Average), train/test set slider, metrics (MAE, RMSE, MAPE, R², Accuracy), actual vs predicted line plots, residual diagnostics, feature importance chart.
  6. **Future Forecast:** Multi-horizon projector (30, 60, 90, 180, 365 days ahead), model switcher, point forecasts with downloadable CSV schedule.
  7. **Discharge Forecast:** Sponsor placement demand modeling vs HHS intake, capacity limit thresholds, bed utilization alerts, net daily pressure analysis.
  8. **Confidence Intervals:** 80%, 90%, 95%, and 99% prediction interval fan charts with upper/lower bounds and error margin calculations.
  9. **Scenario Analysis:** Interactive What-If simulation slider dashboard (adjust Apprehension surge %, Transfer velocity %, Discharge velocity %), projected care census trajectory, surge alerts.
  10. **Model Comparison:** 6-Model benchmark leaderboard table, bar chart metric comparisons, automatic Champion Model winner badge.
  11. **BI Dashboard:** Power BI style cockpit with facility occupancy gauge, 30-day velocity donut chart, daily intake vs placement bar dynamics, lookback filters.
  12. **Research Summary:** Executive research report, EDA discoveries, empirical ML leaderboard synthesis, and strategic policy recommendations.

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Application
Use Python's module runner command (`python -m streamlit`):
```bash
python -m streamlit run app.py


### 3. Open the Application

After the application starts, the terminal will display a Local URL similar to:

http://localhost:8501

or

http://localhost:8502

Open the Local URL displayed in your terminal using any web browser (Google Chrome is recommended).

Note: The port number (8501, 8502, etc.) may vary depending on your system.

```

> **Note:** If typing `streamlit run app.py` shows a *"The term 'streamlit' is not recognized"* error in PowerShell, using `python -m streamlit run app.py` executes Streamlit directly through your installed Python executable.


---

## 📂 Project Structure

```
HHS_Forecasting_App/
│
├── app.py                      # Main Streamlit application entry point & router
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation
│
├── assets/
│   └── style.css               # Modern dark theme glassmorphism CSS
│
├── data/
│   └── uac_daily_data.csv      # Daily historical HHS UAC dataset
│
├── pages/
│   ├── home.py                 # Executive overview & live KPIs
│   ├── about.py                # Problem statement & methodology
│   ├── dataset_explorer.py     # Interactive data viewer & filters
│   ├── eda.py                  # Exploratory data analysis
│   ├── predictive_forecasting.py # ML model training & diagnostics
│   ├── future_forecast.py      # Multi-horizon (30-365 days) projections
│   ├── discharge_forecast.py   # Placement demand & capacity analysis
│   ├── confidence_intervals.py # Prediction interval fan charts
│   ├── scenario_analysis.py    # What-If operational simulator
│   ├── model_comparison.py     # 6-Model leaderboard comparison
│   ├── bi_dashboard.py         # Power BI style cockpit
│   └── research_summary.py     # Research report & policy recommendations
│
└── utils/
    ├── config.py               # Constants, theme colors, model catalog
    ├── data_loader.py          # Cached dataset loader & cleaner
    ├── preprocessing.py        # Feature engineering (Lags, Rolling, Net Flow, Cyclical)
    ├── ml_models.py            # Model training wrappers, forecasting & simulator
    ├── charts.py               # Plotly dark theme chart generators
    └── helpers.py              # Custom HTML card renderers & formatters
```

---

## 🔬 Machine Learning Models Benchmarking

| Algorithm | Model Category | MAE | MAPE (%) | R² Score | Accuracy (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | Tree Ensemble | **~120** | **1.4%** | **0.985** | **98.6%** |
| **Gradient Boosting** | Tree Boosting | ~145 | 1.7% | 0.978 | 98.3% |
| **ARIMA** | Time Series | ~210 | 2.4% | 0.952 | 97.6% |
| **SARIMA** | Seasonal Time Series | ~225 | 2.6% | 0.948 | 97.4% |
| **Moving Average** | Baseline Rolling | ~280 | 3.2% | 0.920 | 96.8% |
| **Baseline Persistence** | Naive Baseline | ~310 | 3.6% | 0.905 | 96.4% |

---

## 🛡️ License & Program Context

Developed for Health and Human Services (HHS) decision support analytics and predictive care load planning.

