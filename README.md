# 🏛️ HHS UAC Predictive Forecasting Platform

Predictive Forecasting of Care Load & Placement Demand Using Machine Learning & Business Intelligence.

## 📋 Overview
This web application transforms operational data from the **U.S. Department of Health and Human Services (HHS) Unaccompanied Alien Children (UAC) Program** into predictive intelligence for proactive resource planning.

## 🚀 Features
- **🏠 Home**: Executive KPIs, workflow diagrams, and live system highlights.
- **📊 Dataset Explorer**: Interactive data browser, range filters, data quality audit & export tools.
- **🔍 Exploratory Data Analysis (EDA)**: Timeline metrics, correlation matrix, seasonality heatmaps & boxplots.
- **🤖 Predictive Forecasting**: Training & validation of Random Forest & SARIMA time series models.
- **🔮 Future Care Load Forecast**: Multi-horizon projections (7, 14, 30, 60, 90 days).
- **📉 Discharge Demand Forecast**: Sponsor placement demand forecasting.
- **⚖️ Model Comparison**: Benchmarking MAE, RMSE, R², and residual analysis.
- **📐 Confidence Intervals**: Prediction bounds (80% and 95% uncertainty intervals).
- **🧪 Scenario Analysis**: Policy simulation with dynamic parameter sliders.
- **📈 BI Dashboard**: Executive Power BI-style multi-panel operational dashboard.
- **📋 Research Summary**: Research methodology, findings, policy impact & recommendations.
- **👤 About**: Project details, tech stack, and attribution.

## ⚙️ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python -m streamlit run app.py
```

## 🛠️ Technology Stack
- **Framework**: Streamlit
- **Analytics & ML**: Pandas, NumPy, Scikit-learn, Statsmodels
- **Visualization**: Plotly, Matplotlib
- **Styling**: Custom CSS (Glassmorphism & Modern Dark Theme)
