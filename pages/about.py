"""
About Page Component for HHS UAC Forecasting Application.
"""

import streamlit as st
from utils.helpers import render_header, render_alert

def render():
    render_header(
        title="About the Project & Technical Methodology",
        subtitle="Comprehensive System Architecture, Research Background, and Data Governance Framework"
    )

    t1, t2, t3, t4 = st.tabs(["Problem Statement", "Methodology & Pipeline", "Dataset Architecture", "Technology Stack"])

    with t1:
        st.markdown("""
        <div class="section-box">
            <h3>Operational Problem Statement</h3>
            <p style='line-height: 1.7; color: #e2e8f0;'>
                The US Department of Health and Human Services (HHS) Office of Refugee Resettlement (ORR) manages the care and placement of unaccompanied children arriving at US borders. Operating under statutory mandates (TVPRA 2008), HHS must safely house children transferred from CBP within 72 hours until they can be safely discharged to vetted family sponsors.
            </p>
            <p style='line-height: 1.7; color: #e2e8f0;'>
                <b>Challenges faced by operational leaders:</b>
            </p>
            <ul style='line-height: 1.8; color: #cbd5e1;'>
                <li><b>Extreme Demand Volatility:</b> Seasonal border spikes lead to rapid surges in intake within days.</li>
                <li><b>Capacity Bottlenecks:</b> Bed capacity constraints cause dangerous overcrowding at CBP facilities if HHS transfers slow down.</li>
                <li><b>Sponsor Vetting Latency:</b> Discharge velocities depend on background check processing times, causing extended length of stay (LOS).</li>
                <li><b>Reactive Resource Allocation:</b> Lack of predictive visibility forces costly emergency shelter openings.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        render_alert("🎯 <b>Core Project Objective:</b> Build a machine learning & time-series intelligence system to accurately project HHS Care Load and Discharge Demand, allowing proactive facility planning up to 1 year in advance.", "info")

    with t2:
        st.markdown("""
        <div class="section-box">
            <h3>Machine Learning & Forecasting Pipeline</h3>
            <ol style='line-height: 1.9; color: #e2e8f0; padding-left: 1.3rem;'>
                <li><b>Data Ingestion & Sanitization:</b> Automated string cleaning, comma stripping, chronological alignment, duplicate suppression, and imputation.</li>
                <li><b>Feature Engineering:</b> Generating temporal lag variables (Lag1, Lag7, Lag14), rolling window statistics (7/14-day mean & std), operational net flow indicators, calendar date parts, sine/cosine cyclical phase features, and US holiday flags.</li>
                <li><b>Multi-Model Training Suite:</b>
                    <ul>
                        <li><b>Supervised ML:</b> Random Forest Regressor, Gradient Boosting Regressor.</li>
                        <li><b>Statistical Time Series:</b> ARIMA(2,1,2), SARIMAX(1,1,1)x(1,0,1,7).</li>
                        <li><b>Baselines:</b> Persistence Naive Model, 7-Day Moving Average.</li>
                    </ul>
                </li>
                <li><b>Evaluation & Leaderboard benchmarking:</b> Comparing models against MAE, RMSE, MAPE, R², and Forecast Accuracy percentage.</li>
                <li><b>Interactive Simulation Engine:</b> Running What-If scenarios with dynamic parameter sliders to simulate border surge scenarios.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with t3:
        st.markdown("""
        <div class="section-box">
            <h3>Dataset Variables Specification</h3>
            <table style='width:100%; border-collapse: collapse; color: #e2e8f0;'>
                <thead>
                    <tr style='border-bottom: 2px solid #334155; text-align: left;'>
                        <th style='padding: 10px;'>Variable Name</th>
                        <th style='padding: 10px;'>Type</th>
                        <th style='padding: 10px;'>Description</th>
                        <th style='padding: 10px;'>Role</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style='border-bottom: 1px solid #1e293b;'>
                        <td style='padding: 10px;'><b>Date</b></td>
                        <td style='padding: 10px;'>Datetime</td>
                        <td style='padding: 10px;'>Daily reporting date timestamp</td>
                        <td style='padding: 10px;'>Temporal Index</td>
                    </tr>
                    <tr style='border-bottom: 1px solid #1e293b;'>
                        <td style='padding: 10px;'><b>Apprehended in CBP</b></td>
                        <td style='padding: 10px;'>Integer</td>
                        <td style='padding: 10px;'>Children apprehended by border patrol daily</td>
                        <td style='padding: 10px;'>System Inflow</td>
                    </tr>
                    <tr style='border-bottom: 1px solid #1e293b;'>
                        <td style='padding: 10px;'><b>In CBP Custody</b></td>
                        <td style='padding: 10px;'>Integer</td>
                        <td style='padding: 10px;'>Total children currently held in CBP facilities</td>
                        <td style='padding: 10px;'>Holding Buffer</td>
                    </tr>
                    <tr style='border-bottom: 1px solid #1e293b;'>
                        <td style='padding: 10px;'><b>Transferred from CBP</b></td>
                        <td style='padding: 10px;'>Integer</td>
                        <td style='padding: 10px;'>Children transferred from CBP into HHS shelter network</td>
                        <td style='padding: 10px;'>HHS Intake</td>
                    </tr>
                    <tr style='border-bottom: 1px solid #1e293b;'>
                        <td style='padding: 10px;'><b>In HHS Care</b></td>
                        <td style='padding: 10px;'>Integer</td>
                        <td style='padding: 10px;'>Total active census of children in HHS shelters</td>
                        <td style='padding: 10px;'>Primary Target</td>
                    </tr>
                    <tr style='border-bottom: 1px solid #1e293b;'>
                        <td style='padding: 10px;'><b>Discharged from HHS</b></td>
                        <td style='padding: 10px;'>Integer</td>
                        <td style='padding: 10px;'>Children released to vetted sponsors/relatives</td>
                        <td style='padding: 10px;'>System Outflow Target</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with t4:
        st.markdown("""
        <div class="section-box">
            <h3>Technology Stack & Libraries</h3>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;'>
                <div style='background: #1e293b; padding: 15px; border-radius: 10px;'>
                    <h4 style='color: #60a5fa; margin-top:0;'>Frontend Dashboard</h4>
                    <p style='color: #cbd5e1;'>Streamlit, HTML5, Custom CSS, Plotly Interactive Visuals</p>
                </div>
                <div style='background: #1e293b; padding: 15px; border-radius: 10px;'>
                    <h4 style='color: #34d399; margin-top:0;'>Data & ML Engines</h4>
                    <p style='color: #cbd5e1;'>Pandas, NumPy, Scikit-learn, Statsmodels (ARIMA/SARIMA)</p>
                </div>
                <div style='background: #1e293b; padding: 15px; border-radius: 10px;'>
                    <h4 style='color: #f472b6; margin-top:0;'>Analytics & Support</h4>
                    <p style='color: #cbd5e1;'>Python 3.14+, GitHub Versioning, Plotly Dark Templates</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__" or True:
    render()

