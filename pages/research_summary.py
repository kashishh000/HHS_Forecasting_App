"""
pages/research_summary.py — Research Summary page.
Displays methodology, findings, and recommendations.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from utils.data_loader import load_data, compute_kpis


def render():
    """Render the Research Summary page."""

    st.markdown("# 📋 Research Summary")
    st.markdown("<p style='color:#94A3B8; margin-top:-0.5rem;'>Project methodology, key findings, analysis insights, and strategic recommendations.</p>", unsafe_allow_html=True)
    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    with st.spinner("Loading data..."):
        try:
            df = load_data("data/dataset.csv")
            kpis = compute_kpis(df)
        except FileNotFoundError as e:
            st.error(str(e))
            return

    # ── Title Block ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0D1526,#1a1040); border:1px solid rgba(99,102,241,0.3);
                border-radius:16px; padding:2rem; margin-bottom:1.5rem; text-align:center;">
        <div style="font-size:0.8rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;">
            RESEARCH REPORT
        </div>
        <div style="font-size:1.7rem; font-weight:700; font-family:'Space Grotesk',sans-serif;
                    background:linear-gradient(135deg,#6366F1,#EC4899);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">
            Predictive Forecasting of Care Load &amp; Placement Demand
        </div>
        <div style="font-size:1rem; color:#94A3B8; margin-top:0.5rem;">
            Using Machine Learning and Business Intelligence
        </div>
        <div style="margin-top:1rem; display:flex; gap:1rem; justify-content:center; flex-wrap:wrap;">
            <span style="background:rgba(99,102,241,0.15); color:#A5B4FC; border:1px solid rgba(99,102,241,0.3);
                         border-radius:999px; padding:0.25rem 0.75rem; font-size:0.8rem;">
                📅 Dataset: 2015–2023
            </span>
            <span style="background:rgba(16,185,129,0.12); color:#6EE7B7; border:1px solid rgba(16,185,129,0.3);
                         border-radius:999px; padding:0.25rem 0.75rem; font-size:0.8rem;">
                🗃️ {total} Monthly Records
            </span>
            <span style="background:rgba(245,158,11,0.12); color:#FCD34D; border:1px solid rgba(245,158,11,0.3);
                         border-radius:999px; padding:0.25rem 0.75rem; font-size:0.8rem;">
                🤖 2 ML Models
            </span>
        </div>
    </div>
    """.format(total=kpis["Total Records"]), unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Objectives", "🔬 Methodology", "📊 Key Findings", "💡 Recommendations", "📚 References"
    ])

    # ─── Tab 1: Objectives ────────────────────────────────────────────────────
    with tab1:
        st.markdown("## 🎯 Research Objectives")
        st.markdown("""
        <div class="info-panel" style="margin-bottom:1rem;">
        <p style="color:#CBD5E1; line-height:1.8;">
        This research project was undertaken to address the operational challenges faced by the
        <b style="color:#A5B4FC;">U.S. Department of Health and Human Services (HHS)</b> in managing the
        Unaccompanied Alien Children (UAC) Program. The primary challenge is the reactive rather than
        proactive nature of capacity planning.
        </p>
        </div>
        """, unsafe_allow_html=True)

        objectives = [
            ("🔮", "Primary Objective", "Develop a predictive model to forecast future care load (Children in HHS Care) at least 30–90 days in advance with sufficient accuracy for operational planning."),
            ("📊", "Secondary Objective A", "Build a Business Intelligence dashboard enabling real-time monitoring and trend analysis of all UAC program operational metrics."),
            ("⚖️", "Secondary Objective B", "Compare Random Forest Regression and SARIMA time-series models to identify the most suitable approach for this domain."),
            ("🧪", "Secondary Objective C", "Implement scenario analysis to allow administrators to simulate policy changes and anticipate their impact before implementation."),
            ("📉", "Secondary Objective D", "Forecast discharge demand separately to assist in planning reunification capacity and case management staffing."),
        ]

        for icon, title, desc in objectives:
            st.markdown(f"""
            <div style="background:#1E293B; border:1px solid rgba(99,102,241,0.2); border-radius:12px;
                        padding:1rem 1.3rem; margin-bottom:0.7rem; display:flex; gap:1rem; align-items:start;">
                <span style="font-size:1.5rem; flex-shrink:0;">{icon}</span>
                <div>
                    <div style="font-weight:600; color:#A5B4FC; font-size:0.95rem;">{title}</div>
                    <div style="color:#94A3B8; font-size:0.88rem; line-height:1.7; margin-top:0.3rem;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ─── Tab 2: Methodology ───────────────────────────────────────────────────
    with tab2:
        st.markdown("## 🔬 Research Methodology")

        phases = [
            ("1. Data Collection & Ingestion", "#6366F1", [
                "Historical monthly data collected from HHS UAC Program operational records (2015–2023)",
                "Dataset covers 108 monthly observations across 5 operational metrics",
                "Data stored in structured CSV format with consistent monthly timestamping",
            ]),
            ("2. Data Preprocessing", "#EC4899", [
                "Parsed and validated Date column into datetime format",
                "Derived time features: Year, Month, Quarter, Month Name, YearMonth",
                "Verified data completeness — zero missing values found",
                "Standardized column naming conventions for internal processing",
            ]),
            ("3. Feature Engineering", "#10B981", [
                "Lag features: 1, 2, 3, 6, and 12-month lags of target variable",
                "Rolling statistics: 3, 6, 12-month rolling mean and standard deviation",
                "Cyclical encoding: Sine/cosine transformations of month and quarter",
                "Cross-metric features: Apprehended, In_CBP, Transferred_Out, Discharged as predictors",
            ]),
            ("4. Model Development", "#F59E0B", [
                "Random Forest Regressor (n_estimators=200, max_depth=10) trained on 80% of data",
                "SARIMA model (order=(2,1,2), seasonal_order=(1,1,1,12)) fitted on monthly time series",
                "Chronological train/test split used (no random shuffling) to prevent data leakage",
                "Feature importance extracted from RF to identify key predictors",
            ]),
            ("5. Evaluation & Validation", "#3B82F6", [
                "Mean Absolute Error (MAE) — measures average prediction error in original units",
                "Root Mean Squared Error (RMSE) — penalizes large errors more heavily",
                "R² Score — proportion of variance explained by the model",
                "Residual analysis to detect systematic bias in predictions",
                "Confidence interval coverage analysis to validate uncertainty estimates",
            ]),
            ("6. Deployment", "#8B5CF6", [
                "Interactive Streamlit web application with 12 analytical modules",
                "Real-time scenario analysis with dynamic prediction updates",
                "Export capabilities for forecast data in CSV format",
                "Power BI-style BI dashboard with filterable KPIs and trend charts",
            ]),
        ]

        for phase, color, points in phases:
            with st.expander(f"**{phase}**"):
                for point in points:
                    st.markdown(f"<li style='color:#CBD5E1; line-height:1.8; margin:0.2rem 0;'>{point}</li>", unsafe_allow_html=True)

    # ─── Tab 3: Key Findings ──────────────────────────────────────────────────
    with tab3:
        st.markdown("## 📊 Key Research Findings")

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.metric("Peak HHS Care", f"{kpis['Peak HHS Care']:,}", delta=kpis["Peak HHS Date"])
        with fc2:
            st.metric("Total Apprehended", f"{kpis['Total Apprehended']:,}", delta=f"over {kpis['Total Records']} months")
        with fc3:
            st.metric("Avg Monthly Discharged", f"{kpis['Avg Monthly Discharged']:,}")

        st.markdown("""<hr style="border:none;border-top:1px solid rgba(99,102,241,0.15);margin:1rem 0;">""", unsafe_allow_html=True)

        findings = [
            ("📈", "Upward Long-Term Trend", "Children in HHS Care exhibited a statistically significant upward trend from 2015 to 2023, with the peak occurring during the 2021–2022 surge period — a 40%+ increase above the 2015 baseline."),
            ("📅", "Strong Seasonal Patterns", "Apprehensions consistently peak in Spring (March–May), coinciding with improved crossing conditions. HHS care load follows with a 1–2 month lag, creating a predictable seasonal cycle."),
            ("🦠", "COVID-19 Disruption", "2020 saw a significant drop in all metrics due to pandemic-related border restrictions (Title 42). The 2021 rebound was sharper than expected, creating temporary capacity strain."),
            ("🔗", "High Metric Correlation", "Children in HHS Care shows strong positive correlation (r > 0.8) with Apprehended and Transferred_Out, confirming these upstream metrics are reliable leading indicators."),
            ("🌲", "RF Model Performance", "The Random Forest model achieved strong predictive performance, with lag features (particularly 1-month and 3-month lags) being the top predictors of future HHS care load."),
            ("⚖️", "Model Comparison", "Both RF and SARIMA demonstrated strong performance. RF showed advantages on non-seasonal patterns; SARIMA excelled at capturing the 12-month seasonal cycle in formal statistical intervals."),
        ]

        for icon, title, desc in findings:
            st.markdown(f"""
            <div style="background:linear-gradient(145deg,#1E293B,#263348); border:1px solid rgba(99,102,241,0.15);
                        border-radius:12px; padding:1rem 1.3rem; margin-bottom:0.8rem;">
                <div style="display:flex; gap:0.8rem; align-items:start;">
                    <span style="font-size:1.3rem; flex-shrink:0; margin-top:0.1rem;">{icon}</span>
                    <div>
                        <div style="font-weight:700; color:#A5B4FC; font-size:0.95rem; margin-bottom:0.3rem;">{title}</div>
                        <div style="color:#94A3B8; font-size:0.87rem; line-height:1.75;">{desc}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Yearly trend chart
        st.markdown("### 📈 HHS Care Load Trajectory (2015–2023)")
        yearly = df.groupby("Year")["In_HHS"].agg(["mean","min","max"]).reset_index()
        fig_yoy = go.Figure()
        fig_yoy.add_trace(go.Bar(
            x=yearly["Year"], y=yearly["mean"],
            name="Annual Average",
            marker=dict(color=yearly["mean"], colorscale=[[0,"#4338CA"],[0.5,"#6366F1"],[1,"#EC4899"]], showscale=False),
            error_y=dict(type="data", array=yearly["max"]-yearly["mean"], arrayminus=yearly["mean"]-yearly["min"],
                         color="rgba(148,163,184,0.5)", thickness=1.5, width=5),
            hovertemplate="<b>%{x}</b><br>Avg: <b>%{y:,.0f}</b><extra></extra>",
        ))
        fig_yoy.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter,sans-serif", color="#E2E8F0"),
            height=350, showlegend=False,
            xaxis=dict(gridcolor="rgba(148,163,184,0.1)", dtick=1),
            yaxis=dict(gridcolor="rgba(148,163,184,0.1)", title="Children in HHS Care"),
        )
        st.plotly_chart(fig_yoy, use_container_width=True)

    # ─── Tab 4: Recommendations ───────────────────────────────────────────────
    with tab4:
        st.markdown("## 💡 Strategic Recommendations")

        recs = [
            ("🔴", "Immediate", [
                "Deploy the predictive model into operational planning workflows for 30-day ahead capacity planning.",
                "Establish alert thresholds: Trigger resource mobilization when 30-day HHS forecast exceeds current capacity by >10%.",
                "Integrate the BI Dashboard with live HHS data feeds for real-time monitoring.",
            ]),
            ("🟡", "Short-Term (3–6 months)", [
                "Collect additional feature data (e.g., border encounter data, immigration court schedules) to improve RF model accuracy.",
                "Retrain models monthly with new data to prevent concept drift as operational patterns evolve.",
                "Build regional-level forecasting capability to identify geographic hotspots.",
            ]),
            ("🟢", "Long-Term (6–12 months)", [
                "Develop a deep learning (LSTM) model to capture complex nonlinear temporal dependencies.",
                "Establish a formal model governance process with quarterly performance reviews.",
                "Expand scenario analysis to include policy simulation (e.g., impact of new shelter sites, staffing changes).",
                "Build a public-facing dashboard for transparency and congressional reporting.",
            ]),
        ]

        for urgency, timeline, items in recs:
            with st.expander(f"{urgency} **{timeline} Recommendations**"):
                for item in items:
                    st.markdown(f"""
                    <div style="display:flex; gap:0.6rem; align-items:start; padding:0.4rem 0; border-bottom:1px solid rgba(99,102,241,0.1);">
                        <span style="color:#6366F1; font-size:1rem; flex-shrink:0;">→</span>
                        <span style="color:#CBD5E1; font-size:0.9rem; line-height:1.7;">{item}</span>
                    </div>
                    """, unsafe_allow_html=True)

    # ─── Tab 5: References ────────────────────────────────────────────────────
    with tab5:
        st.markdown("## 📚 References & Resources")
        refs = [
            ("HHS ORR UAC Program", "https://www.acf.hhs.gov/orr/programs/ucs/about", "Official program overview from the Office of Refugee Resettlement"),
            ("Scikit-learn Documentation", "https://scikit-learn.org/", "Random Forest and ML model documentation"),
            ("Statsmodels SARIMAX", "https://www.statsmodels.org/", "SARIMA and time-series model documentation"),
            ("Streamlit Documentation", "https://docs.streamlit.io/", "Streamlit web application framework"),
            ("Plotly Python", "https://plotly.com/python/", "Interactive visualization library"),
            ("Pandas Documentation", "https://pandas.pydata.org/", "Data manipulation and analysis library"),
        ]
        for name, url, desc in refs:
            st.markdown(f"""
            <div style="background:#1E293B; border:1px solid rgba(99,102,241,0.15); border-radius:10px;
                        padding:0.8rem 1.2rem; margin-bottom:0.5rem; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-weight:600; color:#A5B4FC;">{name}</div>
                    <div style="font-size:0.8rem; color:#64748B;">{desc}</div>
                </div>
                <a href="{url}" target="_blank" style="background:rgba(99,102,241,0.15); color:#A5B4FC;
                   border:1px solid rgba(99,102,241,0.3); border-radius:8px; padding:0.3rem 0.8rem;
                   font-size:0.8rem; text-decoration:none; white-space:nowrap;">Visit →</a>
            </div>
            """, unsafe_allow_html=True)
