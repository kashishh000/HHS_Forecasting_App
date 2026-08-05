"""
pages/home.py — Home page for HHS UAC Predictive Forecasting Application.
Displays project overview, objectives, workflow, technology stack, and live KPI cards.
"""

import streamlit as st
from utils.data_loader import load_data, compute_kpis


def render():
    """Render the Home page."""

    # ── Hero Banner ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <div style="font-size:3.5rem; margin-bottom:0.5rem;">🏛️</div>
        <h1>Predictive Forecasting of Care Load &amp; Placement Demand</h1>
        <p class="subtitle">
            Transforming historical operational data from the <b>U.S. Department of Health and Human Services
            Unaccompanied Alien Children (UAC) Program</b> into predictive intelligence for
            proactive resource planning using Machine Learning and Business Intelligence.
        </p>
        <div style="margin-top:1.2rem; display:flex; gap:0.5rem; justify-content:center; flex-wrap:wrap;">
            <span class="badge badge-primary">🤖 Machine Learning</span>
            <span class="badge badge-success">📊 ARIMA Forecasting</span>
            <span class="badge badge-warning">📈 Business Intelligence</span>
            <span class="badge badge-danger">🔮 Predictive Analytics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Data and Compute KPIs ─────────────────────────────────────────────
    with st.spinner("Loading dataset..."):
        try:
            df = load_data("data/dataset.csv")
            kpis = compute_kpis(df)
            data_loaded = True
        except FileNotFoundError as e:
            st.error(f"⚠️ {e}. Please ensure `data/dataset.csv` exists.")
            data_loaded = False

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    if data_loaded:
        st.markdown('<div class="section-header"><span class="section-icon">📊</span><h2>Live Key Performance Indicators</h2></div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                label="📅 Date Range",
                value=kpis["Date Range"].split("–")[0].strip(),
                delta=f"to {kpis['Date Range'].split('–')[1].strip()}",
            )
        with c2:
            st.metric(
                label="👶 Current HHS Care",
                value=f"{kpis['Current HHS Care']:,}",
                delta=f"{kpis['HHS Care Change %']:+.1f}% MoM",
                delta_color="inverse" if kpis["HHS Care Change %"] < 0 else "normal",
            )
        with c3:
            st.metric(
                label="🚔 Current Apprehended",
                value=f"{kpis['Current Apprehended']:,}",
                delta=f"{kpis['Apprehended Change %']:+.1f}% MoM",
                delta_color="inverse",
            )
        with c4:
            st.metric(
                label="✅ Current Discharged",
                value=f"{kpis['Current Discharged']:,}",
                delta=f"{kpis['Discharged Change %']:+.1f}% MoM",
            )

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.metric(label="📈 Total Records", value=f"{kpis['Total Records']:,}", delta="Monthly data points")
        with c6:
            st.metric(label="🏆 Peak HHS Care", value=f"{kpis['Peak HHS Care']:,}", delta=kpis["Peak HHS Date"])
        with c7:
            st.metric(label="📊 Avg Monthly Apprehended", value=f"{kpis['Avg Monthly Apprehended']:,}")
        with c8:
            st.metric(label="📉 Avg Monthly Discharged", value=f"{kpis['Avg Monthly Discharged']:,}")

        st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Project Overview ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><span class="section-icon">📋</span><h2>Project Overview</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="info-panel">
        <p style="line-height:1.8; color:#CBD5E1;">
        The <b style="color:#A5B4FC;">HHS UAC Predictive Forecasting</b> project aims to leverage historical
        operational data from the U.S. Department of Health and Human Services Unaccompanied Alien Children
        (UAC) Program to develop robust machine learning models capable of predicting future care load
        and placement demand.
        </p>
        <p style="line-height:1.8; color:#CBD5E1; margin-top:0.8rem;">
        By applying <b style="color:#A5B4FC;">Random Forest Regression</b> and <b style="color:#A5B4FC;">SARIMA time-series</b>
        forecasting, combined with a comprehensive <b style="color:#A5B4FC;">Business Intelligence dashboard</b>,
        this tool empowers program administrators to anticipate demand surges and optimize resource allocation
        proactively — reducing response times, improving placement efficiency, and ensuring child welfare.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-panel" style="height:100%;">
            <div style="font-size:0.8rem; color:#94A3B8; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.8rem;">📁 DATASET COLUMNS</div>
            <ul style="list-style:none; padding:0; margin:0;">
                <li style="padding:0.35rem 0; border-bottom:1px solid rgba(99,102,241,0.1); color:#CBD5E1; font-size:0.9rem;">
                    📅 <b>Date</b> <span style="color:#64748B;">— Monthly timestamp</span>
                </li>
                <li style="padding:0.35rem 0; border-bottom:1px solid rgba(99,102,241,0.1); color:#CBD5E1; font-size:0.9rem;">
                    🚔 <b>Children Apprehended</b> <span style="color:#64748B;">— Placed in CBP custody</span>
                </li>
                <li style="padding:0.35rem 0; border-bottom:1px solid rgba(99,102,241,0.1); color:#CBD5E1; font-size:0.9rem;">
                    🏛️ <b>Children in CBP Custody</b> <span style="color:#64748B;">— Current count</span>
                </li>
                <li style="padding:0.35rem 0; border-bottom:1px solid rgba(99,102,241,0.1); color:#CBD5E1; font-size:0.9rem;">
                    🔄 <b>Transferred Out</b> <span style="color:#64748B;">— Out of CBP custody</span>
                </li>
                <li style="padding:0.35rem 0; border-bottom:1px solid rgba(99,102,241,0.1); color:#CBD5E1; font-size:0.9rem;">
                    🏥 <b>Children in HHS Care</b> <span style="color:#64748B;">— Current care load</span>
                </li>
                <li style="padding:0.35rem 0; color:#CBD5E1; font-size:0.9rem;">
                    ✅ <b>Children Discharged</b> <span style="color:#64748B;">— Released from HHS care</span>
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ── Objectives ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><span class="section-icon">🎯</span><h2>Project Objectives</h2></div>', unsafe_allow_html=True)

    obj_cols = st.columns(3)
    objectives = [
        ("📊", "Demand Forecasting", "Predict future care load in HHS facilities using ML models trained on historical UAC program data."),
        ("🔍", "Pattern Discovery", "Identify seasonal, cyclical, and trend-driven patterns in child placement and discharge dynamics."),
        ("🤖", "Model Benchmarking", "Compare Random Forest Regression vs. SARIMA time-series to identify the optimal forecasting approach."),
        ("📈", "BI Reporting", "Build a Power BI-style interactive dashboard for real-time operational intelligence."),
        ("🧪", "Scenario Planning", "Enable administrators to run what-if analyses to anticipate demand under varying policy conditions."),
        ("🔮", "Proactive Planning", "Provide multi-horizon (7/14/30/60-day) forecasts to support staffing, shelter, and resource decisions."),
    ]
    for i, (icon, title, desc) in enumerate(objectives):
        with obj_cols[i % 3]:
            st.markdown(f"""
            <div class="workflow-step" style="margin-bottom:0.8rem;">
                <span style="font-size:1.5rem;">{icon}</span>
                <div class="step-title" style="margin-top:0.4rem;">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Workflow ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><span class="section-icon">⚙️</span><h2>Project Workflow</h2></div>', unsafe_allow_html=True)

    wf_cols = st.columns(5)
    steps = [
        ("1", "Data Ingestion", "Load and validate monthly HHS UAC historical dataset from CSV."),
        ("2", "Preprocessing", "Clean, parse dates, engineer lag/rolling features for ML."),
        ("3", "EDA", "Explore trends, seasonality, correlations, and distribution patterns."),
        ("4", "Modeling", "Train Random Forest & SARIMA; evaluate with MAE, RMSE, R²."),
        ("5", "Forecasting & BI", "Generate forecasts, confidence intervals, and BI dashboards."),
    ]
    for col, (num, title, desc) in zip(wf_cols, steps):
        with col:
            st.markdown(f"""
            <div class="workflow-step">
                <div class="step-number">{num}</div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Technology Stack ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><span class="section-icon">🛠️</span><h2>Technology Stack</h2></div>', unsafe_allow_html=True)

    tech_cols = st.columns(7)
    techs = [
        ("🐍", "Python 3.10+", "Core language"),
        ("🌊", "Streamlit", "Web framework"),
        ("📊", "Plotly", "Interactive charts"),
        ("🐼", "Pandas", "Data wrangling"),
        ("🔢", "NumPy", "Numerical ops"),
        ("🤖", "Scikit-learn", "ML models"),
        ("📈", "Statsmodels", "SARIMA / ARIMA"),
    ]
    for col, (icon, name, desc) in zip(tech_cols, techs):
        with col:
            st.markdown(f"""
            <div class="tech-card">
                <span class="tech-icon">{icon}</span>
                <div class="tech-name">{name}</div>
                <div class="tech-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin-top:2rem; padding:1rem;
                background:linear-gradient(135deg,rgba(99,102,241,0.05),rgba(236,72,153,0.05));
                border:1px solid rgba(99,102,241,0.2); border-radius:12px;">
        <p style="color:#94A3B8; font-size:0.88rem; margin:0;">
            💡 <b style="color:#A5B4FC;">Navigate</b> using the sidebar to explore each module of the application.
            Start with <b style="color:#A5B4FC;">Dataset Explorer</b> to preview data, then proceed to
            <b style="color:#A5B4FC;">EDA</b>, <b style="color:#A5B4FC;">Forecasting</b>, and the <b style="color:#A5B4FC;">BI Dashboard</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
