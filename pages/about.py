"""
pages/about.py — About page for HHS UAC Forecasting Application.
Displays author, role, organization, and project details.
"""

import streamlit as st
import plotly.graph_objects as go


def render():
    """Render the About page."""

    st.markdown("# 👤 About")
    st.markdown("<p style='color:#94A3B8; margin-top:-0.5rem;'>Author, organization, and project information.</p>", unsafe_allow_html=True)
    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Author Card ────────────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("""
        <div style="background:linear-gradient(145deg,#1E293B,#263348); border:1px solid rgba(99,102,241,0.35);
                    border-radius:20px; padding:2rem; text-align:center; position:relative; overflow:hidden;">
            <div style="position:absolute; top:-40px; right:-40px; width:150px; height:150px; border-radius:50%;
                        background:radial-gradient(circle, rgba(99,102,241,0.15), transparent 70%);"></div>
            <div style="font-size:5rem; line-height:1; margin-bottom:1rem;">👩‍💼</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.5rem; font-weight:700;
                        background:linear-gradient(135deg,#6366F1,#EC4899);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">
                Kashish Thakare
            </div>
            <div style="color:#A5B4FC; font-size:0.9rem; font-weight:600; margin-top:0.4rem;">
                Research Analyst Intern
            </div>
            <div style="color:#64748B; font-size:0.82rem; margin-top:0.2rem;">
                Unified Mentor
            </div>
            <hr style="border:none; border-top:1px solid rgba(99,102,241,0.2); margin:1.2rem 0;">
            <div style="display:flex; flex-direction:column; gap:0.5rem; text-align:left;">
                <div style="display:flex; gap:0.6rem; align-items:center;">
                    <span style="font-size:1rem;">🏢</span>
                    <span style="color:#CBD5E1; font-size:0.88rem;"><b style="color:#94A3B8;">Organization:</b> Unified Mentor</span>
                </div>
                <div style="display:flex; gap:0.6rem; align-items:center;">
                    <span style="font-size:1rem;">🎓</span>
                    <span style="color:#CBD5E1; font-size:0.88rem;"><b style="color:#94A3B8;">Role:</b> Research Analyst Intern</span>
                </div>
                <div style="display:flex; gap:0.6rem; align-items:center;">
                    <span style="font-size:1rem;">📊</span>
                    <span style="color:#CBD5E1; font-size:0.88rem;"><b style="color:#94A3B8;">Domain:</b> Data Science & ML</span>
                </div>
                <div style="display:flex; gap:0.6rem; align-items:center;">
                    <span style="font-size:1rem;">🏛️</span>
                    <span style="color:#CBD5E1; font-size:0.88rem;"><b style="color:#94A3B8;">Project:</b> HHS UAC Forecasting</span>
                </div>
            </div>
            <div style="margin-top:1.2rem; display:flex; gap:0.5rem; justify-content:center; flex-wrap:wrap;">
                <span style="background:rgba(99,102,241,0.15); color:#A5B4FC; border:1px solid rgba(99,102,241,0.3);
                             border-radius:999px; padding:0.2rem 0.7rem; font-size:0.75rem;">Python</span>
                <span style="background:rgba(16,185,129,0.12); color:#6EE7B7; border:1px solid rgba(16,185,129,0.3);
                             border-radius:999px; padding:0.2rem 0.7rem; font-size:0.75rem;">ML</span>
                <span style="background:rgba(245,158,11,0.12); color:#FCD34D; border:1px solid rgba(245,158,11,0.3);
                             border-radius:999px; padding:0.2rem 0.7rem; font-size:0.75rem;">BI</span>
                <span style="background:rgba(236,72,153,0.12); color:#F9A8D4; border:1px solid rgba(236,72,153,0.3);
                             border-radius:999px; padding:0.2rem 0.7rem; font-size:0.75rem;">Forecasting</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("## 📋 Project Information")
        project_info = [
            ("📌", "Project Title", "Predictive Forecasting of Care Load & Placement Demand Using Machine Learning and Business Intelligence"),
            ("🏛️", "Organization", "Unified Mentor"),
            ("👤", "Author", "Kashish Thakare"),
            ("🎭", "Role", "Research Analyst Intern"),
            ("🏢", "Client Agency", "U.S. Department of Health and Human Services (HHS)"),
            ("📊", "Program", "Unaccompanied Alien Children (UAC) Program"),
            ("🗃️", "Dataset", "Monthly UAC operational data — 2015 to 2023 (108 records)"),
            ("🤖", "Models Used", "Random Forest Regression + SARIMA Time-Series"),
            ("🛠️", "Tech Stack", "Python · Streamlit · Plotly · Pandas · NumPy · Scikit-learn · Statsmodels"),
            ("🎯", "Primary Goal", "Predict HHS care load 30–90 days ahead for proactive resource planning"),
        ]
        for icon, key, value in project_info:
            st.markdown(f"""
            <div style="display:flex; gap:0.8rem; align-items:start; padding:0.65rem 0;
                        border-bottom:1px solid rgba(99,102,241,0.1);">
                <span style="font-size:1.1rem; flex-shrink:0; margin-top:0.05rem;">{icon}</span>
                <div>
                    <span style="font-size:0.75rem; font-weight:600; color:#64748B; text-transform:uppercase; letter-spacing:0.04em;">{key}</span><br>
                    <span style="color:#CBD5E1; font-size:0.9rem;">{value}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Application Features ───────────────────────────────────────────────────
    st.markdown("## 🗂️ Application Modules")

    features = [
        ("🏠", "Home", "Project overview, objectives, workflow, technology stack, and live KPI cards."),
        ("📊", "Dataset Explorer", "Dataset preview, search, filters, missing values, statistics, and download."),
        ("🔍", "EDA", "Correlation heatmap, timeline, box plots, monthly/yearly analysis, rolling averages."),
        ("🤖", "Predictive Forecasting", "Random Forest and SARIMA training with MAE, RMSE, R² metrics and prediction charts."),
        ("🔮", "Future Care Load", "7, 14, 30, 60, 90-day horizon forecasts with confidence bands."),
        ("📉", "Discharge Forecast", "Dedicated discharge demand forecasting with seasonal analysis."),
        ("⚖️", "Model Comparison", "Side-by-side RF vs SARIMA with radar chart and overlay predictions."),
        ("📐", "Confidence Intervals", "Prediction uncertainty bands with coverage analysis."),
        ("🧪", "Scenario Analysis", "Interactive what-if analysis with preset scenarios and dynamic predictions."),
        ("📈", "BI Dashboard", "Power BI-style dashboard with filterable KPIs, multi-panel charts, treemap."),
        ("📋", "Research Summary", "Full methodology, findings, recommendations, and references."),
        ("👤", "About", "Author, organization, and project metadata."),
    ]

    cols = st.columns(3)
    for i, (icon, name, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:linear-gradient(145deg,#1E293B,#263348); border:1px solid rgba(99,102,241,0.2);
                        border-radius:12px; padding:1rem; margin-bottom:0.8rem; transition:all 0.3s;">
                <div style="font-size:1.4rem; margin-bottom:0.4rem;">{icon}</div>
                <div style="font-weight:600; color:#A5B4FC; font-size:0.92rem; margin-bottom:0.3rem;">{name}</div>
                <div style="font-size:0.78rem; color:#64748B; line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # ── Tech Stack Details ─────────────────────────────────────────────────────
    st.markdown("## 🛠️ Technical Stack Details")

    tech_data = [
        ("Python 3.10+", "Core language", "🐍", "1991–present", "General purpose"),
        ("Streamlit", "Web framework", "🌊", "≥ 1.28", "UI & deployment"),
        ("Plotly", "Visualization", "📊", "≥ 5.17", "Interactive charts"),
        ("Pandas", "Data wrangling", "🐼", "≥ 2.0", "Data manipulation"),
        ("NumPy", "Numerics", "🔢", "≥ 1.24", "Array computing"),
        ("Scikit-learn", "ML models", "🤖", "≥ 1.3", "Random Forest, metrics"),
        ("Statsmodels", "Time series", "📈", "≥ 0.14", "SARIMA / SARIMAX"),
        ("Joblib", "Model I/O", "💾", "≥ 1.3", "Serialization"),
    ]

    tech_df_data = {
        "Library": [t[0] for t in tech_data],
        "Purpose": [t[1] for t in tech_data],
        "Icon": [t[2] for t in tech_data],
        "Version": [t[3] for t in tech_data],
        "Use Case": [t[4] for t in tech_data],
    }
    import pandas as pd
    tech_df = pd.DataFrame(tech_df_data)
    st.dataframe(tech_df, use_container_width=True, hide_index=True)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:2rem 0 1rem; margin-top:1rem;">
        <div style="font-size:2.5rem; margin-bottom:0.5rem;">🏛️</div>
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.2rem; font-weight:700;
                    background:linear-gradient(135deg,#6366F1,#EC4899);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">
            HHS UAC Predictive Forecasting Platform
        </div>
        <div style="color:#64748B; font-size:0.85rem; margin-top:0.5rem; line-height:1.8;">
            Built with ❤️ by <b style="color:#94A3B8;">Kashish Thakare</b> — Research Analyst Intern, Unified Mentor<br>
            Powered by <b style="color:#6366F1;">Streamlit</b> · <b style="color:#EC4899;">Plotly</b> · <b style="color:#10B981;">Scikit-learn</b> · <b style="color:#F59E0B;">Statsmodels</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
