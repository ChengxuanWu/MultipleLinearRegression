"""
Interactive CRISP-DM Machine Learning Dashboard
Featuring Unified Visual Design, Colors & Typography:
1. 🏡 Boston Housing Price Prediction (Multi-Method Feature Selection: Lasso, RFECV, mRMR, Backward Elimination & High-R² Engine)
2. 💼 50 Startups Profit Prediction (One-Hot Encoding, Backward Elimination & Retrained Final Model)
Built with Streamlit, Plotly, and Scikit-Learn
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Configure page
st.set_page_config(
    page_title="CRISP-DM Studio - Boston Housing & 50 Startups",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Unified Modern Typography and High-Contrast Responsive Color Palette
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], .stMarkdown, .stText, h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    /* Hidden Theme Synchronizer Helper */
    iframe[title="streamlit.components.v1.html"] {
        display: none !important;
        position: absolute !important;
        width: 0 !important;
        height: 0 !important;
        border: none !important;
    }
    
    /* -------------------------------------------------------------
       LIGHT THEME (Default - High Contrast, Deep Sharp Typography)
       ------------------------------------------------------------- */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a !important;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #334155 !important;
        margin-bottom: 1.5rem;
        font-weight: 600 !important;
    }
    .metric-card {
        background-color: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .stat-number {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a !important;
    }
    .stat-label {
        font-size: 0.88rem;
        color: #475569 !important;
        font-weight: 700 !important;
        margin-top: 4px;
    }
    .recommendation-box {
        background-color: #f0fdf4 !important;
        border: 1.5px solid #86efac !important;
        border-left: 6px solid #059669 !important;
        padding: 18px 22px !important;
        border-radius: 8px !important;
        margin-top: 10px !important;
        color: #065f46 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    .diag-title {
        color: #065f46 !important;
        font-weight: 800 !important;
        font-size: 1.12rem !important;
        margin-bottom: 10px !important;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .diag-item {
        color: #0f172a !important;
        font-size: 0.95rem !important;
        margin-bottom: 7px !important;
        line-height: 1.55 !important;
    }
    .diag-item b, .diag-item strong {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    .diag-highlight {
        color: #047857 !important;
        font-weight: 700 !important;
    }
    .diag-muted {
        color: #64748b !important;
    }
    .info-card {
        background-color: #f8fafc !important;
        padding: 18px 20px;
        border-radius: 8px;
        border: 1.5px solid #cbd5e1 !important;
        margin-bottom: 15px;
        color: #0f172a !important;
    }
    .pill-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-keep { background-color: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
    .badge-drop { background-color: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    
    div[data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 700 !important;
    }
    .brand-title {
        font-size: 1.18rem;
        font-weight: 800;
        color: #1e3a8a !important;
        letter-spacing: -0.01em;
    }
    .brand-subtitle {
        font-size: 0.78rem;
        font-weight: 600;
        color: #475569 !important;
        letter-spacing: 0.02em;
        margin-top: 2px;
    }
    .brand-header {
        display: flex; 
        align-items: center; 
        gap: 12px; 
        padding: 6px 0 16px 0; 
        margin-bottom: 8px;
        border-bottom: 1.5px solid #cbd5e1 !important;
    }
    .sidebar-nav-title {
        color: #1e3a8a !important;
        font-weight: 800 !important;
        margin: 10px 0 6px 0;
        font-size: 1.05rem;
    }

    /* -------------------------------------------------------------
       DARK MODE: ONLY ACTIVE WHEN STREAMLIT IS EXPLICITLY DARK
       ------------------------------------------------------------- */
    .dark-mode .main-header {
        color: #60a5fa !important;
    }
    .dark-mode .sub-header {
        color: #cbd5e1 !important;
    }
    .dark-mode div[data-testid="stMetricValue"],
    .dark-mode .stat-number {
        color: #ffffff !important;
    }
    .dark-mode div[data-testid="stMetricLabel"],
    .dark-mode .stat-label {
        color: #cbd5e1 !important;
    }
    .dark-mode .metric-card {
        background-color: #1e293b !important;
        border-color: #334155 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4) !important;
    }
    .dark-mode .recommendation-box {
        background-color: #064e3b !important;
        border-color: #059669 !important;
        border-left: 6px solid #10b981 !important;
        color: #ffffff !important;
    }
    .dark-mode .diag-title {
        color: #6ee7b7 !important;
    }
    .dark-mode .diag-item,
    .dark-mode .diag-item b,
    .dark-mode .diag-item strong {
        color: #ffffff !important;
    }
    .dark-mode .diag-highlight {
        color: #a7f3d0 !important;
    }
    .dark-mode .diag-muted {
        color: #cbd5e1 !important;
    }
    .dark-mode .info-card {
        background-color: #1e293b !important;
        border-color: #334155 !important;
        color: #ffffff !important;
    }
    .dark-mode .brand-title {
        color: #ffffff !important;
    }
    .dark-mode .brand-subtitle {
        color: #cbd5e1 !important;
    }
    .dark-mode .brand-header {
        border-bottom-color: #334155 !important;
    }
    .dark-mode .sidebar-nav-title {
        color: #ffffff !important;
    }
    .dark-mode .plotly text {
        fill: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Precise Theme Synchronizer (Inspects Streamlit's actual body text color)
components.html(
    """
    <script>
    function syncTheme() {
        try {
            var doc = window.parent.document;
            if (!doc || !doc.body) return;
            var stApp = doc.querySelector('.stApp');
            if (!stApp) return;

            // In Streamlit Light mode, body color is dark rgb(49, 51, 63)
            // In Streamlit Dark mode, body color is light rgb(250, 250, 250)
            var bodyColor = window.parent.getComputedStyle(doc.body).color;
            var rgb = bodyColor.match(/\\d+/g);
            var isDark = false;
            if (rgb && rgb.length >= 3) {
                var lum = parseInt(rgb[0]) * 0.299 + parseInt(rgb[1]) * 0.587 + parseInt(rgb[2]) * 0.114;
                isDark = (lum > 140);
            }
            if (isDark) {
                stApp.classList.add('dark-mode');
            } else {
                stApp.classList.remove('dark-mode');
            }
        } catch (e) {}
    }
    syncTheme();
    setInterval(syncTheme, 300);
    </script>
    """,
    height=0,
    width=0
)

# Sidebar Project Switcher & Brand Header
st.sidebar.markdown("""
<div class="brand-header">
    <div style="
        width: 44px; 
        height: 44px; 
        border-radius: 12px; 
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%); 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        box-shadow: 0 4px 10px rgba(30, 58, 138, 0.22);
        color: #ffffff;
        font-size: 22px;
        flex-shrink: 0;
    ">
        🔬
    </div>
    <div style="line-height: 1.2;">
        <div class="brand-title">
            Data Mining Studio
        </div>
        <div class="brand-subtitle">
            CRISP-DM Analytics Lab
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<h4 class='sidebar-nav-title'>🎛️ Navigation</h4>", unsafe_allow_html=True)
selected_project = st.sidebar.radio(
    "Choose CRISP-DM Project:",
    ["🏡 Boston Housing Price Prediction", "💼 50 Startups Profit Prediction"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
**CRISP-DM 6-Phase Lifecycle:**
1. Business Understanding
2. Data Understanding
3. Data Preparation (Imputation/Encoding)
4. Modeling & Feature Selection
5. Evaluation & Diagnostics
6. Deployment Pipeline
""")

# =====================================================================================
# PROJECT 1: BOSTON HOUSING PRICE PREDICTION
# =====================================================================================
if selected_project == "🏡 Boston Housing Price Prediction":
    @st.cache_data
    def load_boston_data():
        return pd.read_csv("HousingData.csv")

    def load_boston_pipeline():
        if os.path.exists("boston_housing_pipeline.joblib"):
            return joblib.load("boston_housing_pipeline.joblib")
        return None

    df_boston = load_boston_data()
    boston_bundle = load_boston_pipeline()

    st.markdown('<div class="main-header">🏡 Boston Housing: Econometric Pricing Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Method Feature Selection (Lasso, RFECV, mRMR, Backward Elimination) & High-R² Champion Engine</div>', unsafe_allow_html=True)

    tab_b_overview, tab_b_impute, tab_b_fs, tab_b_eval, tab_b_predict = st.tabs([
        "📌 Project Overview",
        "🧹 Data Audit & Imputation",
        "🎯 Feature Selection Benchmark",
        "📈 Econometric Diagnostics",
        "🔮 Interactive Price Predictor"
    ])

    # --- TAB 1: OVERVIEW ---
    with tab_b_overview:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="stat-number">506</div>
                <div class="stat-label">Census Tracts Analyzed</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="stat-number">7 / 13</div>
                <div class="stat-label">Champion Selected Features</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="stat-number">80.3%</div>
                <div class="stat-label">Champion Test R² (Non-Linear)</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div class="stat-number">$3.80k</div>
                <div class="stat-label">Champion Test RMSE</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Executive Summary & Methodology")
        st.info("""
        - **Target Variable**: `MEDV` (Median value of owner-occupied homes in $1,000s).
        - **Multi-Method Feature Selection Benchmark**: Evaluated 4 distinct feature selection paradigms:
          1. **L1 Regularization (Lasso / Elastic Net)**: Shrinkage and coefficient sparsity.
          2. **RFECV**: Recursive Feature Elimination with 5-Fold Cross-Validation.
          3. **mRMR**: Minimum Redundancy Maximum Relevance using mutual information.
          4. **Original Methods**: Econometric Backward Elimination ($p < 0.05$) and Univariate $F$-Score.
        - **Champion Feature Subset (k=7)**: `['RM', 'DIS', 'LSTAT', 'PTRATIO', 'NOX', 'RAD', 'TAX']`.
        - **Breaking the Linear $R^2$ Ceiling**: Pure linear regression caps at $R^2 \\approx 0.669$ because features like `LSTAT` (inverse socioeconomic curve) and `RM` (quadratic dwelling space) are non-linear. Fitting a regularized polynomial interaction engine on the 7 selected features boosts Test $R^2$ to **0.8029** and lowers Test RMSE from **$4.93k down to $3.80k**!
        """)

        st.write("#### Exploratory Visualizations")
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists("output_plots/boston_01_correlation_heatmap.png"):
                st.image("output_plots/boston_01_correlation_heatmap.png", caption="Predictor Correlation Heatmap", use_container_width=True)
        with c2:
            if os.path.exists("output_plots/boston_02_distributions_and_scatters.png"):
                st.image("output_plots/boston_02_distributions_and_scatters.png", caption="Target & Key Feature Distributions", use_container_width=True)

    # --- TAB 2: IMPUTATION & PREPARATION ---
    with tab_b_impute:
        st.subheader("🧹 Missing Value Treatment & Categorical Dummy Variable")
        st.markdown("""
        The Kaggle Boston Housing dataset contains missing entries across 6 predictors. In accordance with CRISP-DM Phase 3,
        we engineered a robust statistical preprocessor embedded directly inside the Scikit-Learn Pipeline.
        """)
        
        c_imp1, c_imp2 = st.columns([1, 1.2])
        with c_imp1:
            st.write("##### Missing Values Audit in Dataset:")
            null_counts = df_boston.isnull().sum()
            null_pct = (null_counts / len(df_boston)) * 100
            audit_df = pd.DataFrame({'Missing Records': null_counts, 'Missing Pct (%)': null_pct})
            st.dataframe(audit_df[audit_df['Missing Records'] > 0], use_container_width=True)
            
            st.markdown("""
            **Imputation Rules Applied:**
            1. **Continuous Skewed Predictors (`CRIM`, `ZN`, `INDUS`, `AGE`, `LSTAT`)**: Imputed with training set **Median** to avoid outlier distortion.
            2. **Categorical / Dummy Predictor (`CHAS`)**: Imputed with training set **Mode** (most frequent class: 0).
            """)

        with c_imp2:
            st.write("##### Feature Dictionary & Preprocessing Strategy:")
            feat_desc = pd.DataFrame([
                {"Feature": "CRIM", "Description": "Per capita crime rate by town", "Type": "Continuous", "Strategy": "Median Imputer"},
                {"Feature": "ZN", "Description": "Proportion of residential land zoned > 25k sq.ft.", "Type": "Continuous", "Strategy": "Median Imputer"},
                {"Feature": "INDUS", "Description": "Proportion of non-retail business acres", "Type": "Continuous", "Strategy": "Median Imputer"},
                {"Feature": "CHAS", "Description": "Charles River dummy (1 if bounds river; 0 otherwise)", "Type": "Binary Dummy", "Strategy": "Mode Imputer"},
                {"Feature": "NOX", "Description": "Nitric oxides concentration (parts per 10 million)", "Type": "Continuous", "Strategy": "Standardized"},
                {"Feature": "RM", "Description": "Average number of rooms per dwelling", "Type": "Continuous", "Strategy": "Dominant Predictor"},
                {"Feature": "AGE", "Description": "Proportion of owner-occupied units built < 1940", "Type": "Continuous", "Strategy": "Median Imputer"},
                {"Feature": "DIS", "Description": "Weighted distances to five employment centres", "Type": "Continuous", "Strategy": "Standardized"},
                {"Feature": "RAD", "Description": "Index of accessibility to radial highways", "Type": "Ordinal/Discrete", "Strategy": "Numerical"},
                {"Feature": "TAX", "Description": "Full-value property-tax rate per $10,000", "Type": "Continuous", "Strategy": "Standardized"},
                {"Feature": "PTRATIO", "Description": "Pupil-teacher ratio by town school district", "Type": "Continuous", "Strategy": "Standardized"},
                {"Feature": "B", "Description": "1000(Bk - 0.63)^2 demographic proportion", "Type": "Continuous", "Strategy": "Continuous"},
                {"Feature": "LSTAT", "Description": "% Lower socioeconomic status of population", "Type": "Continuous", "Strategy": "Dominant Predictor"},
            ])
            st.dataframe(feat_desc, use_container_width=True, hide_index=True)

    # --- TAB 3: FEATURE SELECTION BENCHMARK ---
    with tab_b_fs:
        st.subheader("🎯 Comparative Validation Curves: 4 Feature Selection Methods")
        st.markdown("""
        Compare performance trajectories across all 4 feature selection paradigms for subset sizes $k=1$ to $13$.
        Hover over any dot or line to view method names, exact feature lists, Test RMSE, and $R^2$ scores.
        """)

        # Data arrays across k=1..13 for all 4 methods
        k_vals = list(range(1, 14))
        
        # Test RMSE
        rmse_lasso = [6.792995, 6.661919, 5.602943, 5.199083, 5.025395, 5.037040, 4.930230, 5.069503, 5.094858, 5.080633, 4.964933, 5.006027, 4.999938]
        rmse_rfecv = [6.792995, 6.661919, 5.602943, 5.199083, 5.025395, 5.037040, 4.930230, 5.069503, 5.094858, 5.080633, 4.964933, 5.006027, 4.999938]
        rmse_mrmr  = [5.737346, 5.218374, 5.206710, 5.207757, 5.309522, 5.318701, 5.304411, 5.317561, 5.467257, 5.303824, 5.124972, 5.029677, 4.999938]
        rmse_orig  = [5.737346, 5.610319, 5.234907, 5.199083, 5.025395, 5.012891, 5.124717, 5.043986, 4.952293, 5.080633, 4.964933, 5.006027, 4.999938]

        # Test R2
        r2_lasso = [0.370757, 0.394806, 0.571917, 0.631405, 0.655621, 0.654023, 0.668541, 0.649549, 0.646035, 0.648009, 0.663858, 0.658271, 0.659101]
        r2_rfecv = [0.370757, 0.394806, 0.571917, 0.631405, 0.655621, 0.654023, 0.668541, 0.649549, 0.646035, 0.648009, 0.663858, 0.658271, 0.659101]
        r2_mrmr  = [0.551133, 0.628665, 0.630323, 0.630174, 0.615579, 0.614249, 0.616319, 0.614414, 0.592399, 0.616404, 0.641838, 0.655034, 0.659101]
        r2_orig  = [0.551133, 0.570789, 0.626308, 0.631405, 0.655621, 0.657333, 0.641874, 0.653069, 0.665567, 0.648009, 0.663858, 0.658271, 0.659101]

        order_lasso = ['RM', 'DIS', 'LSTAT', 'PTRATIO', 'NOX', 'RAD', 'TAX', 'B', 'CRIM', 'CHAS', 'ZN', 'AGE', 'INDUS']
        order_mrmr  = ['LSTAT', 'PTRATIO', 'NOX', 'RM', 'CRIM', 'INDUS', 'AGE', 'TAX', 'B', 'DIS', 'RAD', 'ZN', 'CHAS']
        order_orig  = ['LSTAT', 'RM', 'PTRATIO', 'DIS', 'NOX', 'CHAS', 'CRIM', 'RAD', 'TAX', 'B', 'ZN', 'AGE', 'INDUS']

        fig_compare = make_subplots(
            rows=1, cols=2,
            subplot_titles=("<b>Test RMSE by Feature Count (k)</b>", "<b>Test R-squared by Feature Count (k)</b>"),
            horizontal_spacing=0.10
        )

        def make_custom_data(method_name, r2_list, rmse_list, feat_order):
            return list(zip(
                [method_name] * 13,
                [f"{r:.4f}" for r in r2_list],
                [f"${m:.3f}k" for m in rmse_list],
                ['<br>• '.join(feat_order[:k]) for k in range(1, 14)]
            ))

        h_template = (
            "<span style='font-size:13px; font-weight:bold; color:#38bdf8;'>%{customdata[0]} (k=%{x})</span><br>"
            "<b>Test RMSE:</b> %{customdata[2]}<br>"
            "<b>Test R²:</b> %{customdata[1]}<br>"
            "<b>Features Included:</b><br>• %{customdata[3]}<extra></extra>"
        )

        # Traces for Lasso
        c_lasso = make_custom_data("L1 Regularization (Lasso)", r2_lasso, rmse_lasso, order_lasso)
        fig_compare.add_trace(go.Scatter(x=k_vals, y=rmse_lasso, mode='lines+markers', name='L1 Regularization (Lasso)', line=dict(color='#2563eb', width=2.5), marker=dict(size=8, color='#2563eb'), customdata=c_lasso, hovertemplate=h_template), row=1, col=1)
        fig_compare.add_trace(go.Scatter(x=k_vals, y=r2_lasso, mode='lines+markers', name='L1 Regularization (Lasso)', line=dict(color='#2563eb', width=2.5), marker=dict(size=8, color='#2563eb'), customdata=c_lasso, hovertemplate=h_template, showlegend=False), row=1, col=2)

        # Traces for RFECV
        c_rfecv = make_custom_data("RFECV (Cross-Validated)", r2_rfecv, rmse_rfecv, order_lasso)
        fig_compare.add_trace(go.Scatter(x=k_vals, y=rmse_rfecv, mode='lines+markers', name='RFECV (Cross-Validated)', line=dict(color='#059669', width=2.0, dash='dot'), marker=dict(size=7, color='#059669'), customdata=c_rfecv, hovertemplate=h_template), row=1, col=1)
        fig_compare.add_trace(go.Scatter(x=k_vals, y=r2_rfecv, mode='lines+markers', name='RFECV (Cross-Validated)', line=dict(color='#059669', width=2.0, dash='dot'), marker=dict(size=7, color='#059669'), customdata=c_rfecv, hovertemplate=h_template, showlegend=False), row=1, col=2)

        # Traces for mRMR
        c_mrmr = make_custom_data("mRMR (Min-Redundancy)", r2_mrmr, rmse_mrmr, order_mrmr)
        fig_compare.add_trace(go.Scatter(x=k_vals, y=rmse_mrmr, mode='lines+markers', name='mRMR (Min-Redundancy)', line=dict(color='#d97706', width=2.2), marker=dict(size=7, color='#d97706'), customdata=c_mrmr, hovertemplate=h_template), row=1, col=1)
        fig_compare.add_trace(go.Scatter(x=k_vals, y=r2_mrmr, mode='lines+markers', name='mRMR (Min-Redundancy)', line=dict(color='#d97706', width=2.2), marker=dict(size=7, color='#d97706'), customdata=c_mrmr, hovertemplate=h_template, showlegend=False), row=1, col=2)

        # Traces for Original
        c_orig = make_custom_data("Original (Backward Elim)", r2_orig, rmse_orig, order_orig)
        fig_compare.add_trace(go.Scatter(x=k_vals, y=rmse_orig, mode='lines+markers', name='Original (Backward Elim)', line=dict(color='#7c3aed', width=2.2), marker=dict(size=7, color='#7c3aed'), customdata=c_orig, hovertemplate=h_template), row=1, col=1)
        fig_compare.add_trace(go.Scatter(x=k_vals, y=r2_orig, mode='lines+markers', name='Original (Backward Elim)', line=dict(color='#7c3aed', width=2.2), marker=dict(size=7, color='#7c3aed'), customdata=c_orig, hovertemplate=h_template, showlegend=False), row=1, col=2)

        # Highlight Champion Linear Point (k=7)
        fig_compare.add_annotation(
            x=7, y=4.930230, text="<b>Champion Linear: k=7<br>RMSE = $4.93k</b>",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#dc2626",
            ax=45, ay=-40, font=dict(color="#991b1b", size=11),
            bgcolor="rgba(254, 242, 242, 0.95)", bordercolor="#f87171", borderwidth=1, row=1, col=1
        )
        fig_compare.add_annotation(
            x=7, y=0.668541, text="<b>Champion Linear: k=7<br>R² = 0.6685</b>",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#dc2626",
            ax=-45, ay=40, font=dict(color="#991b1b", size=11),
            bgcolor="rgba(254, 242, 242, 0.95)", bordercolor="#f87171", borderwidth=1, row=1, col=2
        )

        fig_compare.for_each_annotation(
            lambda a: a.update(font=dict(size=14, color='#1e3a8a', family='Inter, sans-serif'))
            if getattr(a, 'text', '') in ["<b>Test RMSE by Feature Count (k)</b>", "<b>Test R-squared by Feature Count (k)</b>"]
            else None
        )

        fig_compare.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=480,
            hovermode="closest",
            margin=dict(l=30, r=30, t=50, b=30),
            legend=dict(orientation='h', yanchor='bottom', y=-0.28, xanchor='center', x=0.5, font=dict(color='#0f172a', size=11)),
            hoverlabel=dict(bgcolor="#0f172a", font_size=12, font_color="#ffffff", bordercolor="#38bdf8")
        )
        fig_compare.update_xaxes(title_text="<b>Number of Features (k)</b>", title_font=dict(color='#0f172a', size=12), tickfont=dict(color='#0f172a'), dtick=1, gridcolor='rgba(148, 163, 184, 0.25)', linecolor='#64748b')
        fig_compare.update_yaxes(title_text="<b>Test RMSE ($1,000s)</b>", title_font=dict(color='#0f172a', size=12), tickfont=dict(color='#0f172a'), row=1, col=1, gridcolor='rgba(148, 163, 184, 0.25)', linecolor='#64748b')
        fig_compare.update_yaxes(title_text="<b>Test R-squared (R²)</b>", title_font=dict(color='#0f172a', size=12), tickfont=dict(color='#0f172a'), row=1, col=2, gridcolor='rgba(148, 163, 184, 0.25)', linecolor='#64748b')

        st.plotly_chart(fig_compare, use_container_width=True)

        st.write("##### 📋 Head-to-Head Comparison Table of the 4 Methods:")
        summary_table = pd.DataFrame([
            {"Feature Selection Method": "L1 Regularization (Lasso / Elastic Net)", "Best k": 7, "Peak Test R²": "0.6685", "Test Adj R²": "0.6439", "Min Test RMSE": "$4.930k", "5-Fold CV RMSE": "$5.102k", "Optimal Feature Subset": "RM, DIS, LSTAT, PTRATIO, NOX, RAD, TAX", "Status": "⭐ Optimal Linear Subset"},
            {"Feature Selection Method": "RFECV (Cross-Validated RFE)", "Best k": 7, "Peak Test R²": "0.6685", "Test Adj R²": "0.6439", "Min Test RMSE": "$4.930k", "5-Fold CV RMSE": "$5.102k", "Optimal Feature Subset": "RM, DIS, LSTAT, PTRATIO, NOX, RAD, TAX", "Status": "⭐ Optimal Linear Subset"},
            {"Feature Selection Method": "Original (Backward Elimination & F-Score)", "Best k": 9, "Peak Test R²": "0.6656", "Test Adj R²": "0.6329", "Min Test RMSE": "$4.952k", "5-Fold CV RMSE": "$4.980k", "Optimal Feature Subset": "LSTAT, RM, PTRATIO, DIS, NOX, CHAS, CRIM, RAD, TAX", "Status": "Econometric Parsimonious"},
            {"Feature Selection Method": "mRMR (Min-Redundancy Max-Relevance)", "Best k": 13, "Peak Test R²": "0.6591", "Test Adj R²": "0.6087", "Min Test RMSE": "$5.000k", "5-Fold CV RMSE": "$4.934k", "Optimal Feature Subset": "All 13 Features", "Status": "Full Feature Set"},
        ])
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.write("##### 🚀 Breaking the R² Barrier: Why Non-Linearity Matters")
        
        breakthrough_c1, breakthrough_c2 = st.columns([1.1, 1])
        with breakthrough_c1:
            st.info("""
            **Why is pure linear $R^2$ capped around 0.67?**
            - **`LSTAT` (Socioeconomic Status)**: Follows an exponential/inverse decay curve with home value, not a rigid straight line.
            - **`RM` (Number of Rooms)**: Possesses convex quadratic scaling (moving from 7 to 8 rooms adds vastly more market value than moving from 4 to 5).
            - **Synergy Effects**: Interaction terms between structural space (`RM`) and school district quality (`PTRATIO`) account for substantial variance that linear equations cannot capture.
            """)
        with breakthrough_c2:
            st.markdown("""
            <div class="recommendation-box">
                <div class="diag-title">🏆 Champion Model Verification:</div>
                <div class="diag-item">• <b>Base Linear Model (7 Features):</b> Test R² = 0.6685 | RMSE = $4.930k</div>
                <div class="diag-item">• <b>Champion Non-Linear Engine (7 Features):</b> <span class="diag-highlight">Test R² = 0.8029 (+0.1344 Lift!)</span></div>
                <div class="diag-item">• <b>Error Reduction:</b> <span class="diag-highlight">Test RMSE = $3.802k (-$1.128k Drop!)</span></div>
                <div class="diag-item">• <b>Mean Absolute Error:</b> $2.414k</div>
                <div class="diag-item">• <b>Deployment Status:</b> Serialized in <code>boston_housing_pipeline.joblib</code> for real-time predictions.</div>
            </div>
            """, unsafe_allow_html=True)

    # --- TAB 4: EVALUATION & DIAGNOSTICS ---
    with tab_b_eval:
        st.subheader("📈 Econometric Evaluation & Assumption Testing")
        
        c_diag1, c_diag2 = st.columns([1, 1.2])
        with c_diag1:
            st.write("##### Model Performance Comparison:")
            comp_table = pd.DataFrame({
                "Metric": ["Features Included", "Model Architecture", "Train R²", "Test R²", "Test Adjusted R²", "Test RMSE ($1k)", "Test MAE ($1k)"],
                "Initial Full Model": ["13 Features", "Linear Regression", "0.7427", "0.6593", "0.6090", "$4.9983k", "$3.1476k"],
                "Selected Linear Champion": ["7 Features (Lasso/RFECV)", "Linear Regression", "0.7239", "0.6685", "0.6439", "$4.9302k", "$3.1491k"],
                "Champion High-R² Model": ["7 Features (Lasso/RFECV)", "Polynomial + Ridge Regularization", "0.8585", "0.8029", "0.7882", "$3.8021k", "$2.4140k"],
            })
            st.dataframe(comp_table, use_container_width=True, hide_index=True)
            st.success("By isolating the 7 essential features and applying regularized interaction modeling, out-of-sample R² jumps from 0.66 to 0.803!")

        with c_diag2:
            st.write("##### Multicollinearity Check: Variance Inflation Factor (VIF):")
            vif_df = pd.DataFrame([
                {"Feature": "TAX", "VIF": "7.08", "Multicollinearity Status": "Acceptable (< 10)"},
                {"Feature": "RAD", "VIF": "7.03", "Multicollinearity Status": "Acceptable (< 10)"},
                {"Feature": "NOX", "VIF": "3.74", "Multicollinearity Status": "Low (< 5)"},
                {"Feature": "DIS", "VIF": "2.61", "Multicollinearity Status": "Low (< 5)"},
                {"Feature": "LSTAT", "VIF": "2.36", "Multicollinearity Status": "Low (< 5)"},
                {"Feature": "RM", "VIF": "1.76", "Multicollinearity Status": "Low (< 5)"},
                {"Feature": "PTRATIO", "VIF": "1.57", "Multicollinearity Status": "Low (< 5)"},
            ])
            st.dataframe(vif_df, use_container_width=True, hide_index=True)

        st.write("##### Regression Assumption Plots:")
        d1, d2 = st.columns(2)
        with d1:
            if os.path.exists("output_plots/boston_04_actual_vs_predicted.png"):
                st.image("output_plots/boston_04_actual_vs_predicted.png", caption="Actual vs. Predicted MEDV", use_container_width=True)
            if os.path.exists("output_plots/boston_06_residuals_normality_qq.png"):
                st.image("output_plots/boston_06_residuals_normality_qq.png", caption="Residuals Normality Q-Q Plot", use_container_width=True)
        with d2:
            if os.path.exists("output_plots/boston_05_residuals_vs_fitted.png"):
                st.image("output_plots/boston_05_residuals_vs_fitted.png", caption="Homoscedasticity Check (Residuals vs Fitted)", use_container_width=True)
            if os.path.exists("output_plots/boston_07_standardized_coefficients.png"):
                st.image("output_plots/boston_07_standardized_coefficients.png", caption="Standardized Coefficients (Feature Magnitude)", use_container_width=True)

    # --- TAB 5: PREDICTOR ---
    with tab_b_predict:
        st.subheader("🔮 Live Boston Property Valuation Predictor")
        st.markdown("Select prediction engine and input tract socioeconomic and structural characteristics:")

        model_mode = st.radio(
            "Select Prediction Engine:",
            [
                "⭐ Champion High-R² Engine (Polynomial Ridge, Test R² = 0.803, RMSE = $3.80k)",
                "🎯 Optimal Linear Model (Lasso/RFECV 7 Features, Test R² = 0.669, RMSE = $4.93k)"
            ],
            index=0
        )

        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            in_rm = st.slider("🛏️ Rooms per Dwelling (RM)", min_value=3.5, max_value=9.0, value=6.5, step=0.1)
            in_lstat = st.slider("👥 Lower Status Population (% LSTAT)", min_value=1.5, max_value=38.0, value=8.5, step=0.5)
            in_ptratio = st.slider("🎓 Pupil-Teacher Ratio (PTRATIO)", min_value=12.0, max_value=23.0, value=17.0, step=0.2)

        with col_in2:
            in_dis = st.slider("🚗 Distance to Employment Centres (DIS)", min_value=1.0, max_value=12.5, value=4.0, step=0.2)
            in_nox = st.slider("🏭 Nitric Oxides Concentration (NOX)", min_value=0.38, max_value=0.88, value=0.52, step=0.01)

        with col_in3:
            in_tax = st.slider("🏛️ Property Tax Rate per $10k (TAX)", min_value=180.0, max_value=720.0, value=310.0, step=10.0)
            in_rad = st.slider("🛣️ Highway Accessibility Index (RAD)", min_value=1, max_value=24, value=4, step=1)

        with st.expander("🛠️ Advanced Auxiliary Predictors (CRIM, CHAS, B, ZN, INDUS, AGE)"):
            c_aux1, c_aux2, c_aux3 = st.columns(3)
            with c_aux1:
                in_crim = st.number_input("🚨 Crime Rate (CRIM)", min_value=0.0, max_value=90.0, value=0.25, step=0.05)
                in_chas = st.selectbox("🌊 Bounds River (CHAS)", [0.0, 1.0], format_func=lambda x: "Yes" if x == 1.0 else "No")
            with c_aux2:
                in_b = st.slider("Demographic Index B", min_value=0.0, max_value=400.0, value=391.44, step=5.0)
                in_zn = st.slider("Zoned Lots ZN (%)", min_value=0.0, max_value=100.0, value=0.0, step=5.0)
            with c_aux3:
                in_indus = st.slider("Non-Retail Acres INDUS (%)", min_value=0.0, max_value=30.0, value=9.69, step=0.5)
                in_age = st.slider("Older Homes AGE (%)", min_value=0.0, max_value=100.0, value=77.5, step=2.5)

        if boston_bundle:
            is_poly = "High-R²" in model_mode
            pipe_b = boston_bundle['pipeline'] if is_poly else boston_bundle.get('linear_pipeline', boston_bundle['pipeline'])
            
            # Universal dictionary providing all 13 features to guarantee full backward/forward compatibility
            universal_input = {
                'CRIM': float(in_crim),
                'ZN': float(in_zn),
                'INDUS': float(in_indus),
                'CHAS': float(in_chas),
                'NOX': float(in_nox),
                'RM': float(in_rm),
                'AGE': float(in_age),
                'DIS': float(in_dis),
                'RAD': float(in_rad),
                'TAX': float(in_tax),
                'PTRATIO': float(in_ptratio),
                'B': float(in_b),
                'LSTAT': float(in_lstat)
            }
            
            input_b_df = pd.DataFrame([universal_input])
            
            # If the pipeline specifies exact input features, align input columns accordingly
            if hasattr(pipe_b, 'feature_names_in_'):
                expected_cols = [c for c in pipe_b.feature_names_in_ if c in input_b_df.columns]
                input_b_df = input_b_df[expected_cols]
            elif 'selected_features' in boston_bundle:
                sel_f = [c for c in boston_bundle['selected_features'] if c in input_b_df.columns]
                # Only subset if ColumnTransformer does not require other columns
                try:
                    pipe_b.predict(input_b_df[sel_f])
                    input_b_df = input_b_df[sel_f]
                except Exception:
                    # Keep universal input DataFrame
                    pass
            
            pred_b_medv = float(pipe_b.predict(input_b_df)[0])
            pred_b_medv = max(5.0, min(50.0, pred_b_medv))
            pred_b_dollars = pred_b_medv * 1000.0

            st.markdown("---")
            p_res1, p_res2 = st.columns([1, 2])
            with p_res1:
                st.metric(label="Predicted Median Home Price (MEDV)", value=f"${pred_b_medv:.2f}k")
                st.caption(f"Estimated Market Valuation: **${pred_b_dollars:,.2f}**")
            with p_res2:
                engine_lbl = "Champion High-R² Engine (Polynomial Regularized)" if is_poly else "Optimal Linear 7-Feature Model"
                st.markdown(f"""
                <div class="recommendation-box">
                    <div class="diag-title">📊 Econometric Valuation Breakdown ({engine_lbl})</div>
                    <div class="diag-item">• <b>Rooms Driver (RM = {in_rm:.1f}):</b> Primary value amplifier with non-linear returns on larger dwellings.</div>
                    <div class="diag-item">• <b>Socioeconomic Curve (% LSTAT = {in_lstat:.1f}%):</b> Quadratic negative impact tapering at lower wealth thresholds.</div>
                    <div class="diag-item">• <b>School Quality (PTRATIO = {in_ptratio:.1f}):</b> Classroom density pricing adjustment.</div>
                    <div class="diag-item">• <b>Model Status:</b> 7 Core Features selected via L1 Lasso & RFECV (Noise predictors eliminated).</div>
                    <div class="diag-item">• <b>Model Reliability:</b> {'Test R² = 0.8029, Test RMSE = $3.80k' if is_poly else 'Test R² = 0.6685, Test RMSE = $4.93k'}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Model bundle not loaded. Run `compare_feature_selection_methods.py` to create the pipeline.")

# =====================================================================================
# PROJECT 2: 50 STARTUPS PROFIT PREDICTION
# =====================================================================================
else:
    @st.cache_data
    def load_startups_dataset():
        return pd.read_csv("50_Startups.csv")

    @st.cache_resource
    def load_startups_pipeline():
        if os.path.exists("50_startups_pipeline.joblib"):
            return joblib.load("50_startups_pipeline.joblib")
        return None

    df_startups = load_startups_dataset()
    pipeline_startups = load_startups_pipeline()

    # Unified Header Style
    st.markdown('<div class="main-header">💼 50 Startups: Econometric Profit Prediction Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">One-Hot Encoding, Backward Elimination, RFE & Production Retraining (CRISP-DM)</div>', unsafe_allow_html=True)

    tab_s_overview, tab_s_ohe, tab_s_fs, tab_s_eval, tab_s_predict = st.tabs([
        "📌 Project Overview",
        "🔀 Data Audit & Encoding",
        "🎯 Feature Selection Curves",
        "📈 Econometric Diagnostics",
        "🔮 Interactive Profit Predictor"
    ])

    with tab_s_overview:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="stat-number">50</div>
                <div class="stat-label">Startups Analyzed</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="stat-number">2 / 5</div>
                <div class="stat-label">Optimal Selected Features</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="stat-number">91.7%</div>
                <div class="stat-label">Test Set R² (Out-of-Sample)</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div class="stat-number">$6,469</div>
                <div class="stat-label">Test Mean Absolute Error (MAE)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Executive Summary & Methodology")
        st.info("""
        - **Target Variable**: `Profit` (Annual net earnings in USD).
        - **Categorical Preprocessing**: Applied **One-Hot Encoding** to `State` (California, Florida, New York) with `drop='first'` to strictly avoid the **Dummy Variable Trap**.
        - **Systematic Feature Selection**: Automated Backward Elimination pruned non-significant variables (`State_New York` p=0.999, `State_Florida` p=0.776, `Administration` p=0.257), eliminating 60% of predictors while boosting Test Adjusted $R^2$ from **0.772 up to 0.893** (+12.1% lift).
        - **Econometric Integrity**: Multicollinearity resolved; residuals satisfy classical Gauss-Markov linear regression assumptions with homoscedasticity and normality.
        """)

        st.write("#### Exploratory Visualizations")
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists("output_plots/01_correlation_heatmap.png"):
                st.image("output_plots/01_correlation_heatmap.png", caption="Predictor Correlation Heatmap", use_container_width=True)
        with c2:
            if os.path.exists("output_plots/03_profit_by_state.png"):
                st.image("output_plots/03_profit_by_state.png", caption="Profit Distribution Across States", use_container_width=True)

    with tab_s_ohe:
        st.subheader("🔀 Categorical Dummy Encoding & Data Audit")
        st.markdown("""
        The `State` attribute contains nominal qualitative labels: **California**, **Florida**, and **New York**. 
        Linear regression requires numerical feature vectors. Applying **One-Hot Encoding** converts nominal labels into binary dummy vectors ($0$ or $1$).
        """)
        
        c_enc1, c_enc2 = st.columns([1, 1.2])
        with c_enc1:
            st.write("##### Raw Categorical Column ('State'):")
            st.dataframe(df_startups[['State', 'Profit']].head(6), use_container_width=True)
            st.caption("Distribution: New York (17), California (17), Florida (16)")

        with c_enc2:
            ohe = OneHotEncoder(drop='first', sparse_output=False)
            encoded_states = ohe.fit_transform(df_startups[['State']])
            encoded_df = pd.DataFrame(encoded_states, columns=ohe.get_feature_names_out(['State']))
            st.write("##### Transformed Binary Matrix (`drop='first'`):")
            st.dataframe(encoded_df.head(6), use_container_width=True)
            st.caption("California serves as the baseline reference category.")

        st.markdown(r"""
        > **Addressing the Dummy Variable Trap:**  
        > If $m$ categories are encoded into $m$ dummy variables, their sum equals $1$ for every row ($\sum D_i = 1$). 
        > This introduces perfect multicollinearity with the intercept $\beta_0$. 
        > By dropping the first category (`drop='first'`), $m-1 = 2$ dummy variables (`State_Florida` and `State_New York`) represent the 3 states without collinearity.
        """)

    with tab_s_fs:
        st.subheader("🎯 Systematic Feature Selection & Validation Curves")
        st.markdown("""
        Moving the cursor over any dot or line displays rich interactive tooltips including exact feature names,
        training and testing RMSE, and $R^2$ scores.
        """)

        k_s = [1, 2, 3, 4, 5]
        rmse_te_s = [8274.868018, 8198.797191, 8309.059683, 8409.916714, 9137.990153]
        r2_te_s = [0.946459, 0.947439, 0.946015, 0.944697, 0.934707]
        rmse_tr_s = [9493.582, 9078.621, 9069.684, 9064.120, 9031.670]
        r2_tr_s = [0.9450, 0.9497, 0.9498, 0.9498, 0.9502]
        
        feature_subsets_s = [
            "[R&D Spend]",
            "[R&D Spend, Marketing Spend]",
            "[R&D Spend, Marketing Spend, State_New York]",
            "[R&D Spend, Marketing Spend, State_New York, State_Florida]",
            "[R&D Spend, Marketing Spend, State_New York, State_Florida, State_California]"
        ]
        
        statuses_s = [
            "Under-specified base model",
            "⭐ OPTIMAL MODEL (Minimum RMSE & Peak R²)",
            "Noise added (RMSE climbs)",
            "Degrading (Redundant state dummies)",
            "Severe Overfitting & Multicollinearity"
        ]

        custom_s = list(zip(
            statuses_s,
            [f"{r:.4f}" for r in r2_te_s],
            [f"${m:,.2f}" for m in rmse_te_s],
            [f"${m:,.2f}" for m in rmse_tr_s],
            [f"{r:.4f}" for r in r2_tr_s],
            [f.replace(', ', '<br>• ') for f in feature_subsets_s]
        ))

        fig_s_plotly = make_subplots(
            rows=1, cols=2,
            subplot_titles=("<b>RMSE by Number of Features</b>", "<b>R-squared by Number of Features</b>"),
            horizontal_spacing=0.10
        )

        hover_template_s = (
            "<span style='font-size:13px; font-weight:bold; color:#38bdf8;'>Features: %{x}</span><br>"
            "<b>Status:</b> %{customdata[0]}<br>"
            "<b>Test RMSE:</b> %{customdata[2]}<br>"
            "<b>Test R²:</b> %{customdata[1]}<br>"
            "<b>Train RMSE:</b> %{customdata[3]}<br>"
            "<b>Train R²:</b> %{customdata[4]}<br>"
            "<b>Predictor Set:</b><br>%{customdata[5]}"
            "<extra></extra>"
        )

        fig_s_plotly.add_trace(
            go.Scatter(
                x=k_s, y=rmse_te_s,
                mode='lines+markers',
                name='Test RMSE ($)',
                line=dict(color='#0284c7', width=2.5),
                marker=dict(size=10, color=['#0284c7', '#10b981', '#0284c7', '#0284c7', '#dc2626'], line=dict(color='#0f172a', width=1)),
                customdata=custom_s,
                hovertemplate=hover_template_s
            ),
            row=1, col=1
        )
        fig_s_plotly.add_trace(
            go.Scatter(
                x=k_s, y=rmse_tr_s,
                mode='lines+markers',
                name='Train RMSE ($)',
                line=dict(color='#94a3b8', width=1.8, dash='dash'),
                marker=dict(size=6, color='#94a3b8'),
                customdata=custom_s,
                hovertemplate=hover_template_s
            ),
            row=1, col=1
        )

        fig_s_plotly.add_trace(
            go.Scatter(
                x=k_s, y=r2_te_s,
                mode='lines+markers',
                name='Test R²',
                line=dict(color='#0284c7', width=2.5),
                marker=dict(size=10, color=['#0284c7', '#10b981', '#0284c7', '#0284c7', '#dc2626'], line=dict(color='#0f172a', width=1)),
                customdata=custom_s,
                hovertemplate=hover_template_s
            ),
            row=1, col=2
        )
        fig_s_plotly.add_trace(
            go.Scatter(
                x=k_s, y=r2_tr_s,
                mode='lines+markers',
                name='Train R²',
                line=dict(color='#94a3b8', width=1.8, dash='dash'),
                marker=dict(size=6, color='#94a3b8'),
                customdata=custom_s,
                hovertemplate=hover_template_s
            ),
            row=1, col=2
        )

        fig_s_plotly.add_annotation(
            x=2, y=8198.797191,
            text="<b>k=2 (Optimal Selected)<br>RMSE = $8,198.80</b>",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#059669",
            ax=45, ay=-45,
            font=dict(color="#065f46", size=11),
            bgcolor="rgba(240, 253, 244, 0.95)",
            bordercolor="#86efac", borderwidth=1,
            row=1, col=1
        )
        fig_s_plotly.add_annotation(
            x=2, y=0.947439,
            text="<b>k=2 (Optimal Selected)<br>R² = 0.9474</b>",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#059669",
            ax=-45, ay=45,
            font=dict(color="#065f46", size=11),
            bgcolor="rgba(240, 253, 244, 0.95)",
            bordercolor="#86efac", borderwidth=1,
            row=1, col=2
        )

        fig_s_plotly.for_each_annotation(
            lambda a: a.update(font=dict(size=15, color='#1e3a8a', family='Inter, sans-serif'))
            if getattr(a, 'text', '') in ["<b>RMSE by Number of Features</b>", "<b>R-squared by Number of Features</b>"]
            else None
        )

        fig_s_plotly.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=460,
            hovermode="closest",
            margin=dict(l=30, r=30, t=50, b=30),
            legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5, font=dict(color='#0f172a', size=11)),
            hoverlabel=dict(bgcolor="#0f172a", font_size=12, font_color="#ffffff", bordercolor="#38bdf8")
        )

        fig_s_plotly.update_xaxes(title_text="<b>Number of Features</b>", title_font=dict(color='#0f172a', size=12), tickfont=dict(color='#0f172a'), dtick=1, gridcolor='rgba(148, 163, 184, 0.25)', linecolor='#64748b')
        fig_s_plotly.update_yaxes(title_text="<b>RMSE ($)</b>", title_font=dict(color='#0f172a', size=12), tickfont=dict(color='#0f172a'), row=1, col=1, gridcolor='rgba(148, 163, 184, 0.25)', linecolor='#64748b')
        fig_s_plotly.update_yaxes(title_text="<b>R-squared (R²)</b>", title_font=dict(color='#0f172a', size=12), tickfont=dict(color='#0f172a'), row=1, col=2, gridcolor='rgba(148, 163, 184, 0.25)', linecolor='#64748b')

        st.plotly_chart(fig_s_plotly, use_container_width=True)

        st.write("##### 📋 Complete Feature Subset Performance Benchmark Table:")
        bench_df_s = pd.DataFrame({
            "Number of Features": k_s,
            "Selected Features": feature_subsets_s,
            "Test RMSE ($)": [f"${x:,.2f}" for x in rmse_te_s],
            "Test R-squared": [f"{x:.4f}" for x in r2_te_s],
            "Train R-squared": [f"{x:.4f}" for x in r2_tr_s],
            "Status": statuses_s
        })
        st.dataframe(bench_df_s, use_container_width=True, hide_index=True)

    with tab_s_eval:
        st.subheader("📈 Econometric Evaluation & Assumption Testing")
        comp_col1, comp_col2 = st.columns([1, 1.2])
        with comp_col1:
            st.write("##### Head-to-Head Test Set Performance:")
            metrics_table = pd.DataFrame({
                "Metric": ["Features Count", "Test R²", "Test Adjusted R²", "Test RMSE", "Test MAE", "Test MAPE"],
                "Initial Full Model": ["5 Features", "0.8987", "0.7721", "$9,055.96", "$6,961.48", "10.28%"],
                "Final Selected Model": ["2 Features (R&D, Mkt)", "0.9168", "0.8931", "$8,206.33", "$6,469.18", "8.84%"],
                "Lift / Change": ["-60% Complexity", "+1.8%", "+12.1% (Massive)", "-$849.63 (Better)", "-$492.30 (Better)", "-1.44%"]
            })
            st.dataframe(metrics_table, use_container_width=True, hide_index=True)
            st.success("By eliminating the 3 noisy features, the test error decreased and Adjusted R² jumped from 77.2% to 89.3%!")

        with comp_col2:
            if os.path.exists("output_plots/08_feature_selection_comparison.png"):
                st.image("output_plots/08_feature_selection_comparison.png", caption="Model Generalization Lift Comparison", use_container_width=True)

        st.write("##### Diagnostic Plots (Final Model)")
        diag_c1, diag_c2 = st.columns(2)
        with diag_c1:
            if os.path.exists("output_plots/04_actual_vs_predicted_final.png"):
                st.image("output_plots/04_actual_vs_predicted_final.png", caption="Actual vs. Predicted Profit (Final Model)", use_container_width=True)
            if os.path.exists("output_plots/06_residuals_normality_qq.png"):
                st.image("output_plots/06_residuals_normality_qq.png", caption="Normality of Residuals & Q-Q Plot", use_container_width=True)
        with diag_c2:
            if os.path.exists("output_plots/07_standardized_coefficients.png"):
                st.image("output_plots/07_standardized_coefficients.png", caption="Standardized Regression Coefficients", use_container_width=True)

    with tab_s_predict:
        st.subheader("🔮 Live Startup Profit Valuation Predictor")
        st.markdown("Test different operational allocation scenarios using the retrained production pipeline:")
        
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            in_rd = st.slider("🔬 R&D Spend ($)", min_value=0.0, max_value=200000.0, value=120000.0, step=2500.0)
            in_admin = st.slider("🏢 Administration Spend ($) [Non-significant Overhead]", min_value=0.0, max_value=200000.0, value=100000.0, step=2500.0)
        with pcol2:
            in_mkt = st.slider("📢 Marketing Spend ($)", min_value=0.0, max_value=500000.0, value=250000.0, step=5000.0)
            in_state = st.selectbox("📍 State of Incorporation [Non-significant Location]", options=["California", "Florida", "New York"])
        
        if pipeline_startups:
            input_df = pd.DataFrame([{
                'R&D Spend': in_rd,
                'Administration': in_admin,
                'Marketing Spend': in_mkt,
                'State': in_state
            }])
            
            predicted_val = pipeline_startups.predict(input_df)[0]
            
            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.metric(label="Predicted Annual Profit", value=f"${predicted_val:,.2f}")
                st.caption(f"Estimated Net Earnings: **${predicted_val:,.2f}**")
            with res_col2:
                st.markdown(f"""
                <div class="recommendation-box">
                    <div class="diag-title">📊 Econometric Valuation Breakdown</div>
                    <div class="diag-item">• <b>Base Expected Profit (Intercept):</b> $45,542.39 baseline capital return</div>
                    <div class="diag-item">• <b>R&D Driver (${in_rd:,.0f} × 0.7834):</b> <span class="diag-highlight">+${(in_rd * 0.783374):,.2f}</span> value addition</div>
                    <div class="diag-item">• <b>Marketing Driver (${in_mkt:,.0f} × 0.0392):</b> <span class="diag-highlight">+${(in_mkt * 0.039240):,.2f}</span> growth amplification</div>
                    <div class="diag-item">• <b>Administration & Location:</b> <span class="diag-muted" style="font-style: italic;">Filtered out (Zero coefficient bias)</span></div>
                    <div class="diag-item">• <b>Estimated Return on Capital (Profit / Total Spend):</b> <b>{(predicted_val / max(1.0, (in_rd + in_admin + in_mkt)) * 100):.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Model pipeline file not found. Run `crisp_dm_50_startups.py` to compile the pipeline.")
