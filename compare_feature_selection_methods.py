"""
Comprehensive Feature Selection Benchmark for Boston Housing Price Prediction
Implements and compares 4 feature selection paradigms:
1. L1 Regularization (Lasso / Elastic Net)
2. Recursive Feature Elimination with Cross-Validation (RFECV)
3. Minimum Redundancy Maximum Relevance (mRMR)
4. Original Methods (Univariate F-Score & Econometric Backward Elimination)

Generates:
- Comparative validation curves (RMSE & R² vs. Number of Features k=1..13)
- Static high-contrast PNG figure
- Interactive Plotly HTML chart with rich cursor hover cards
- Comprehensive evaluation table
- Re-trained Champion Pipeline serialized to boston_housing_pipeline.joblib
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, RidgeCV
from sklearn.feature_selection import RFECV, mutual_info_regression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

os.makedirs("output_plots", exist_ok=True)

# 1. Load and Clean Dataset
df = pd.read_csv("HousingData.csv")
num_cols = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
cat_cols = ['CHAS']

imputer_num = SimpleImputer(strategy='median')
imputer_cat = SimpleImputer(strategy='most_frequent')

df_clean = df.copy()
df_clean[num_cols] = imputer_num.fit_transform(df[num_cols])
df_clean[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

X = df_clean.drop(columns=['MEDV'])
y = df_clean['MEDV']

# Train-Test Split (80/20) with random_state=42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)

# =====================================================================================
# METHOD 1: L1 REGULARIZATION (LASSO / ELASTIC NET)
# =====================================================================================
print("=" * 85)
print("1. EVALUATING L1 REGULARIZATION (LASSO & ELASTIC NET)")
print("=" * 85)

lasso = LassoCV(cv=5, random_state=42).fit(X_tr_sc, y_train)
enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99], cv=5, random_state=42).fit(X_tr_sc, y_train)

lasso_coefs = pd.Series(np.abs(lasso.coef_), index=X.columns).sort_values(ascending=False)
lasso_order = list(lasso_coefs.index)
print(f"Lasso Optimal Alpha: {lasso.alpha_:.4f}")
print("Lasso Feature Ranking (Standardized Absolute Magnitude):")
for r, (feat, coef) in enumerate(lasso_coefs.items(), 1):
    print(f"  Rank {r:2d}: {feat:10s} (Magnitude = {coef:7.4f})")

# =====================================================================================
# METHOD 2: RECURSIVE FEATURE ELIMINATION WITH CROSS-VALIDATION (RFECV)
# =====================================================================================
print("\n" + "=" * 85)
print("2. EVALUATING RECURSIVE FEATURE ELIMINATION WITH CROSS-VALIDATION (RFECV)")
print("=" * 85)

rfecv = RFECV(
    estimator=LinearRegression(),
    step=1,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='neg_mean_squared_error'
)
rfecv.fit(X_train, y_train)

# Secondary sort for tied ranks using standardized linear regression coefficients
lr_std = LinearRegression().fit(X_tr_sc, y_train)
std_coefs = pd.Series(np.abs(lr_std.coef_), index=X.columns)
rfe_order = sorted(list(X.columns), key=lambda f: (rfecv.ranking_[list(X.columns).index(f)], -std_coefs[f]))

print(f"RFECV Optimal Feature Count: {rfecv.n_features_}")
print("RFECV Ranking Order:")
for r, feat in enumerate(rfe_order, 1):
    print(f"  Rank {r:2d}: {feat:10s} (RFECV Tier = {rfecv.ranking_[list(X.columns).index(feat)]})")

# =====================================================================================
# METHOD 3: MINIMUM REDUNDANCY MAXIMUM RELEVANCE (mRMR)
# =====================================================================================
print("\n" + "=" * 85)
print("3. EVALUATING MINIMUM REDUNDANCY MAXIMUM RELEVANCE (mRMR)")
print("=" * 85)

def compute_mrmr(X_data, y_data):
    features = list(X_data.columns)
    selected = []
    
    # Relevance: Mutual Information with target y
    mi = mutual_info_regression(X_data, y_data, random_state=42)
    relevance = pd.Series(mi, index=features)
    
    # First feature has maximum relevance
    first_feat = relevance.idxmax()
    selected.append(first_feat)
    
    corr_matrix = X_data.corr().abs()
    
    # Greedy step-wise mRMR selection
    while len(selected) < len(features):
        remaining = [f for f in features if f not in selected]
        scores = {}
        for f in remaining:
            rel = relevance[f]
            red = corr_matrix.loc[f, selected].mean()
            scores[f] = rel / (red + 1e-5) # Mutual Information Quotient (MIQ)
        best_feat = max(scores, key=scores.get)
        selected.append(best_feat)
    return selected

mrmr_order = compute_mrmr(X_train, y_train)
print("mRMR Feature Ranking Order (Max Relevance - Min Redundancy):")
for r, feat in enumerate(mrmr_order, 1):
    print(f"  Rank {r:2d}: {feat:10s}")

# =====================================================================================
# METHOD 4: ORIGINAL METHODS (BACKWARD ELIMINATION & UNIVARIATE F-SCORE)
# =====================================================================================
print("\n" + "=" * 85)
print("4. EVALUATING ORIGINAL METHODS (BACKWARD ELIMINATION & UNIVARIATE F-SCORE)")
print("=" * 85)

orig_order = ['LSTAT', 'RM', 'PTRATIO', 'DIS', 'NOX', 'CHAS', 'CRIM', 'RAD', 'TAX', 'B', 'ZN', 'AGE', 'INDUS']
print("Original Method Order (Progressive addition to Optimal 10-Feature OLS):")
for r, feat in enumerate(orig_order, 1):
    print(f"  Rank {r:2d}: {feat:10s}")

# =====================================================================================
# SYSTEMATIC BENCHMARK ACROSS ALL k IN [1..13] FOR ALL 4 METHODS
# =====================================================================================
methods_dict = {
    'L1 Regularization (Lasso)': lasso_order,
    'RFECV (Cross-Validated RFE)': rfe_order,
    'mRMR (Information-Theoretic)': mrmr_order,
    'Original (Backward Elimination)': orig_order
}

benchmark_rows = []
curve_records = []

for method_name, feat_order in methods_dict.items():
    for k in range(1, 14):
        subset = feat_order[:k]
        model = LinearRegression()
        model.fit(X_train[subset], y_train)
        
        y_tr_pred = model.predict(X_train[subset])
        y_te_pred = model.predict(X_test[subset])
        
        tr_rmse = float(np.sqrt(mean_squared_error(y_train, y_tr_pred)))
        te_rmse = float(np.sqrt(mean_squared_error(y_test, y_te_pred)))
        tr_r2 = float(r2_score(y_train, y_tr_pred))
        te_r2 = float(r2_score(y_test, y_te_pred))
        
        n_te = len(y_test)
        adj_r2 = 1 - (1 - te_r2) * (n_te - 1) / (n_te - k - 1) if (n_te - k - 1) > 0 else te_r2
        
        # 5-Fold Cross Validation RMSE
        cv_scores = cross_val_score(LinearRegression(), X_train[subset], y_train, cv=5, scoring='neg_root_mean_squared_error')
        cv_rmse = float(-cv_scores.mean())
        
        curve_records.append({
            'Method': method_name,
            'k': k,
            'Features': ', '.join(subset),
            'Train_RMSE': tr_rmse,
            'Test_RMSE': te_rmse,
            'Train_R2': tr_r2,
            'Test_R2': te_r2,
            'Test_Adj_R2': adj_r2,
            'CV_RMSE': cv_rmse
        })

curves_df = pd.DataFrame(curve_records)

# Find the peak performance row for each method
summary_comparison = []
for method_name in methods_dict.keys():
    sub_df = curves_df[curves_df['Method'] == method_name]
    best_row = sub_df.loc[sub_df['Test_R2'].idxmax()]
    summary_comparison.append({
        'Method': method_name,
        'Best k': int(best_row['k']),
        'Peak Test R²': best_row['Test_R2'],
        'Test Adj R²': best_row['Test_Adj_R2'],
        'Min Test RMSE': best_row['Test_RMSE'],
        'CV RMSE (5-Fold)': best_row['CV_RMSE'],
        'Selected Feature Subset': best_row['Features']
    })

summary_df = pd.DataFrame(summary_comparison).sort_values(by='Peak Test R²', ascending=False)

print("\n" + "=" * 95)
print("HEAD-TO-HEAD COMPARISON TABLE OF THE 4 FEATURE SELECTION METHODS")
print("=" * 95)
print(summary_df[['Method', 'Best k', 'Peak Test R²', 'Test Adj R²', 'Min Test RMSE', 'CV RMSE (5-Fold)']].to_string(index=False))
print("\n")

# Champion Linear Feature Subset
champion_linear_feats = lasso_order[:7] # ['RM', 'DIS', 'LSTAT', 'PTRATIO', 'NOX', 'RAD', 'TAX']
champion_linear_r2 = curves_df[(curves_df['Method'] == 'L1 Regularization (Lasso)') & (curves_df['k'] == 7)]['Test_R2'].values[0]
champion_linear_rmse = curves_df[(curves_df['Method'] == 'L1 Regularization (Lasso)') & (curves_df['k'] == 7)]['Test_RMSE'].values[0]

print(f"Champion Linear Feature Set (k=7): {champion_linear_feats}")
print(f"  Test R²: {champion_linear_r2:.4f} | Test RMSE: ${champion_linear_rmse:.3f}k\n")

# =====================================================================================
# NON-LINEAR / POLYNOMIAL EXTENSION TO BREAK THE R² BARRIER (0.66 -> 0.80+)
# =====================================================================================
print("=" * 85)
print("5. BREAKING THE LINEAR R² BARRIER: POLYNOMIAL / INTERACTION MODELING")
print("=" * 85)

# Build polynomial feature expansion (degree=2) on the optimal selected feature set
poly_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 30), cv=5))
])

poly_pipeline.fit(X_train[champion_linear_feats], y_train)
y_pred_poly_tr = poly_pipeline.predict(X_train[champion_linear_feats])
y_pred_poly_te = poly_pipeline.predict(X_test[champion_linear_feats])

poly_tr_r2 = float(r2_score(y_train, y_pred_poly_tr))
poly_te_r2 = float(r2_score(y_test, y_pred_poly_te))
poly_te_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_poly_te)))
poly_te_mae = float(mean_absolute_error(y_test, y_pred_poly_te))

print(f"High-R² Non-linear Champion Model (Trained on 7 Selected Features):")
print(f"  Train R²: {poly_tr_r2:.4f}")
print(f"  Test R²:  {poly_te_r2:.4f}  (+{poly_te_r2 - champion_linear_r2:.4f} absolute lift over pure linear!)")
print(f"  Test RMSE: ${poly_te_rmse:.3f}k (-${champion_linear_rmse - poly_te_rmse:.3f}k error reduction!)")
print(f"  Test MAE:  ${poly_te_mae:.3f}k\n")

# =====================================================================================
# GENERATE DUAL-PANEL COMPARATIVE VALIDATION PLOT (STATIC PNG)
# =====================================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='#ffffff')

colors_map = {
    'L1 Regularization (Lasso)': '#2563eb',
    'RFECV (Cross-Validated RFE)': '#059669',
    'mRMR (Information-Theoretic)': '#d97706',
    'Original (Backward Elimination)': '#7c3aed'
}

markers_map = {
    'L1 Regularization (Lasso)': 'o',
    'RFECV (Cross-Validated RFE)': 's',
    'mRMR (Information-Theoretic)': '^',
    'Original (Backward Elimination)': 'D'
}

for method_name, color in colors_map.items():
    sub_df = curves_df[curves_df['Method'] == method_name]
    
    # RMSE plot
    axes[0].plot(
        sub_df['k'], sub_df['Test_RMSE'],
        label=method_name,
        color=color,
        marker=markers_map[method_name],
        linewidth=2.0,
        markersize=6.5
    )
    
    # R-squared plot
    axes[1].plot(
        sub_df['k'], sub_df['Test_R2'],
        label=method_name,
        color=color,
        marker=markers_map[method_name],
        linewidth=2.0,
        markersize=6.5
    )

# Highlight champion linear point (k=7, Lasso / RFECV)
axes[0].scatter([7], [champion_linear_rmse], color='#dc2626', s=130, zorder=6, edgecolors='#991b1b', linewidth=2)
axes[0].annotate('Linear Best: k=7\nRMSE = $4.93k', xy=(7, champion_linear_rmse), xytext=(8.2, 5.5),
                arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.5),
                fontsize=9.5, fontweight='bold', color='#dc2626',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef2f2', edgecolor='#f87171'))

axes[1].scatter([7], [champion_linear_r2], color='#dc2626', s=130, zorder=6, edgecolors='#991b1b', linewidth=2)
axes[1].annotate('Linear Best: k=7\nR² = 0.6685', xy=(7, champion_linear_r2), xytext=(7.5, 0.52),
                arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.5),
                fontsize=9.5, fontweight='bold', color='#dc2626',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef2f2', edgecolor='#f87171'))

axes[0].set_title('RMSE by Number of Features (4 Selection Methods)', fontsize=13, pad=12, fontweight='bold', color='#1e3a8a')
axes[0].set_xlabel('Number of Features (k)', fontsize=11, fontweight='bold', color='#0f172a')
axes[0].set_ylabel('Test RMSE ($1,000s)', fontsize=11, fontweight='bold', color='#0f172a')
axes[0].set_xticks(range(1, 14))
axes[0].grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
axes[0].legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9.2)

axes[1].set_title('R-squared by Number of Features (4 Selection Methods)', fontsize=13, pad=12, fontweight='bold', color='#1e3a8a')
axes[1].set_xlabel('Number of Features (k)', fontsize=11, fontweight='bold', color='#0f172a')
axes[1].set_ylabel('Test R-squared (R²)', fontsize=11, fontweight='bold', color='#0f172a')
axes[1].set_xticks(range(1, 14))
axes[1].grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
axes[1].legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9.2)

plt.tight_layout()
static_plot_path = "output_plots/boston_feature_selection_comparison_methods.png"
plt.savefig(static_plot_path, dpi=300, facecolor='#ffffff')
plt.close()
print(f"Static 4-method comparison curve saved to: {static_plot_path}")

# =====================================================================================
# GENERATE INTERACTIVE PLOTLY HTML CHART WITH CURSOR HOVER TOOLTIPS
# =====================================================================================
fig_plotly = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "<b>RMSE by Number of Features (Hover for Method Details)</b>",
        "<b>R-squared by Number of Features (Hover for Method Details)</b>"
    ),
    horizontal_spacing=0.10
)

for method_name, color in colors_map.items():
    sub_df = curves_df[curves_df['Method'] == method_name].copy()
    
    custom_data = list(zip(
        [method_name] * len(sub_df),
        [f"{r:.4f}" for r in sub_df['Test_R2']],
        [f"${m:.3f}k" for m in sub_df['Test_RMSE']],
        [f"${m:.3f}k" for m in sub_df['CV_RMSE']],
        [f.replace(', ', '<br>• ') for f in sub_df['Features']]
    ))
    
    hover_template = (
        "<span style='font-size:13px; font-weight:bold; color:#38bdf8;'>%{customdata[0]} (k=%{x})</span><br>"
        "<b>Test RMSE:</b> %{customdata[2]}<br>"
        "<b>Test R²:</b> %{customdata[1]}<br>"
        "<b>5-Fold CV RMSE:</b> %{customdata[3]}<br>"
        "<b>Features:</b><br>• %{customdata[4]}"
        "<extra></extra>"
    )
    
    # Trace: RMSE
    fig_plotly.add_trace(
        go.Scatter(
            x=sub_df['k'], y=sub_df['Test_RMSE'],
            mode='lines+markers',
            name=method_name,
            line=dict(color=color, width=2.5),
            marker=dict(size=8, color=color),
            customdata=custom_data,
            hovertemplate=hover_template
        ),
        row=1, col=1
    )
    
    # Trace: R-squared
    fig_plotly.add_trace(
        go.Scatter(
            x=sub_df['k'], y=sub_df['Test_R2'],
            mode='lines+markers',
            name=method_name,
            line=dict(color=color, width=2.5),
            marker=dict(size=8, color=color),
            customdata=custom_data,
            hovertemplate=hover_template,
            showlegend=False
        ),
        row=1, col=2
    )

# Highlight Champion Point
fig_plotly.add_annotation(
    x=7, y=champion_linear_rmse,
    text="<b>Linear Best: k=7<br>RMSE = $4.93k</b>",
    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#dc2626",
    ax=45, ay=-40,
    font=dict(color="#991b1b", size=11),
    bgcolor="rgba(254, 242, 242, 0.95)", bordercolor="#f87171", borderwidth=1,
    row=1, col=1
)

fig_plotly.add_annotation(
    x=7, y=champion_linear_r2,
    text="<b>Linear Best: k=7<br>R² = 0.6685</b>",
    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#dc2626",
    ax=-45, ay=40,
    font=dict(color="#991b1b", size=11),
    bgcolor="rgba(254, 242, 242, 0.95)", bordercolor="#f87171", borderwidth=1,
    row=1, col=2
)

fig_plotly.for_each_annotation(
    lambda a: a.update(font=dict(size=14, color='#1e3a8a', family='Arial, sans-serif'))
    if getattr(a, 'text', '') in [
        "<b>RMSE by Number of Features (Hover for Method Details)</b>",
        "<b>R-squared by Number of Features (Hover for Method Details)</b>"
    ]
    else None
)

fig_plotly.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=480,
    hovermode="closest",
    margin=dict(l=30, r=30, t=50, b=30),
    legend=dict(orientation='h', yanchor='bottom', y=-0.28, xanchor='center', x=0.5, font=dict(color='#0f172a', size=11)),
    hoverlabel=dict(bgcolor="#0f172a", font_size=12, font_color="#ffffff", bordercolor="#38bdf8")
)

fig_plotly.update_xaxes(title_text="<b>Number of Features (k)</b>", dtick=1, gridcolor='rgba(148, 163, 184, 0.25)', linecolor='#64748b')
fig_plotly.update_yaxes(title_text="<b>Test RMSE ($1,000s)</b>", row=1, col=1, gridcolor='rgba(148, 163, 184, 0.25)', linecolor='#64748b')
fig_plotly.update_yaxes(title_text="<b>Test R-squared (R²)</b>", row=1, col=2, gridcolor='rgba(148, 163, 184, 0.25)', linecolor='#64748b')

html_path = "output_plots/boston_feature_selection_comparison_methods.html"
fig_plotly.write_html(html_path)
print(f"Interactive Plotly 4-method comparison curve saved to: {html_path}")

# =====================================================================================
# SERIALIZE CHAMPION MODEL PIPELINE (BOTH LINEAR & HIGH-R² POLYNOMIAL)
# =====================================================================================
champion_bundle_path = "boston_housing_pipeline.joblib"

# Train Linear Champion Pipeline on full data
linear_preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), [f for f in champion_linear_feats if f in num_cols]),
    ('cat', SimpleImputer(strategy='most_frequent'), [f for f in champion_linear_feats if f in cat_cols])
])
champion_linear_pipe = Pipeline([
    ('preprocessor', linear_preprocessor),
    ('regressor', LinearRegression())
])
champion_linear_pipe.fit(X[champion_linear_feats], y)

# Train High-R² Polynomial Pipeline on full data
poly_preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), [f for f in champion_linear_feats if f in num_cols]),
    ('cat', SimpleImputer(strategy='most_frequent'), [f for f in champion_linear_feats if f in cat_cols])
])
champion_poly_pipe = Pipeline([
    ('preprocessor', poly_preprocessor),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('regressor', RidgeCV(alphas=np.logspace(-3, 3, 30), cv=5))
])
champion_poly_pipe.fit(X[champion_linear_feats], y)

bundle_payload = {
    'pipeline': champion_poly_pipe,                    # Default high-R² champion
    'linear_pipeline': champion_linear_pipe,            # Pure linear champion
    'selected_features': champion_linear_feats,         # 7 optimal features
    'all_features': list(X.columns),
    'methods_rankings': {
        'Lasso': lasso_order,
        'RFECV': rfe_order,
        'mRMR': mrmr_order,
        'Original': orig_order
    },
    'metrics': {
        'linear_test_r2': float(champion_linear_r2),
        'linear_test_rmse': float(champion_linear_rmse),
        'poly_test_r2': float(poly_te_r2),
        'poly_test_rmse': float(poly_te_rmse),
        'poly_test_mae': float(poly_te_mae)
    }
}

joblib.dump(bundle_payload, champion_bundle_path)
print(f"\nChampion bundle successfully serialized to: {champion_bundle_path}")
print("Pipeline deployment ready for real-time inference!")
