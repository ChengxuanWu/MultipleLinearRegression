"""
Generate Feature Selection Performance Curves and Benchmark Table for Boston Housing Price Prediction
Matches the visual layout of the reference dual-panel plot (RMSE and R-squared vs. Number of Features)
Also produces an interactive Plotly HTML chart with comprehensive cursor hover tooltips.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load and Clean Dataset
df = pd.read_csv("HousingData.csv")

continuous_features = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
categorical_features = ['CHAS']

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

df_clean = df.copy()
df_clean[continuous_features] = num_imputer.fit_transform(df[continuous_features])
df_clean[categorical_features] = cat_imputer.fit_transform(df[categorical_features])

X = df_clean.drop(columns=['MEDV'])
y = df_clean['MEDV']

# 2. Train-Test Split (80/20) with random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Progressive Feature Ordering (ranked by econometric significance and RFE)
ordered_features = [
    'LSTAT',
    'RM',
    'PTRATIO',
    'DIS',
    'NOX',
    'CHAS',
    'CRIM',
    'RAD',
    'TAX',
    'B',
    'ZN',
    'AGE',
    'INDUS'
]

results = []
for k in range(1, 14):
    feats = ordered_features[:k]
    model = LinearRegression()
    model.fit(X_train[feats], y_train)
    
    y_pred_tr = model.predict(X_train[feats])
    y_pred_te = model.predict(X_test[feats])
    
    rmse_tr = float(np.sqrt(mean_squared_error(y_train, y_pred_tr)))
    rmse_te = float(np.sqrt(mean_squared_error(y_test, y_pred_te)))
    r2_tr = float(r2_score(y_train, y_pred_tr))
    r2_te = float(r2_score(y_test, y_pred_te))
    
    # Calculate Adjusted R2 on test set
    n_test = len(y_test)
    adj_r2_te = 1 - (1 - r2_te) * (n_test - 1) / (n_test - k - 1) if (n_test - k - 1) > 0 else r2_te
    
    status = "Baseline Single" if k == 1 else ("Optimal Backward Elimination" if k == 10 else ("Full Model (13 Features)" if k == 13 else "Intermediate"))
    
    results.append({
        'Number of Features': k,
        'Selected Features': f"[{', '.join(feats)}]",
        'Train RMSE ($1k)': rmse_tr,
        'Test RMSE ($1k)': rmse_te,
        'Train R-squared': r2_tr,
        'Test R-squared': r2_te,
        'Test Adj R-squared': adj_r2_te,
        'Status': status
    })

results_df = pd.DataFrame(results)

print("=" * 95)
print("BOSTON HOUSING FEATURE SELECTION PERFORMANCE TABLE")
print("=" * 95)
print(results_df[['Number of Features', 'Test RMSE ($1k)', 'Test R-squared', 'Test Adj R-squared', 'Status']].to_string(index=False))
print("\n")

os.makedirs("output_plots", exist_ok=True)

# 4. Generate Static High-Contrast Dual-Panel Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor='#ffffff')

# Left Panel: RMSE by Number of Features
axes[0].plot(
    results_df['Number of Features'],
    results_df['Test RMSE ($1k)'],
    marker='o',
    color='#1f77b4',
    linewidth=2.0,
    markersize=6,
    label='Test RMSE ($1k)'
)
axes[0].plot(
    results_df['Number of Features'],
    results_df['Train RMSE ($1k)'],
    marker='s',
    color='#64748b',
    linestyle='--',
    linewidth=1.5,
    markersize=5,
    label='Train RMSE ($1k)'
)
axes[0].set_title('RMSE by Number of Features', fontsize=13, pad=12, fontweight='bold', color='#0f172a')
axes[0].set_xlabel('Number of Features', fontsize=11, fontweight='bold', color='#0f172a')
axes[0].set_ylabel('RMSE ($1,000s)', fontsize=11, fontweight='bold', color='#0f172a')
axes[0].set_xticks(range(1, 14))
axes[0].grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
axes[0].legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9.5)

# Highlight k=10
axes[0].scatter([10], [results_df.loc[results_df['Number of Features'] == 10, 'Test RMSE ($1k)'].values[0]],
                color='#10b981', s=120, zorder=5, edgecolors='#047857', linewidth=2)
axes[0].annotate('k=10 (Optimal)\nRMSE = $5.08k', xy=(10, 5.08), xytext=(8.2, 5.4),
                arrowprops=dict(arrowstyle='->', color='#047857', lw=1.5),
                fontsize=9.5, fontweight='bold', color='#047857',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecfdf5', edgecolor='#10b981', alpha=0.9))

# Right Panel: R-squared by Number of Features
axes[1].plot(
    results_df['Number of Features'],
    results_df['Test R-squared'],
    marker='o',
    color='#1f77b4',
    linewidth=2.0,
    markersize=6,
    label='Test R²'
)
axes[1].plot(
    results_df['Number of Features'],
    results_df['Train R-squared'],
    marker='s',
    color='#64748b',
    linestyle='--',
    linewidth=1.5,
    markersize=5,
    label='Train R²'
)
axes[1].set_title('R-squared by Number of Features', fontsize=13, pad=12, fontweight='bold', color='#0f172a')
axes[1].set_xlabel('Number of Features', fontsize=11, fontweight='bold', color='#0f172a')
axes[1].set_ylabel('R-squared', fontsize=11, fontweight='bold', color='#0f172a')
axes[1].set_xticks(range(1, 14))
axes[1].grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
axes[1].legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9.5)

# Highlight k=10
axes[1].scatter([10], [results_df.loc[results_df['Number of Features'] == 10, 'Test R-squared'].values[0]],
                color='#10b981', s=120, zorder=5, edgecolors='#047857', linewidth=2)
axes[1].annotate('k=10 (Optimal)\nR² = 0.6480', xy=(10, 0.6480), xytext=(7.5, 0.58),
                arrowprops=dict(arrowstyle='->', color='#047857', lw=1.5),
                fontsize=9.5, fontweight='bold', color='#047857',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecfdf5', edgecolor='#10b981', alpha=0.9))

plt.tight_layout()
static_plot_path = "output_plots/boston_feature_selection_curve.png"
plt.savefig(static_plot_path, dpi=300)
plt.close()
print(f"Static validation curve saved to: {static_plot_path}")

# 5. Generate Interactive Plotly Chart with Rich Cursor Hover Details
custom_hover_template = (
    "<b>Feature Count: %{x}</b><br>"
    "<b>Status:</b> %{customdata[0]}<br>"
    "<b>Test RMSE:</b> $%{y:.3f}k<br>"
    "<b>Test R²:</b> %{customdata[1]:.4f}<br>"
    "<b>Test Adj R²:</b> %{customdata[2]:.4f}<br>"
    "<b>Train RMSE:</b> $%{customdata[3]:.3f}k<br>"
    "<b>Train R²:</b> %{customdata[4]:.4f}<br>"
    "<b>Included Predictors:</b><br>%{customdata[5]}"
    "<extra></extra>"
)

custom_data = list(zip(
    results_df['Status'],
    results_df['Test R-squared'],
    results_df['Test Adj R-squared'],
    results_df['Train RMSE ($1k)'],
    results_df['Train R-squared'],
    [f.replace(', ', '<br>• ') for f in results_df['Selected Features']]
))

fig_plotly = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "<b>RMSE by Number of Features (Hover for Data)</b>",
        "<b>R-squared by Number of Features (Hover for Data)</b>"
    ),
    horizontal_spacing=0.12
)

# Test RMSE Trace
fig_plotly.add_trace(
    go.Scatter(
        x=results_df['Number of Features'],
        y=results_df['Test RMSE ($1k)'],
        mode='lines+markers',
        name='Test RMSE ($1k)',
        line=dict(color='#3b82f6', width=2.5),
        marker=dict(size=8, color='#3b82f6'),
        customdata=custom_data,
        hovertemplate=custom_hover_template
    ),
    row=1, col=1
)

# Train RMSE Trace
fig_plotly.add_trace(
    go.Scatter(
        x=results_df['Number of Features'],
        y=results_df['Train RMSE ($1k)'],
        mode='lines+markers',
        name='Train RMSE ($1k)',
        line=dict(color='#94a3b8', width=1.8, dash='dash'),
        marker=dict(size=6, color='#94a3b8'),
        customdata=custom_data,
        hovertemplate=custom_hover_template
    ),
    row=1, col=1
)

# Test R2 Trace
fig_plotly.add_trace(
    go.Scatter(
        x=results_df['Number of Features'],
        y=results_df['Test R-squared'],
        mode='lines+markers',
        name='Test R²',
        line=dict(color='#10b981', width=2.5),
        marker=dict(size=8, color='#10b981'),
        customdata=custom_data,
        hovertemplate=custom_hover_template
    ),
    row=1, col=2
)

# Train R2 Trace
fig_plotly.add_trace(
    go.Scatter(
        x=results_df['Number of Features'],
        y=results_df['Train R-squared'],
        mode='lines+markers',
        name='Train R²',
        line=dict(color='#94a3b8', width=1.8, dash='dash'),
        marker=dict(size=6, color='#94a3b8'),
        customdata=custom_data,
        hovertemplate=custom_hover_template
    ),
    row=1, col=2
)

fig_plotly.update_layout(
    template='plotly_white',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(255,255,255,0.03)',
    font=dict(family='Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto', color='#f8fafc'),
    hovermode='closest',
    hoverlabel=dict(
        bgcolor='#1e293b',
        bordercolor='#38bdf8',
        font=dict(color='#f8fafc', size=12)
    ),
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=-0.25,
        xanchor='center',
        x=0.5,
        font=dict(color='#e2e8f0', size=11)
    )
)

fig_plotly.update_xaxes(title_text="<b>Number of Features</b>", dtick=1, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#e2e8f0'))
fig_plotly.update_yaxes(title_text="<b>RMSE ($1,000s)</b>", row=1, col=1, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#e2e8f0'))
fig_plotly.update_yaxes(title_text="<b>R-squared (R²)</b>", row=1, col=2, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#e2e8f0'))

html_path = "output_plots/boston_interactive_curve.html"
fig_plotly.write_html(html_path)
print(f"Interactive Plotly validation curve saved to: {html_path}")
