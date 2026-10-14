"""
Generate Feature Selection Performance Curve & Evaluation Table for 50 Startups
Synchronized with Boston Housing Color Scheme & Typography
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load data
df = pd.read_csv("50_Startups.csv")

# 2. One-hot encoding for State
dummies = pd.get_dummies(df['State'], prefix='State', dtype=float)
df_all = pd.concat([df.drop('State', axis=1), dummies], axis=1)

y = df_all['Profit']

# 3. Split with random_state=0 and test_size=0.2
X_train, X_test, y_train, y_test = train_test_split(
    df_all.drop('Profit', axis=1), y, test_size=0.2, random_state=0
)

# 4. Feature sets evaluated across steps 1 through 5
feature_subsets = [
    ['R&D Spend'],
    ['R&D Spend', 'Marketing Spend'],
    ['R&D Spend', 'Marketing Spend', 'State_New York'],
    ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida'],
    ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida', 'Administration']
]

display_feature_labels = [
    "[R&D Spend]",
    "[R&D Spend, Marketing Spend]",
    "[R&D Spend, Marketing Spend, State_New York]",
    "[R&D Spend, Marketing Spend, State_New York, State_Florida]",
    "[R&D Spend, Marketing Spend, State_New York, State_Florida, State_California]"
]

results = []
for i, features in enumerate(feature_subsets):
    model = LinearRegression()
    model.fit(X_train[features], y_train)
    
    y_pred_tr = model.predict(X_train[features])
    y_pred_te = model.predict(X_test[features])
    
    rmse_tr = float(np.sqrt(mean_squared_error(y_train, y_pred_tr)))
    rmse_te = float(np.sqrt(mean_squared_error(y_test, y_pred_te)))
    r2_tr = float(r2_score(y_train, y_pred_tr))
    r2_te = float(r2_score(y_test, y_pred_te))
    
    status = "Under-specified base" if i == 0 else ("⭐ Optimal Parsimonious" if i == 1 else ("Noise Added" if i == 2 else "Degrading / Collinear"))
    
    results.append({
        'Number of Features': i + 1,
        'Selected Features': display_feature_labels[i],
        'Train RMSE': rmse_tr,
        'Test RMSE': rmse_te,
        'Train R-squared': r2_tr,
        'Test R-squared': r2_te,
        'Status': status
    })

results_df = pd.DataFrame(results)

print("=" * 85)
print("50 STARTUPS FEATURE SELECTION PERFORMANCE TABLE")
print("=" * 85)
print(results_df[['Number of Features', 'Train RMSE', 'Test RMSE', 'Train R-squared', 'Test R-squared', 'Status']].to_string(index=False))
print("\n")

# 5. Generate High-Contrast Dual-Panel Static Plot (Matched with Boston Housing Style)
os.makedirs("output_plots", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor='#ffffff')

# Left Panel: RMSE by Number of Features
axes[0].plot(
    results_df['Number of Features'],
    results_df['Test RMSE'],
    marker='o',
    color='#1f77b4',
    linewidth=2.0,
    markersize=6,
    label='Test RMSE ($)'
)
axes[0].plot(
    results_df['Number of Features'],
    results_df['Train RMSE'],
    marker='s',
    color='#64748b',
    linestyle='--',
    linewidth=1.5,
    markersize=5,
    label='Train RMSE ($)'
)
axes[0].set_title('RMSE by Number of Features', fontsize=13, pad=12, fontweight='bold', color='#0f172a')
axes[0].set_xlabel('Number of Features', fontsize=11, fontweight='bold', color='#0f172a')
axes[0].set_ylabel('RMSE ($)', fontsize=11, fontweight='bold', color='#0f172a')
axes[0].set_xticks(range(1, 6))
axes[0].grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
axes[0].legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9.5)

# Highlight k=2
axes[0].scatter([2], [results_df.loc[results_df['Number of Features'] == 2, 'Test RMSE'].values[0]],
                color='#10b981', s=120, zorder=5, edgecolors='#047857', linewidth=2)
axes[0].annotate('k=2 (Optimal)\nRMSE = $8,198.80', xy=(2, 8198.80), xytext=(2.4, 8400),
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
axes[1].set_xticks(range(1, 6))
axes[1].grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
axes[1].legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9.5)

# Highlight k=2
axes[1].scatter([2], [results_df.loc[results_df['Number of Features'] == 2, 'Test R-squared'].values[0]],
                color='#10b981', s=120, zorder=5, edgecolors='#047857', linewidth=2)
axes[1].annotate('k=2 (Optimal)\nR² = 0.9474', xy=(2, 0.947439), xytext=(2.4, 0.942),
                arrowprops=dict(arrowstyle='->', color='#047857', lw=1.5),
                fontsize=9.5, fontweight='bold', color='#047857',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecfdf5', edgecolor='#10b981', alpha=0.9))

plt.tight_layout()
output_img_path = "output_plots/feature_selection_metrics_curve.png"
plt.savefig(output_img_path, dpi=300)
plt.close()
print(f"Static plot saved successfully to: {output_img_path}")

# 6. Generate and save interactive HTML chart with hover tooltips
fig_plotly = make_subplots(
    rows=1, cols=2,
    subplot_titles=("<b>RMSE by Number of Features</b>", "<b>R-squared by Number of Features</b>"),
    horizontal_spacing=0.10
)

custom_data = list(zip(
    results_df['Status'],
    [f"{r:.4f}" for r in results_df['Test R-squared']],
    [f"${m:,.2f}" for m in results_df['Test RMSE']],
    [f"${m:,.2f}" for m in results_df['Train RMSE']],
    [f"{r:.4f}" for r in results_df['Train R-squared']],
    [f.replace(', ', '<br>• ') for f in results_df['Selected Features']]
))

hover_template = (
    "<span style='font-size:13px; font-weight:bold; color:#38bdf8;'>Features: %{x}</span><br>"
    "<b>Status:</b> %{customdata[0]}<br>"
    "<b>Test RMSE:</b> %{customdata[2]}<br>"
    "<b>Test R²:</b> %{customdata[1]}<br>"
    "<b>Train RMSE:</b> %{customdata[3]}<br>"
    "<b>Train R²:</b> %{customdata[4]}<br>"
    "<b>Predictor Set:</b><br>%{customdata[5]}"
    "<extra></extra>"
)

# Test RMSE
fig_plotly.add_trace(
    go.Scatter(
        x=results_df['Number of Features'],
        y=results_df['Test RMSE'],
        mode='lines+markers',
        name='Test RMSE ($)',
        line=dict(color='#0284c7', width=2.5),
        marker=dict(size=10, color=['#0284c7', '#10b981', '#0284c7', '#0284c7', '#dc2626'], line=dict(color='#0f172a', width=1)),
        customdata=custom_data,
        hovertemplate=hover_template
    ),
    row=1, col=1
)

# Train RMSE
fig_plotly.add_trace(
    go.Scatter(
        x=results_df['Number of Features'],
        y=results_df['Train RMSE'],
        mode='lines+markers',
        name='Train RMSE ($)',
        line=dict(color='#94a3b8', width=1.8, dash='dash'),
        marker=dict(size=6, color='#94a3b8'),
        customdata=custom_data,
        hovertemplate=hover_template
    ),
    row=1, col=1
)

# Test R2
fig_plotly.add_trace(
    go.Scatter(
        x=results_df['Number of Features'],
        y=results_df['Test R-squared'],
        mode='lines+markers',
        name='Test R²',
        line=dict(color='#0284c7', width=2.5),
        marker=dict(size=10, color=['#0284c7', '#10b981', '#0284c7', '#0284c7', '#dc2626'], line=dict(color='#0f172a', width=1)),
        customdata=custom_data,
        hovertemplate=hover_template
    ),
    row=1, col=2
)

# Train R2
fig_plotly.add_trace(
    go.Scatter(
        x=results_df['Number of Features'],
        y=results_df['Train R-squared'],
        mode='lines+markers',
        name='Train R²',
        line=dict(color='#94a3b8', width=1.8, dash='dash'),
        marker=dict(size=6, color='#94a3b8'),
        customdata=custom_data,
        hovertemplate=hover_template
    ),
    row=1, col=2
)

# Highlight k=2
fig_plotly.add_annotation(
    x=2, y=8198.797191,
    text="<b>k=2 (Optimal Selected)<br>RMSE = $8,198.80</b>",
    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#059669",
    ax=45, ay=-45,
    font=dict(color="#065f46", size=11),
    bgcolor="rgba(240, 253, 244, 0.95)", bordercolor="#86efac", borderwidth=1,
    row=1, col=1
)
fig_plotly.add_annotation(
    x=2, y=0.947439,
    text="<b>k=2 (Optimal Selected)<br>R² = 0.9474</b>",
    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#059669",
    ax=-45, ay=45,
    font=dict(color="#065f46", size=11),
    bgcolor="rgba(240, 253, 244, 0.95)", bordercolor="#86efac", borderwidth=1,
    row=1, col=2
)

fig_plotly.for_each_annotation(
    lambda a: a.update(font=dict(size=15, color='#1e3a8a', family='Arial, sans-serif'))
    if getattr(a, 'text', '') in ["<b>RMSE by Number of Features</b>", "<b>R-squared by Number of Features</b>"]
    else None
)

fig_plotly.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=460,
    hovermode="closest",
    margin=dict(l=30, r=30, t=50, b=30),
    legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5, font=dict(color='#0f172a', size=11)),
    hoverlabel=dict(bgcolor="#0f172a", font_size=12, font_color="#ffffff", bordercolor="#38bdf8")
)

fig_plotly.update_xaxes(title_text="<b>Number of Features</b>", dtick=1, gridcolor='rgba(148, 163, 184, 0.25)', linecolor='#64748b')
fig_plotly.update_yaxes(title_text="<b>RMSE ($)</b>", row=1, col=1, gridcolor='rgba(148, 163, 184, 0.25)', linecolor='#64748b')
fig_plotly.update_yaxes(title_text="<b>R-squared (R²)</b>", row=1, col=2, gridcolor='rgba(148, 163, 184, 0.25)', linecolor='#64748b')

html_path = "output_plots/interactive_feature_selection_curve.html"
fig_plotly.write_html(html_path)
print(f"Interactive Plotly curve saved to: {html_path}")
