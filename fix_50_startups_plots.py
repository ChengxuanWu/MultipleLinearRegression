"""
Re-generate all 50 Startups plots with high-contrast typography, crisp card background,
and crystal-clear text visibility across both light and dark display modes.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

os.makedirs("output_plots", exist_ok=True)

# Universal high-contrast card styling: solid white card background with bold slate typography
plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#f8fafc',
    'savefig.facecolor': '#ffffff',
    'savefig.edgecolor': 'none',
    'text.color': '#0f172a',
    'axes.labelcolor': '#0f172a',
    'xtick.color': '#0f172a',
    'ytick.color': '#0f172a',
    'font.size': 11,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titleweight': 'bold',
    'lines.linewidth': 2.2,
    'grid.color': '#cbd5e1',
    'grid.linestyle': ':',
    'grid.alpha': 0.7
})

# 1. Load data & prep variables
df = pd.read_csv("50_Startups.csv")
dummies = pd.get_dummies(df['State'], prefix='State', dtype=float)
df_all = pd.concat([df.drop('State', axis=1), dummies], axis=1)

y = df_all['Profit']
X_train, X_test, y_train, y_test = train_test_split(
    df_all.drop('Profit', axis=1), y, test_size=0.2, random_state=42
)

final_features = ['R&D Spend', 'Marketing Spend']
final_model = LinearRegression().fit(X_train[final_features], y_train)
y_pred_final = final_model.predict(X_test[final_features])

# -------------------------------------------------------------------------------------
# PLOT 1: Correlation Heatmap
# -------------------------------------------------------------------------------------
plt.figure(figsize=(7.5, 5.5), facecolor='#ffffff')
corr = df[['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']].corr()
ax = sns.heatmap(
    corr, annot=True, cmap="coolwarm", fmt=".3f", vmin=-1, vmax=1,
    annot_kws={"size": 12, "weight": "bold", "color": "#0f172a"},
    linewidths=1.2, linecolor='#ffffff',
    cbar_kws={'label': 'Correlation Coefficient'}
)
plt.title("Correlation Heatmap (Numerical Features)", fontsize=13, weight='bold', pad=14, color='#1e3a8a')
plt.tight_layout()
plt.savefig("output_plots/01_correlation_heatmap.png", dpi=300, facecolor='#ffffff')
plt.close()
print("Saved: 01_correlation_heatmap.png")

# -------------------------------------------------------------------------------------
# PLOT 2: Profit by State
# -------------------------------------------------------------------------------------
plt.figure(figsize=(7.5, 5.0), facecolor='#ffffff')
state_palette = {'New York': '#38bdf8', 'California': '#34d399', 'Florida': '#fbbf24'}
sns.boxplot(
    data=df, x='State', y='Profit', palette=state_palette,
    boxprops=dict(alpha=0.9, edgecolor='#0f172a', linewidth=1.5),
    medianprops=dict(color='#0f172a', linewidth=2.0)
)
sns.stripplot(data=df, x='State', y='Profit', color='#0f172a', alpha=0.75, jitter=0.2, size=7.5)
plt.title("Profit Distribution Across States", fontsize=13, weight='bold', pad=14, color='#1e3a8a')
plt.xlabel("State of Operation", fontsize=11, weight='bold', color='#0f172a')
plt.ylabel("Annual Profit ($)", fontsize=11, weight='bold', color='#0f172a')
plt.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
plt.tight_layout()
plt.savefig("output_plots/03_profit_by_state.png", dpi=300, facecolor='#ffffff')
plt.close()
print("Saved: 03_profit_by_state.png")

# -------------------------------------------------------------------------------------
# PLOT 3: Actual vs Predicted (Final Model)
# -------------------------------------------------------------------------------------
r2_final = r2_score(y_test, y_pred_final)
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))

plt.figure(figsize=(7.2, 5.8), facecolor='#ffffff')
plt.scatter(
    y_test, y_pred_final,
    color='#059669', edgecolors='#064e3b', s=95, alpha=0.9, linewidth=1.5,
    label='Test Observations (R&D + Mkt)'
)
min_v = min(y_test.min(), y_pred_final.min()) * 0.92
max_v = max(y_test.max(), y_pred_final.max()) * 1.08
plt.plot([min_v, max_v], [min_v, max_v], color='#dc2626', linestyle='--', lw=2.4, label='Perfect Fit Line (y = x)')

plt.title(f"Final Model: Actual vs Predicted Profit\n(Test R² = {r2_final:.4f}, RMSE = ${rmse_final:,.2f})",
          fontsize=12.5, weight='bold', pad=12, color='#1e3a8a')
plt.xlabel("Actual Profit ($)", fontsize=11, weight='bold', color='#0f172a')
plt.ylabel("Predicted Profit ($)", fontsize=11, weight='bold', color='#0f172a')
plt.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
plt.tight_layout()
plt.savefig("output_plots/04_actual_vs_predicted_final.png", dpi=300, facecolor='#ffffff')
plt.close()
print("Saved: 04_actual_vs_predicted_final.png")

# -------------------------------------------------------------------------------------
# PLOT 4: Residuals Distribution & Q-Q Plot
# -------------------------------------------------------------------------------------
residuals = y_train - final_model.predict(X_train[final_features])
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor='#ffffff')

sns.histplot(residuals, kde=True, ax=axes[0], color='#2563eb', edgecolor='#1e3a8a', alpha=0.75, bins=12)
axes[0].set_title("Residuals Distribution (Final Model)", fontsize=12, weight='bold', color='#1e3a8a', pad=10)
axes[0].set_xlabel("Residual ($)", fontsize=11, weight='bold', color='#0f172a')
axes[0].set_ylabel("Frequency", fontsize=11, weight='bold', color='#0f172a')
axes[0].grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')

sm.qqplot(residuals, line='s', ax=axes[1])
axes[1].get_lines()[0].set_markerfacecolor('#0284c7')
axes[1].get_lines()[0].set_markeredgecolor('#0f172a')
axes[1].get_lines()[0].set_markersize(7)
axes[1].get_lines()[1].set_color('#dc2626')
axes[1].get_lines()[1].set_linewidth(2.2)
axes[1].set_title("Normal Q-Q Plot of Residuals", fontsize=12, weight='bold', color='#1e3a8a', pad=10)
axes[1].set_xlabel("Theoretical Quantiles", fontsize=11, weight='bold', color='#0f172a')
axes[1].set_ylabel("Sample Quantiles", fontsize=11, weight='bold', color='#0f172a')
axes[1].grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')

plt.tight_layout()
plt.savefig("output_plots/06_residuals_normality_qq.png", dpi=300, facecolor='#ffffff')
plt.close()
print("Saved: 06_residuals_normality_qq.png")

# -------------------------------------------------------------------------------------
# PLOT 5: Feature Selection Comparison Bar Chart
# -------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.0), facecolor='#ffffff')
models = ['Initial Full Model (5 Features)', 'Final Selected Model (2 Features)']
r2_vals = [0.8987, 0.9168]
adj_r2_vals = [0.7721, 0.8931]
x_pos = np.arange(len(models))
width = 0.32

rects1 = ax.bar(x_pos - width/2, r2_vals, width, label='Test R²', color='#2563eb', edgecolor='#1e3a8a', alpha=0.9)
rects2 = ax.bar(x_pos + width/2, adj_r2_vals, width, label='Test Adjusted R²', color='#059669', edgecolor='#064e3b', alpha=0.9)

ax.set_ylabel('Score', fontsize=11, weight='bold', color='#0f172a')
ax.set_title('Generalization Lift via Feature Selection', fontsize=13, weight='bold', pad=14, color='#1e3a8a')
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=10.5, weight='bold', color='#0f172a')
ax.set_ylim([0.65, 1.0])
ax.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                textcoords="offset points", ha='center', va='bottom', fontsize=10.5, weight='bold', color='#1e3a8a')
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                textcoords="offset points", ha='center', va='bottom', fontsize=10.5, weight='bold', color='#065f46')

ax.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=10)
plt.tight_layout()
plt.savefig("output_plots/08_feature_selection_comparison.png", dpi=300, facecolor='#ffffff')
plt.close()
print("Saved: 08_feature_selection_comparison.png")

# -------------------------------------------------------------------------------------
# PLOT 6: Standardized Beta Coefficients
# -------------------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_train[final_features]), columns=final_features)
std_reg = LinearRegression().fit(X_scaled, y_train)
std_coefs = pd.Series(std_reg.coef_, index=final_features).sort_values(ascending=True)

plt.figure(figsize=(7.5, 4.5), facecolor='#ffffff')
colors = ['#10b981' if c > 0 else '#ef4444' for c in std_coefs.values]
bars = plt.barh(std_coefs.index, std_coefs.values, color=colors, edgecolor='#0f172a', linewidth=1.2, height=0.5)
plt.axvline(0, color='#64748b', linestyle='--', lw=1.2)
plt.title("Standardized Regression Coefficients (Feature Impact Magnitude)", fontsize=12, pad=12, weight='bold', color='#1e3a8a')
plt.xlabel("Standardized Effect on Profit (Beta)", fontsize=11, weight='bold', color='#0f172a')
plt.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')

for bar in bars:
    w = bar.get_width()
    ha = 'left' if w >= 0 else 'right'
    offset = 1200 if w >= 0 else -1200
    plt.text(w + offset, bar.get_y() + bar.get_height()/2, f"{w:+,.1f}",
             va='center', ha=ha, fontsize=10, weight='bold', color='#0f172a')

plt.tight_layout()
plt.savefig("output_plots/07_standardized_coefficients.png", dpi=300, facecolor='#ffffff')
plt.close()
print("Saved: 07_standardized_coefficients.png")

print("All 50 Startups plots regenerated with high-contrast card styling and crystal-clear text!")
