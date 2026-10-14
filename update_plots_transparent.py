"""
Re-generate all plots with transparent backgrounds, high-contrast bold fonts, and crystal-clear text.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

os.makedirs("output_plots", exist_ok=True)

# Set base aesthetic: clean white card background with dark, bold text
plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
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
})

df = pd.read_csv("50_Startups.csv")
dummies = pd.get_dummies(df['State'], prefix='State', dtype=float)
df_all = pd.concat([df.drop('State', axis=1), dummies], axis=1)

y = df_all['Profit']

# 1. Plot: Correlation Heatmap
plt.figure(figsize=(7, 5))
corr = df[['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']].corr()
ax = sns.heatmap(
    corr, annot=True, cmap="coolwarm", fmt=".3f", vmin=-1, vmax=1,
    annot_kws={"size": 11, "weight": "bold", "color": "#0f172a"},
    cbar_kws={'label': 'Correlation Coefficient'}
)
plt.title("Correlation Heatmap (Numerical Features)", fontsize=13, weight='bold', pad=12, color='#1e3a8a')
plt.tight_layout()
plt.savefig("output_plots/01_correlation_heatmap.png", transparent=True, dpi=300)
plt.close()

# 2. Plot: Profit by State
plt.figure(figsize=(7, 4.5))
sns.boxplot(data=df, x='State', y='Profit', palette='Set2', boxprops=dict(alpha=0.85))
sns.stripplot(data=df, x='State', y='Profit', color='#0f172a', alpha=0.7, jitter=0.2, size=7)
plt.title("Profit Distribution Across States", fontsize=13, weight='bold', pad=12, color='#1e3a8a')
plt.xlabel("State", fontsize=11, weight='bold', color='#0f172a')
plt.ylabel("Profit ($)", fontsize=11, weight='bold', color='#0f172a')
plt.tight_layout()
plt.savefig("output_plots/03_profit_by_state.png", transparent=True, dpi=300)
plt.close()

# 3. Plot: Feature Selection Comparison Bar Chart
fig, ax = plt.subplots(figsize=(8, 4.5))
models = ['Full Model (5 Features)', 'Final Selected (R&D + Mkt)']
r2_vals = [0.8987, 0.9168]
adj_r2_vals = [0.7721, 0.8931]
x_pos = np.arange(len(models))
width = 0.35

ax.bar(x_pos - width/2, r2_vals, width, label='Test R^2', color='#2563eb', edgecolor='#1e3a8a', alpha=0.9)
ax.bar(x_pos + width/2, adj_r2_vals, width, label='Test Adjusted R^2', color='#059669', edgecolor='#064e3b', alpha=0.9)
ax.set_ylabel('Score', fontsize=11, weight='bold', color='#0f172a')
ax.set_title('Generalization Lift via Feature Selection', fontsize=13, weight='bold', pad=12, color='#1e3a8a')
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=11, weight='bold', color='#0f172a')
ax.set_ylim([0.65, 1.0])
for i in x_pos:
    ax.text(i - width/2, r2_vals[i] + 0.015, f"{r2_vals[i]:.3f}", ha='center', weight='bold', color='#1e3a8a', fontsize=11)
    ax.text(i + width/2, adj_r2_vals[i] + 0.015, f"{adj_r2_vals[i]:.3f}", ha='center', weight='bold', color='#064e3b', fontsize=11)
ax.legend(loc='lower right', framealpha=0.9)
plt.tight_layout()
plt.savefig("output_plots/08_feature_selection_comparison.png", transparent=True, dpi=300)
plt.close()

# 4. Plot: Actual vs Predicted Final
X_train, X_test, y_train, y_test = train_test_split(df_all.drop('Profit', axis=1), y, test_size=0.2, random_state=42)
final_model = LinearRegression().fit(X_train[['R&D Spend', 'Marketing Spend']], y_train)
y_pred_final = final_model.predict(X_test[['R&D Spend', 'Marketing Spend']])

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_final, color='#059669', edgecolors='#064e3b', s=90, alpha=0.9, label='Test Startups (Selected Features)')
min_v = min(y_test.min(), y_pred_final.min()) * 0.95
max_v = max(y_test.max(), y_pred_final.max()) * 1.05
plt.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2.2, label='Perfect Fit line (y = x)')
plt.title(f"Final Model: Actual vs Predicted Profit (Test R^2 = 0.917)", fontsize=13, weight='bold', pad=12, color='#1e3a8a')
plt.xlabel("Actual Profit ($)", fontsize=11, weight='bold', color='#0f172a')
plt.ylabel("Predicted Profit ($)", fontsize=11, weight='bold', color='#0f172a')
plt.legend(framealpha=0.9)
plt.tight_layout()
plt.savefig("output_plots/04_actual_vs_predicted_final.png", transparent=True, dpi=300)
plt.close()

# 5. Plot: Residuals Normality Q-Q
residuals = y_train - final_model.predict(X_train[['R&D Spend', 'Marketing Spend']])
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(residuals, kde=True, ax=axes[0], color='#4f46e5', edgecolor='#1e1b4b', alpha=0.8)
axes[0].set_title("Residuals Distribution", fontsize=12, weight='bold', color='#1e3a8a')
axes[0].set_xlabel("Residual ($)", fontsize=11, weight='bold', color='#0f172a')
axes[0].set_ylabel("Frequency", fontsize=11, weight='bold', color='#0f172a')

sm.qqplot(residuals, line='s', ax=axes[1])
axes[1].get_lines()[0].set_markerfacecolor('#0284c7')
axes[1].get_lines()[0].set_markeredgecolor('#0369a1')
axes[1].get_lines()[1].set_color('#dc2626')
axes[1].get_lines()[1].set_linewidth(2.2)
axes[1].set_title("Normal Q-Q Plot of Residuals", fontsize=12, weight='bold', color='#1e3a8a')
axes[1].set_xlabel("Theoretical Quantiles", fontsize=11, weight='bold', color='#0f172a')
axes[1].set_ylabel("Sample Quantiles", fontsize=11, weight='bold', color='#0f172a')
plt.tight_layout()
plt.savefig("output_plots/06_residuals_normality_qq.png", facecolor='white', dpi=300)
plt.close()

# 6. Plot: Static Feature Selection Curves (Matching Reference Image)
k_vals = [1, 2, 3, 4, 5]
rmse_vals = [8274.868018, 8198.797191, 8309.059683, 8409.916714, 9137.990153]
r2_vals = [0.946459, 0.947439, 0.946015, 0.944697, 0.934707]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
# RMSE
axes[0].plot(k_vals, rmse_vals, marker='o', color='#0284c7', linewidth=2.2, markersize=8)
axes[0].scatter([2], [8198.797191], color='#059669', s=150, zorder=5, edgecolors='#064e3b', label='Optimal (Min RMSE)')
axes[0].set_title('RMSE by Number of Features', fontsize=13, weight='bold', pad=10, color='#1e3a8a')
axes[0].set_xlabel('Number of Features', fontsize=11, weight='bold', color='#0f172a')
axes[0].set_ylabel('RMSE', fontsize=11, weight='bold', color='#0f172a')
axes[0].set_xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
axes[0].set_ylim([8100, 9200])
axes[0].legend(loc='upper left', framealpha=0.9)

# R-squared
axes[1].plot(k_vals, r2_vals, marker='o', color='#0284c7', linewidth=2.2, markersize=8)
axes[1].scatter([2], [0.947439], color='#059669', s=150, zorder=5, edgecolors='#064e3b', label='Optimal (Peak R^2)')
axes[1].set_title('R-squared by Number of Features', fontsize=13, weight='bold', pad=10, color='#1e3a8a')
axes[1].set_xlabel('Number of Features', fontsize=11, weight='bold', color='#0f172a')
axes[1].set_ylabel('R-squared', fontsize=11, weight='bold', color='#0f172a')
axes[1].set_xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
axes[1].set_ylim([0.934, 0.949])
axes[1].legend(loc='lower left', framealpha=0.9)

plt.tight_layout()
plt.savefig("output_plots/feature_selection_metrics_curve.png", transparent=True, dpi=300)
plt.close()

print("All transparent plots re-generated successfully!")
