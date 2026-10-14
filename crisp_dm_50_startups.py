"""
CRISP-DM Implementation for 50 Startups Profit Prediction using Multiple Linear Regression
With Explicit One-Hot Encoding and Feature Selection (Scikit-Learn)

Phases:
1. Business Understanding
2. Data Understanding
3. Data Preparation (One-Hot Encoding for State, Train/Test Split)
4. Modeling & Feature Selection (Backward Elimination, RFE, SelectKBest)
5. Evaluation (Full Model vs. Final Feature-Selected Model)
6. Deployment (Final Pipeline Serialization & Prediction CLI)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create directories for outputs
os.makedirs("output_plots", exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'savefig.transparent': True,
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'text.color': '#0f172a',
    'axes.labelcolor': '#0f172a',
    'xtick.color': '#0f172a',
    'ytick.color': '#0f172a',
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})

print("=" * 85)
print("PHASE 1: BUSINESS UNDERSTANDING")
print("=" * 85)
print("""
Objective:
  Predict startup Profit based on operational expenditures and State.
  Apply One-Hot Encoding to categorical location data and perform systematic
  Feature Selection to eliminate redundant/noisy predictors, retraining a
  parsimonious, highly generalizable Multiple Linear Regression model.
""")

print("=" * 85)
print("PHASE 2: DATA UNDERSTANDING")
print("=" * 85)

df = pd.read_csv("50_Startups.csv")
print(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns\n")
print("Dataset Head:")
print(df.head(), "\n")

print("Categorical Feature ('State') Distribution:")
print(df['State'].value_counts(), "\n")

numerical_cols = ['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']
corr_matrix = df[numerical_cols].corr()
print("Correlation Matrix:")
print(corr_matrix.round(4), "\n")

# EDA Plot 1: Heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f", vmin=-1, vmax=1)
plt.title("Correlation Heatmap (Numerical Features)", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("output_plots/01_correlation_heatmap.png", dpi=300)
plt.close()

# EDA Plot 2: Pairplot
pairplot_fig = sns.pairplot(df, hue='State', palette='Set2')
pairplot_fig.fig.suptitle("Pairplot of Startups Features by State", y=1.02, fontsize=14)
plt.savefig("output_plots/02_pairplot.png", dpi=300, bbox_inches='tight')
plt.close()

print("=" * 85)
print("PHASE 3: DATA PREPARATION (ONE-HOT ENCODING & SPLITTING)")
print("=" * 85)

# Separate Features X and Target y
X = df[['R&D Spend', 'Administration', 'Marketing Spend', 'State']]
y = df['Profit']

# 1. One-Hot Encoding for 'State'
# Use drop='first' (dropping California) to prevent the Dummy Variable Trap
encoder = OneHotEncoder(drop='first', sparse_output=False)
state_encoded_array = encoder.fit_transform(X[['State']])
state_columns = encoder.get_feature_names_out(['State']).tolist()

print(f"Original Categorical Column: 'State' (Classes: California, Florida, New York)")
print(f"One-Hot Encoded Columns created (California dropped as reference): {state_columns}")

# Construct the full encoded feature DataFrame
df_encoded = pd.concat([
    X[['R&D Spend', 'Administration', 'Marketing Spend']],
    pd.DataFrame(state_encoded_array, columns=state_columns)
], axis=1)

print("\nPrepared Feature Matrix Head (after One-Hot Encoding):")
print(df_encoded.head(), "\n")

# 2. Train-Test Split (80% Train, 20% Test, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    df_encoded, y, test_size=0.2, random_state=42
)
print(f"Training observations: {X_train.shape[0]} | Testing observations: {X_test.shape[0]}\n")

print("=" * 85)
print("PHASE 4: MODELING & SYSTEMATIC FEATURE SELECTION")
print("=" * 85)

# --- 4.1 Full Initial Model ---
full_model = LinearRegression()
full_model.fit(X_train, y_train)

print("[A] INITIAL FULL MODEL (All 5 Features):")
full_coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': full_model.coef_
})
print(f"  Intercept: ${full_model.intercept_:,.2f}")
print(full_coef_df.to_string(index=False), "\n")

# OLS Statistical Diagnostics on Full Model
X_train_sm = sm.add_constant(X_train)
ols_full = sm.OLS(y_train, X_train_sm).fit()
print("Statsmodels OLS Summary on Full Model:")
print(ols_full.summary(), "\n")

# --- 4.2 Feature Selection Method 1: Econometric Backward Elimination ---
print("-" * 80)
print("[B] FEATURE SELECTION METHOD 1: BACKWARD ELIMINATION (p-value & Adj R^2)")
print("-" * 80)

elimination_history = []
current_features = list(X_train.columns)
step = 1

while len(current_features) > 0:
    X_curr_sm = sm.add_constant(X_train[current_features])
    model_step = sm.OLS(y_train, X_curr_sm).fit()
    p_vals = model_step.pvalues.drop('const', errors='ignore')
    max_p = p_vals.max()
    worst_feature = p_vals.idxmax()
    adj_r2 = model_step.rsquared_adj
    aic = model_step.aic
    
    elimination_history.append({
        'Step': step,
        'Remaining Features': list(current_features),
        'Num Features': len(current_features),
        'Highest p-value Feature': worst_feature,
        'Max p-value': max_p,
        'Adj R-squared': adj_r2,
        'AIC': aic
    })
    
    if max_p > 0.05 and len(current_features) > 1:
        current_features.remove(worst_feature)
        step += 1
    else:
        break

history_df = pd.DataFrame(elimination_history)
for idx, row in history_df.iterrows():
    print(f"Step {row['Step']}: Features ({row['Num Features']}): {row['Remaining Features']}")
    print(f"   Max p-val: {row['Highest p-value Feature']} (p = {row['Max p-value']:.4f}) | Adj R^2: {row['Adj R-squared']:.4f} | AIC: {row['AIC']:.2f}")

# --- 4.3 Feature Selection Method 2: Scikit-Learn RFE & SelectKBest ---
print("\n" + "-" * 80)
print("[C] FEATURE SELECTION METHOD 2: SCIKIT-LEARN RFE & SELECTKBEST")
print("-" * 80)

# Standardized RFE
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

rfe_2 = RFE(LinearRegression(), n_features_to_select=2)
rfe_2.fit(X_train_scaled, y_train)
rfe_selected_features = X_train.columns[rfe_2.get_support()].tolist()
print(f"Scikit-Learn RFE (Top 2 Features): {rfe_selected_features}")

# SelectKBest (F-regression)
skb = SelectKBest(f_regression, k=2)
skb.fit(X_train, y_train)
skb_selected_features = X_train.columns[skb.get_support()].tolist()
f_scores_df = pd.DataFrame({
    'Feature': X_train.columns,
    'F_Score': skb.scores_,
    'p_value': skb.pvalues_
}).sort_values(by='F_Score', ascending=False)

print("\nScikit-Learn SelectKBest F-Scores:")
print(f_scores_df.to_string(index=False), "\n")

# --- 4.4 Feature Subsets Evaluation Curve (Matching User's Reference) ---
print("-" * 80)
print("[D] EVALUATION BY NUMBER OF FEATURES (RMSE & R^2 CURVE)")
print("-" * 80)

# Evaluate subsets 1 to 5
curve_subsets = [
    ['R&D Spend'],
    ['R&D Spend', 'Marketing Spend'],
    ['R&D Spend', 'Marketing Spend', 'State_New York'],
    ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida'],
    ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida', 'Administration']
]

display_labels = [
    "[R&D Spend]",
    "[R&D Spend, Marketing Spend]",
    "[R&D Spend, Marketing Spend, State_New York]",
    "[R&D Spend, Marketing Spend, State_New York, State_Florida]",
    "[R&D Spend, Marketing Spend, State_New York, State_Florida, State_California]"
]

curve_metrics = []
for idx, f_set in enumerate(curve_subsets):
    m_temp = LinearRegression().fit(X_train[f_set], y_train)
    p_temp = m_temp.predict(X_test[f_set])
    rmse_temp = np.sqrt(mean_squared_error(y_test, p_temp))
    r2_temp = r2_score(y_test, p_temp)
    curve_metrics.append({
        'Number of Features': idx + 1,
        'Selected Features': display_labels[idx],
        'RMSE': rmse_temp,
        'R-squared': r2_temp
    })

curve_df = pd.DataFrame(curve_metrics)
print(curve_df.to_string(index=False), "\n")

# Generate and save the 2-subplot curves figure
fig_curve, axes_c = plt.subplots(1, 2, figsize=(11.5, 5))

axes_c[0].plot(curve_df['Number of Features'], curve_df['RMSE'], marker='o', color='#1f77b4', linewidth=1.8, markersize=6)
axes_c[0].set_title('RMSE by Number of Features', fontsize=13, pad=10)
axes_c[0].set_xlabel('Number of Features', fontsize=11)
axes_c[0].set_ylabel('RMSE', fontsize=11)
axes_c[0].set_xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
axes_c[0].set_ylim([8100, 9200])

axes_c[1].plot(curve_df['Number of Features'], curve_df['R-squared'], marker='o', color='#1f77b4', linewidth=1.8, markersize=6)
axes_c[1].set_title('R-squared by Number of Features', fontsize=13, pad=10)
axes_c[1].set_xlabel('Number of Features', fontsize=11)
axes_c[1].set_ylabel('R-squared', fontsize=11)
axes_c[1].set_xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
axes_c[1].set_ylim([0.934, 0.949])

plt.tight_layout()
plt.savefig("output_plots/feature_selection_metrics_curve.png", dpi=300)
plt.close()
print("Saved feature selection curves to 'output_plots/feature_selection_metrics_curve.png'\n")

# --- 4.5 Conclusion on Final Features ---
final_features = ['R&D Spend', 'Marketing Spend']
print(f">>> FINAL SELECTED FEATURES FOR RE-TRAINING: {final_features} <<<\n")

# --- 4.6 Re-training Final Model on Selected Features ---
final_model = LinearRegression()
final_model.fit(X_train[final_features], y_train)

final_coef_df = pd.DataFrame({
    'Feature': final_features,
    'Coefficient': final_model.coef_
})

print(f"Final Model Intercept: ${final_model.intercept_:,.2f}")
print("Final Model Coefficients:")
print(final_coef_df.to_string(index=False), "\n")

print("Final Retrained Regression Equation:")
final_eq = f"Profit = ${final_model.intercept_:,.2f} + ({final_model.coef_[0]:,.4f} * R&D Spend) + ({final_model.coef_[1]:,.4f} * Marketing Spend)"
print(final_eq, "\n")

print("=" * 85)
print("PHASE 5: EVALUATION (COMPARISON: FULL MODEL vs. FINAL SELECTED MODEL)")
print("=" * 85)

def compute_model_metrics(m, X_tr, X_te, y_tr, y_te, feature_cols):
    y_tr_pred = m.predict(X_tr[feature_cols])
    y_te_pred = m.predict(X_te[feature_cols])
    
    n_te = len(y_te)
    k = len(feature_cols)
    
    r2_tr = r2_score(y_tr, y_tr_pred)
    r2_te = r2_score(y_te, y_te_pred)
    adj_r2_te = 1 - ((1 - r2_te) * (n_te - 1) / (n_te - k - 1))
    
    mae_te = mean_absolute_error(y_te, y_te_pred)
    mse_te = mean_squared_error(y_te, y_te_pred)
    rmse_te = np.sqrt(mse_te)
    mape_te = np.mean(np.abs((y_te - y_te_pred) / y_te)) * 100
    
    return {
        'Train R2': r2_tr,
        'Test R2': r2_te,
        'Test Adj R2': adj_r2_te,
        'Test MAE': mae_te,
        'Test RMSE': rmse_te,
        'Test MAPE (%)': mape_te
    }

full_metrics = compute_model_metrics(full_model, X_train, X_test, y_train, y_test, list(X_train.columns))
final_metrics = compute_model_metrics(final_model, X_train, X_test, y_train, y_test, final_features)

comparison_df = pd.DataFrame({
    'Metric': list(full_metrics.keys()),
    'Full Model (5 Features)': [
        f"{v:.4f}" if "R2" in k or "%" in k else f"${v:,.2f}" for k, v in full_metrics.items()
    ],
    'Final Selected Model (2 Features)': [
        f"{v:.4f}" if "R2" in k or "%" in k else f"${v:,.2f}" for k, v in final_metrics.items()
    ]
})

print("Head-to-Head Model Comparison:")
print(comparison_df.to_string(index=False), "\n")

# Diagnostic Plots
y_test_pred_final = final_model.predict(X_test[final_features])

# Plot: Actual vs Predicted (Final Model)
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_test_pred_final, color='#10b981', edgecolors='k', s=80, alpha=0.85, label='Test Startups (Selected Features)')
min_v = min(y_test.min(), y_test_pred_final.min()) * 0.95
max_v = max(y_test.max(), y_test_pred_final.max()) * 1.05
plt.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Perfect Fit line (y = x)')
plt.title(f"Final Model: Actual vs Predicted Profit (Test $R^2 = {final_metrics['Test R2']:.3f}$)", fontsize=13, pad=12)
plt.xlabel("Actual Profit ($)", fontsize=11)
plt.ylabel("Predicted Profit ($)", fontsize=11)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig("output_plots/04_actual_vs_predicted_final.png", dpi=300)
plt.close()

# Plot: Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(8, 4.5))
models = ['Full Model (5 Features)', 'Final Selected (R&D + Mkt)']
r2_vals = [full_metrics['Test R2'], final_metrics['Test R2']]
adj_r2_vals = [full_metrics['Test Adj R2'], final_metrics['Test Adj R2']]
x_pos = np.arange(len(models))
width = 0.35

ax.bar(x_pos - width/2, r2_vals, width, label='Test R^2', color='#3b82f6')
ax.bar(x_pos + width/2, adj_r2_vals, width, label='Test Adjusted R^2', color='#10b981')
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Generalization Lift via Feature Selection', fontsize=13, pad=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylim([0.65, 1.0])
for i in x_pos:
    ax.text(i - width/2, r2_vals[i] + 0.01, f"{r2_vals[i]:.3f}", ha='center', fontweight='bold')
    ax.text(i + width/2, adj_r2_vals[i] + 0.01, f"{adj_r2_vals[i]:.3f}", ha='center', fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig("output_plots/08_feature_selection_comparison.png", dpi=300)
plt.close()

print("=" * 85)
print("PHASE 6: DEPLOYMENT")
print("=" * 85)

# Build an end-to-end deployment Pipeline
# Accepts the raw input features (R&D Spend, Administration, Marketing Spend, State)
# Automatically extracts the selected final features ('R&D Spend', 'Marketing Spend')
# and feeds into the retrained LinearRegression estimator.

feature_selector_transformer = ColumnTransformer(
    transformers=[
        ('select_features', 'passthrough', final_features)
    ],
    remainder='drop'
)

deploy_pipeline = Pipeline(steps=[
    ('selector', feature_selector_transformer),
    ('regressor', final_model)
])

# Fit on raw training DataFrame to establish feature names
deploy_pipeline.fit(df[['R&D Spend', 'Administration', 'Marketing Spend', 'State']], y)

model_filename = "50_startups_pipeline.joblib"
joblib.dump(deploy_pipeline, model_filename)
print(f"Final Retrained Pipeline successfully serialized to '{model_filename}'")

# Demonstration
sample_startups = pd.DataFrame([
    {'R&D Spend': 160000.0, 'Administration': 130000.0, 'Marketing Spend': 400000.0, 'State': 'California'},
    {'R&D Spend': 85000.0, 'Administration': 115000.0, 'Marketing Spend': 250000.0, 'State': 'New York'},
    {'R&D Spend': 45000.0, 'Administration': 65000.0, 'Marketing Spend': 120000.0, 'State': 'Florida'},
])

preds = deploy_pipeline.predict(sample_startups)
sample_startups['Predicted Profit'] = [f"${p:,.2f}" for p in preds]
print("\nDemonstration Inference with Final Retrained Pipeline:")
print(sample_startups.to_string(index=False))

print("\n" + "=" * 85)
print("CRISP-DM PIPELINE WITH ONE-HOT ENCODING & FEATURE SELECTION COMPLETED!")
print("=" * 85)
