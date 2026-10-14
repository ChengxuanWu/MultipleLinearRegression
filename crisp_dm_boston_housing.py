"""
CRISP-DM Implementation for Boston Housing Price Prediction using Multiple Linear Regression
With Systematic Missing Value Imputation, Categorical Encoding, and Feature Selection (Scikit-Learn & Statsmodels)

Phases:
1. Business Understanding
2. Data Understanding
3. Data Preparation (Median & Mode Imputation, Categorical Dummy Handling, Train/Test Split)
4. Modeling & Feature Selection (Backward Elimination, RFE, SelectKBest)
5. Evaluation & Econometric Diagnostics (Full Model vs. Final Feature-Selected Model, Linearity, VIF, Normality, Homoscedasticity)
6. Deployment (Final Scikit-Learn Pipeline Serialization & Prediction CLI)
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
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

# Ensure output directory exists
os.makedirs("output_plots", exist_ok=True)

# Plot styling: High-contrast typography with clean readable background
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#f8fafc',
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
  Predict the median value of owner-occupied homes (MEDV, in $1,000s) across Boston census tracts.
  The model must capture socioeconomic indicators (LSTAT, PTRATIO), structural characteristics (RM, AGE),
  environmental conditions (NOX, CHAS), and accessibility metrics (DIS, RAD, TAX, CRIM).
  
Methodology:
  - Clean missing data via robust statistical imputation (Median for skewed continuous variables, Mode for binary dummy).
  - Perform systematic Feature Selection (Backward Elimination, RFE, F-Regression) to eliminate noise and multicollinearity.
  - Rigorously validate econometric assumptions (Linearity, Homoscedasticity, Normality, VIF).
  - Package the optimal solution into a production-ready Scikit-Learn Pipeline.
""")

print("=" * 85)
print("PHASE 2: DATA UNDERSTANDING")
print("=" * 85)

df = pd.read_csv("HousingData.csv")
print(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns\n")
print("Dataset Head:")
print(df.head(), "\n")

print("Missing Value Audit:")
null_summary = df.isnull().sum()
null_percent = (null_summary / len(df)) * 100
missing_df = pd.DataFrame({'Missing_Count': null_summary, 'Percentage (%)': null_percent})
print(missing_df[missing_df['Missing_Count'] > 0], "\n")

print("Descriptive Statistics:")
print(df.describe().round(2), "\n")

# Correlation Matrix
corr_matrix = df.corr()
print("Correlations with Target (MEDV):")
print(corr_matrix['MEDV'].sort_values(ascending=False).round(4), "\n")

# EDA Plot 1: Correlation Heatmap
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={'label': 'Pearson Correlation'})
plt.title("Boston Housing Correlation Heatmap", fontsize=14, pad=12, fontweight='bold', color='#0f172a')
plt.tight_layout()
plt.savefig("output_plots/boston_01_correlation_heatmap.png", dpi=300)
plt.close()

# EDA Plot 2: Target and Key Predictor Distributions
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
sns.histplot(df['MEDV'].dropna(), kde=True, ax=axes[0, 0], color='#2563eb', bins=30)
axes[0, 0].set_title("Target: Median House Value (MEDV)", fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel("MEDV ($1,000s)")

sns.scatterplot(data=df, x='RM', y='MEDV', ax=axes[0, 1], color='#059669', alpha=0.7)
axes[0, 1].set_title("Rooms per Dwelling (RM) vs. MEDV", fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel("Average Number of Rooms (RM)")
axes[0, 1].set_ylabel("MEDV ($1,000s)")

sns.scatterplot(data=df, x='LSTAT', y='MEDV', ax=axes[1, 0], color='#dc2626', alpha=0.7)
axes[1, 0].set_title("Lower Status Population (% LSTAT) vs. MEDV", fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel("% Lower Status (LSTAT)")
axes[1, 0].set_ylabel("MEDV ($1,000s)")

sns.boxplot(data=df, x='CHAS', y='MEDV', ax=axes[1, 1], palette=['#64748b', '#0284c7'])
axes[1, 1].set_title("Charles River Bounds (CHAS Dummy) vs. MEDV", fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel("Tract Bounds River (0 = No, 1 = Yes)")
axes[1, 1].set_ylabel("MEDV ($1,000s)")

plt.tight_layout()
plt.savefig("output_plots/boston_02_distributions_and_scatters.png", dpi=300)
plt.close()

print("=" * 85)
print("PHASE 3: DATA PREPARATION (IMPUTATION & TRAIN/TEST SPLIT)")
print("=" * 85)

# Separate predictors and target
X = df.drop(columns=['MEDV'])
y = df['MEDV']

# Define feature types
continuous_features = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
categorical_features = ['CHAS']

# Train/Test Split (80/20) with random_state=42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"Training observations: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
print(f"Testing observations:  {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)\n")

# Imputation Preprocessor
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Fit imputers strictly on training data
X_train_imputed = X_train.copy()
X_test_imputed = X_test.copy()

X_train_imputed[continuous_features] = num_imputer.fit_transform(X_train[continuous_features])
X_train_imputed[categorical_features] = cat_imputer.fit_transform(X_train[categorical_features])

X_test_imputed[continuous_features] = num_imputer.transform(X_test[continuous_features])
X_test_imputed[categorical_features] = cat_imputer.transform(X_test[categorical_features])

print("Imputation Complete: 0 nulls remain across training and test sets.")

print("=" * 85)
print("PHASE 4: MODELING & SYSTEMATIC FEATURE SELECTION")
print("=" * 85)

# --- 1. Fit Full Model (All 13 Features) ---
full_model = LinearRegression()
full_model.fit(X_train_imputed, y_train)

y_pred_tr_full = full_model.predict(X_train_imputed)
y_pred_te_full = full_model.predict(X_test_imputed)

r2_tr_full = r2_score(y_train, y_pred_tr_full)
r2_te_full = r2_score(y_test, y_pred_te_full)
rmse_te_full = np.sqrt(mean_squared_error(y_test, y_pred_te_full))
mae_te_full = mean_absolute_error(y_test, y_pred_te_full)

print(f"Full Model (13 Features):")
print(f"  Train R^2: {r2_tr_full:.4f} | Test R^2: {r2_te_full:.4f} | Test RMSE: {rmse_te_full:.4f} | Test MAE: {mae_te_full:.4f}\n")

# --- 2. Feature Selection Method 1: Univariate F-Regression (SelectKBest) ---
kbest = SelectKBest(score_func=f_regression, k=8)
kbest.fit(X_train_imputed, y_train)
f_scores = pd.Series(kbest.scores_, index=X.columns).sort_values(ascending=False)
print("SelectKBest (F-Scores Ranking):")
for feat, score in f_scores.items():
    print(f"  {feat:10s}: F = {score:8.2f} (p = {kbest.pvalues_[list(X.columns).index(feat)]:.2e})")
print()

# --- 3. Feature Selection Method 2: Recursive Feature Elimination (RFE) ---
rfe = RFE(estimator=LinearRegression(), n_features_to_select=10)
rfe.fit(X_train_imputed, y_train)
rfe_ranking = pd.DataFrame({'Feature': X.columns, 'RFE_Rank': rfe.ranking_, 'Selected': rfe.support_})
print("RFE Ranking (Top 10 Selected):")
print(rfe_ranking.sort_values(by='RFE_Rank').to_string(index=False), "\n")

# --- 4. Feature Selection Method 3: Econometric Backward Elimination (OLS p-value < 0.05) ---
def backward_elimination_ols(X_data, y_data, sl=0.05):
    features = list(X_data.columns)
    step = 1
    print("Executing Backward Elimination using Ordinary Least Squares (OLS):")
    while len(features) > 0:
        X_with_const = sm.add_constant(X_data[features])
        model_ols = sm.OLS(y_data, X_with_const).fit()
        p_vals = model_ols.pvalues.drop('const')
        max_p = p_vals.max()
        worst_feat = p_vals.idxmax()
        if max_p > sl:
            print(f"  Step {step}: Dropped '{worst_feat}' (p-value = {max_p:.4f} > {sl})")
            features.remove(worst_feat)
            step += 1
        else:
            print(f"\nConvergence achieved in {step-1} reduction steps.")
            print(f"Final Parsimonious Feature Subset ({len(features)} features): {features}\n")
            return features, model_ols

selected_features, ols_reduced = backward_elimination_ols(X_train_imputed, y_train, sl=0.05)

# Fit Final Scikit-Learn Model with Selected Features
final_model = LinearRegression()
final_model.fit(X_train_imputed[selected_features], y_train)

y_pred_tr_sel = final_model.predict(X_train_imputed[selected_features])
y_pred_te_sel = final_model.predict(X_test_imputed[selected_features])

r2_tr_sel = r2_score(y_train, y_pred_tr_sel)
r2_te_sel = r2_score(y_test, y_pred_te_sel)
rmse_te_sel = np.sqrt(mean_squared_error(y_test, y_pred_te_sel))
mae_te_sel = mean_absolute_error(y_test, y_pred_te_sel)

n_test = len(y_test)
p_full = X_train_imputed.shape[1]
p_sel = len(selected_features)
adj_r2_full = 1 - (1 - r2_te_full) * (n_test - 1) / (n_test - p_full - 1)
adj_r2_sel = 1 - (1 - r2_te_sel) * (n_test - 1) / (n_test - p_sel - 1)

print("=" * 85)
print("PHASE 5: EVALUATION & ECONOMETRIC DIAGNOSTICS")
print("=" * 85)

eval_df = pd.DataFrame({
    'Metric': ['Train R-Squared', 'Test R-Squared', 'Test Adjusted R-Squared', 'Test RMSE ($1k)', 'Test MAE ($1k)', 'AIC', 'BIC', 'Feature Count'],
    'Full Model (13 Features)': [
        f"{r2_tr_full:.4f}", f"{r2_te_full:.4f}", f"{adj_r2_full:.4f}",
        f"${rmse_te_full:.4f}k", f"${mae_te_full:.4f}k",
        f"{sm.OLS(y_train, sm.add_constant(X_train_imputed)).fit().aic:.1f}",
        f"{sm.OLS(y_train, sm.add_constant(X_train_imputed)).fit().bic:.1f}",
        13
    ],
    'Selected Model (10 Features)': [
        f"{r2_tr_sel:.4f}", f"{r2_te_sel:.4f}", f"{adj_r2_sel:.4f}",
        f"${rmse_te_sel:.4f}k", f"${mae_te_sel:.4f}k",
        f"{ols_reduced.aic:.1f}",
        f"{ols_reduced.bic:.1f}",
        len(selected_features)
    ]
})
print("Comprehensive Model Comparison Table:")
print(eval_df.to_string(index=False), "\n")

# Diagnostic 1: Actual vs Predicted
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_te_sel, color='#2563eb', alpha=0.75, edgecolors='k', s=65, label='Test Observations')
min_val = min(y_test.min(), y_pred_te_sel.min())
max_val = max(y_test.max(), y_pred_te_sel.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.2, label='Perfect Prediction (1:1)')
plt.title(f"Boston Housing: Actual vs. Predicted MEDV\n(Test R² = {r2_te_sel:.4f}, RMSE = ${rmse_te_sel:.2f}k)", fontsize=12, pad=12)
plt.xlabel("Actual MEDV ($1,000s)", fontsize=11)
plt.ylabel("Predicted MEDV ($1,000s)", fontsize=11)
plt.legend(frameon=True, facecolor='#ffffff')
plt.tight_layout()
plt.savefig("output_plots/boston_04_actual_vs_predicted.png", dpi=300)
plt.close()

# Diagnostic 2: Residuals vs. Fitted
residuals = y_test - y_pred_te_sel
plt.figure(figsize=(7, 5))
plt.scatter(y_pred_te_sel, residuals, color='#0284c7', alpha=0.75, edgecolors='k', s=60)
plt.axhline(0, color='#dc2626', linestyle='--', lw=2)
plt.title("Residuals vs. Fitted Values (Homoscedasticity Check)", fontsize=12, pad=12)
plt.xlabel("Fitted Values ($1,000s)", fontsize=11)
plt.ylabel("Residuals ($1,000s)", fontsize=11)
plt.tight_layout()
plt.savefig("output_plots/boston_05_residuals_vs_fitted.png", dpi=300)
plt.close()

# Diagnostic 3: Normality Q-Q Plot
plt.figure(figsize=(7, 5))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Normal Q-Q Plot of Residuals", fontsize=12, pad=12)
plt.tight_layout()
plt.savefig("output_plots/boston_06_residuals_normality_qq.png", dpi=300)
plt.close()

# Diagnostic 4: Multicollinearity (Variance Inflation Factor - VIF)
X_sel_const = sm.add_constant(X_train_imputed[selected_features])
vif_data = pd.DataFrame()
vif_data["Feature"] = selected_features
vif_data["VIF"] = [variance_inflation_factor(X_sel_const.values, i + 1) for i in range(len(selected_features))]
print("Variance Inflation Factor (VIF) on Selected Features:")
print(vif_data.sort_values(by="VIF", ascending=False).to_string(index=False), "\n")

# Diagnostic 5: Standardized Beta Coefficients
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed[selected_features]), columns=selected_features)
std_reg = LinearRegression().fit(X_scaled, y_train)
std_coefs = pd.Series(std_reg.coef_, index=selected_features).sort_values(key=abs, ascending=True)

plt.figure(figsize=(8, 6))
colors = ['#10b981' if c > 0 else '#ef4444' for c in std_coefs.values]
bars = plt.barh(std_coefs.index, std_coefs.values, color=colors, edgecolor='none', height=0.6)
plt.axvline(0, color='#64748b', linestyle='--', lw=1)
plt.title("Standardized Regression Coefficients (Feature Impact Magnitude)", fontsize=12, pad=12)
plt.xlabel("Standardized Effect on MEDV (Beta)", fontsize=11)
for bar in bars:
    width = bar.get_width()
    ha = 'left' if width >= 0 else 'right'
    offset = 0.1 if width >= 0 else -0.1
    plt.text(width + offset, bar.get_y() + bar.get_height()/2, f"{width:+.2f}",
             va='center', ha=ha, fontsize=9, fontweight='bold', color='#0f172a')
plt.tight_layout()
plt.savefig("output_plots/boston_07_standardized_coefficients.png", dpi=300)
plt.close()

# Statistical Normality & Heteroscedasticity Tests
shapiro_stat, shapiro_p = stats.shapiro(residuals)
bp_stat, bp_pval, _, _ = het_breuschpagan(residuals, sm.add_constant(y_pred_te_sel))
print(f"Normality Test (Shapiro-Wilk): W = {shapiro_stat:.4f}, p = {shapiro_p:.4e}")
print(f"Heteroscedasticity Test (Breusch-Pagan): LM = {bp_stat:.4f}, p = {bp_pval:.4e}\n")

print("=" * 85)
print("PHASE 6: DEPLOYMENT (PIPELINE SERIALIZATION & TESTING)")
print("=" * 85)

# Build Complete End-to-End Production Pipeline using Scikit-Learn
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), [f for f in selected_features if f in continuous_features]),
        ('cat', SimpleImputer(strategy='most_frequent'), [f for f in selected_features if f in categorical_features])
    ],
    remainder='drop'
)

production_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit on entire dataset with selected features for maximum deployment reliability
production_pipeline.fit(X[selected_features], y)

model_path = "boston_housing_pipeline.joblib"
joblib.dump({
    'pipeline': production_pipeline,
    'selected_features': selected_features,
    'all_features': list(X.columns),
    'metrics': {
        'test_r2': float(r2_te_sel),
        'test_rmse': float(rmse_te_sel),
        'test_mae': float(mae_te_sel),
        'aic': float(ols_reduced.aic),
        'bic': float(ols_reduced.bic)
    }
}, model_path)
print(f"Production pipeline successfully serialized to: {model_path}")

# Sanity Verification via Reload
saved_bundle = joblib.load(model_path)
loaded_pipe = saved_bundle['pipeline']
sample_input = pd.DataFrame([{
    'CRIM': 0.05,
    'CHAS': 0.0,
    'NOX': 0.5,
    'RM': 6.5,
    'DIS': 4.0,
    'RAD': 4.0,
    'TAX': 300.0,
    'PTRATIO': 16.0,
    'B': 390.0,
    'LSTAT': 8.0
}])
sample_pred = loaded_pipe.predict(sample_input)[0]
print(f"Deployment Verification Inference:")
print(f"  Input: RM=6.5, LSTAT=8.0%, PTRATIO=16.0, NOX=0.5, CRIM=0.05, CHAS=0")
print(f"  Predicted Boston Median House Value: ${sample_pred*1000:,.2f} (MEDV = {sample_pred:.2f})\n")

print("CRISP-DM Execution Complete for Boston Housing Dataset.")
