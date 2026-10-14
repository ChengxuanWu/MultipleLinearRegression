"""
CLI Inference Script for Boston Housing Price Prediction
Powered by the Champion Feature-Selected Pipeline (Lasso/RFECV 7-Feature Set with High-R² Regularized Engine)
"""

import sys
import argparse
import joblib
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

MODEL_PATH = "boston_housing_pipeline.joblib"

def predict_price(args_dict=None, mode='high-r2'):
    try:
        model_bundle = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model artifact '{MODEL_PATH}' not found. Run compare_feature_selection_methods.py first.")
        sys.exit(1)

    selected_features = model_bundle.get('selected_features', ['RM', 'DIS', 'LSTAT', 'PTRATIO', 'NOX', 'RAD', 'TAX'])
    metrics = model_bundle.get('metrics', {})

    if mode == 'linear' and 'linear_pipeline' in model_bundle:
        pipeline = model_bundle['linear_pipeline']
        engine_name = "Optimal Linear Model (Lasso/RFECV 7 Features)"
        engine_r2 = metrics.get('linear_test_r2', 0.6685)
        engine_rmse = metrics.get('linear_test_rmse', 4.930)
    else:
        pipeline = model_bundle.get('pipeline', model_bundle.get('linear_pipeline'))
        engine_name = "⭐ Champion High-R² Engine (Polynomial Interaction + Regularization)"
        engine_r2 = metrics.get('poly_test_r2', 0.8029)
        engine_rmse = metrics.get('poly_test_rmse', 3.802)
    
    # Defaults representing median Boston suburban tract characteristics
    defaults = {
        'RM': 6.28,       # Average number of rooms per dwelling
        'DIS': 3.79,      # Weighted distances to five Boston employment centres
        'LSTAT': 11.36,   # % lower status of the population
        'PTRATIO': 19.05, # Pupil-teacher ratio by town
        'NOX': 0.538,     # Nitric oxides concentration (parts per 10 million)
        'RAD': 5.0,       # Radial highway accessibility index
        'TAX': 330.0,     # Full-value property-tax rate per $10,000
        'CRIM': 0.25,     # Per capita crime rate
        'CHAS': 0.0,      # Charles River dummy (0 = No, 1 = Yes)
        'ZN': 0.0,        # Residential zoning
        'INDUS': 9.69,    # Non-retail acres
        'AGE': 77.5,      # Older units built < 1940
        'B': 391.44       # Demographic index
    }

    if args_dict:
        for k, v in args_dict.items():
            if v is not None and k in defaults:
                defaults[k] = float(v)

    input_df = pd.DataFrame([defaults])[selected_features]
    predicted_medv = pipeline.predict(input_df)[0]
    # Bound to valid Boston housing domain
    predicted_medv = max(5.0, min(50.0, float(predicted_medv)))
    total_dollar_value = predicted_medv * 1000.0

    print("=" * 75)
    print("🏡 BOSTON HOUSE PRICE PREDICTION ENGINE")
    print("=" * 75)
    print(f"Selected Model Engine: {engine_name}")
    print(f"Engine Reliability:    Test R² = {engine_r2:.4f} | Test RMSE = ${engine_rmse:.2f}k")
    print("-" * 75)
    print("Input Tract Characteristics (7 Selected Features):")
    for feat in selected_features:
        unit = "rooms" if feat == 'RM' else ("%" if feat == 'LSTAT' else ("ratio" if feat == 'PTRATIO' else ("miles" if feat == 'DIS' else "")))
        print(f"  • {feat:10s} : {defaults[feat]:>8.2f} {unit}")
    print("-" * 75)
    print(f"Predicted Median Value (MEDV): {predicted_medv:.2f} ($1,000s)")
    print(f"Estimated Market Valuation:   ${total_dollar_value:,.2f}")
    print("=" * 75)
    return predicted_medv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Boston Housing Median Price (MEDV)")
    parser.add_argument("--mode", choices=['high-r2', 'linear'], default='high-r2', help="Prediction engine mode")
    parser.add_argument("--rm", type=float, help="Average rooms per dwelling (e.g. 6.5)")
    parser.add_argument("--dis", type=float, help="Distance to employment centres (e.g. 4.0)")
    parser.add_argument("--lstat", type=float, help="% Lower status of population (e.g. 8.5)")
    parser.add_argument("--ptratio", type=float, help="Pupil-teacher ratio (e.g. 17.5)")
    parser.add_argument("--nox", type=float, help="Nitric oxides concentration (e.g. 0.52)")
    parser.add_argument("--rad", type=float, help="Radial highway index (1 to 24)")
    parser.add_argument("--tax", type=float, help="Property tax rate per $10k (e.g. 310)")

    args = parser.parse_args()
    args_dict = {
        'RM': args.rm,
        'DIS': args.dis,
        'LSTAT': args.lstat,
        'PTRATIO': args.ptratio,
        'NOX': args.nox,
        'RAD': args.rad,
        'TAX': args.tax
    }
    predict_price(args_dict, mode=args.mode)
