"""
Predict Profit for New Startups using the Retrained Final Scikit-Learn Pipeline
Selected Features: R&D Spend, Marketing Spend
"""

import sys
import joblib
import pandas as pd

MODEL_PATH = "50_startups_pipeline.joblib"

def load_pipeline():
    try:
        pipeline = joblib.load(MODEL_PATH)
        return pipeline
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found. Please run 'crisp_dm_50_startups.py' first.")
        sys.exit(1)

def predict_startup_profit(rd_spend: float, admin_spend: float = 0.0, marketing_spend: float = 0.0, state: str = "California") -> float:
    """
    Predict expected profit given spending and state using the retrained final model.
    """
    pipeline = load_pipeline()
    input_data = pd.DataFrame([{
        'R&D Spend': float(rd_spend),
        'Administration': float(admin_spend),
        'Marketing Spend': float(marketing_spend),
        'State': state
    }])
    
    predicted_val = pipeline.predict(input_data)[0]
    return predicted_val

if __name__ == "__main__":
    print("=" * 65)
    print("50 STARTUPS - PROFIT PREDICTION ENGINE (RETRAINED FINAL MODEL)")
    print("Features Selected: R&D Spend, Marketing Spend")
    print("=" * 65)
    
    sample_cases = [
        {"name": "Tech Innovator A", "rd": 160000.0, "admin": 130000.0, "mkt": 400000.0, "state": "California"},
        {"name": "E-Commerce Startup B", "rd": 85000.0, "admin": 115000.0, "mkt": 250000.0, "state": "New York"},
        {"name": "Lean Operations C", "rd": 45000.0, "admin": 65000.0, "mkt": 120000.0, "state": "Florida"},
    ]
    
    print("\n[Evaluating 3 Sample Startup Scenarios]\n")
    for s in sample_cases:
        est_profit = predict_startup_profit(s["rd"], s["admin"], s["mkt"], s["state"])
        print(f"Company: {s['name']}")
        print(f"  - R&D Spend:        ${s['rd']:>10,.2f}")
        print(f"  - Administration:   ${s['admin']:>10,.2f}  (Non-significant / eliminated)")
        print(f"  - Marketing Spend:  ${s['mkt']:>10,.2f}")
        print(f"  - State:            {s['state']}         (Non-significant / eliminated)")
        print(f"  ===> Predicted Profit: ${est_profit:,.2f}\n")
