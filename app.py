import re
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Load model ────────────────────────────────────────────────────────────
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

LABEL_MAP = {0: "Good", 1: "Poor", 2: "Standard"}

# ── Categorical encodings (same order as LabelEncoder fit on train+test) ──
# LabelEncoder sorts alphabetically, so we sort each list too
ENCODINGS = {
    "Occupation": sorted([
        "Accountant","Architect","Developer","Doctor","Engineer",
        "Entrepreneur","Journalist","Lawyer","Manager","Mechanic",
        "Musician","Scientist","Teacher","Writer"
    ]),
    "Type_of_Loan": sorted([
        "Auto Loan","Credit-Builder Loan","Home Equity Loan",
        "Mortgage Loan","Not Specified","Payday Loan",
        "Personal Loan","Student Loan"
    ]),
    "Credit_Mix": sorted(["Bad","Good","Standard"]),
    "Payment_of_Min_Amount": sorted(["NM","No","Yes"]),
    "Payment_Behaviour": sorted([
        "High_spent_Large_value_payments","High_spent_Medium_value_payments",
        "High_spent_Small_value_payments","Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments","Low_spent_Small_value_payments"
    ]),
}

def encode(col, val):
    """Replicate LabelEncoder: return index of val in sorted list."""
    classes = ENCODINGS.get(col, [])
    val = str(val)
    return classes.index(val) if val in classes else 0

def credit_history_to_months(x):
    """'3 Years and 6 Months' → 42"""
    m = re.search(r"(\d+)\s+Years?\s+and\s+(\d+)\s+Months?", str(x))
    return int(m.group(1)) * 12 + int(m.group(2)) if m else 0

FEATURE_COLS = [
    "Age","Occupation","Annual_Income","Monthly_Inhand_Salary",
    "Num_Bank_Accounts","Num_Credit_Card","Interest_Rate","Num_of_Loan",
    "Type_of_Loan","Delay_from_due_date","Num_of_Delayed_Payment",
    "Changed_Credit_Limit","Num_Credit_Inquiries","Credit_Mix",
    "Outstanding_Debt","Credit_Utilization_Ratio","Credit_History_Age",
    "Payment_of_Min_Amount","Total_EMI_per_month","Amount_invested_monthly",
    "Payment_Behaviour","Monthly_Balance"
]

def preprocess(data: dict) -> pd.DataFrame:
    row = {}
    for col in FEATURE_COLS:
        val = data.get(col, 0)

        if col == "Credit_History_Age":
            row[col] = credit_history_to_months(val)
        elif col in ENCODINGS:
            row[col] = encode(col, val)
        else:
            try:
                row[col] = float(val)
            except:
                row[col] = 0.0

    return pd.DataFrame([row])[FEATURE_COLS]

# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "CatBoost"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    try:
        df   = preprocess(data)
        pred = int(model.predict(df).flatten()[0])
        prob = model.predict_proba(df)[0]

        return jsonify({
            "predicted_label": pred,
            "credit_score":    LABEL_MAP[pred],
            "confidence":      round(float(prob[pred]), 4),
            "probabilities": {
                "Good":     round(float(prob[0]), 4),
                "Poor":     round(float(prob[1]), 4),
                "Standard": round(float(prob[2]), 4),
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Run ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("✅ Running at http://localhost:5000")
    app.run(debug=True, port=5000)
