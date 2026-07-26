# 💳 Credit Score Classification System

An end-to-end machine learning project that predicts a customer's **credit score category** — *Good*, *Standard*, or *Poor* — from their financial and behavioural data. The project covers data cleaning, exploratory analysis, model training and comparison, and deployment as a web application with a Flask REST API and an interactive HTML front end.

---

## 📌 Overview

Banks and financial institutions need a fast, reliable way to assess a customer's creditworthiness. This project trains and compares several machine learning classifiers on a real-world credit dataset, selects the best-performing model, and deploys it behind a simple web app so predictions can be made in real time from a browser.

**Target classes:**

| Label | Meaning |
|:---:|---|
| `Good` | Healthy credit profile |
| `Standard` | Average credit profile |
| `Poor` | Weak credit profile |

---

## 🚀 Demo / Features

- 🔍 Predicts credit score from 21 financial/behavioural inputs
- 📊 Returns prediction confidence and full class probability breakdown
- 🌐 Simple web UI (`index.html`) to submit customer data and see results instantly
- ⚡ Lightweight Flask REST API (`/predict`, `/health`)
- 🧠 Powered by a tuned **CatBoost** classifier (best of 4 models tested)

---

## 🗂️ Project Structure

```
├── final_project_2.ipynb   # Notebook: EDA, preprocessing, model training & evaluation
├── best_model.pkl          # Final trained CatBoost model (pickled)
├── app.py                  # Flask backend serving predictions
├── index.html               # Web front end
└── README.md                # Project documentation
```

---

## 🧪 Dataset & Preprocessing

The training and test datasets contained ~27–28 columns of numeric and categorical customer data. Key preprocessing steps:

- Dropped identifier columns (`ID`, `Customer_ID`, `Month`, `SSN`, `Name`)
- Cleaned and converted `Age` to numeric, filled missing values with the median
- Filled missing values in `Monthly_Inhand_Salary` (mean), `Type_of_Loan` / `Num_of_Delayed_Payment` (mode), `Num_Credit_Inquiries`, `Amount_invested_monthly`, and `Monthly_Balance`
- Parsed `Credit_History_Age` (e.g. `"3 Years and 6 Months"`) into total months using regex
- Label-encoded all categorical columns (`Occupation`, `Type_of_Loan`, `Credit_Mix`, `Payment_of_Min_Amount`, `Payment_Behaviour`), fit on combined train + test values for consistent mappings
- Split data 75% / 25% for training and evaluation (`random_state=42`)

### Features used (21 total)

`Age`, `Occupation`, `Annual_Income`, `Monthly_Inhand_Salary`, `Num_Bank_Accounts`, `Num_Credit_Card`, `Interest_Rate`, `Num_of_Loan`, `Type_of_Loan`, `Delay_from_due_date`, `Num_of_Delayed_Payment`, `Changed_Credit_Limit`, `Num_Credit_Inquiries`, `Credit_Mix`, `Outstanding_Debt`, `Credit_Utilization_Ratio`, `Credit_History_Age`, `Payment_of_Min_Amount`, `Total_EMI_per_month`, `Amount_invested_monthly`, `Payment_Behaviour`, `Monthly_Balance`

---

## 🤖 Models Compared

| Model | Training Accuracy | Test Accuracy | Notes |
|---|:---:|:---:|---|
| Decision Tree | 75.45% | 69.02% | Baseline (entropy, max depth 40) |
| Random Forest | 86.98% | 75.85% | 500 trees, max depth 20 |
| Random Forest + GridSearchCV | 86.98% | 75.85% | 5-fold CV hyper-parameter tuning |
| XGBoost | 83.12% | 75.43% | 1000 estimators, lr 0.03 |
| **CatBoost (Final Model)** | – | **77.52%** | 2000 iterations, depth 10 ✅ Selected for deployment |

**CatBoost** was chosen as the production model for its highest test accuracy and strong performance across all three classes.

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **ML Libraries:** scikit-learn, CatBoost, XGBoost, pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Backend:** Flask, Flask-CORS
- **Frontend:** HTML, CSS, JavaScript

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install flask flask-cors pandas numpy scikit-learn catboost xgboost
   ```

4. **Run the app**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://localhost:5000
   ```

---

## 📡 API Reference

### `GET /health`
Simple health check.

```json
{ "status": "ok", "model": "CatBoost" }
```

### `POST /predict`
Accepts a JSON object of customer features and returns the predicted credit score.

**Sample request body:**
```json
{
  "Age": 34,
  "Occupation": "Engineer",
  "Annual_Income": 45000,
  "Monthly_Inhand_Salary": 3200,
  "Num_Bank_Accounts": 3,
  "Num_Credit_Card": 4,
  "Interest_Rate": 12,
  "Num_of_Loan": 2,
  "Type_of_Loan": "Auto Loan",
  "Delay_from_due_date": 5,
  "Num_of_Delayed_Payment": 3,
  "Changed_Credit_Limit": 2.5,
  "Num_Credit_Inquiries": 4,
  "Credit_Mix": "Good",
  "Outstanding_Debt": 1200,
  "Credit_Utilization_Ratio": 32.5,
  "Credit_History_Age": "5 Years and 2 Months",
  "Payment_of_Min_Amount": "Yes",
  "Total_EMI_per_month": 150,
  "Amount_invested_monthly": 200,
  "Payment_Behaviour": "High_spent_Medium_value_payments",
  "Monthly_Balance": 500
}
```

**Sample response:**
```json
{
  "predicted_label": 0,
  "credit_score": "Good",
  "confidence": 0.82,
  "probabilities": {
    "Good": 0.82,
    "Poor": 0.06,
    "Standard": 0.12
  }
}
```

---

## 📓 Notebook

`final_project_2.ipynb` contains the full workflow:

1. Library imports & dataset loading
2. Exploratory data analysis (train & test)
3. Missing value handling & feature cleaning
4. Label encoding of categorical features
5. Train/test split
6. Training & evaluating Decision Tree, Random Forest, GridSearchCV-tuned RF, XGBoost, and CatBoost
7. Confusion matrices & classification reports for each model
8. Saving the best model (`best_model.pkl`)

---

## 📈 Future Improvements

- Add cross-validation-based model comparison for more robust benchmarking
- Handle class imbalance (e.g. SMOTE) to improve minority-class recall
- Add input validation and better error handling on the API
- Containerize the app with Docker for easier deployment
- Add authentication and rate limiting for production use

---

## 📄 License

This project is available for educational and personal use. Add a license file (e.g. MIT) if you plan to distribute or open-source it publicly.

---

## 🙋 Author

Feel free to open an issue or submit a pull request if you'd like to contribute or report a bug.
