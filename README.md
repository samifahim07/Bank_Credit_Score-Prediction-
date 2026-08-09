# Credit Score Classification System

An end-to-end machine learning project that predicts a customer's **credit score category** — *Good*, *Standard*, or *Poor* — using financial and behavioural data. The project covers data preprocessing, exploratory data analysis, model training and comparison, evaluation, and deployment as a web application using Flask and an interactive HTML frontend.

## Overview

Banks and financial institutions need efficient methods for assessing customer creditworthiness. This project uses machine learning classification algorithms to predict a customer's credit score category based on financial history, credit behaviour, loan information, and payment patterns.

The project compares multiple classification models, selects the best-performing model based on test accuracy, and deploys it through a Flask REST API that can be accessed from a web interface.

### Target Classes

|    Label   | Meaning                |
| :--------: | :--------------------- |
|   `Good`   | Healthy credit profile |
| `Standard` | Average credit profile |
|   `Poor`   | Weak credit profile    |

## Features

* Predicts credit score from 21 financial and behavioural features
* Provides the predicted credit score category
* Returns prediction confidence
* Returns probability for each credit score class
* Interactive HTML-based frontend
* Flask REST API for prediction
* Health-check endpoint for API monitoring
* Multiple machine learning models compared before selecting the final model
* Final model saved using Pickle for deployment

## Project Structure

```text
├── final_project_2.ipynb   # EDA, preprocessing, model training and evaluation
├── best_model.pkl          # Final trained CatBoost model
├── app.py                  # Flask backend and REST API
├── index.html              # Web frontend
└── README.md               # Project documentation
```

## Dataset and Preprocessing

The dataset contains approximately 27–28 columns consisting of numerical and categorical customer information.

The following preprocessing steps were performed:

* Removed unnecessary identifier columns such as `ID`, `Customer_ID`, `Month`, `SSN`, and `Name`
* Converted `Age` to numeric values
* Handled missing values in numerical and categorical features
* Filled missing numerical values using appropriate statistical methods
* Filled missing categorical values using the mode
* Converted `Credit_History_Age` from text format into total months
* Applied regular expressions to extract years and months from `Credit_History_Age`
* Label-encoded categorical features
* Used consistent categorical mappings across the dataset
* Split the data into training and testing sets using a 75/25 ratio
* Used `random_state=42` for reproducibility

### Features Used

The final model uses the following 21 features:

```text
Age
Occupation
Annual_Income
Monthly_Inhand_Salary
Num_Bank_Accounts
Num_Credit_Card
Interest_Rate
Num_of_Loan
Type_of_Loan
Delay_from_due_date
Num_of_Delayed_Payment
Changed_Credit_Limit
Num_Credit_Inquiries
Credit_Mix
Outstanding_Debt
Credit_Utilization_Ratio
Credit_History_Age
Payment_of_Min_Amount
Total_EMI_per_month
Amount_invested_monthly
Payment_Behaviour
Monthly_Balance
```

## Machine Learning Models

Several classification algorithms were trained and evaluated to determine the best-performing model.

| Model                        | Training Accuracy | Test Accuracy | Configuration                       |
| ---------------------------- | :---------------: | :-----------: | ----------------------------------- |
| Decision Tree                |       75.45%      |     69.02%    | Entropy, max depth 40               |
| Random Forest                |       86.98%      |     75.85%    | 500 trees, max depth 20             |
| Random Forest + GridSearchCV |       86.98%      |     75.85%    | 5-fold cross-validation             |
| XGBoost                      |       83.12%      |     75.43%    | 1000 estimators, learning rate 0.03 |
| **CatBoost**                 |         —         |   **77.52%**  | 2000 iterations, depth 10           |

### Final Model

**CatBoost** was selected as the final model because it achieved the highest test accuracy among the evaluated models.


## Model Evaluation

The notebook includes several evaluation methods to compare model performance:

* Training accuracy
* Test accuracy
* Confusion matrix
* Classification report
* Precision
* Recall
* F1-score
* Model comparison

These evaluations provide a more comprehensive understanding of model performance beyond accuracy alone.

## Technology Stack

### Programming Language

* Python 3

### Machine Learning

* Scikit-learn
* CatBoost
* XGBoost

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Backend

* Flask
* Flask-CORS

### Frontend

* HTML
* CSS
* JavaScript


```

## Notebook Workflow

The `final_project_2.ipynb` notebook contains the complete machine learning workflow:

1. Import required libraries
2. Load the dataset
3. Perform exploratory data analysis
4. Analyze numerical and categorical features
5. Handle missing values
6. Clean and transform feature values
7. Convert `Credit_History_Age` into numerical months
8. Encode categorical variables
9. Split the dataset into training and testing sets
10. Train multiple classification models
11. Evaluate model performance
12. Generate confusion matrices and classification reports
13. Compare model accuracy
14. Select the best-performing model
15. Save the final CatBoost model as `best_model.pkl`

## Deployment Architecture

The application follows a simple machine learning deployment architecture:

```text
User
  |
  v
HTML / CSS / JavaScript Interface
  |
  v
Flask REST API
  |
  v
Data Preprocessing
  |
  v
CatBoost Model
  |
  v
Prediction + Class Probabilities
  |
  v
Web Interface
```

The frontend collects customer information and sends it to the Flask `/predict` endpoint. The backend processes the input, passes it to the trained CatBoost model, and returns the predicted credit score and probability distribution.

## Future Improvements

Several improvements can be made to make the system more robust and production-ready:

* Perform cross-validation for more reliable model benchmarking
* Improve handling of class imbalance
* Experiment with SMOTE and other resampling techniques
* Improve minority-class recall
* Add stronger API input validation
* Add comprehensive error handling
* Separate preprocessing from model inference into a reusable pipeline
* Containerize the application using Docker
* Add authentication and API rate limiting
* Deploy the application to a cloud platform
* Add model monitoring and logging
* Add automated testing for the API and prediction pipeline
