# Customer Churn Prediction

## Project Overview

This project predicts customer churn using machine learning.

Customer churn occurs when a customer stops using a company’s services. Predicting churn can help a business identify at-risk customers early and take retention actions before they leave.

The project uses the Telco Customer Churn dataset and follows a full machine learning workflow, including data understanding, exploratory data analysis, preprocessing, model training, cross-validation, final evaluation, threshold tuning, and business recommendations.

## Business Problem

Customer churn is costly for telecom companies because acquiring new customers is often more expensive than retaining existing ones.

The goal of this project is to build a classification model that predicts whether a customer is likely to churn based on customer demographics, account information, subscribed services, contract type, payment method, tenure, and charges.

## Dataset

Dataset: Telco Customer Churn  
Source: Kaggle  
Target variable: `Churn`

Each row represents one customer. The target variable indicates whether the customer churned or not.

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Joblib
- Git and GitHub

## Project Workflow

1. Data Understanding
2. Exploratory Data Analysis
3. Data Cleaning
4. Preprocessing
5. Baseline Modeling
6. Advanced Model Comparison
7. Cross-Validation
8. Final Model Evaluation
9. Threshold Tuning
10. Feature Importance
11. Business Recommendations

## Notebooks

- [01_data_understanding.ipynb](notebooks/01_data_understanding.ipynb): data loading, data understanding, cleaning checks, and exploratory data analysis.
- [02_preprocessing_modeling.ipynb](notebooks/02_preprocessing_modeling.ipynb): preprocessing, model training, cross-validation, final evaluation, threshold tuning, feature importance, and business recommendations.

## Models Tested

- Dummy Classifier
- Logistic Regression
- Logistic Regression with class weights
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

## Final Model

The selected final model is:

**Logistic Regression with `class_weight="balanced"`**
## Model Results

The final model was evaluated on the test set using a threshold of 0.50.

| Metric | Score |
|---|---:|
| Accuracy | [0.738] |
| Precision | [0.504] |
| Recall | [0.783] |
| F1-score | [0.613] |
| ROC-AUC | [0.841] |

The model achieved strong recall for the churn class, meaning it was able to identify a large proportion of customers who actually churned.

This model was selected because it achieved strong recall for the churn class during cross-validation.

## Key Business Insights

The analysis showed that churn is associated with several factors:

- Month-to-month contracts
- Lower tenure
- Higher monthly charges
- Lack of tech support
- Lack of online security
- Certain payment methods
- Internet service type

## Business Recommendations

- Target month-to-month customers with retention offers.
- Create onboarding campaigns for new customers with low tenure.
- Promote tech support and online security services.
- Monitor customers with high monthly charges.
- Use predicted churn probabilities to prioritize retention campaigns.

## Project Limitations

- The dataset is relatively small and may not represent all telecom customers.
- The model is trained on historical data.
- The project does not include exact financial costs for false positives and false negatives.
- Additional customer behavior data could improve prediction quality.

## Future Improvements

- Hyperparameter tuning with RandomizedSearchCV or GridSearchCV
- SHAP explainability
- Cost-sensitive threshold optimization
- Streamlit dashboard
- Model monitoring