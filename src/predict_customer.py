import joblib
import pandas as pd

# Load the trained model
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
model_path = BASE_DIR / "models" / "final_churn_model.pkl"

model = joblib.load(model_path)

# Example new customer
new_customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 425.0
}])

# Predict churn class
prediction = model.predict(new_customer)

# Predict churn probability
probability = model.predict_proba(new_customer)[:, 1]

print("Prediction:", prediction[0])
print("Churn probability:", round(probability[0], 3))

if prediction[0] == 1:
    print("Result: This customer is likely to churn.")
else:
    print("Result: This customer is not likely to churn.")