import joblib
import pandas as pd

model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# Minimal input
new_raw = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 80,
    "TotalCharges": 400
}])

# One-hot encode all categorical features
new_data = pd.get_dummies(new_raw, drop_first=False)

# Add missing columns (those in feature_columns but not in new_data) with 0
for col in feature_columns:
    if col not in new_data.columns:
        new_data[col] = 0

# Reorder columns to match training
new_data = new_data[feature_columns]

# Scale numeric columns
numeric_features = scaler.feature_names_in_  # numeric columns seen during training
new_data[numeric_features] = scaler.transform(new_data[numeric_features])

# Predict
prediction = model.predict(new_data)
print("Prediction (0=No Churn, 1=Churn):", prediction)

