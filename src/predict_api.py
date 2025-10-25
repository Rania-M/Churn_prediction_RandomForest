# predict_api.py
import joblib
import pandas as pd

# Load trained model, scaler, and feature columns
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")  # saved X_train columns

# Example: new customer data (can be partial, only key info)
new_data = pd.DataFrame([
    {
        "tenure": 5,
        "MonthlyCharges": 80,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        # You can leave out other features; they will be filled with 0
    }
])

# Encode categorical features using get_dummies
new_data = pd.get_dummies(new_data, drop_first=True)

# Align columns with training features, fill missing with 0
new_data = new_data.reindex(columns=feature_columns, fill_value=0)

# Scale numeric features
numeric_features = new_data.select_dtypes(include=['int64', 'float64']).columns
new_data[numeric_features] = scaler.transform(new_data[numeric_features])

# Make prediction
prediction = model.predict(new_data)
print("Prediction (0=No Churn, 1=Churn):", prediction)
