# train_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from preprocessing import preprocess_data
from data_ingestion import load_processed_data

# Load data
df = load_processed_data()

# Preprocess
X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data(df)

# Train model
model = RandomForestClassifier(n_estimators=150,max_features="log2",class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)

# Save model and scaler
joblib.dump(model, "models/random_forest_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Model and scaler saved.")
