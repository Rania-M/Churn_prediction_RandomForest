# evaluate.py
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from preprocessing import preprocess_data
from data_ingestion import load_processed_data

# Load processed data
df = load_processed_data()
X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data(df)

# Load trained model
model = joblib.load("models/random_forest_model.pkl")

# Make predictions
y_pred = model.predict(X_val)

# Metrics
acc = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
print(f"Validation Accuracy: {acc:.4f}")
print(f"Validation F1 Score: {f1:.4f}")
print("Validation Classification Report:\n", classification_report(y_val, y_pred))

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
print("Validation Confusion Matrix:\n", cm)
