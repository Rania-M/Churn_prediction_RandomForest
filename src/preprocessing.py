# preprocessing.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

def preprocess_data(df):
    # Separate features and target
    X = df.drop(columns=['Churn', 'Churn_numeric'])
    y = df['Churn_numeric']

    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Encode categorical features (one-hot)
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    # Scale numeric features
    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])

    # Train/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

    # Second split: temp → validation + test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler  # return scaler for future use

if __name__ == "__main__":
    from data_ingestion import load_processed_data
    df = load_processed_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    print("Preprocessing done.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    joblib.dump(X_train.columns, "models/feature_columns.pkl")

