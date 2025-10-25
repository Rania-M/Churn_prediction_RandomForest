import pandas as pd
import os

def load_processed_data(filename="data/processed/cleaned_telco_churn.csv"):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found. Make sure the processed CSV exists.")
    
    df = pd.read_csv(filename)
    
    expected_columns = ['tenure', 'MonthlyCharges', 'Contract', 'PaymentMethod', 'Churn_numeric']
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in processed data: {missing}")
    
    return df

if __name__ == "__main__":
    df = load_processed_data()
    print(df.head())
