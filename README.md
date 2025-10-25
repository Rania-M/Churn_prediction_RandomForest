# 📊 Customer Churn Prediction with Random Forest

## 🔍 Project Overview

This project aims to predict customer churn for a telecommunications company using a **Random Forest classifier**. The goal is to identify customers likely to leave, enabling proactive retention strategies.

## 📁 Dataset

The dataset is sourced from Kaggle:

- **Name**: Telco Customer Churn
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

It contains 7,043 customer records with 21 attributes, including:

- **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account Info**: `tenure`, `Contract`, `MonthlyCharges`, `TotalCharges`
- **Services**: `PhoneService`, `InternetService`, `OnlineSecurity`, etc.
- **Target Variable**: `Churn` (Yes/No)

> **Note**: The dataset is not included in this repository due to size constraints. Please download it from the provided Kaggle link and place it in the `data/raw` directory.

## 🛠️ Preprocessing Steps

- **Numeric Scaling**: Applied `StandardScaler` to numerical features.
- **Categorical Encoding**: Used one-hot encoding for categorical variables.
- **Class Imbalance**: Addressed using `class_weight='balanced'` in the Random Forest model.

## 🤖 Model

- **Algorithm**: Random Forest Classifier
- **Hyperparameters Tuned**: `n_estimators`, `max_depth`, `max_features`, `class_weight`
- **Metrics**: Accuracy, F1-Score, Precision, Recall

## 📈 Results

| Metric     | Score |
|------------|-------|
| Accuracy   | 0.79  |
| F1-Score   | 0.56  |

> These are baseline results. Future work includes implementing SMOTE for better handling of class imbalance.

## 🚀 Getting Started

1. Clone this repository:

   ```bash
   git clone https://github.com/Rania-M/Churn_prediction_RandomForest
