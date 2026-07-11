# Telco Customer Churn Prediction Pipeline

## Internship Task

DevelopersHub Corporation  
AI/ML Engineering Advanced Internship  
Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API

## Objective

The objective of this project is to build a reusable and production-ready machine learning pipeline for predicting telecom customer churn.

## Dataset

IBM Telco Customer Churn dataset in Excel format.

Target:

- `Churn Value = 0`: Customer stayed
- `Churn Value = 1`: Customer churned

## Features

The model uses customer service, billing, contract, and subscription-related attributes.

Leakage and unnecessary columns such as `Churn Score`, `Churn Reason`, `Churn Label`, and `CLTV` were removed.

## Preprocessing

- Median imputation for numerical missing values
- Standard scaling for numerical columns
- Most-frequent imputation for categorical columns
- One-hot encoding for categorical variables
- Scikit-learn `ColumnTransformer`
- Complete reusable `Pipeline`

## Models

- Logistic Regression
- Random Forest
- Tuned Random Forest

## Hyperparameter Tuning

`GridSearchCV` was used with five-fold cross-validation.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

## Results

The model evaluation results are available in the notebook and screenshots.

## Project Structure

```text
task2/
├── app.py
├── churn_pipeline.joblib
├── model_info.json
├── Task_2_Customer_Churn_Pipeline.ipynb
├── requirements.txt
├── README.md
└── .gitignore