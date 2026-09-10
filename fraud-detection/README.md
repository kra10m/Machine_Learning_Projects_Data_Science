# Credit Card Fraud Detection System

An end-to-end machine learning system for detecting fraudulent credit card transactions under severe class imbalance.

The project covers exploratory data analysis, feature engineering, imbalance handling, model comparison, threshold optimization, business-cost evaluation, model interpretability, MLflow experiment tracking, model serialization, FastAPI deployment, API validation, and automated testing.

## Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent only a very small fraction of all transactions.

A model that predicts every transaction as legitimate can achieve extremely high accuracy while being ineffective at detecting fraud.

This project therefore focuses on precision-recall performance, probability thresholds, and financial costs associated with false positives and false negatives.

### Objective

Given the features of a credit card transaction, predict whether the transaction is:

- `0` → Not Fraud
- `1` → Fraud

The system also produces a fraud probability and applies a business-optimized decision threshold.

## Dataset

The project uses the Credit Card Fraud Detection dataset containing European card transactions.

Dataset characteristics:

- **284,807 transactions**
- **492 fraudulent transactions**
- Fraud rate: approximately **0.1727%**
- 30 original input features
- `Time`
- `V1`–`V28`
- `Amount`
- `Class` — target variable

The dataset is excluded from version control and should be placed at:

```text
data/creditcard.csv
```

## Project Architecture

```text
creditcard.csv
      |
      v
EDA & Data Analysis
      |
      v
Feature Engineering
      |
      +-- Time features
      +-- Cyclical hour encoding
      +-- Transaction volume
      |
      v
Preprocessing
      |
      +-- RobustScaler
      |
      v
Imbalance Handling
      |
      +-- SMOTE
      |
      v
LightGBM Classifier
      |
      v
Threshold Optimization
      |
      +-- Business cost
      |
      +-------------------+
      |                   |
      v                   v
SHAP                FastAPI
Interpretability        |
                        v
                  Fraud Prediction
```

## Machine Learning Workflow

### 1. Exploratory Data Analysis

The first stage investigates:

- Dataset dimensions and schema
- Missing values
- Class distribution
- Class imbalance
- Feature distributions
- Outliers
- `Amount` distribution
- `Time` distribution
- Correlations
- Baseline performance

The dataset has an extreme class imbalance:

```text
Fraud rate ≈ 0.1727%
```

Because of this imbalance, accuracy is not used as the primary model-selection metric.

### 2. Train/Test Split

The dataset is divided using a stratified split:

```text
80% Training
20% Test
```

with:

```text
random_state = 42
```

Stratification preserves the fraud/non-fraud ratio between training and test sets.

The test set remains untouched until final evaluation.

### 3. Feature Engineering

The following time-based features are created:

- `Time_Hours`
- `Hour`
- `Hour_Sin`
- `Hour_Cos`

A transaction-volume feature is also created:

```text
Transactions_Per_Hour
```

The hourly transaction statistics are learned from the training data.

Feature engineering is implemented in:

```text
src/features.py
```

### 4. Preprocessing

`Time` and `Amount` are scaled using:

```text
RobustScaler
```

The preprocessing transformer is implemented in:

```text
src/preprocessing.py
```

## Model Comparison

Several approaches were evaluated using stratified cross-validation.

### Logistic Regression

- Class-weighted Logistic Regression

### LightGBM

- Class weighting
- SMOTE
- Random undersampling

### XGBoost

- Class weighting
- SMOTE
- Random undersampling

Model selection focused primarily on **PR-AUC**.

The best-performing approach was:

```text
LightGBM + SMOTE
```

## Handling Class Imbalance

The project evaluates:

```text
Class Weighting
SMOTE
Random Undersampling
```

SMOTE was configured with:

```text
sampling_strategy = 0.10
```

The final selected model uses:

```text
LightGBM + SMOTE
```

## Evaluation Metrics

Because fraud represents only approximately 0.17% of transactions, several metrics are used.

### PR-AUC

The primary model-selection metric.

### ROC-AUC

Measures the model's ability to rank fraudulent transactions above legitimate transactions across thresholds.

### Precision

Of the transactions predicted as fraud, how many were actually fraudulent?

### Recall

Of the actual fraudulent transactions, how many were detected?

### F1 Score

The harmonic mean of precision and recall.

### Financial Cost

The project assigns different costs to false positives and false negatives.

```text
False Positive Cost = 1
False Negative Cost = 10
```

The cost function is:

```text
Total Cost = FP × 1 + FN × 10
```

## Threshold Optimization

The selected operating threshold is:

```text
0.239
```

Therefore:

```text
probability >= 0.239 → FRAUD
probability <  0.239 → NOT_FRAUD
```

The threshold was selected using out-of-fold training predictions and business-cost analysis.

The untouched test set was reserved for final evaluation.

## Final Model Performance

The final LightGBM + SMOTE model achieved the following results on the untouched test set:

| Metric | Result |
|---|---:|
| PR-AUC | **0.8744** |
| ROC-AUC | **0.9793** |
| Precision | **76.99%** |
| Recall | **88.78%** |
| F1 Score | **82.46%** |
| Threshold | **0.239** |
| Financial Cost | **136** |

### Confusion Matrix

| | Predicted Not Fraud | Predicted Fraud |
|---|---:|---:|
| Actual Not Fraud | 56,838 | 26 |
| Actual Fraud | 11 | 87 |

```text
True Positives  = 87
False Positives = 26
False Negatives = 11
True Negatives  = 56,838
```

## Model Interpretability

The project uses two levels of interpretability.

### Global Feature Importance

LightGBM feature importance is used to understand which features contribute most strongly to the model.

### SHAP

SHAP is used for:

- Global feature analysis
- Individual transaction explanations

## MLflow

MLflow is used for experiment tracking.

The project uses the experiment:

```text
credit-card-fraud-detection
```

The experiment notebook is:

```text
notebooks/05_mlflow_experiments.ipynb
```

## Production Prediction Pipeline

The trained model is saved as:

```text
models/fraud_detection_lgbm_smote.joblib
```

The saved artifact contains:

- Trained ML pipeline
- Learned hourly transaction statistics
- Decision threshold
- Model metadata
- Evaluation metrics

The prediction logic is implemented in:

```text
src/predict.py
```

## FastAPI

The trained model is exposed through a REST API using FastAPI.

### Start the API

From the project root:

```bash
python -m uvicorn api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### Health Check

```text
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

### Fraud Prediction

```text
POST /predict
```

Example response:

```json
{
  "fraud_probability": 0.000016850736747101812,
  "threshold": 0.239,
  "prediction": 0,
  "decision": "NOT_FRAUD"
}
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## API Input Validation

Pydantic is used to validate incoming API requests.

The API rejects:

- Missing required features
- Invalid numeric values
- Negative transaction amounts
- Incorrect data types

Invalid requests return:

```text
422 Unprocessable Content
```

before reaching the machine learning model.

## Automated Testing

The project includes automated tests using pytest.

Run:

```bash
python -m pytest
```

Current test suite:

```text
6 passed
```

Tests cover:

- FastAPI health endpoint
- Missing API fields
- Invalid transaction amounts
- Time feature engineering
- Transaction-volume feature engineering
- Preprocessing

## Project Structure

```text
fraud-detection/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   ├── 01_eda_and_baseline.ipynb
│   ├── 02_preprocessing_and_feature_engineering.ipynb
│   ├── 03_model_comparison_and_imbalance.ipynb
│   ├── 04_threshold_tuning_and_shap.ipynb
│   └── 05_mlflow_experiments.ipynb
│
├── src/
│   ├── __init__.py
│   ├── features.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── models/
│   └── fraud_detection_lgbm_smote.joblib
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_preprocessing.py
│
├── requirements.txt
└── README.md
```

## Installation

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Place the dataset at:

```text
data/creditcard.csv
```

To train the model:

```bash
python src/train.py
```

The trained artifact will be saved to:

```text
models/fraud_detection_lgbm_smote.joblib
```

## Running the Prediction Module

```bash
python -m src.predict
```

## Running the API

```bash
python -m uvicorn api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Key Engineering Decisions

### Why PR-AUC instead of accuracy?

Fraud represents only approximately 0.17% of transactions.

A model predicting every transaction as legitimate could achieve extremely high accuracy while detecting no fraud.

PR-AUC provides a more useful evaluation of performance on the rare positive class.

### Why SMOTE?

SMOTE increases representation of the minority class during training, allowing the model to learn fraud patterns without modifying the untouched test set.

### Why RobustScaler?

`Time` and `Amount` can contain extreme values. RobustScaler reduces the influence of those outliers.

### Why a threshold of 0.239?

A default probability threshold of `0.5` is not necessarily optimal for fraud detection.

The threshold was selected using out-of-fold predictions and a business-cost function where missing a fraudulent transaction is considered more costly than incorrectly flagging a legitimate transaction.

### Why LightGBM?

LightGBM + SMOTE provided the strongest PR-AUC among the evaluated candidate approaches.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- LightGBM
- XGBoost
- SHAP
- MLflow
- FastAPI
- Pydantic
- Pytest

## Reproducibility

The project uses:

```text
random_state = 42
```

for the primary train/test split and cross-validation procedures.

The production prediction artifact stores the trained pipeline, learned training statistics, threshold, and model metadata required for inference.

## Limitations

This project uses a public historical dataset and does not represent a production banking environment.

A real-world fraud detection system would require additional considerations such as:

- Real-time transaction streams
- Concept drift
- Data drift monitoring
- Model monitoring
- Automated retraining
- Feedback from fraud investigations
- Business-specific cost calibration
- Latency requirements
- Authentication and authorization
- Privacy and regulatory requirements
- Production observability

## Future Improvements

Potential extensions include:

- Real-time streaming inference
- Model and data drift detection
- Automated model retraining
- Probability calibration
- Production cloud deployment
- API authentication
- CI/CD pipeline
- Advanced monitoring and observability

## Summary

This project demonstrates an end-to-end approach to highly imbalanced machine learning classification.

The complete workflow is:

```text
EDA
→ Feature Engineering
→ Preprocessing
→ Imbalance Handling
→ Model Comparison
→ Cross-Validation
→ Threshold Optimization
→ Business Cost Evaluation
→ SHAP Interpretability
→ MLflow Tracking
→ Model Serialization
→ FastAPI
→ Pydantic Validation
→ Automated Testing
```

The final LightGBM + SMOTE model achieves:

- **0.8744 PR-AUC**
- **0.9793 ROC-AUC**
- **76.99% precision**
- **88.78% recall**
- **82.46% F1**
- **136 financial cost**
- **0.239 operating threshold**

on the untouched test set.
