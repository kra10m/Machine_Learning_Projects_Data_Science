from pathlib import Path

import joblib
import pandas as pd

from src.features import add_time_features


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "fraud_detection_lgbm_smote.joblib"
)


# ============================================================
# Model loading
# ============================================================

def load_model():
    """Load the trained model artifact."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# Feature preparation
# ============================================================

def prepare_transaction(
    transaction: pd.DataFrame,
    hourly_counts: dict,
) -> pd.DataFrame:
    """
    Apply the same feature engineering used during training.

    Parameters
    ----------
    transaction : pd.DataFrame
        Transaction data containing the original 30 input features.

    hourly_counts : dict
        Training-derived hourly transaction counts stored
        inside the model artifact.

    Returns
    -------
    pd.DataFrame
        Transaction with the 35 model features.
    """

    transaction = transaction.copy()

    # Deterministic time features
    transaction = add_time_features(transaction)

    # Recreate the hourly bucket used during training
    transaction["Time_Hour_Bucket"] = (
        transaction["Time"] // 3600
    ).astype(int)

    # Apply training-derived transaction volume mapping
    transaction["Transactions_Per_Hour"] = (
        transaction["Time_Hour_Bucket"]
        .map(hourly_counts)
        .fillna(0)
        .astype(float)
    )

    # Bucket is only an intermediate calculation
    transaction.drop(
        columns="Time_Hour_Bucket",
        inplace=True,
    )

    return transaction


# ============================================================
# Prediction
# ============================================================

def predict_transaction(
    transaction: pd.DataFrame,
) -> dict:
    """
    Predict whether a transaction is fraudulent.

    Returns
    -------
    dict
        Fraud probability, threshold, and binary prediction.
    """

    artifact = load_model()

    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]
    hourly_counts = artifact["hourly_counts"]

    prepared_transaction = prepare_transaction(
        transaction,
        hourly_counts,
    )

    probability = pipeline.predict_proba(
        prepared_transaction
    )[:, 1]

    prediction = int(
        probability[0] >= threshold
    )

    return {
        "fraud_probability": float(probability[0]),
        "prediction": prediction,
        "threshold": float(threshold),
    }


# ============================================================
# Manual test
# ============================================================

if __name__ == "__main__":

    data_path = PROJECT_ROOT / "data" / "creditcard.csv"

    df = pd.read_csv(data_path)

    # Use one transaction as an inference example.
    transaction = (
        df
        .drop(columns="Class")
        .iloc[[0]]
    )

    result = predict_transaction(transaction)

    print("Prediction result:")
    print(
        f"Fraud probability: "
        f"{result['fraud_probability']:.6f}"
    )
    print(
        f"Threshold:         "
        f"{result['threshold']:.3f}"
    )
    print(
        f"Prediction:        "
        f"{result['prediction']}"
    )

    if result["prediction"] == 1:
        print("Decision: FRAUD")
    else:
        print("Decision: NOT FRAUD")