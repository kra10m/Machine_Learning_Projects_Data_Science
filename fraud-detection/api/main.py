from fastapi import FastAPI

from api.schemas import (
    PredictionResponse,
    TransactionRequest,
)

from src.predict import predict_transaction


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Production API for predicting whether a credit card "
        "transaction is fraudulent."
    ),
    version="1.0.0",
)


# ============================================================
# Health check
# ============================================================

@app.get("/health")
def health_check():
    """Check whether the API is running."""

    return {
        "status": "healthy"
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: TransactionRequest):
    """Predict whether a credit card transaction is fraudulent."""

    transaction = request.model_dump()

    # Convert the validated request into a DataFrame
    import pandas as pd

    transaction_df = pd.DataFrame(
        [transaction]
    )

    result = predict_transaction(
        transaction_df
    )

    decision = (
        "FRAUD"
        if result["prediction"] == 1
        else "NOT_FRAUD"
    )

    return {
        "fraud_probability": result["fraud_probability"],
        "threshold": result["threshold"],
        "prediction": result["prediction"],
        "decision": decision,
    }