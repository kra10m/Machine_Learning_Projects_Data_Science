from pathlib import Path

import joblib
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.features import add_time_features, add_training_volume_feature
from src.preprocessing import make_preprocessor


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "fraud_detection_lgbm_smote.joblib"


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

SMOTE_SAMPLING_STRATEGY = 0.10

# Threshold selected during Stage 4
SELECTED_THRESHOLD = 0.239

# Financial cost assumptions from Stage 1/4
FP_COST = 1
FN_COST = 10

# Winning LightGBM configuration from Stage 3
LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}


# ============================================================
# Load data
# ============================================================

def load_data():
    """Load the credit card fraud dataset."""

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns="Class")
    y = df["Class"]

    return X, y


# ============================================================
# Build model pipeline
# ============================================================

def build_pipeline():
    """
    Build the production training pipeline.

    Order:
        preprocessing
        -> SMOTE
        -> LightGBM
    """

    preprocessor = make_preprocessor()

    smote = SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        random_state=RANDOM_STATE,
    )

    model = LGBMClassifier(**LGBM_PARAMS)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", smote),
            ("model", model),
        ]
    )

    return pipeline


# ============================================================
# Training
# ============================================================

def train_model():
    """Train the final LightGBM + SMOTE model."""

    print("Loading data...")

    X, y = load_data()

    print(f"Dataset shape: {X.shape}")
    print(f"Fraud transactions: {int(y.sum())}")
    print(f"Fraud rate: {y.mean() * 100:.4f}%")

    # --------------------------------------------------------
    # Stratified train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print()
    print("Train/test split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    print()
    print("Applying feature engineering...")

    X_train = add_time_features(X_train)
    X_test = add_time_features(X_test)

    X_train, X_test, hourly_counts = add_training_volume_feature(
        X_train,
        X_test,
    )

    print(f"Feature count after engineering: {X_train.shape[1]}")
    print(f"Hourly buckets learned: {len(hourly_counts)}")

    # --------------------------------------------------------
    # Build and train pipeline
    # --------------------------------------------------------

    pipeline = build_pipeline()

    print()
    print("Training LightGBM + SMOTE...")

    pipeline.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    test_probabilities = pipeline.predict_proba(X_test)[:, 1]

    test_predictions = (
        test_probabilities >= SELECTED_THRESHOLD
    ).astype(int)

    pr_auc = average_precision_score(
        y_test,
        test_probabilities,
    )

    roc_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        test_predictions,
    ).ravel()

    financial_cost = (
        fp * FP_COST
        + fn * FN_COST
    )

    print()
    print("=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"PR-AUC:          {pr_auc:.4f}")
    print(f"ROC-AUC:         {roc_auc:.4f}")
    print(f"Threshold:       {SELECTED_THRESHOLD:.3f}")
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Negatives:  {tn}")
    print(f"Financial cost:  {financial_cost:.0f}")

    print()
    print("Classification report:")

    print(
        classification_report(
            y_test,
            test_predictions,
            digits=4,
        )
    )

    # --------------------------------------------------------
    # Save production artifact
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "pipeline": pipeline,

        # Training-derived feature statistics
        "hourly_counts": hourly_counts.to_dict(),

        # Decision threshold
        "threshold": SELECTED_THRESHOLD,

        # Model metadata
        "model_name": "LightGBM_SMOTE",
        "random_state": RANDOM_STATE,
        "smote_sampling_strategy": SMOTE_SAMPLING_STRATEGY,

        # Feature metadata
        "feature_count": X_train.shape[1],

        # Cost assumptions
        "fp_cost": FP_COST,
        "fn_cost": FN_COST,

        # Evaluation metrics
        "test_pr_auc": pr_auc,
        "test_roc_auc": roc_auc,
        "test_precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "test_recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "test_f1": (
            2 * tp / (2 * tp + fp + fn)
            if (2 * tp + fp + fn) > 0
            else 0.0
        ),
        "test_financial_cost": financial_cost,
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
    )

    print()
    print(f"Model saved to: {MODEL_PATH}")

    return pipeline, artifact


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    train_model()