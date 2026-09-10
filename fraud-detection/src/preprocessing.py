from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler


SCALED_FEATURES = ["Time", "Amount"]


def make_preprocessor():
    """
    Create the preprocessing transformer for the fraud detection model.

    Time and Amount are scaled using RobustScaler.
    All other features are passed through unchanged.
    """

    return ColumnTransformer(
        transformers=[
            (
                "robust_scaler",
                RobustScaler(),
                SCALED_FEATURES,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )