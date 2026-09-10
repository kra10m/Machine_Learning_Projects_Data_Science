import numpy as np
import pandas as pd

from src.features import (
    add_time_features,
    add_training_volume_feature,
)
from src.preprocessing import make_preprocessor


def test_time_features():
    data = pd.DataFrame(
        {
            "Time": [0.0, 3600.0, 7200.0],
            "Amount": [10.0, 20.0, 30.0],
        }
    )

    result = add_time_features(data)

    assert "Time_Hours" in result.columns
    assert "Hour" in result.columns
    assert "Hour_Sin" in result.columns
    assert "Hour_Cos" in result.columns

    assert result.loc[0, "Hour"] == 0
    assert result.loc[1, "Hour"] == 1
    assert result.loc[2, "Hour"] == 2

    assert np.isclose(result.loc[0, "Hour_Sin"], 0.0)
    assert np.isclose(result.loc[0, "Hour_Cos"], 1.0)


def test_training_volume_feature():
    train = pd.DataFrame(
        {
            "Time": [0.0, 10.0, 3600.0],
            "Amount": [10.0, 20.0, 30.0],
        }
    )

    other = pd.DataFrame(
        {
            "Time": [20.0, 3600.0],
            "Amount": [40.0, 50.0],
        }
    )

    train_result, other_result, hourly_counts = (
        add_training_volume_feature(train, other)
    )

    assert "Transactions_Per_Hour" in train_result.columns
    assert "Transactions_Per_Hour" in other_result.columns

    assert hourly_counts[0] == 2
    assert hourly_counts[1] == 1

    assert train_result.loc[0, "Transactions_Per_Hour"] == 2
    assert other_result.loc[0, "Transactions_Per_Hour"] == 2


def test_preprocessor():
    data = pd.DataFrame(
        {
            "Time": [0.0, 100.0, 200.0],
            "Amount": [10.0, 20.0, 30.0],
            "V1": [1.0, 2.0, 3.0],
        }
    )

    preprocessor = make_preprocessor()

    transformed = preprocessor.fit_transform(data)

    assert transformed.shape == data.shape
    assert np.isfinite(transformed).all()