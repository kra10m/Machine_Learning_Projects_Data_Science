import numpy as np
import pandas as pd


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features to transaction data.

    Features created:
    - Time_Hours: elapsed time expressed in hours
    - Hour: hour of day (0-23)
    - Hour_Sin: cyclical sine encoding of hour
    - Hour_Cos: cyclical cosine encoding of hour
    """

    data = data.copy()

    data["Time_Hours"] = data["Time"] / 3600.0
    data["Hour"] = ((data["Time"] // 3600) % 24).astype(int)

    data["Hour_Sin"] = np.sin(2 * np.pi * data["Hour"] / 24)
    data["Hour_Cos"] = np.cos(2 * np.pi * data["Hour"] / 24)

    return data

def add_training_volume_feature(
    train_data: pd.DataFrame,
    other_data: pd.DataFrame,
):
    """
    Add a transaction-volume feature using statistics learned
    from training data only.

    Returns:
    - transformed training data
    - transformed other data
    - hourly transaction counts learned from training data
    """

    train_data = train_data.copy()
    other_data = other_data.copy()

    train_data["Time_Hour_Bucket"] = (
        train_data["Time"] // 3600
    ).astype(int)

    other_data["Time_Hour_Bucket"] = (
        other_data["Time"] // 3600
    ).astype(int)

    hourly_counts = train_data["Time_Hour_Bucket"].value_counts()

    train_data["Transactions_Per_Hour"] = (
        train_data["Time_Hour_Bucket"]
        .map(hourly_counts)
        .astype(float)
    )

    other_data["Transactions_Per_Hour"] = (
        other_data["Time_Hour_Bucket"]
        .map(hourly_counts)
        .fillna(0)
        .astype(float)
    )

    train_data.drop(columns="Time_Hour_Bucket", inplace=True)
    other_data.drop(columns="Time_Hour_Bucket", inplace=True)

    return train_data, other_data, hourly_counts