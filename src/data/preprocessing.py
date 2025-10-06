"""Data preprocessing functionality for aquarium data."""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Prepare features (X) and targets (y) from aquarium data.
    
    Args:
        df: Raw aquarium data DataFrame
        
    Returns:
        Tuple of (X_scaled, y, scaler) where:
        - X_scaled: Scaled feature DataFrame
        - y: Target DataFrame (next hour predictions)
        - scaler: Fitted StandardScaler object
    """
    # Select relevant columns
    df = df[["tank_id", "ph", "temperature", "turbidity", "recorder_at"]]

    # Shift data per tank → predict next values
    df["ph_next"] = df.groupby("tank_id")["ph"].shift(-1)
    df["temp_next"] = df.groupby("tank_id")["temperature"].shift(-1)
    df["turb_next"] = df.groupby("tank_id")["turbidity"].shift(-1)

    # Remove rows with NaN values (last row of each tank)
    df = df.dropna()

    # Features
    X = df[["tank_id", "ph", "temperature", "turbidity"]]

    # Targets
    y = df[["ph_next", "temp_next", "turb_next"]]

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[["ph", "temperature", "turbidity"]] = scaler.fit_transform(
        X[["ph", "temperature", "turbidity"]]
    )

    return X_scaled, y, scaler
