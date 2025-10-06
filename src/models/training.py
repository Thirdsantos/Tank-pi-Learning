"""Model training functionality for aquarium prediction."""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Any


def train_model(X: pd.DataFrame, y: pd.DataFrame) -> RandomForestRegressor:
    """
    Train a Random Forest model for aquarium parameter prediction.
    
    Args:
        X: Feature DataFrame
        y: Target DataFrame
        
    Returns:
        Trained RandomForestRegressor model
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
    """
    Evaluate the trained model and return R² score.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        R² score
    """
    return model.score(X_test, y_test)


def save_model_artifacts(model: RandomForestRegressor, scaler: StandardScaler, 
                        model_path: str = "aquarium_predictor.pkl", 
                        scaler_path: str = "scaler.pkl") -> None:
    """
    Save trained model and scaler to disk.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        model_path: Path to save the model
        scaler_path: Path to save the scaler
    """
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


def load_model_artifacts(model_path: str = "aquarium_predictor.pkl", 
                        scaler_path: str = "scaler.pkl") -> Tuple[RandomForestRegressor, StandardScaler]:
    """
    Load trained model and scaler from disk.
    
    Args:
        model_path: Path to the saved model
        scaler_path: Path to the saved scaler
        
    Returns:
        Tuple of (model, scaler)
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def make_predictions(model: RandomForestRegressor, X: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions for aquarium parameters.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        
    Returns:
        DataFrame with predictions for each tank
    """
    predictions = []
    
    for tank_id, group in X.groupby("tank_id"):
        # Get the last available reading for this tank
        last_row = group.tail(1)
        prediction = model.predict(last_row)[0]
        
        predictions.append({
            'tank_id': tank_id,
            'predicted_ph': prediction[0],
            'predicted_temperature': prediction[1],
            'predicted_turbidity': prediction[2]
        })
    
    return pd.DataFrame(predictions)
