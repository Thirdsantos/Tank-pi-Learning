"""Main script for training and running the aquarium prediction model."""
import requests
import json
from sklearn.model_selection import train_test_split
from src.data import get_aquarium_logs, preprocess_data
from src.models import (
    train_model, 
    evaluate_model, 
    save_model_artifacts, 
    make_predictions
)
from flask import Flask, jsonify, json

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello, this is the index!"


@app.route("/ml_predictions", methods=["GET"])
def main():
    """Main function to orchestrate the aquarium prediction pipeline."""
    print("Aquarium Learning - Prediction Pipeline")
    print("=" * 50)
    
    # Step 1: Load data
    print("Loading aquarium data...")
    df = get_aquarium_logs()
    print(f"Loaded {len(df)} records from {df['tank_id'].nunique()} tanks")
    
    # Step 2: Preprocess data
    print("\nPreprocessing data...")
    X, y, scaler = preprocess_data(df)
    print(f"Prepared {len(X)} samples for training")
    
    # Step 3: Split data
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Step 4: Train model
    print("\nTraining model...")
    model = train_model(X_train, y_train)
    print("Model training completed!")
    
    # Step 5: Evaluate model
    print("\nEvaluating model...")
    score = evaluate_model(model, X_test, y_test)
    print(f"Model R² score: {score:.3f}")
    
    # Step 6: Save model artifacts
    print("\nSaving model and scaler...")
    save_model_artifacts(model, scaler)
    
    # Step 7: Make predictions
    print("\nMaking predictions for next hour...")
    predictions = make_predictions(model, X)

    # Convert predictions DataFrame to list of dicts
    predictions_list = predictions.to_dict(orient="records")

    # Step 8: Send to backend
    print("\nSending predictions to AquaCare backend...")
    url = "https://aquacare-5cyr.onrender.com/ml"
    print(predictions_list)

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(predictions_list)
        )

        if response.status_code == 200:
            print("Predictions successfully sent to backend!")
            result = response.json()
            return jsonify({"Message" : result})
        else:
            print(f"Failed to send data. Status code: {response.status_code}")
            print("Response:", response.text)
    except Exception as e:
        print(f"Error sending data: {e}")
        return jsonify({"message" : e})

    
    print("\n=== Next Hour Predictions Per Tank ===")
    for _, row in predictions.iterrows():
        print(
            f"Tank {int(row['tank_id'])} → "
            f"pH: {row['predicted_ph']:.2f}, "
            f"Temp: {row['predicted_temperature']:.2f}°C, "
            f"Turbidity: {row['predicted_turbidity']:.2f}"
        )

    print("\nPipeline completed successfully!")

  

if __name__ == "__main__":
  app.run(host="0.0.0.0")
