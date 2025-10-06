import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import data_loader

def test_pipeline():
    df = data_loader.get_aquarium_logs()
    assert df is not None and not df.empty

    X, y, scaler = data_loader.preprocess_data(df)
    assert X.shape[0] == y.shape[0]

    model = data_loader.train_model(X, y)
    assert model is not None

    # Predict the next hour for each tank
    print("\n=== NEXT HOUR PREDICTIONS PER TANK ===")
    for tank_id, group in X.groupby("tank_id"):
        last_row = group.tail(1)  # last reading of this tank
        prediction = model.predict(last_row)[0]

        print(
            f"Tank {tank_id} → "
            f"pH: {prediction[0]:.2f}, "
            f"Temp: {prediction[1]:.2f}, "
            f"Turbidity: {prediction[2]:.2f}"
        )

    # Extra assertion to make sure prediction works
    for tank_id, group in X.groupby("tank_id"):
        last_row = group.tail(1)
        prediction = model.predict(last_row)
        assert prediction.shape == (1, 3)
