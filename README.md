## Aqua Learning — Aquarium Prediction Pipeline

A compact machine learning pipeline that predicts the next-hour water quality metrics for aquariums per tank:
- pH
- Temperature (°C)
- Turbidity

It uses a Random Forest multi-output regressor trained on historical logs. Data can be pulled from Supabase if credentials are provided, or a built-in dummy dataset is used for quick testing.

### Key Features
- **End-to-end pipeline**: load → preprocess → split → train → evaluate → save → predict
- **Multi-output regression**: predicts pH, temperature, and turbidity together
- **Per-tank next-step prediction**: targets are the next-hour values, computed via group-wise shift
- **Artifacts**: persists trained model and scaler (`aquarium_predictor.pkl`, `scaler.pkl`)

---

## Project Structure

- `main.py`: Orchestrates the full pipeline (training, evaluation, saving artifacts, predictions)
- `src/data/loader.py`: Loads aquarium logs from Supabase or generates a dummy dataset
- `src/data/preprocessing.py`: Builds features and next-hour targets; scales numeric features
- `src/models/training.py`: Training, evaluation, artifact save/load, and batch predictions
- `src/config/database.py`: Supabase client configuration and availability checks
- `src/data/__init__.py`: Exposes `get_aquarium_logs` and `preprocess_data`
- `src/models/__init__.py`: Exposes training/evaluation/artifact/prediction functions
- `data/data_loader.py`: Self-contained legacy/utility script mirroring the pipeline for quick runs
- `test/first_test.py`: Sanity test of the pipeline using the utility loader
- `aquarium_predictor.pkl`: Saved model (created after running the pipeline)
- `scaler.pkl`: Saved `StandardScaler` (created after running the pipeline)

---

## Requirements

- Python 3.10+
- See `requirements.txt` for pinned packages (scikit-learn, pandas, numpy, supabase, python-dotenv, etc.)

### Quick setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Environment Configuration (optional)

If you have a Supabase table `aquarium_logs`, set these in a `.env` file at the project root:

```dotenv
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_or_service_role_key
```

- When present, data is fetched from Supabase and sorted by `tank_id` and `recorder_at`.
- If missing or connection fails, the pipeline automatically falls back to a dummy dataset.

---

## Expected Data Schema

The pipeline expects a tabular format with at least:
- `tank_id` (int)
- `ph` (float)
- `temperature` (float, °C)
- `turbidity` (float)
- `recorder_at` (datetime)

During preprocessing, next-hour targets are created per tank using group-wise shifts:
- `ph_next`, `temp_next`, `turb_next`

Features used for training:
- `tank_id` (left unscaled)
- `ph`, `temperature`, `turbidity` (scaled via `StandardScaler`)

---

## How It Works

1. **Load data**
   - `src/data/loader.py#get_aquarium_logs()` reads from Supabase (if configured) or generates dummy data.
2. **Preprocess**
   - `src/data/preprocessing.py#preprocess_data()` creates next-hour targets per tank and scales numeric features.
3. **Split**
   - `train_test_split` with a fixed `random_state=42`.
4. **Train**
   - `src/models/training.py#train_model()` builds a `RandomForestRegressor(n_estimators=100, random_state=42)` for multi-output regression.
5. **Evaluate**
   - `src/models/training.py#evaluate_model()` reports R² on the test split.
6. **Save Artifacts**
   - `src/models/training.py#save_model_artifacts()` writes `aquarium_predictor.pkl` and `scaler.pkl` to the project root.
7. **Predict**
   - `src/models/training.py#make_predictions()` predicts the next hour for each tank using the last available row per tank.

---

## Run the Full Pipeline

From the project root:
```powershell
python main.py
```
You will see:
- Dataset summary (count of records and tanks)
- Train/test split sizes
- Model training completion
- R² score
- A per-tank summary of next-hour predictions

Artifacts (`aquarium_predictor.pkl`, `scaler.pkl`) will be saved in the project root.

---

## Programmatic Usage

Use the modules directly if you want to embed the pipeline in another script:

```python
from src.data import get_aquarium_logs, preprocess_data
from src.models import train_model, evaluate_model, save_model_artifacts, load_model_artifacts, make_predictions
from sklearn.model_selection import train_test_split

# Load and preprocess
df = get_aquarium_logs()
X, y, scaler = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate
model = train_model(X_train, y_train)
print("R2:", evaluate_model(model, X_test, y_test))

# Save and later load
save_model_artifacts(model, scaler)
model, scaler = load_model_artifacts()

# Predict next-hour per tank using latest rows
pred_df = make_predictions(model, X)
print(pred_df.head())
```

---

## Tests

A simple test ensures end-to-end functionality using the utility loader under `data/`:

```powershell
pip install pytest
python -m pytest -q
```

The test prints sample per-tank predictions and asserts that the model returns a 1×3 array for each tank’s next-hour prediction.

---

## Notes & Assumptions

- The model predicts one step ahead (next hour) only; it does not model longer horizons.
- `tank_id` is included as a feature and kept unscaled; `ph`, `temperature`, and `turbidity` are scaled.
- `make_predictions()` relies on the last available row per tank from the (scaled) features `X`.
- If you change preprocessing, ensure you apply the same transformations and column ordering before calling `model.predict`.

---


