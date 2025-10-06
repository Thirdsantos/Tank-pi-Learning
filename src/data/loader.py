"""Data loading functionality for aquarium logs."""

import pandas as pd
import numpy as np
from typing import Optional

from src.config import get_supabase_client, is_supabase_available


def get_aquarium_logs() -> pd.DataFrame:
    """Fetch aquarium logs from Supabase (or fallback to dummy data)."""
    if is_supabase_available():
        try:
            supabase = get_supabase_client()
            response = supabase.table("aquarium_logs").select("*").execute()
            if response.data:  # if not empty
                df = pd.DataFrame(response.data)
                df["recorder_at"] = pd.to_datetime(df["recorder_at"])
                df = df.sort_values(["tank_id", "recorder_at"])
                return df
        except Exception as e:
            print(f"Failed to connect to Supabase: {e}")
            print("Falling back to dummy dataset")

    # === Fake dataset for testing ===
    print("Using dummy dataset (Supabase not available)")
    return _generate_dummy_data()


def _generate_dummy_data() -> pd.DataFrame:
    """Generate dummy aquarium data for testing purposes."""
    rng = np.random.default_rng(42)
    dummy = {
        "tank_id": [1]*50 + [2]*50,
        "ph": rng.uniform(6.5, 8.0, 100),
        "temperature": rng.uniform(24, 30, 100),
        "turbidity": rng.uniform(0, 10, 100),
        "recorder_at": pd.date_range("2023-01-01", periods=100, freq="H"),
    }
    return pd.DataFrame(dummy)

