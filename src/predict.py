"""
Inference Logic
------------------
Loads the trained model and makes delivery time predictions on new orders.
"""

import os
from functools import lru_cache

import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "xgb_delivery_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "..", "models", "feature_columns.pkl")


VALID_VALUES = {
    "vehicle_type": {"bike", "scooter", "car", "van"},
    "weather_condition": {"clear", "rain", "fog", "storm"},
    "traffic_density": {"low", "medium", "high"},
}


@lru_cache(maxsize=1)
def load_model():
    """Load the model once and reuse it for every prediction."""
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return model, feature_cols


def prepare_input(order, feature_cols):
    """
    Convert a raw order dict into a feature row matching training format.

    order: dict with keys distance_km, order_hour, vehicle_type,
           weather_condition, traffic_density
    """
    df = pd.DataFrame([order])

    df["hour_sin"] = np.sin(2 * np.pi * df["order_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["order_hour"] / 24)
    df["is_rush_hour"] = df["order_hour"].isin([7, 8, 11, 12, 13, 17, 18, 19, 20]).astype(int)

    df = pd.get_dummies(df, columns=["vehicle_type", "weather_condition", "traffic_density"])

    # Align with training features (fill missing one-hot columns with 0)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]

    return df


def validate_order(order):
    """Raise ValueError if the order has an out-of-range or unknown value."""
    for field, allowed in VALID_VALUES.items():
        if order.get(field) not in allowed:
            raise ValueError(f"{field} must be one of {sorted(allowed)}")
    hour = order.get("order_hour")
    if not isinstance(hour, int) or not 0 <= hour <= 23:
        raise ValueError("order_hour must be an integer between 0 and 23")
    distance = order.get("distance_km")
    if not isinstance(distance, (int, float)) or distance <= 0:
        raise ValueError("distance_km must be a positive number")


def predict_delivery_time(order):
    validate_order(order)
    model, feature_cols = load_model()
    X = prepare_input(order, feature_cols)
    prediction = model.predict(X)[0]
    return round(float(prediction), 1)


if __name__ == "__main__":
    sample_order = {
        "distance_km": 8.5,
        "order_hour": 18,
        "vehicle_type": "bike",
        "weather_condition": "rain",
        "traffic_density": "high"
    }

    predicted_time = predict_delivery_time(sample_order)
    print(f"Sample order: {sample_order}")
    print(f"Predicted delivery time: {predicted_time} minutes")
