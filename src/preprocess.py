"""
Preprocessing and Feature Engineering
---------------------------------------
Prepares delivery order data for model training: encodes categorical
features and adds derived features.
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "delivery_orders.csv")


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def engineer_features(df):
    df = df.copy()

    # Cyclical hour encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["order_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["order_hour"] / 24)

    # Rush hour flag
    df["is_rush_hour"] = df["order_hour"].isin([7, 8, 11, 12, 13, 17, 18, 19, 20]).astype(int)

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=["vehicle_type", "weather_condition", "traffic_density"], drop_first=False)

    return df


def get_feature_columns(df):
    exclude = ["order_id", "delivery_time_minutes", "order_hour"]
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    df = load_data()
    df = engineer_features(df)
    print("Feature engineering complete.")
    print(f"Total features: {len(get_feature_columns(df))}")
    print(df.head())
