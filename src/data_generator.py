"""
Delivery Order Data Simulator
--------------------------------
Generates realistic delivery order data with features that influence
delivery time: distance, traffic, weather, vehicle type, and time of day.
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "delivery_orders.csv")


def generate_delivery_data(n_orders=5000, seed=42):
    np.random.seed(seed)

    vehicle_types = ["bike", "scooter", "car", "van"]
    weather_conditions = ["clear", "rain", "fog", "storm"]
    traffic_levels = ["low", "medium", "high"]

    records = []
    for i in range(n_orders):
        distance_km = round(np.random.exponential(scale=6) + 0.5, 2)
        distance_km = min(distance_km, 40)

        order_hour = np.random.choice(range(24), p=_hour_weights())
        vehicle = np.random.choice(vehicle_types, p=[0.35, 0.30, 0.25, 0.10])
        weather = np.random.choice(weather_conditions, p=[0.60, 0.25, 0.10, 0.05])
        traffic = np.random.choice(traffic_levels, p=_traffic_weights(order_hour))

        # Base speed by vehicle type (km/h)
        base_speed = {"bike": 18, "scooter": 28, "car": 35, "van": 30}[vehicle]

        # Traffic slowdown
        traffic_factor = {"low": 1.0, "medium": 0.75, "high": 0.5}[traffic]

        # Weather slowdown
        weather_factor = {"clear": 1.0, "rain": 0.85, "fog": 0.80, "storm": 0.65}[weather]

        effective_speed = base_speed * traffic_factor * weather_factor
        travel_time_min = (distance_km / effective_speed) * 60

        # Add fixed prep/handoff time + noise
        prep_time = np.random.normal(8, 2)
        noise = np.random.normal(0, 3)

        delivery_time_min = max(5, travel_time_min + prep_time + noise)

        records.append({
            "order_id": f"ORD{i+1:05d}",
            "distance_km": distance_km,
            "order_hour": order_hour,
            "vehicle_type": vehicle,
            "weather_condition": weather,
            "traffic_density": traffic,
            "delivery_time_minutes": round(delivery_time_min, 1)
        })

    return pd.DataFrame(records)


def _hour_weights():
    """Higher probability of orders during lunch and dinner rush."""
    weights = np.ones(24)
    weights[11:14] = 3.0   # lunch rush
    weights[18:21] = 3.5   # dinner rush
    weights[0:6] = 0.2     # late night low volume
    return weights / weights.sum()


def _traffic_weights(hour):
    """Traffic density probabilities depend on hour of day."""
    if hour in [7, 8, 17, 18, 19]:
        return [0.1, 0.3, 0.6]   # rush hour -> more high traffic
    elif hour in [0, 1, 2, 3, 4, 5]:
        return [0.8, 0.15, 0.05]  # night -> mostly low traffic
    else:
        return [0.4, 0.4, 0.2]


if __name__ == "__main__":
    df = generate_delivery_data()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df)} delivery orders")
    print(df.head(10))
    print(f"\nDelivery time stats:\n{df['delivery_time_minutes'].describe()}")
