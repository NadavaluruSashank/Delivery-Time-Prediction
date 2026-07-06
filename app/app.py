"""
Delivery Time Prediction REST API
--------------------------------------
Flask API exposing a /predict endpoint for real-time delivery time
prediction.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from flask import Flask, request, jsonify
from predict import predict_delivery_time

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Delivery Time Prediction API",
        "usage": "POST /predict with JSON: {distance_km, order_hour, vehicle_type, weather_condition, traffic_density}"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        required_fields = ["distance_km", "order_hour", "vehicle_type", "weather_condition", "traffic_density"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        predicted_minutes = predict_delivery_time(data)

        return jsonify({
            "predicted_delivery_time_minutes": predicted_minutes,
            "input": data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
