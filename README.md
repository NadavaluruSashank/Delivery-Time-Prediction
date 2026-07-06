# Delivery Time Prediction System

An end-to-end machine learning application that predicts delivery time for logistics orders using order and route features. Built to improve planning accuracy and reduce late deliveries through real-time predictive inference, exposed as a REST API.

---

## Problem Statement

Logistics companies struggle with inaccurate delivery time estimates, leading to poor planning, customer dissatisfaction, and operational inefficiency. This project builds a predictive model that estimates delivery time based on distance, vehicle type, weather, traffic, and time of day — served through a real-time REST API.

---

## Solution Overview

| Stage | Details |
|---|---|
| Data Simulation | Realistic delivery order generator — 5,000 orders with distance, traffic, weather, vehicle type |
| Feature Engineering | Cyclical hour encoding, rush-hour flags, one-hot categorical encoding |
| Model Training | XGBoost Regressor with train/test split and evaluation |
| Evaluation | MAE, RMSE, R² on held-out test set |
| Deployment | Flask REST API — accepts JSON input, returns predicted delivery time |

---

## Tech Stack

- **Language:** Python 3.12
- **ML:** XGBoost, Scikit-learn
- **Data:** Pandas, NumPy
- **API:** Flask
- **Tools:** Jupyter Notebook, Git

---

## Project Structure

```
delivery-time-prediction/
│
├── data/
│   └── delivery_orders.csv          # Simulated delivery order dataset (5,000 records)
│
├── notebooks/
│   ├── 01_EDA.ipynb                 # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Training.ipynb
│
├── src/
│   ├── data_generator.py            # Delivery order data simulation
│   ├── preprocess.py                # Data cleaning and feature engineering
│   ├── train.py                     # Model training and evaluation
│   └── predict.py                   # Inference logic
│
├── app/
│   └── app.py                       # Flask REST API
│
├── models/
│   ├── xgb_delivery_model.pkl       # Trained model (generated on run)
│   └── feature_columns.pkl          # Feature schema for inference
│
├── requirements.txt
└── README.md
```

---

## Model Performance

Trained and evaluated on 5,000 simulated delivery orders (80/20 train-test split):

| Metric | Value |
|---|---|
| MAE (Mean Absolute Error) | 3.25 minutes |
| RMSE | 4.08 minutes |
| R² Score | 0.974 |

**Top predictive features:** traffic density, distance, vehicle type (bike vs. car/van), and weather conditions.

---

## API Usage

**Endpoint:** `POST /predict`

**Request:**
```json
{
  "distance_km": 5.2,
  "order_hour": 12,
  "vehicle_type": "scooter",
  "weather_condition": "clear",
  "traffic_density": "medium"
}
```

**Response:**
```json
{
  "predicted_delivery_time_minutes": 22.8,
  "input": {
    "distance_km": 5.2,
    "order_hour": 12,
    "vehicle_type": "scooter",
    "weather_condition": "clear",
    "traffic_density": "medium"
  }
}
```

**Valid values:**
- `vehicle_type`: bike, scooter, car, van
- `weather_condition`: clear, rain, fog, storm
- `traffic_density`: low, medium, high
- `order_hour`: 0–23

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/NadavaluruSashank/Delivery-Time-Prediction.git
cd Delivery-Time-Prediction

# Install dependencies
pip install -r requirements.txt

# 1. Generate simulated delivery order data
python src/data_generator.py

# 2. Train the model
cd src && python train.py && cd ..

# 3. Test a single prediction
cd src && python predict.py && cd ..

# 4. Start the REST API
python app/app.py
```

Once running, test the API with curl:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_km": 8.5, "order_hour": 18, "vehicle_type": "bike", "weather_condition": "rain", "traffic_density": "high"}'
```

---

## Use Cases

- Logistics and last-mile delivery planning
- ETA prediction for food delivery and e-commerce platforms
- Operational dashboards for delivery fleet management

---

## Author

**Nadavaluru Sashank**
Data Scientist · MSc Data Science & AI @ DSTI School of Engineering, Paris
[LinkedIn](https://www.linkedin.com/in/sashank-nadavaluru) · [GitHub](https://github.com/NadavaluruSashank)
