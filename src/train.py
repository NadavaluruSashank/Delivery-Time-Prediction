"""
Delivery Time Prediction Model Training
-------------------------------------------
Trains an XGBoost regressor to predict delivery time in minutes.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from preprocess import load_data, engineer_features, get_feature_columns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "xgb_delivery_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "..", "models", "feature_columns.pkl")


def train_model(df, feature_cols, target_col="delivery_time_minutes", tune=False):
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if tune:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1],
        }
        base_model = XGBRegressor(random_state=42)
        search = GridSearchCV(base_model, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print(f"Best params: {search.best_params_}")
    else:
        model = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"MAE:  {mae:.2f} minutes")
    print(f"RMSE: {rmse:.2f} minutes")
    print(f"R2:   {r2:.3f}")

    return model, {"mae": mae, "rmse": rmse, "r2": r2}


def get_feature_importance(model, feature_cols):
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    return importance


if __name__ == "__main__":
    df = load_data()
    df = engineer_features(df)
    feature_cols = get_feature_columns(df)

    model, metrics = train_model(df, feature_cols)

    print("\nFeature importance:")
    print(get_feature_importance(model, feature_cols))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
