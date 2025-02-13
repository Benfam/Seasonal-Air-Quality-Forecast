import h5py
import joblib
import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ✅ Load the trained model and feature importances
def load_rf_model(model_filename):
    with h5py.File(model_filename, "r") as h5f:
        model_binary = h5f["random_forest_model"][()]
        rf_model = joblib.load(io.BytesIO(model_binary))

        # Load feature importance
        feature_importance = h5f["feature_importance"][:]
        feature_names = [name.decode("utf-8") for name in h5f["feature_names"][:]]

    return rf_model, feature_importance, feature_names

# ✅ Select the seasonal model to visualize
season = "winter"  # Change to "summer" for summer model
model_filename = f"{season}_rf_model.h5"

# ✅ Load model and features
rf_model, feature_importance, feature_names = load_rf_model(model_filename)

# ✅ Load dataset for visualization
df = pd.read_csv('Daily_PM25.csv', parse_dates=['Date'], dayfirst=True)
df = df[df['Date'].dt.month.isin([12, 1, 2] if season == "winter" else [6, 7, 8])].dropna()
X = df[['ws', 'wd', 'Humidity', 'Temp', 'Rain', 'Pressure']]
y_actual = df['PM2.5']
y_pred = rf_model.predict(X)

# ✅ Plot Predicted vs. Actual PM2.5
plt.figure(figsize=(8, 6))
plt.scatter(y_actual, y_pred, color='blue', alpha=0.5, label='Predicted vs Actual')
plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], color='red', linestyle='--', label='Perfect Fit')
plt.xlabel("Actual PM2.5 (µg/m³)")
plt.ylabel("Predicted PM2.5 (µg/m³)")
plt.title(f"{season.capitalize()} PM2.5 Predictions")
plt.legend()
plt.grid(True)
plt.show()

# ✅ Plot Feature Importance
plt.figure(figsize=(8, 5))
plt.barh(feature_names, feature_importance, color='green')
plt.xlabel("Feature Importance")
plt.ylabel("Weather Features")
plt.title(f"{season.capitalize()} Model Feature Importance")
plt.grid(True)
plt.show()
