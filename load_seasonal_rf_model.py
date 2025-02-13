import h5py
import joblib
import io
import numpy as np

# ✅ Allow the user to choose the model (Winter or Summer)
season = input("\n❄️🌞 Enter season to load the model ('winter' or 'summer'): ").strip().lower()
if season not in ["winter", "summer"]:
    raise ValueError("Invalid input! Choose either 'winter' or 'summer'.")

h5_filename = f"{season}_rf_model.h5"

# ✅ Load the trained model
with h5py.File(h5_filename, "r") as h5f:
    if "random_forest_model" in h5f:
        model_binary = h5f["random_forest_model"][()]
        rf_model = joblib.load(io.BytesIO(model_binary))
        print(f"\n✅ {season.capitalize()} Random Forest Model Loaded Successfully!")
    else:
        print("\n❌ Model not found in HDF5 file.")
        exit()

    # Print performance metrics
    print("\n📊 Model Performance Metrics:")
    print(f"R² (Train): {h5f.attrs['r2_train']:.4f}, R² (Test): {h5f.attrs['r2_test']:.4f}")
    print(f"RMSE (Train): {h5f.attrs['rmse_train']:.4f}, RMSE (Test): {h5f.attrs['rmse_test']:.4f}")

# ✅ Example Prediction
predict_choice = input("\n📡 Do you want to predict PM2.5 for new data? (yes/no): ").strip().lower()
if predict_choice == "yes":
    try:
        # Example input (Replace with real data)
        X_new = np.array([[3.5, 120, 60, 15, 0.2, 1015]])  # ws, wd, Humidity, Temp, Rain, Pressure
        predictions = rf_model.predict(X_new)
        print(f"\n🌍 Predicted PM2.5 for {season}: {predictions[0]:.2f} µg/m³")
    except Exception as e:
        print(f"\n❌ Error in prediction: {e}")

