# 📄 Deployment Documentation for Seasonal Random Forest Model

**📌 Author:** Abimbola Folami  
**📌 Version:** 1.0  
**📌 Last Updated:** 13 Feb 2025  

## 🚀 Overview

This project builds a **Random Forest model** for predicting PM2.5 levels during **Winter and Summer** seasons.  
The pipeline consists of **three scripts**:

1️⃣ **`train_seasonal_rf_model.py`** → Trains and saves the seasonal models.  
2️⃣ **`load_seasonal_rf_model.py`** → Loads trained models for predictions.  
3️⃣ **`visualize_seasonal_model.py`** → Plots and visualizes model performance.

---

## 📂 File Structure

```
📁 Seasonal_RF_Model/
│── 📜 train_seasonal_rf_model.py    # Train and save the models
│── 📜 load_seasonal_rf_model.py     # Load trained models for predictions
│── 📜 visualize_seasonal_model.py   # Visualize model performance
│── 📜 save_rf_model.py              # Utility function for saving models
│── 📜 check_h5_contents.py          # Debugging tool to check .h5 file contents
│── 📄 README.md                     # Deployment Documentation
│── 📊 winter_rf_model.h5            # Saved Winter Model
│── 📊 summer_rf_model.h5            # Saved Summer Model
│── 📊 Daily_PM25.csv                 # Raw dataset
```

---

## 1️⃣ Train the Seasonal Models

### **📜 `train_seasonal_rf_model.py`**
✔ Loads the dataset (`Daily_PM25.csv`)  
✔ Splits data into **Winter** (Dec-Feb) and **Summer** (Jun-Aug)  
✔ Trains **Random Forest models** for each season  
✔ Saves models as `.h5` files  

### 💻 How to Run

```bash
python train_seasonal_rf_model.py
```

### ✅ Expected Output

```
✅ Model saved successfully as 'winter_rf_model.h5'
Winter Model Evaluation:
R² (Train): 0.47, R² (Test): 0.39
RMSE (Train): 2.18, RMSE (Test): 1.91

✅ Model saved successfully as 'summer_rf_model.h5'
Summer Model Evaluation:
R² (Train): 0.38, R² (Test): -1.90
RMSE (Train): 4.94, RMSE (Test): 4.48
```

---

## 2️⃣ Load the Trained Models for Predictions

### **📜 `load_seasonal_rf_model.py`**
✔ Loads the **Winter or Summer** `.h5` model  
✔ Prints **performance metrics (R², RMSE, etc.)**  
✔ Allows **new predictions** for PM2.5  

### 💻 How to Run

```bash
python load_seasonal_rf_model.py
```

### ✅ Expected Output

```
✅ Random Forest Model Loaded Successfully!
📊 Model Performance Metrics:
R² (Train): 0.4739, R² (Test): 0.3944
RMSE (Train): 2.1777, RMSE (Test): 1.9076
```

### 💡 Making Predictions

To make predictions with new input:

```python
import numpy as np

# Example input (Replace with real data)
X_new = np.array([[3.5, 120, 60, 15, 0.2, 1015]])  # ws, wd, Humidity, Temp, Rain, Pressure
predictions = rf_model.predict(X_new)

print("Predicted PM2.5:", predictions)
```

---

## 3️⃣ Visualize Model Performance

### **📜 `visualize_seasonal_model.py`**
✔ **Loads trained models**  
✔ **Plots:**
- Predicted vs. Actual PM2.5  
- Feature Importance  

### 💻 How to Run

```bash
python visualize_seasonal_model.py
```

### ✅ Expected Output

🔹 **Scatter Plot** - Predicted vs. Measured PM2.5  
🔹 **Feature Importance Plot** - Shows which variables affect PM2.5 levels  

---

## 🔍 Debugging & Checking `.h5` File Contents

### **📜 `check_h5_contents.py`**
✔ **Check attributes & datasets** stored in `.h5` files  

### 💻 How to Run

```bash
python check_h5_contents.py
```

### ✅ Expected Output

```
✅ Attributes in 'winter_rf_model.h5':
r2_test: 0.3944
r2_train: 0.4739
rmse_test: 1.9076
rmse_train: 2.1777

✅ Datasets in the .h5 file:
feature_importance
feature_names
random_forest_model
```

---

## 📌 Deployment Instructions

1️⃣ **Ensure all `.h5` model files are included in the deployment.**  
2️⃣ Use **`load_seasonal_rf_model.py`** to make predictions.  
3️⃣ Run **`visualize_seasonal_model.py`** for analysis & reports.  
4️⃣ Use **`check_h5_contents.py`** for debugging.  

---

## 🎯 Summary

| **Script**                    | **Function**                                              |
| ----------------------------- | --------------------------------------------------------- |
| `train_seasonal_rf_model.py`  | Trains and saves seasonal RF models.                      |
| `load_seasonal_rf_model.py`   | Loads trained models for PM2.5 predictions.               |
| `visualize_seasonal_model.py` | Plots predicted vs. actual values and feature importance. |
| `check_h5_contents.py`        | Debugging tool to inspect `.h5` files.                    |

### 🚀 Ready for Deployment!

✔ **All scripts are tested and working!**  
✔ **Can be integrated into a larger system!**  
✔ **Supports seasonal PM2.5 forecasting!**  

---

## 🔧 Need Help?

📧 Contact: [enitanabimbola@gmail.com](mailto:enitanabimbola@gmail.com)  
📂 GitHub Repo: [Seasonal-Air-Quality-Forecast](https://github.com/Benfam/Seasonal-Air-Quality-Forecast)  
```
