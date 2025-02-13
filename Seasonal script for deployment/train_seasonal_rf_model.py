from save_rf_model import save_rf_model  # Ensure this is at the top

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the dataset and specify the date format
df = pd.read_csv('Daily_PM25.csv', parse_dates=['Date'], dayfirst=True)

# Define winter and summer months
winter_months = [12, 1, 2]  # December, January, February
summer_months = [6, 7, 8]   # June, July, August

# Filter dataframe for winter and summer months
winter_data = df[df['Date'].dt.month.isin(winter_months)]
summer_data = df[df['Date'].dt.month.isin(summer_months)]

# Drop rows with NaN values
winter_data_clean = winter_data.dropna()
summer_data_clean = summer_data.dropna()

# Define features and target for winter and summer
feature_columns = ['ws', 'wd', 'Humidity', 'Temp', 'Rain', 'Pressure']
winter_X = winter_data_clean[feature_columns]
winter_y = winter_data_clean['PM2.5']
summer_X = summer_data_clean[feature_columns]
summer_y = summer_data_clean['PM2.5']

# Function to train and save seasonal models
def train_and_save_model(X, y, season):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Compute metrics
    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mse_train = mean_squared_error(y_train, y_pred_train)
    sse_train = np.sum((y_train - y_pred_train) ** 2)
    
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mse_test = mean_squared_error(y_test, y_pred_test)
    sse_test = np.sum((y_test - y_pred_test) ** 2)
    
    # Save the trained model
    model_filename = f"{season}_rf_model.h5"
    save_rf_model(
        model, model_filename,
        r2_train=r2_train, rmse_train=rmse_train, mse_train=mse_train, sse_train=sse_train,
        r2_test=r2_test, rmse_test=rmse_test, mse_test=mse_test, sse_test=sse_test,
        feature_importance=model.feature_importances_,
        feature_names=X.columns
    )
    
    print(f"\n{season.capitalize()} Random Forest Model Evaluation:")
    print(f"R² (Train): {r2_train:.2f}, R² (Test): {r2_test:.2f}")
    print(f"RMSE (Train): {rmse_train:.2f}, RMSE (Test): {rmse_test:.2f}")

# Train and save models for winter and summer
train_and_save_model(winter_X, winter_y, "winter")
train_and_save_model(summer_X, summer_y, "summer")
