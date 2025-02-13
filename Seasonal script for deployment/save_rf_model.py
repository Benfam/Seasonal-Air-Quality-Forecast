import h5py
import joblib
import io
import numpy as np

def save_rf_model(model, filename, r2_train, rmse_train, mse_train, sse_train, 
                  r2_test=None, rmse_test=None, mse_test=None, sse_test=None, 
                  feature_importance=None, feature_names=None):
    """
    Saves a trained Random Forest model along with evaluation metrics and feature importances into an HDF5 (.h5) file.

    Parameters:
    - model: Trained Random Forest model
    - filename: Name of the .h5 file to save
    - r2_train, rmse_train, mse_train, sse_train: Training performance metrics
    - r2_test, rmse_test, mse_test, sse_test: (Optional) Test performance metrics
    - feature_importance: Feature importance scores
    - feature_names: Feature names
    """

    model_buffer = io.BytesIO()
    joblib.dump(model, model_buffer)
    model_binary = model_buffer.getvalue()

    with h5py.File(filename, 'w') as h5f:
        # Save model as binary
        h5f.create_dataset('random_forest_model', data=np.void(model_binary))

        # Save training metrics
        h5f.attrs['r2_train'] = r2_train
        h5f.attrs['rmse_train'] = rmse_train
        h5f.attrs['mse_train'] = mse_train
        h5f.attrs['sse_train'] = sse_train

        # Save test metrics if provided
        if r2_test is not None:
            h5f.attrs['r2_test'] = r2_test
            h5f.attrs['rmse_test'] = rmse_test
            h5f.attrs['mse_test'] = mse_test
            h5f.attrs['sse_test'] = sse_test

        # Save feature importance if provided
        if feature_importance is not None and feature_names is not None:
            h5f.create_dataset('feature_importance', data=feature_importance)
            h5f.create_dataset('feature_names', data=np.array(feature_names, dtype='S'))

    print(f"✅ Model saved successfully as '{filename}'")

