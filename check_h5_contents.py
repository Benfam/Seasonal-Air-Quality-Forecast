import h5py

# Specify the model file to inspect
h5_filename = "summer_rf_model.h5"  # Change to "summer_rf_model.h5" to check summer model

# Load and display the attributes and datasets
with h5py.File(h5_filename, "r") as h5f:
    print(f"\n✅ Attributes in '{h5_filename}':")
    for attr in h5f.attrs:
        print(f"{attr}: {h5f.attrs[attr]}")

    print("\n✅ Datasets in the .h5 file:")
    for key in h5f.keys():
        print(key)

