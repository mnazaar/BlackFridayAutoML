import logging
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
from Cal_house_pricing_hyperparameter_tuning import optimize_autokeras
from Cal_house_pricing_drift_detection import detect_drift_ks  # Import the improved method

# Configure logging
logging.basicConfig(filename="logs/house_pricing_model_drift.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Load California Housing dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Define X (features) and y (target)
X = df.drop(columns=["MedHouseVal"])  # All features except target
y = df["MedHouseVal"]  # Target variable

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simulate drift in test data
X_test_drifted = X_test.copy()
X_test_drifted["MedInc"] *= 1.2  # Increase median income by 20%
X_test_drifted["AveRooms"] *= 0.8  # Decrease AveRooms by 20%

# Use the improved detect_drift_ks function
drift_results, drift_detected = detect_drift_ks(X_train, X_test_drifted)


# If drift is detected, retrain the model using AutoKeras
if drift_detected:
    logging.info("Drift detected! Retraining the model...")

    # Combine both train and drifted test data for retraining
    X_combined = pd.concat([X_train, X_test_drifted])
    y_combined = pd.concat([y_train, y_test])

    # Call optimize_autokeras to retrain the model
    best_model = optimize_autokeras(X_combined, y_combined)

else:
    logging.info("No drift detected. No retraining required.")
