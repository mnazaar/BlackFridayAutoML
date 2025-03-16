import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from black_friday_dataset import load_dataset, save_dataset, load_transformed_dataset
from black_friday_drift_detection import detect_drift_ks  # Import the improved method
from black_friday_feature_engg import drop_id_fields, replace_missing_values_with_unknown, one_hot_encode, \
    scale_features, convert_bool_to_int
from black_friday_hyperparameter_tuning import optimize_autokeras_model_using_optuna

# Configure logging
logging.basicConfig(filename="logs/black_friday_model_drift.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

df = load_dataset()

# Simulate drift in test data
df_drifted = df.sample(20000).copy()
df_drifted["Occupation"] *= 5  # Create drift in occupation data in sample of 20K rows
df_combined = pd.concat([df, df_drifted])

# Use the improved detect_drift_ks function
drift_results, drift_detected = detect_drift_ks(df, df_combined)

# If drift is detected, retrain the model using AutoKeras
if drift_detected:
    logging.info("Drift detected! Retraining the model...")
    df_combined = drop_id_fields(df_combined)
    df_combined = replace_missing_values_with_unknown(df_combined)
    df_combined = one_hot_encode(df_combined)
    df_combined = convert_bool_to_int(df_combined)
    df_combined = scale_features(df_combined)

    save_dataset(df_combined)

    df = load_transformed_dataset()

    df = df.sample(50000)

    # Define X (features) and y (target)
    X = df.drop(columns=["Purchase"])  # All features except target
    y = df["Purchase"]  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("Dataset split into train and test sets.")

    # Hyperparameter tuning using Optuna
    tuned_model, tuning_final_params, tuned_predictions = optimize_autokeras_model_using_optuna(X_train, y_train,
                                                                                                timeout=1800,
                                                                                                max_trials=5,
                                                                                                max_epochs=50)
    logging.info(f"Hyperparameter tuning completed. Best parameters: {tuning_final_params}")


else:
    logging.info("No drift detected. No retraining required.")

logging.info("Drift simulations completed successfully.")
