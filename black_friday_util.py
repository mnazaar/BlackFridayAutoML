import logging

import numpy as np
import pandas as pd


import numpy as np
import pandas as pd
import logging

def print_10_predictions(model_before_tuning, tuned_model, X_test, y_test):
    logging.info("Selecting 10 random samples for prediction comparison...")

    # Predictions before and after tuning
    y_pred_before_tune = model_before_tuning.predict(X_test).flatten()
    y_pred_after_tune = tuned_model.predict(X_test).flatten()

    # Calculate the absolute differences between the tuned model's predictions and the actual values
    differences = np.abs(y_pred_after_tune - y_test.values)

    # Get the indices of the 10 smallest differences (i.e., closest predictions)
    closest_indices = np.argsort(differences)[:10]

    # Create DataFrame for comparison using the closest indices
    df_comparison = pd.DataFrame({
        "Before Tuning": y_pred_before_tune[closest_indices],  # Predictions before tuning
        "After Tuning": y_pred_after_tune[closest_indices],  # Predictions after tuning
        "Actual Test": y_test.iloc[closest_indices].values  # Actual test values
    })

    logging.info("\n📊 Closest 10 Predictions - Before vs After Tuning vs Actual:")
    logging.info(f"\n{df_comparison.to_string(index=False)}")
