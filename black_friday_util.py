import logging

import numpy as np
import pandas as pd


def print_10_predictions(model_before_tuning, tuned_model, X_test, y_test):
    logging.info("Selecting 10 random samples for prediction comparison...")

    y_pred_before_tune = model_before_tuning.predict(X_test).flatten()
    y_pred_after_tune = tuned_model.predict(X_test).flatten()

    valid_positions = np.arange(len(y_test))  # Ensure positional indexing
    random_positions = np.random.choice(valid_positions, size=10, replace=False)

    random_indices = y_test.index[random_positions]  # Convert positions to dataset indices


    df_comparison = pd.DataFrame({
        "Before Tuning": y_pred_before_tune[random_positions],  # Predictions before tuning
        "After Tuning": y_pred_after_tune[random_positions],  # Predictions after tuning
        "Actual Test": y_test.loc[random_indices].values  # Actual test values
    })

    logging.info("\n📊 Random 10 Predictions - Before vs After Tuning vs Actual:")
    logging.info(f"\n{df_comparison.to_string(index=False)}")
