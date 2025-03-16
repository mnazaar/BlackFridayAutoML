import logging
import numpy as np
import autokeras as ak
from sklearn.metrics import mean_squared_error, accuracy_score
import tensorflow as tf
import pandas as pd
from tabulate import tabulate


def find_autokeras_model(X_train, X_test, y_train, y_test, task="regression", max_trials=10, epochs=20, patience=5):
    print("Starting AutoML training with AutoKeras...")

    # Convert DataFrames to NumPy arrays
    X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
    X_test = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test
    y_train = y_train.to_numpy().reshape(-1, 1) if isinstance(y_train, pd.Series) else y_train
    y_test = y_test.to_numpy().reshape(-1, 1) if isinstance(y_test, pd.Series) else y_test

    # Define AutoKeras model
    model = ak.AutoModel(
        inputs=ak.Input(),
        outputs=ak.RegressionHead() if task == "regression" else ak.ClassificationHead(),
        max_trials=max_trials,
        overwrite=True
    )

    # Track trials using a custom callback
    class TrialLogger(tf.keras.callbacks.Callback):
        trial_count = 0  # Track trial numbers

        def on_train_begin(self, logs=None):
            TrialLogger.trial_count += 1
            print(f"Trial #{TrialLogger.trial_count} started...")

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            print(
                f"Epoch {epoch + 1}/{epochs} - Loss: {logs.get('loss', 'N/A')} - Val Loss: {logs.get('val_loss', 'N/A')}")

        def on_train_end(self, logs=None):
            print(f"Trial #{TrialLogger.trial_count} completed!")

            # Extract model details after the trial ends
            trial_model = self.model
            if trial_model is not None:
                num_layers = len(trial_model.layers)
                print(f"Trial #{TrialLogger.trial_count} Model has {num_layers} layers.")

                model_layers = [[layer.name, str(layer.output.shape), f"{layer.count_params():,}"] for layer in
                                trial_model.layers]
                table = tabulate(model_layers, headers=["Layer (type)", "Output Shape", "Param #"], tablefmt="grid")
                print(f"\nTrial #{TrialLogger.trial_count} Model Summary:\n{table}")

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    # Train the model and track trials
    model.fit(X_train, y_train, epochs=epochs, callbacks=[TrialLogger(), early_stopping], verbose=1)

    # Get the best model
    best_model = model.export_model()
    best_model.summary()

    # Extract model summary and log it
    model_layers = [[layer.name, str(layer.output.shape), f"{layer.count_params():,}"] for layer in best_model.layers]

    table = tabulate(
        model_layers,
        headers=["Layer (type)", "Output Shape", "Param #"],
        tablefmt="grid"
    )

    print(f"\n Best Model Summary:\n{table}")

    # Make Predictions
    final_predictions = best_model.predict(X_test)

    # Ensure predictions are 1D for regression
    if task == "regression":
        final_predictions = final_predictions.flatten()
        rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
        print(f"RMSE on Test Set: {rmse:.4f}")
    else:  # Classification
        final_predictions = np.argmax(final_predictions, axis=1)
        metric = accuracy_score(y_test, final_predictions)
        print(f"Accuracy on Test Set: {metric:.4f}")

    print("🏆 Best Model Found and Evaluated!")

    best_model.save("models/best_autokeras_model.keras")
    print("Completed AutoML training with AutoKeras...")

    return best_model, final_predictions