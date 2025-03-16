# Define Optuna objective function
import keras
import mlflow.tensorflow
from autokeras.preprocessors.common import AddOneDimension
from sklearn.model_selection import train_test_split
from tabulate import tabulate

keras.saving.register_keras_serializable()(AddOneDimension)

import logging
import numpy as np
import mlflow
import tensorflow as tf
import optuna
from sklearn.metrics import mean_squared_error

current_trial = 0;


def objective(trial, X_train, X_val, y_train, y_val, max_epochs=30):
    try:
        # Convert Pandas DataFrame to NumPy array
        X_train_np, X_val_np = X_train.to_numpy(), X_val.to_numpy()
        y_train_np, y_val_np = y_train.to_numpy(), y_val.to_numpy()

        # Check if NaNs exist after conversion to NumPy
        print("NaNs in X_train_np:", np.isnan(X_train_np).sum())
        print("NaNs in X_val_np:", np.isnan(X_val_np).sum())
        print("NaNs in y_train_np:", np.isnan(y_train_np).sum())
        print("NaNs in y_val_np:", np.isnan(y_val_np).sum())
        print("Inf in X_train_np:", np.isinf(X_train_np).sum())
        print("Inf in X_val_np:", np.isinf(X_val_np).sum())
        print("Inf in y_train_np:", np.isinf(y_train_np).sum())
        print("Inf in y_val_np:", np.isinf(y_val_np).sum())
        # Suggest hyperparameters
        epochs = trial.suggest_int("epochs", 1, max_epochs)  # More epochs for better tuning
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])  # Optimize batch size
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop", "sgd"])  # Test different optimizers
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)  # Prevent overfitting
        global current_trial
        current_trial = current_trial + 1
        print(f"Starting hyperparameter search trial  {current_trial}")
        print(f"Trial params: epochs={epochs}, lr={learning_rate}, "
                     f"batch_size={batch_size}, optimizer={optimizer_name}, dropout={dropout_rate}")

        # Select optimizer
        if optimizer_name == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

        best_model = tf.keras.models.load_model("models/best_autokeras_model.keras")

        # Compile the modified model
        best_model.compile(optimizer=optimizer, loss="mse")

        # Train the model
        best_model.fit(X_train_np, y_train_np, epochs=epochs, batch_size=batch_size, verbose=0)

        # Evaluate the model
        y_pred = best_model.predict(X_val_np).flatten()
        rmse = np.sqrt(mean_squared_error(y_val_np, y_pred))

        print(f"Trial {current_trial} RMSE: {rmse:.4f}")

        # Log parameters and metrics to MLflow
        mlflow.set_experiment("AutoML Hyperparameter Optimization")
        with mlflow.start_run(nested=True):
            mlflow.log_params({
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "optimizer": optimizer_name,
                "dropout_rate": dropout_rate})
            mlflow.log_metric("rmse", rmse)

        return rmse  # Optuna minimizes RMSE

    except Exception as e:
        logging.error(f"Trial failed: {e}")
        return np.inf


def check_nan(df, name):
    nan_cols = df.columns[df.isna().any()].tolist()  # Get columns with NaNs
    if nan_cols:
        print(f"⚠️ {name} contains NaN values in columns: {nan_cols}")
    else:
        print(f"✅ {name} has no NaN values.")


# Function to run Optuna tuning
def optimize_autokeras_model_using_optuna(X, y, timeout=600, max_trials=20, max_epochs=20):
    print("\n Starting Optuna hyperparameter tuning for AutoKeras...")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    study = optuna.create_study(direction="minimize")

    # Check each dataset
    check_nan(X_train, "X_train")
    check_nan(X_val, "X_test")
    check_nan(y_train.to_frame(), "y_train")  # Convert Series to DataFrame
    check_nan(y_val.to_frame(), "y_test")  # Convert Series to DataFrame

    study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val, max_epochs=max_epochs),
                   timeout=timeout, n_trials=max_trials)

    print(f"Best Parameters: {study.best_params}")
    mlflow.set_experiment("AutoML Hyperparameter Optimization")
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_rmse", study.best_value)

    # Retrain the best model with optimal hyperparameters
    best_model, best_preds = train_best_model(X_train, X_val, y_train, y_val, study.best_params)

    # Export the model before logging
    # Ensure `best_model` is an AutoKeras model before calling export_model()
    if hasattr(best_model, "export_model"):
        exported_model = best_model.export_model()  # AutoKeras models need exporting
    else:
        exported_model = best_model  # Already a Keras model
    # Log the trained model to MLflow
    mlflow.tensorflow.log_model(
        exported_model,
        artifact_path="autokeras_model"
    )

    return best_model, study.best_params, best_preds


def train_best_model(X_train, X_test, y_train, y_test, best_params):
    print("\nTraining the best AutoKeras model with optimized hyperparameters...")

    # Convert DataFrame to NumPy array
    X_train_np, X_test_np = X_train.to_numpy(), X_test.to_numpy()
    y_train_np, y_test_np = y_train.to_numpy(), y_test.to_numpy()
    # Check if NaNs exist after conversion to NumPy
    print("NaNs in X_train_np:", np.isnan(X_train_np).sum())
    print("NaNs in X_val_np:", np.isnan(X_test_np).sum())
    print("NaNs in y_train_np:", np.isnan(y_train_np).sum())
    print("NaNs in y_val_np:", np.isnan(y_test_np).sum())
    print("Inf in X_train_np:", np.isinf(X_train_np).sum())
    print("Inf in X_val_np:", np.isinf(X_test_np).sum())
    print("Inf in y_train_np:", np.isinf(y_train_np).sum())
    print("Inf in y_val_np:", np.isinf(y_test_np).sum())
    # 🔹 **Load the best AutoKeras model**
    best_model = tf.keras.models.load_model("models/best_autokeras_model.keras")

    # 🔹 Modify Dropout layers (if present and dropout_rate exists in best_params)
    if "dropout_rate" in best_params:
        for layer in best_model.layers:
            if "dropout" in layer.name.lower():
                layer.rate = best_params["dropout_rate"]

    # 🔹 **Compile with best hyperparameters**
    best_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["learning_rate"]),
        loss="mse"
    )

    # 🔹 **Train the model using best parameters**
    best_model.fit(
        X_train_np, y_train_np,
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        verbose=1  # Show training progress
    )
    # Get the best model
    best_model.summary()

    # Extract model summary and log it
    model_layers = [[layer.name, str(layer.output.shape), f"{layer.count_params():,}"] for layer in best_model.layers]

    table = tabulate(
        model_layers,
        headers=["Layer (type)", "Output Shape", "Param #"],
        tablefmt="grid"
    )

    print(f"\n Best Model after tuning Summary:\n{table}")

    # 🔹 **Evaluate the final model**
    final_predictions = best_model.predict(X_test_np).flatten()
    best_model.save("models/best_autokeras_model.keras")
    print("\n Completed Training the best AutoKeras model with optimized hyperparameters...")

    return best_model, final_predictions
