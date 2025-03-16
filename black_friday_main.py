import logging

from sklearn.model_selection import train_test_split

from black_friday_EDA import eda_pandas_profiling, eda_dataprep
from black_friday_automl import find_autokeras_model
from black_friday_dataset import load_transformed_dataset, load_dataset, save_dataset
from black_friday_explainability import get_model_for_explain, shap_explain, lime_explain
from black_friday_feature_engg import scale_features, one_hot_encode, replace_missing_values_with_unknown, \
    drop_id_fields, convert_bool_to_int
from black_friday_hyperparameter_tuning import optimize_autokeras_model_using_optuna
from black_friday_util import print_10_predictions

# Configure logging
logging.basicConfig(filename="logs/black_friday_main.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

logging.info("Started...")
print("Started...")

# Download dataset
df= load_dataset()
df = df.sample(50000)


logging.info("Dataset loaded successfully.")

logging.info(f"Sample Data:\n{df.sample(10)}")

##Perform EDA below
eda_pandas_profiling(df, "Before")
#eda_dataprep(df, "Before")



df = drop_id_fields(df)
df = replace_missing_values_with_unknown(df)
df = one_hot_encode(df)
df = convert_bool_to_int(df)
df = scale_features(df)

#Visualize after EDA
eda_pandas_profiling(df, "After")
#eda_dataprep(df, "After")


## Saving and loading again as a checkpoint after transformation
save_dataset(df)

df = load_transformed_dataset()

##Original dataset has more than 500000 rows, so taking a sample of 50000 rows to do the model.
df = df.sample(50000)

# Define X (features) and y (target)
X = df.drop(columns=["Purchase"])  # All features except target
y = df["Purchase"]  # Target variable

model = get_model_for_explain(X, y)
logging.info("Base model trained for explanations.")

shap_explain(model, X)
lime_explain(model, X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info("Dataset split into train and test sets.")




# Run AutoML
best_model_before_tuning, automl_predictions_before_tuning = find_autokeras_model(X_train, X_test, y_train, y_test, task="regression",
                                                                    max_trials=5, epochs=50, patience=2)
logging.info("AutoML model training completed.")

# Hyperparameter tuning using Optuna
tuned_model, tuning_final_params, tuned_predictions = optimize_autokeras_model_using_optuna(X_train, y_train,
                                                                                            timeout=1800, max_trials=5,
                                                                                            max_epochs=50)
logging.info(f"Hyperparameter tuning completed. Best parameters: {tuning_final_params}")

# Log the comparison instead of printing
print_10_predictions(best_model_before_tuning, tuned_model,X_test, y_test)

logging.info("Process completed successfully.")