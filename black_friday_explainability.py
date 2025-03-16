import logging
import os

import joblib
import lightgbm as lgb
import lime
import matplotlib
import numpy as np
import shap
from matplotlib import pyplot as plt


# SHAP Global Explanation
# Do for a sample as bigger dataset runs forever
def shap_explain(model, X):
    if len(X) > 5000:
        X = X.sample(5000)
    print("Generating explanations using SHAP...")
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X)
    matplotlib.use("Agg")  # Use a non-interactive backend

    save_to_path = "reports/shap_summary_plot.png"
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(save_to_path)
    plt.close()

    print(f"SHAP summary plot saved at {save_to_path}")
    return shap_values


# Load or Train Model for Explanation
def get_model_for_explain(X, y, model_path="models/explanation_model.pkl"):
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = joblib.load(model_path)
    else:
        print("Fitting the model for explanations...")
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

    return model


# LIME Local Explanation
def lime_explain(model, X):
    print("Generating local explanations using LIME...")

    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns.tolist(),
        mode="regression"
    )

    random_indices = np.random.choice(X.index, size=10, replace=False)
    random_instances = X.loc[random_indices]

    for i, (index, instance) in enumerate(random_instances.iterrows()):
        exp = explainer_lime.explain_instance(instance.values, model.predict)
        file_path = f"reports/lime_instance_{index}.html"
        exp.save_to_file(file_path)

        print(f"Local Explanation for Instance {index} saved to {file_path}")

    return exp
