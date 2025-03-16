import logging
import pandas as pd
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset


def detect_drift_ks(train_data: pd.DataFrame, new_data: pd.DataFrame):
    """
    Detects drift in numerical features using EvidentlyAI's DataDriftTestPreset.

    Parameters:
    - train_data (pd.DataFrame): Reference dataset (original training data)
    - new_data (pd.DataFrame): Incoming dataset (new batch or test data)

    Returns:
    - drift_results (pd.DataFrame): Summary of drift detection per feature
    - drift_detected (bool): Flag indicating whether any drift was detected
    """
    drift_detected = False

    # Run EvidentlyAI's built-in drift detection
    test_suite = TestSuite(tests=[DataDriftTestPreset()])
    test_suite.run(reference_data=train_data, current_data=new_data)

    # Extract test results
    test_results = test_suite.as_dict()

    # Debugging: Print the entire result structure
    import json
    print(json.dumps(test_results, indent=2))

    # Adjusted condition to check if the expected structure exists
    if not test_results.get("tests"):
        logging.warning("No test results found in Evidently output. Possibly incorrect API format.")
        return pd.DataFrame(columns=["Feature", "p-value", "Drift Detected"]), False

    # Extract drift results safely
    drift_detected = False
    for test in test_results["tests"]:
        if test["name"] == "Drift per Column":  # Ensure we process the drift test
            drift_detected = test.get("parameters", {}).get("detected")
            if drift_detected:
                break

    print("\nDrift Detection Results:\n" + json.dumps(test_results, indent=2))

    return test_results, drift_detected
