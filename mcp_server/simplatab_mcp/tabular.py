"""The tabular classification run, as app.run_pipeline of the web application (which lives in
the Flask application): bias assessment, data checks, K-fold training and the external test.
Outputs go to ./Materials. Python 3.9 compatible."""
import json
import os

import pandas as pd
import yaml


def run_tabular_pipeline(input_folder, params):
    from Helpers import DBDM
    from Helpers.data_checks import DataChecker
    from Helpers.pipelines_main import external_test, read_yaml, train_k_fold

    with open(os.path.join(input_folder, "machine_learning_parameters.yaml"), "w") as f:
        yaml.dump(params, f)
    try:
        read_yaml(input_folder)
        if params["BiasAssessment"]:
            print("------------- \n", " Bias Detection Started \n", "-------------")
            for name in ("Train.csv", "Test.csv"):
                try:
                    DBDM.bias_config(file_path=os.path.join(input_folder, name), subgroup_analysis=0,
                                     facet=params["Feature"], outcome="Target", subgroup_col="", label_value=1)
                    print(f"Bias Detection Finished for {name}")
                except Exception as e:
                    print(f"Error in bias detection for {name}: {e}")

        print("------------- \n", "Loading Data \n", "-------------")
        data_checker = DataChecker(input_folder)
        try:
            train, test = data_checker.process_data()
        except (FileNotFoundError, ValueError) as e:
            print(e)
            return f"Error: {e}"
        X_train, y_train = train.drop("Target", axis=1), train["Target"]
        X_test, y_test = test.drop("Target", axis=1), test["Target"]
        columns = pd.read_csv(os.path.join(input_folder, "Train.csv"), nrows=0).columns
        with open(os.path.join("Materials", "run_info.json"), "w") as f:
            json.dump({"automator": "tabular", "classes": sorted(int(c) for c in y_train.unique()),
                       "id_column": next((c for c in ("ID", "patient_id") if c in columns), None),
                       "k_folds": params["number_of_k_folds"],
                       "metric": params["Metric For Threshold Optimization"]}, f, indent=2)
        print("------------- \n", "Data Loaded successfully \n", "-------------")

        print("------------- \n", "Training on K-Fold cross validation \n", "-------------")
        params_dict, _, thresholds, _ = train_k_fold(X_train, y_train, rows=getattr(data_checker, "train_rows", None),
                                                     has_ids=getattr(data_checker, "train_has_ids", True))
        print("------------- \n", "Training on K-Fold cross validation completed successfully \n", "-------------")
        if not params_dict:
            return "Error: every model failed during the K-fold training (see error_log.log)."
        print("------------- \n", "Evaluating algorithms on Test.csv \n", "-------------")
        external_test(X_train, y_train, X_test, y_test, params_dict, thresholds)
        print("Pipeline completed successfully.")
        return "Pipeline completed successfully"
    except Exception as e:
        print(f"Error in pipeline: {e}")
        return f"Error: {e}"
