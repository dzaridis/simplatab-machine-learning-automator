import os
import pandas as pd
import yaml
from Helpers.pipelines_main import train_k_fold, external_test, read_yaml
from tkinter import Tk, filedialog

def get_user_input():
    params = {}

    params["number_of_k_folds"] = int(input("Enter number of k-folds: "))

    params["apply_grid_search"] = {}
    params["apply_grid_search"]["enabled"] = input("Enable grid search (true/false): ").lower() == 'true'
    params["apply_grid_search"]["type"] = {}
    randomized = input("Use randomized grid search (true/false): ").lower() == 'true'
    params["apply_grid_search"]["type"]["Randomized"] = randomized
    params["apply_grid_search"]["type"]["Exhaustive"] = not randomized

    params["Correlation Limit"] = float(input("Enter correlation limit: "))
    
    metrics = ["Specificity", "Sensitivity", "AUC", "F-score", "Accuracy", "Balanced Accuracy"]
    while True:
        metric = input(f"Select metric for threshold optimization ({', '.join(metrics)}): ")
        if metric in metrics:
            params["Metric For Threshold Optimization"] = metric
            break
        else:
            print("Invalid input. Please choose a valid metric.")

    params["Machine Learning Models"] = {}
    params["Machine Learning Models"]["Logistic Regression"] = input("Use Logistic Regression (true/false): ").lower() == 'true'
    params["Machine Learning Models"]["Support Vector Machines"] = input("Use Support Vector Machines (true/false): ").lower() == 'true'
    params["Machine Learning Models"]["Random Forest"] = input("Use Random Forest (true/false): ").lower() == 'true'
    params["Machine Learning Models"]["Stochastic Gradient Descent"] = input("Use Stochastic Gradient Descent (true/false): ").lower() == 'true'
    params["Machine Learning Models"]["Multi-Layer Neural Network"] = input("Use Multi-Layer Neural Network (true/false): ").lower() == 'true'
    params["Machine Learning Models"]["Decision Trees"] = input("Use Decision Trees (true/false): ").lower() == 'true'
    params["Machine Learning Models"]["XGBoost"] = input("Use XGBoost (true/false): ").lower() == 'true'

    return params

def save_yaml(params, yaml_path):
    with open(yaml_path, 'w') as file:
        yaml.dump(params, file)

def select_folder(prompt):
    root = Tk()
    root.withdraw()  # Hide the root window
    folder_selected = filedialog.askdirectory(title=prompt)
    root.destroy()
    return folder_selected

def main(input_folder, output_folder):
    read_yaml(input_folder)

    # Load data
    print("------------- \n", "Loading Data \n", "-------------")
    train = pd.read_csv(os.path.join(input_folder, "Train.csv"), index_col="patient_id")
    test = pd.read_csv(os.path.join(input_folder, "Test.csv"), index_col="patient_id")
    X_train = train.drop('Target', axis=1)
    y_train = train['Target']
    X_test = test.drop('Target', axis=1)
    y_test = test['Target']
    print("------------- \n", "Data Loaded successfully \n", "-------------")

    # Run the pipeline
    print("------------- \n", "Training on K-Fold cross validation with Train.csv file and parameters set on machine_learning_parameters.yaml file \n", "-------------")
    params_dict, scores_storage, thresholds, _ = train_k_fold(X_train, y_train)
    print("------------- \n", "Training on K-Fold cross validation completed successfully \n", "-------------")

    print("------------- \n", "Evaluating algorithms on Test.csv \n", "-------------")
    external_test(X_train, y_train, X_test, y_test, params_dict, thresholds)
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    print("Please select the input folder:")
    input_folder = select_folder("Select the input folder")

    print("Please select the output folder:")
    output_folder = select_folder("Select the output folder")

    params = get_user_input()
    yaml_path = os.path.join(input_folder, "machine_learning_parameters.yaml")
    save_yaml(params, yaml_path)

    main(input_folder, output_folder)
