import os
import pandas as pd
import yaml
from Helpers.pipelines_main import train_k_fold, external_test, read_yaml
from Helpers.data_checks import DataChecker
from tkinter import Tk, filedialog
from Helpers import DBDM

def get_user_input():
    params = {}
    params["BiasAssessment"] = input("Enable bias assessment (true/false): ").lower() == 'true'
    params["Feature"] = str(input("Enter feature from your train and test sets columns for bias assessment: "))
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


def main(input_folder, output_folder, params):
    read_yaml(input_folder)

    # Perform Bias Assessment
    if params["BiasAssessment"]:
        print("------------- \n", " Bias Detection Started \n", "-------------")
        try:
            print("------------- \n", " Bias Detection Started for Train.csv \n", "-------------")
            DBDM.bias_config(
                file_path=os.path.join(input_folder, "Train.csv"),
                subgroup_analysis=0,  # default is 0
                facet=params["Feature"],
                outcome='Target',
                subgroup_col='',  # default is ''
                label_value=1,  # default is 1
            )
            print("------------- \n", " Bias Detection Finished for Train.csv \n", "-------------")
        except:
            pass
        try:
            print("------------- \n", " Bias Detection Started for Test.csv \n", "-------------")
            DBDM.bias_config(
                file_path=os.path.join(input_folder, "Test.csv"), 
                subgroup_analysis=0, # default is 0
                facet=params["Feature"],
                outcome='Target',
                subgroup_col='',  # default is ''
                label_value=1,  # default is 1
            )
            print("------------- \n", " Bias Detection Finished for Test.csv \n", "-------------")
        except:
            pass

    # Load data
    print("------------- \n", "Loading Data \n", "-------------")
    data_checker = DataChecker(input_folder)

# Process the data
    try:
        train, test = data_checker.process_data()
        print("Train and Test data processed successfully.")
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)

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

    main(input_folder, output_folder, params)
