from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import os
import yaml
from Helpers.pipelines_main import train_k_fold, external_test, read_yaml
from Helpers.data_checks import DataChecker
from Helpers import DBDM
import warnings
warnings.simplefilter(action='ignore', category=Warning)

app = Flask(__name__)
# Global variable to track pipeline status
pipeline_status_message = "Not started"


def get_train_columns(input_folder):
    train_file_path = os.path.join(input_folder, "Train.csv")
    df = pd.read_csv(train_file_path)
    
    # Drop columns containing "ID" or "patient_id" if they exist
    columns_to_drop = [col for col in df.columns if 'ID' in col or 'patient_id' in col]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Select only categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return categorical_columns


@app.route('/')
def index():
    input_folder = './input_data' 
    columns = get_train_columns(input_folder)
    return render_template('index.html', columns=columns)


@app.route('/pipeline_status')
def pipeline_status_route():
    # Simple status page to inform user that the pipeline is running
    return render_template('status.html')


@app.route('/submit', methods=['POST'])
def submit():
    # Retrieve the selected facet from the form
    selected_facet = request.form.get('facet')
    bias_assess = request.form["bias_assess"].lower() == 'true'

    input_folder = './input_data'
    output_folder = './Materials'
    
    # Update params with the rest of the form data
    params = {
        "number_of_k_folds": int(request.form["number_of_k_folds"]),
        "apply_grid_search": {
            "enabled": request.form["apply_grid_search"].lower() == 'true',
            "type": {
                "Randomized": request.form["randomized"].lower() == 'true',
                "Exhaustive": not (request.form["randomized"].lower() == 'true')
            }
        },
        "Correlation Limit": float(request.form["correlation_limit"]),
        "Metric For Threshold Optimization": request.form["metric_for_threshold_optimization"],
        "Machine Learning Models": {
            "Logistic Regression": request.form.get("logistic_regression") == 'true',
            "Support Vector Machines": request.form.get("support_vector_machines") == 'true',
            "Random Forest": request.form.get("random_forest") == 'true',
            "Stochastic Gradient Descent": request.form.get("stochastic_gradient_descent") == 'true',
            "Multi-Layer Neural Network": request.form.get("multi_layer_neural_network") == 'true',
            "Decision Trees": request.form.get("decision_trees") == 'true',
            "XGBoost": request.form.get("xgboost") == 'true'
        }
    }

    # Save the parameters to a YAML file
    yaml_path = os.path.join(input_folder, "machine_learning_parameters.yaml")
    with open(yaml_path, 'w') as file:
        yaml.dump(params, file)

    # Redirect to run pipeline
    return redirect(url_for('run_pipeline', selected_facet=selected_facet, bias_assess=bias_assess))

@app.route('/run_pipeline')
def run_pipeline():
    global pipeline_status_message
    input_folder = "./input_data"
    output_folder = "./Materials"
    selected_facet = request.args.get('selected_facet')
    bias_assess = request.args.get('bias_assess').lower() == 'true'
    try:
        import threading
        pipeline_status_message = "Running"
        # Run the main function asynchronously
        threading.Thread(target=main, args=(input_folder, output_folder, selected_facet, bias_assess)).start()
        return redirect(url_for('pipeline_status'))
    except Exception as e:
        pipeline_status_message = f"Error: {e}"
        return f"An error occurred: {e}"


@app.route('/pipeline_status')
def pipeline_status():
    return render_template('status.html', status=pipeline_status_message)


def main(input_folder, output_folder, selected_facet,  bias_assess=False):
    global pipeline_status_message
    # Load parameters from YAML file
    read_yaml(input_folder)
    if bias_assess:
        print("------------- \n", " Bias Detection Started \n", "-------------")
        try:
            print("------------- \n", " Bias Detection Started for Train.csv \n", "-------------")
            DBDM.bias_config(
                file_path=os.path.join(input_folder, "Train.csv"),
                subgroup_analysis=0,  # default is 0
                facet=selected_facet,
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
                facet=selected_facet,
                outcome='Target',
                subgroup_col='',  # default is ''
                label_value=1,  # default is 1
            )
            print("------------- \n", " Bias Detection Finished for Test.csv \n", "-------------")
        except:
            pass
        print("------------- \n", " Bias Detection Finished \n", "-------------")
    # Load data
    print("------------- \n", " Loading Data \n", "-------------")
    data_checker = DataChecker(input_folder)

    # Process the data
    try:
        train, test = data_checker.process_data()
        print("Train and Test data processed successfully.")
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    try:
        X_train = train.drop('Target', axis=1)  # Drop the 'Target' column for X_train
        y_train = train['Target']
        X_test = test.drop('Target', axis=1)  # Drop the 'Target' column for X_test
        y_test = test['Target']
        print("------------- \n", " Data Loaded successfully \n", "-------------")

        # Run the pipeline
        print("------------- \n", " Training on K-Fold cross validation with Train.csv file and parameters set on machine_learning_parameters.yaml file  \n", "-------------")

        params_dict, scores_storage, thresholds, _ = train_k_fold(X_train, y_train)
        print("------------- \n", " Training on K-Fold cross validation with Train.csv file and parameters set on machine_learning_parameters.yaml file completed successfully \n", "-------------")

        print("------------- \n", " Evaluating algorithms on Test.csv \n", "-------------")
        external_test(X_train, y_train, X_test, y_test, params_dict, thresholds)

        pipeline_status_message = "Completed"
    except Exception as e:
        print(e)
        pipeline_status_message = f"Error: {e}"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)