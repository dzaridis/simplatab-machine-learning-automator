from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import os
import yaml
from Helpers.pipelines_main import train_k_fold, external_test, read_yaml
import warnings
warnings.simplefilter(action='ignore', category=Warning)

app = Flask(__name__)
# Global variable to track pipeline status
pipeline_status_message = "Not started"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pipeline_status')
def pipeline_status_route():
    # Simple status page to inform user that the pipeline is running
    return render_template('status.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Get form data
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
            "Logistic Regression": request.form["logistic_regression"].lower() == 'true',
            "Support Vector Machines": request.form["support_vector_machines"].lower() == 'true',
            "Random Forest": request.form["random_forest"].lower() == 'true',
            "Stochastic Gradient Descent": request.form["stochastic_gradient_descent"].lower() == 'true',
            "Multi-Layer Neural Network": request.form["multi_layer_neural_network"].lower() == 'true',
            "Decision Trees": request.form["decision_trees"].lower() == 'true',
            "XGBoost": request.form["xgboost"].lower() == 'true'
        }
    }

    # Save the parameters to a YAML file
    yaml_path = os.path.join("./input_data", "machine_learning_parameters.yaml")
    with open(yaml_path, 'w') as file:
        yaml.dump(params, file)

    # Redirect to run pipeline
    return redirect(url_for('run_pipeline'))

@app.route('/run_pipeline')
def run_pipeline():
    global pipeline_status_message
    input_folder = "./input_data"
    output_folder = "./Materials"
    try:
        import threading
        pipeline_status_message = "Running"
        # Run the main function asynchronously
        threading.Thread(target=main, args=(input_folder, output_folder)).start()
        return redirect(url_for('pipeline_status'))
    except Exception as e:
        pipeline_status_message = f"Error: {e}"
        return f"An error occurred: {e}"

@app.route('/pipeline_status')
def pipeline_status():
    return render_template('status.html', status=pipeline_status_message)

def main(input_folder, output_folder):
    global pipeline_status_message
        # Load parameters from YAML file
    read_yaml()

    # Load data
    print("------------- \n", " Loading Data \n", "-------------")
    train = pd.read_csv(os.path.join(input_folder, "Train.csv"), index_col="patient_id")
    test = pd.read_csv(os.path.join(input_folder, "Test.csv"), index_col="patient_id")
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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

# if __name__ == "__main__":
#     input_folder = "./input_data"
#     output_folder = "./Materials"

#     main(input_folder, output_folder)