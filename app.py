from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import os
from Helpers.pipelines_main import train_k_fold, external_test
import warnings
warnings.simplefilter(action='ignore', category=Warning)

def main(input_folder, output_folder):
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
    print("------------- \n", " Training on K-Fold cross validation with Train.csv file and parameters set on machine_learning_pipelines yaml file  \n", "-------------")
    params_dict, scores_storage, thresholds = train_k_fold(X_train, y_train)
    print("------------- \n", " Training on K-Fold cross validation with Train.csv file and parameters set on machine_learning_pipelines yaml file completed successfully \n", "-------------")

    print("------------- \n", " Evaluating algorithms on Test.csv \n", "-------------")
    external_test(X_train, y_train, X_test, y_test, params_dict, thresholds)

if __name__ == "__main__":
    input_folder = "./input_data"
    output_folder = "./Materials"

    main(input_folder, output_folder)