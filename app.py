from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import os
from Helpers.pipelines_main import train_k_fold, external_test
import warnings
warnings.simplefilter(action='ignore', category=Warning)

def main(input_folder, output_folder):
    # Load data
    train = pd.read_csv(os.path.join(input_folder, "Train.csv"), index_col="patient_id")
    test = pd.read_csv(os.path.join(input_folder, "Test.csv"), index_col="patient_id")
    X_train = train.drop('Target', axis=1)  # Drop the 'Target' column for X_train
    y_train = train['Target']
    X_test = test.drop('Target', axis=1)  # Drop the 'Target' column for X_test
    y_test = test['Target']

    # Run the pipeline
    params_dict, scores_storage, thresholds = train_k_fold(X_train, y_train, k_folds=3)
    external_test(X_train, y_train, X_test, y_test, params_dict, thresholds)

if __name__ == "__main__":
    input_folder = "./input_data"
    output_folder = "./Materials"

    main(input_folder, output_folder)