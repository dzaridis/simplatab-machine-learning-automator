
# Machine Learning Automator

## Overview
Machine Learning Automator is a comprehensive machine learning pipeline designed to automate the process of training, evaluating, and validating classification models. It provides a robust framework for feature selection, preprocessing, hyperparameter tuning, and model evaluation, ensuring efficient and accurate model performance.

## Contents
- [Context](#context)
- [Requirements](#requirements)
- [Inputs](#inputs)
- [Outputs](#outputs)
- [Main Advantages](#main-advantages)
- [Key Concepts](#key-concepts)
- [Feature Selection](#feature-selection)
- [Preprocessing](#preprocessing)
- [Hyperparameter Tuning & Training](#hyperparameter-tuning--training)
- [Evaluation](#evaluation)
- [Testing on the External Set](#testing-on-the-external-set)

## Context
This pipeline trains and evaluates various classification models, providing insights into how different components contribute to building and validating these models.

## Requirements
- **Python 3.x**
- **Docker (for Docker version)**
- **Required Python Libraries**: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `xgboost`, `shap`, `featurewiz`

## Inputs
- **CSV Files**: Must not contain missing values.
  - **Train.csv**: Features and target (last column). Used for K-Fold cross-validation and threshold tuning.
  - **Test.csv**: Features and target (last column). Used for validation of thresholds, metrics, ROC curves, and SHAP analysis.

For the Docker version, place `Train.csv` and `Test.csv` in an input volume folder.

## Outputs
The outputs will be saved in the `Materials` folder (or a specified output volume for Docker version):
- `ROC_CURVES.PNG`: ROC curves for each algorithm on the test set.
- `ShapFeatures` folder: PNG images showing SHAP values for each algorithm.
- Excel files:
  - Metrics for the algorithm on the internal K-Fold.
  - Metrics for the algorithm on the external set.
- `Models` folder: Pickle files containing the models evaluated on the external data. These pipelines can be used directly without manual feature selection or preprocessing.

## Main Advantages
- Automated feature selection & preprocessing.
- K-Fold Stratified Cross-validation on `Train.csv`.
- Automated threshold calculation based on validation splits from K-Fold.
- Hyperparameter tuning on the stratified K-Fold.
- Testing on `Test.csv` with the best hyperparameters from the internal K-Fold and the average threshold across folds.
- Reporting of five metrics on both the internal K-Fold and external set (`Test.csv`):
  - AUC
  - F-Score
  - Accuracy
  - Sensitivity
  - Specificity
- ROC Curves and SHAP Analysis on the external set.

## Key Concepts
- **Hyperparameters**: Set before training to control the behavior of the training algorithm.
- **Cross-validation**: Evaluates model performance by splitting data into multiple folds and training/testing on different combinations.
- **Pipeline**: A sequence of data processing and model training steps applied consistently across all models.

## Feature Selection
Identifies and retains important features based on correlation. Supports various strategies:
- **featurewiz** (Default): Based on correlation matrix and XGBoost selection.
  - `corr_limit` (default: 0.6)
- **rfe**: Recursive Feature Elimination using logistic regression.
  - `n_features_to_select` (default: 5)
- **lasso**
- **random_forest**: Based on correlation matrix and XGBoost selection.
- **xgboost**: Based on correlation matrix and XGBoost selection.

## Preprocessing
Prepares data for training:
- **Tabular Data**: One-hot encoding.
- **Numeric Data**: Z-Score normalization.

## Hyperparameter Tuning & Training
- **Hyperparameter Tuning**: Uses exhaustive grid search to find the best hyperparameters.
- **Training**: Trains the model on the training data.

## Evaluation
- **Threshold Optimizer**: Finds the optimal threshold on the train set for each fold based on the AUC metric.
- **Metrics**: Evaluates the model on validation data on each fold using:
  - AUC
  - F-Score
  - Accuracy
  - Sensitivity
  - Specificity

## Testing on the External Set
- **Retraining**: Models are retrained on the entire `Train.csv` dataset with hyperparameters set based on K-Fold selection.
- **Threshold**: Set as the average of the thresholds from the K-Fold.
- **Metrics**: Computed on the `Test.csv` for the optimal threshold.
- **Shapley Analysis**: Performed on a fraction of the test set (up to 100 instances).
- **ROC Curves**: Reported for each algorithm on the testing dataset.

