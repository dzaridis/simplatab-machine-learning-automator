
# SIMPLATAB: **SI**mplified **M**achine **P**ipe**L**ine **A**utomator for **TAB**ular data
![ML Pipeline](images_materials/MLPipeline.png)
## Overview
Machine Learning Automator is a comprehensive machine learning pipeline designed to automate the process of training, evaluating, and validating classification models. It provides a robust framework for feature selection, preprocessing, hyperparameter tuning, and model evaluation, ensuring efficient and accurate model performance.

## Contents
- [Context](#context)
- [Requirements](#requirements)
- [Docker Image](#Docker-Image-Available-Publicly)
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

## Run as Docker Image
You can pull Directly the Docker Image as   
```bash
docker pull dimzaridis/simplatab-machine-learning-automator:1.0.0-TestVersion
```
and run the docker image as  
```bash
sudo docker run -v /absolute_path/to/your/input_data:/input_data  
                -v /absolute_path/to/your/Empty/Folder/For_the/outcomes/to_be/stored:/Materials  
                dimzaridis/simplatab-machine-learning-automator:1.0.0-TestVersion (or latest)
```
## Run as Python API
1. Clone the repository to the desired folder  
```bash
git clone https://github.com/dzaridis/SIMPLATAB.git  
```
2. Place your Train.csv and Test.csv into a folder  

3. Create a "Materials"folder in the repository workspace  

4. Install the requirements  
```bash
pip install -r requirements
```
5. Run the python API
```bash
python __main__.py 
```

6. An interactive window to select you folders will be opened. Select input folder and the "Materials" folder you previously created as the output folder  
```bash
python __main__.py 
```

7. Follow the steps in the interactive session to fill in the parameters such as k-fold, grid-search (they are simple just true/false values)

8. Wait for the execution to finalize

### Requirements for the Data
1. The input folder must contain a Train.csv and a Test.csv files  
2. The **CSVs** MUST not contain missing values  
3. The target column should be   
   a. **BINARY** with values **0 and 1**  
   b. it should have the name **Target**  
   c. It should contain a column named **"patient_id"** (It will change in the future)
## Inputs
- **CSV Files**: Must not contain missing values.
  - **Train.csv**: Features and target (last column). Used for K-Fold cross-validation and threshold tuning.
  - **Test.csv**: Features and target (last column). Used for validation of thresholds, metrics, ROC curves, and SHAP analysis.

For the Docker version, place `Train.csv` and `Test.csv` in an input volume folder.

## Outputs
The outputs will be saved in the `Materials` folder (or a specified output volume for Docker version):
- `ROC_CURVES.png`: ROC curves for each algorithm on the test set.
- `Precision-Recall curves.png`:Precision-Recall curves for each algorithm on the test set.
- `ShapFeatures` folder: A ShapFeatures folder will be created, Inside model subfolders will be created which contain 3 kind of plots  
    - `Summary Plot`: Top 10 features and their impact on model output
    - `BeeSwarm Plot`: Similar to summary plot but also takes into account the sum of the shap values for all features not just the top 10
    - `Heatmap Plot`: Contains information regarding the impact of each feature (top 10 and the rest as a sum) and how they impact the probabilities of the model's outcome
- Excel files:
  - Metrics for the algorithm on the internal K-Fold.
  - Metrics for the algorithm on the external set.
- `Models` folder: Pickle files containing the models evaluated on the external data. These pipelines can be used directly without manual feature selection or preprocessing.
- `Confusion_Matrices` folder: The confusion matrices for each model on the internal k-fold (mena values of tp, fp, tn , fn) and external set are provided as images

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
  - Balanced Accuracy
- ROC Curves and SHAP Analysis on the external set.

Shap Analysis consists of 3 plots (summary plot, beeswarm, heatmap)

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


## Authors
Main Work Implemented by:  
- **Dimitrios Zaridis** (corresponding), M.Eng, PhD Student @ National Technical University of Athens
- **Eugenia Mylona**, Ph.D
- **Vasileios B. Pezoulas**, Ph.D

Assistance by:  
- **Charalampos Kalantzopoulos**, M.Sc
- **Nikolaos S. Tachos**, Ph.D
- **Dimitrios I. Fotiadis**, Professor of Biomedical Technology, University of Ioannina


