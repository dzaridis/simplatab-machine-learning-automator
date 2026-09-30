
# SIMPLATAB: **SI**mplified **M**achine **P**ipe**L**ine **A**utomator for **TAB**ular data
![ML Pipeline](static/images_materials/MLPipeline.png)
## Overview

- Simplatab is a comprehensive machine learning pipeline designed to automate the process of data bias detection, training, evaluation, and validatiion of typical ML classification models bundled with XAI shap analysis. It provides a robust framework for bias detection, feature selection, preprocessing, hyperparameter tuning, model evaluation, and XAI analysis ensuring efficient and accurate model performance.

- Overall, Simplatab is a comprehensive platform for automated machine learning pipelines with support for both binary and multiclass classification tasks. This tool simplifies the process of building, training, and evaluating machine learning models through an intuitive web interface.


## Context

Simplatab framework runs a complete Machine Learning Pipeline from **Data Bias assessment** to **model train** and **evaluation** and **XAI analysis** with Shap, for a variety of selectable models.
Please navigate to the [Examples Folder](Example) where examplars Train.csv and Test.csv are given along with the outcomes after the execution of the tool


## Features

- **Automated Machine Learning**: Train and evaluate multiple classification models simultaneously
- **Support for Binary and Multiclass Classification**: Automatically adapts to your dataset
- **Comprehensive Model Evaluation**: ROC curves, precision-recall curves, confusion matrices, and more
- **Feature Importance Analysis**: SHAP-based explainability for all models
- **Deep Learning Classifiers**: TabPFNv2, TabTransformer, TabR and TabICL run through the same pipeline as the classical models
- **Bias Detection**: Identify and assess potential biases in your datasets
- **Model Export**: Save trained models for deployment in other applications


## Getting Started

You can run the Machine Learning Automator using either Docker or as a standalone Python application.

### Option 1: Using Docker by Pulling the Image (Recommended)
--- 
**Just Pull the Image and run it :)**

#### Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop) installed on your system

#### Steps

1. Pull the Image
```bash
docker pull dimzaridis/simplatab-machine-learning-automator:1.0.0
```
---
2. Run the Docker Image
```bash
docker run -p 7111:5000 simplatab-machine-learning-automator:1.0.0
```
---
4. Open browser (Chrome, Mozilla) and Access the web interface at ```http://localhost:7111/automl/```


### Option 2: Using Docker by Building it from repository (Recommended)
--- 
Using Docker is the easiest way to run the application without worrying about dependencies.

#### Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop) installed on your system

#### Steps

1. Clone the Repository
```bash
git clone https://github.com/dzaridis/simplatab-machine-learning-automator.git
cd simplatab-machine-learning-automator
```
---
2. Build the Docker Image
```bash
docker build -t simplatab .
```
---
3. Run the Docker Image
```bash
docker run -p 7111:5000 ml-automator
```

4. Open browser (Chrome, Mozilla) and Access the web interface at ```http://localhost:7111/automl/```

### Option 3: Running as a Python Application
---
#### Steps
1. Clone the Repository
```bash
git clone https://github.com/dzaridis/simplatab-machine-learning-automator.git
cd simplatab-machine-learning-automator
```

2. Create and activate a virtual environment (optional but recommended, Python 3.9 or newer):
```bash
python -m venv simplatab
# On Windows
simplatab\Scripts\activate
# On macOS/Linux
source simplatab/bin/activate
```


3. Install Dependencies
```bash
pip install -r requirements.txt
```

4. Create a folder named "Materials" in the project parent folder
```bash
mkdir Materials
```

5. Run API
```bash
python app.py
```

6. Access the web interface at ```http://localhost:5000/automl/```

## Using the Machine Learning Automator
### Dataset Format
--- 
**Your dataset should be prepared as follows:**

File Format: CSV files named Train.csv and Test.csv
Target Column: A column named **Target** containing:

- For binary classification: Values of 0 and 1 (1 is the positive class)
- For multiclass classification: Consecutive integer class labels starting at 0 (0, 1, 2, ..., K-1)
- Train.csv must contain every class, and Test.csv must only use classes of Train.csv. Other labels
  (e.g. 1/2 or text) must be recoded: the metrics assume this numbering, and the parameters page
  shows a warning when the uploaded Target column does not follow it.


- Features: Any number of numeric or categorical columns

### Step-by-Step Usage
---
1. **Upload Datasets:**

Start by uploading your Train.csv and Test.csv files
The system will automatically detect if your task is binary or multiclass classification


2. Configure Parameters:

**Bias Assessment:** Enable/disable bias detection
**Feature for Bias Assessment:** Select the feature to check for bias (Works only for categorical features, else place None)
**Number of K-Folds:** Set stratified cross-validation folds
**Grid Search:** Enable/disable hyperparameter optimization (If false then the randomized has no impace,If true then by selecting randomized a randomized grid search will be aplied for Hyperparameter tuning)
**Correlation Limit:** Set threshold for feature selection (correlation matrix threshold)
**Models:** Select which machine learning models to train
**Deep Learning Models:** Select which deep learning models to train (TabPFNv2, TabTransformer, TabR, TabICL - disabled by default, see [Deep Learning Models](#deep-learning-models))


3. **Run Pipeline:**

Click "Run Pipeline" to start the automated machine learning process
The system will perform:

- Feature selection
- Model training with cross-validation
- Hyperparameter optimization (if enabled)
- Model evaluation
- SHAP analysis for model interpretability




4. ** View Results:**

Once processing is complete, you'll be redirected to the results page
Review performance metrics, visualizations, and download trained models
For multiclass problems, class-specific metrics and visualizations are provided



## Example Datasets

The repository includes example datasets for both binary and multiclass classification tasks:

examples/binary/Train.csv and examples/binary/Test.csv: Binary classification example (Breast Cancer)  
examples/multiclass/Train.csv and examples/multiclass/Test.csv: Multiclass classification example (IRIS multiclass)  

- Their respective results are located in Examples\BreastCancerExample (binary)
- Examples\IrisExample (Multiclass)

---

## Outputs
The outputs will be saved in the `Materials` folder:
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
- Data Bias Detection
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
- ROC and PR Curves
- SHAP Analysis on the external set to identify significant features for Model's Outcomes.

Shap Analysis consists of 3 plots (summary plot, beeswarm, heatmap)

## Key Concepts
- **Data Bias Detection**: User sets a column of his data to check whether there is a bias in respect to the Target column
- **Hyperparameters**: Set before training to control the behavior of the training algorithm.
- **Cross-validation**: Evaluates model performance by splitting data into multiple folds and training/testing on different combinations.
- **Pipeline**: A sequence of data processing and model training steps applied consistently across all models.
- **XAI Analysis** with Shapley Library

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

## Deep Learning Models
Four deep learning classifiers for tabular data can be selected next to the classical models. They are
scikit-learn compatible estimators (`Helpers/dl_classifiers.py`), so they go through exactly the same flow:
feature selection, preprocessing, (randomized) grid search, K-fold threshold optimization, external test,
ROC/PR curves, SHAP analysis and saved pipelines.

| Model | Type | Reference |
|-------|------|-----------|
| **TabPFNv2** | Pretrained foundation model, in-context learning (no training) | Hollmann et al., *Accurate predictions on small data with a tabular foundation model*, Nature 2025 |
| **TabICL** | Pretrained foundation model, in-context learning (no training) | Qu et al., *TabICL: A Tabular Foundation Model for In-Context Learning on Large Data*, ICML 2025 |
| **TabTransformer** | Trained from scratch | Huang et al., *TabTransformer: Tabular Data Modeling Using Contextual Embeddings*, 2020 |
| **TabR** | Trained from scratch, retrieval-augmented | Gorishniy et al., *TabR: Tabular Deep Learning Meets Nearest Neighbors*, ICLR 2024 |

- **Pretrained weights**: TabPFNv2 ([tabpfn](https://github.com/PriorLabs/TabPFN)) and TabICL ([tabicl](https://github.com/soda-inria/tabicl))
  download their checkpoints from the HuggingFace Hub on first use. The Docker image downloads them at build time.
  For offline machines, place the TabPFN checkpoint in the folder set by `TABPFN_MODEL_CACHE_DIR` and the TabICL
  checkpoint in the HuggingFace cache. TabPFNv2 is limited to 10,000 training samples, 500 features and 10 classes.
- **TabTransformer**: columns with few distinct values (e.g. the one-hot encoded categorical features) are
  categorical tokens contextualized by the transformer; the other columns are continuous. As in the original
  architecture, on datasets without categorical features it reduces to an MLP.
- **TabR**: the training data is the retrieval pool at prediction time, so it is stored in the saved pipeline.
- **Training**: TabTransformer and TabR use AdamW with early stopping on a stratified 15% validation split of the training data.
- **Hardware**: a GPU is used automatically when available. On CPU these models are slower than the classical ones,
  especially with grid search enabled.
- **SHAP**: computed with a Kernel explainer on a bounded budget (14 test samples, 5 background samples,
  150 coalitions per sample) to keep the analysis tractable on CPU.
- A model that cannot run on the dataset (e.g. beyond the TabPFNv2 limits, or its weights cannot be downloaded)
  is skipped and reported in `Materials/error_log.log`; the other models still complete.

## Hyperparameter Tuning & Training
- **Hyperparameter Tuning**: Uses exhaustive grid search to find the best hyperparameters.
  Candidates are scored by their cross-validated AUC (macro one-vs-rest AUC for multiclass targets).
- **Training**: Trains the model on the training data.

## Evaluation
- **Threshold Optimizer**: Finds the optimal threshold on the train set for each fold based on the AUC metric.
- **Metrics**: Evaluates the model on validation data on each fold using:
  - AUC
  - F-Score
  - Accuracy
  - Sensitivity
  - Specificity
  - Balanced Accuracy

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


