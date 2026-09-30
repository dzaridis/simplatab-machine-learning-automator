
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
- **Guided Web Interface**: Upload with instant checks, explained settings, live progress and a results dashboard


## Getting Started

You can run the Machine Learning Automator using either Docker or as a standalone Python application.

### Option 1: Using Docker by Pulling the Image (Recommended)
--- 
**Just Pull the Image and run it :)**

#### Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop) installed on your system

The image is published for `linux/amd64` (Linux, Windows, Intel Macs) and `linux/arm64` (Apple Silicon Macs,
ARM Linux): Docker pulls the one matching your machine.

#### Steps

1. Pull the Image
```bash
docker pull dimzaridis/simplatab-machine-learning-automator:latest
```
---
2. Run the Docker Image
```bash
docker run -p 7111:5000 dimzaridis/simplatab-machine-learning-automator:latest
```
To run a specific version, replace `latest` with a version from the
[Releases](https://github.com/dzaridis/simplatab-machine-learning-automator/releases) page (e.g. `1.1.1`).
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
docker run -p 7111:5000 simplatab
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

## Versions and Releases

Every push to `main` whose tests pass is released automatically by the CI (`.github/workflows/cicd.yaml`):

1. The version is the latest release version plus `0.0.1` (e.g. `1.1.1` → `1.1.2`), computed by
   [`ci/next_version.sh`](ci/next_version.sh).
2. The Docker image is built for `linux/amd64` and `linux/arm64` and published as
   `dimzaridis/simplatab-machine-learning-automator:<version>` and `:latest`.
3. A Git tag and a GitHub release with the same `<version>` are created, with notes listing the merged changes.

The version is shown at the bottom of the web interface (`dev` when running from source). To move to a new
minor or major version, create a tag such as `1.2.0` on `main` (for example with a release on GitHub): the next
push to `main` is then released as `1.2.1`.

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
The web interface opens on a landing page listing the **automators**. **Tabular Classification** is available
today. **Image Classification**, **Image Segmentation** and **Longitudinal Forecasting** are shown as
*Coming soon*, with pages describing the data they will take and what they will produce. The tabular automator
guides you through four steps, shown at the top of every page:

1. **Upload.** Drag and drop (or browse for) `Train.csv` and `Test.csv`. The files are checked in the browser
   before anything is sent: the `Target` column, the class labels, missing values, the columns shared by the two
   files and the categorical values. Problems that would stop the pipeline block the upload; the others are shown
   as warnings. A preview of both files is shown, with the `Target` column first.

2. **Configure.** Every setting explains what it does, and the defaults are a good starting point:
   - **Models:** the classical models (all selected by default) and the deep learning models (TabPFNv2,
     TabTransformer, TabR, TabICL; off by default, see [Deep Learning Models](#deep-learning-models)).
   - **Cross-validation folds:** the number of stratified folds, at most the size of the smallest class.
   - **Hyperparameter search:** on or off; *Randomized* tries 40 combinations per model, *Exhaustive* tries
     them all.
   - **Feature correlation limit:** of two features correlated above this limit, only one is kept.
   - **Metric for threshold optimisation** (binary targets only): the probability threshold is chosen to
     maximise this metric. The default is **Balanced Accuracy**.
   - **Data bias assessment:** checks the outcome balance across the groups of a categorical feature.

   A summary of your data (rows, features, class distribution) stays visible next to the settings.

3. **Run.** The pipeline runs in the background and the page follows it live: the current phase, the
   progress, the state of each model in the K-fold and in the external test, and the log. Models that fail
   are skipped without stopping the run, and the reason is shown. Only one run at a time is possible.

4. **Results.** A dashboard with the best model on the test set, the test and K-fold metrics tables,
   the ROC and precision-recall curves, the confusion matrices and the SHAP plots of each model.
   The trained pipelines and all the output files can be downloaded individually or as one zip archive.

> **Note:** each run replaces the results of the previous one in the `Materials` folder. When results exist,
> the configuration page says so and links to their download.

The interface has light and dark themes (following the system setting by default) and works on small screens.
All its assets are served locally, so it works without internet access.



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


