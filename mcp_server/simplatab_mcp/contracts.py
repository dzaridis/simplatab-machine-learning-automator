"""Data contracts of the automators: what the training and test data must look like, how they are
checked, the configuration an experiment accepts and what it produces.

``AUTOMATORS`` is static (the server shows it without starting a worker); ``contract()`` adds the
networks of each automator from the Simplatab registries and the configuration schema, so it runs
in the worker. Python 3.9 compatible.
"""
AUTOMATORS = {
    "tabular": {
        "name": "Tabular Classification",
        "task": "Binary or multiclass classification of rows of a table.",
        "train": "Train.csv: a CSV file",
        "test": "Test.csv: a CSV file with the same columns",
        "model_summary": "7 classical (Logistic Regression, SVM, Random Forest, SGD, MLP, Decision Tree, XGBoost) and "
                  "4 deep learning (TabPFNv2, TabICL, TabTransformer, TabR)",
        "validation": "Stratified K-fold cross-validation, decision threshold tuned on the validation folds",
        "explanations": "SHAP",
    },
    "image-classification": {
        "name": "Image Classification (2D and 3D)",
        "task": "Classification of 2D images, or of 3D studies made of one or more series (e.g. T2 + ADC MRI).",
        "train": "Train.zip (or a folder): one sub-folder per class",
        "test": "Test.zip (or a folder) with the same class folders",
        "model_summary": "10 pretrained 2D networks; 18 3D networks (pretrained and trained from scratch)",
        "validation": "Stratified K-fold cross-validation (grouped by patient in 3D)",
        "explanations": "Grad-CAM (3D Grad-CAM for volumes)",
    },
    "object-detection": {
        "name": "Object Detection (2D and 3D)",
        "task": "Locating and classifying objects with boxes in 2D images or 3D volumes.",
        "train": "Train.zip (or a folder): images and boxes (COCO, YOLO, Pascal VOC, CSV or masks)",
        "test": "Test.zip (or a folder) in the same format",
        "model_summary": "10 COCO-pretrained detectors (torchvision and detection transformers)",
        "validation": "K-fold cross-validation grouped by patient folder, or hold-out",
        "explanations": "D-RISE saliency maps",
    },
    "image-segmentation": {
        "name": "Image Segmentation (2D and 3D)",
        "task": "Pixel/voxel-wise delineation of structures in 2D images or 3D volumes.",
        "train": "Train.zip (or a folder): images/ and masks/ (or the nnU-Net raw layout)",
        "test": "Test.zip (or a folder) with the same layout",
        "model_summary": "The official nnU-Net v2 (2D, 3D full resolution) and 18 U-Net-family networks",
        "validation": "K-fold cross-validation grouped by patient, or hold-out",
        "explanations": "Uncertainty maps (test-time augmentation)",
    },
    "time-series-forecasting": {
        "name": "Time Series Forecasting",
        "task": "Forecasting the next values of many time series (e.g. repeated measurements of patients).",
        "train": "Train.csv in long format (ID, Time, Target and covariates)",
        "test": "Test.csv in the same format",
        "model_summary": "10 neuralforecast networks (NHITS, NBEATSx, TiDE, KAN, DLinear, TFT, PatchTST, BiTCN, TCN, TimesNet)",
        "validation": "Rolling-origin (prequential) validation",
        "explanations": "Integrated gradients",
    },
    "survival-analysis": {
        "name": "Survival Analysis",
        "task": "Time-to-event prediction with censored follow-up (e.g. overall survival, time to relapse).",
        "train": "Train.csv: a CSV file with Time, Event and the features",
        "test": "Test.csv: a CSV file with the same columns",
        "model_summary": "8 models: Cox PH, Weibull AFT, log-normal AFT, XGBoost Cox, XGBoost AFT, DeepSurv, DeepHit, "
                         "Logistic-Hazard (+ a Kaplan-Meier reference)",
        "validation": "K-fold cross-validation stratified on Event, then the external test set",
        "explanations": "Risk groups (Kaplan-Meier, log-rank), calibration, permutation importance, Cox/AFT coefficients",
    },
    "clustering": {
        "name": "Clustering",
        "task": "Finding groups of similar rows in a table, unsupervised; with a Target column of classes the clusters "
                "are also evaluated against them (supervised evaluation; the labels are never used to fit).",
        "train": "Train.csv: a CSV file, one row per sample",
        "test": "Optional Test.csv with the same feature columns (its samples are assigned to the clusters)",
        "model_summary": "12 classical (K-Means, Bisecting K-Means, Gaussian mixture, Bayesian GMM, Ward, BIRCH, spectral, "
                         "affinity propagation, DBSCAN, HDBSCAN, OPTICS, Mean Shift) and 6 deep/neural (DEC, IDEC, DCN, "
                         "VaDE, SCARF + k-means, self-organizing map)",
        "validation": "K-fold: held-out samples assigned by models fitted on the other folds; stability across folds",
        "explanations": "Cluster profiles, PCA/t-SNE projections, SHAP of a surrogate random forest",
    },
}

THRESHOLD_METRICS = ["Balanced Accuracy", "AUC", "F-score", "Accuracy", "Sensitivity", "Specificity"]

TABULAR_MODELS = [
    ("logistic_regression", "Logistic Regression", "Linear baseline with interpretable coefficients.", True),
    ("svm", "Support Vector Machines", "Maximum-margin classifier with linear and kernel variants.", True),
    ("random_forest", "Random Forest", "Ensemble of decision trees, robust to feature scales.", True),
    ("sgd", "Stochastic Gradient Descent", "Regularised linear model trained with stochastic gradient descent.", True),
    ("neural_network", "Multi-Layer Neural Network", "Fully connected neural network (scikit-learn MLP).", True),
    ("decision_trees", "Decision Trees", "Single tree whose decisions can be read directly.", True),
    ("xgboost", "XGBoost", "Gradient-boosted trees, strong on tabular data.", True),
    ("tabpfn", "TabPFNv2", "Pretrained tabular foundation model; up to 10,000 samples, 500 features, 10 classes.", False),
    ("tabtransformer", "TabTransformer", "Transformer contextualising categorical features, trained from scratch.", False),
    ("tabr", "TabR", "Retrieval-augmented network using the nearest training samples.", False),
    ("tabicl", "TabICL", "Pretrained tabular foundation model based on in-context learning.", False),
]

CT_WINDOWS = ["auto", "lung", "soft_tissue", "bone", "brain"]

DATA_RULES = {
    "tabular": {
        "layout": "Two CSV files (comma separated, header row). Give their paths as train and test.",
        "columns": {
            "Target": "Required. Numeric class labels numbered 0, 1, ..., K-1 (binary: 0/1, 1 being the positive class).",
            "ID or patient_id": "Optional identifier column (used as the row index, never as a feature; reported in the splits).",
            "other columns": "Features: numeric or categorical (text). Categorical columns whose values differ between "
                             "Train.csv and Test.csv are dropped.",
        },
        "rules": [
            "Both files have the same columns.",
            "Rows with a missing value are removed.",
            "Every class needs at least K rows in Train.csv for K-fold cross-validation.",
            "TabPFNv2 is limited to 10,000 training rows, 500 features and 10 classes.",
        ],
        "example": "ID,age,sex,biomarker,Target\nP001,54,F,1.25,1\nP002,61,M,0.80,0\n",
    },
    "image-classification": {
        "layout": "A zip file (or a folder) per split with one folder per class; sub-folders below the class "
                  "(e.g. one per patient) are allowed. 2D: every image file is one sample. 3D: "
                  "<class>/<patient>/<study>/<series>, a series being a folder of DICOM slices or a NIfTI / multi-frame "
                  "DICOM file; the study level is optional.",
        "tree_2d": "Train.zip\n  benign/\n    patient_01/img_001.png\n    img_002.dcm\n  malignant/\n    scan_17.nii.gz\n",
        "tree_3d": "Train.zip\n  benign/\n    patient_01/\n      study_1/\n        t2/        (DICOM slices)\n        adc.nii.gz\n"
                   "  malignant/\n    patient_02/study_1/t2/ ...\n",
        "formats": "DICOM (compressed, multi-frame, colour, without extension), NIfTI (.nii, .nii.gz), PNG (8/16-bit), "
                   "JPEG, BMP, TIFF.",
        "rules": [
            "The class folder names are the classes; Test.zip uses the same class names.",
            "Zips of DICOM series / NIfTI volumes only (no PNG/JPEG) are classified in 3D by default (config dim=2 "
            "classifies them one image per file).",
            "3D: the series with the same name in every study (e.g. t2, adc, dwi) are the input channels, aligned in "
            "patient coordinates on the first (reference) series.",
            "Every class needs at least K images (3D: K patients) in Train.zip for K-fold cross-validation.",
            "Up to 5 GB per zip.",
        ],
    },
    "object-detection": {
        "layout": "A zip file (or a folder) per split with the images and the boxes, in one of these formats "
                  "(detected automatically; boxes in pixels of the original image):",
        "formats": {
            "COCO JSON": "annotations.json (images, annotations with bbox = [x, y, width, height], categories) and the images",
            "YOLO": "images/... and labels/... (class cx cy w h, normalised), class names in classes.txt or data.yaml",
            "Pascal VOC": "one .xml file next to each image",
            "CSV": "image,class,x_min,y_min,x_max,y_max (+ z_min,z_max: first and last slice for 3D boxes)",
            "Masks": "masks/<image name> label images (PNG or NIfTI); each connected region of a label is a box; names in classes.txt",
        },
        "rules": [
            "2D: DICOM, NIfTI, PNG (8/16-bit), JPEG, BMP, TIFF. 3D: NIfTI volumes, multi-frame DICOM or a folder of DICOM "
            "slices per series, with CSV boxes or NIfTI masks.",
            "Images without boxes (or listed in the CSV without a box) are negatives.",
            "Sub-folders (one per patient) keep a patient's images in the same fold.",
        ],
        "example": "annotations.csv\nimage,class,x_min,y_min,x_max,y_max\npatient_01/img_001.png,nodule,34,50,61,78\n",
    },
    "image-segmentation": {
        "layout": "A zip file (or a folder) per split, in one of two layouts.",
        "tree_folders": "Train.zip\n  images/case_01.png            masks/case_01.png      (2D)\n"
                        "  images/patient_07/scan.dcm     masks/patient_07/scan.png\n"
                        "  images/case_02.nii.gz          masks/case_02.nii.gz   (3D)\n"
                        "  images/case_03/t2/ (DICOM) + images/case_03/adc.nii.gz   masks/case_03.nii.gz\n"
                        "  labels.json                    {\"1\": \"liver\", \"2\": \"tumour\"}\n",
        "tree_nnunet": "Train.zip\n  imagesTr/case_0000.nii.gz  imagesTr/case_0001.nii.gz   (one file per channel)\n"
                       "  labelsTr/case.nii.gz\n  dataset.json\n(Test.zip: imagesTs/ and labelsTs/)\n",
        "rules": [
            "Mask paths mirror the image paths (suffixes _mask, _seg, _label, ... are accepted).",
            "Masks are label images (0 = background; 0/255 binary masks read as 0/1), palette PNGs or colour (RGB) masks.",
            "Class names: labels.json ({\"1\": \"liver\"}), classes.txt (\"1,liver\") or dataset.json; else class_<value>.",
            "A sub-folder per patient keeps a patient's cases in the same fold.",
            "3D: the series of a case are aligned in patient coordinates on the reference (first) series; the mask is "
            "put on its grid.",
            "Up to 32 classes.",
        ],
    },
    "time-series-forecasting": {
        "layout": "Two CSV files in long format: one row per series and time point.",
        "columns": {
            "ID (or patient_id)": "Required. The series.",
            "Time": "Required. Dates at a regular frequency or integer steps; missing points are filled in.",
            "Target": "Required. The numeric value to forecast.",
            "other columns": "Covariates: constant within a series -> static (categorical ones one-hot encoded); varying -> "
                             "known in advance (future_columns of the configuration, e.g. a scheduled dose) or observed up "
                             "to now (past covariates, e.g. another measurement).",
        },
        "rules": [
            "Test.csv: the last H points (the horizon) of every series are forecast from the points before them.",
            "A Test.csv series whose ID is in Train.csv continues it (Test.csv may hold only its next H points); a new ID "
            "is an unseen series with its own history.",
            "Every training series needs at least two horizons of points.",
        ],
        "example": "ID,Time,Target,Dose,Sex\nP01,2024-01-01,5.4,10,F\nP01,2024-01-02,5.9,10,F\n",
    },
    "survival-analysis": {
        "layout": "Two CSV files (comma separated, header row), one row per patient.",
        "columns": {
            "Time": "Required. Follow-up time: of the event, or of the last contact when censored. Positive, any unit "
                    "(the same in both files; horizons are in this unit).",
            "Event": "Required. 1 if the event happened at Time, 0 if censored.",
            "ID or patient_id": "Optional identifier (never a feature).",
            "other columns": "Features: numeric or categorical (one-hot encoded); missing values imputed inside each fold.",
        },
        "rules": ["Both files have the same feature columns.", "Train.csv needs at least 10 events.",
                  "Horizons must lie within the follow-up (0 < h < longest Time)."],
        "example": "ID,Time,Event,Age,Stage,CEA\nCRC0001,31.2,1,73,III,12.8\nCRC0002,58.0,0,58,II,3.1\n",
    },
    "clustering": {
        "layout": "One CSV file (Train.csv) with one row per sample, and optionally a Test.csv with the same feature "
                  "columns. Give test=null (or leave it out) when there is no test set.",
        "columns": {
            "ID or patient_id": "Optional identifier (never a feature; reported in the cluster tables and the splits).",
            "Target": "Optional class labels (numbers or text). Present: supervised evaluation (ARI, AMI, NMI, purity, "
                      "matched accuracy...) and n_clusters can be the number of classes. Absent: unsupervised clustering. "
                      "The labels never influence the clusters. A Target of continuous values is left out.",
            "other columns": "Features: numeric or categorical (text, one-hot encoded). Missing values are imputed "
                             "(median, or a 'missing' category). Constant, mostly missing (>50%) or identifier-like text "
                             "columns are left out; ignore_columns leaves out others.",
        },
        "rules": [
            "At least 10 rows and at most 100,000.",
            "Test.csv (optional) has every feature column of Train.csv; its Target (optional) evaluates its clusters.",
            "Some algorithms are limited in rows: agglomerative 20,000, spectral 10,000, affinity propagation 5,000, "
            "OPTICS and Mean Shift 20,000, DBSCAN and HDBSCAN 50,000.",
            "Cluster numbers: 0 is the largest cluster of Train.csv; -1 is noise (DBSCAN, HDBSCAN, OPTICS).",
        ],
        "example": "ID,Age,BMI,HbA1c,GADA,Target\nP0001,51,34.4,54,Negative,MOD\nP0002,72,30.7,77,Negative,SIDD\n",
    },
}

OUTPUTS = {
    "common": [
        "<K>_fold_results.xlsx (mean ± SD over the folds) or holdout_results.xlsx: validation metrics per model",
        "test_results.xlsx: metrics on the external test set",
        "Splits/splits.csv and Splits/splits.json: the samples of every fold (to reproduce the validation)",
        "run_info.json: the settings of the run",
        "error_log.log: the errors of models that were skipped",
    ],
    "tabular": ["Models/<model>_pipeline.pkl, Models/thresholds.json", "ROC_Curves/, ConfusionMatrices/, Shap_Features/<model>/"],
    "image-classification": ["Models/<network>.pt (TorchScript)", "Predictions/<network>_test_predictions.csv",
                             "ROC_Curves/, ConfusionMatrices/, GradCAM/<network>/", "classes.csv"],
    "object-detection": ["Models/<network>.pt or .zip", "Predictions/<network>_test_predictions.csv",
                         "Detection_Curves/, Detections/, Explainability/ (D-RISE)"],
    "image-segmentation": ["Models/<network>.pt (TorchScript) or .zip (nnU-Net model folder)",
                           "Predictions/<network>/<case>.png|.nii.gz (predicted masks, original mask values)",
                           "Segmentation_Metrics/test_per_class.csv, test_per_case.csv",
                           "Segmentation_Plots/, Overlays/<network>/ (with the uncertainty map)"],
    "time-series-forecasting": ["Models/<model>.zip (neuralforecast)", "Forecasts/test_forecasts.csv, future_forecasts.csv",
                                "Forecast_Plots/, Metrics_Plots/, Explainability/ (integrated gradients)"],
    "survival-analysis": ["Models/<model>.pkl (cloudpickle: predict_risk(df), predict_survival(df, times))",
                          "Predictions/test_predictions.csv (risk and survival at the horizons per test patient)",
                          "Survival_Plots/ (risk groups, patient curves), Metrics_Plots/ (AUC and Brier over time, "
                          "calibration), Explainability/ (permutation importance, Cox/AFT coefficients)"],
    "clustering": ["train_results.xlsx: metrics of the clusters of Train.csv (test_results.xlsx with a Test.csv)",
                   "Models/<algorithm>.pkl (cloudpickle: model.predict(dataframe) -> cluster; no Simplatab needed)",
                   "Clusters/train_clusters.csv, test_clusters.csv (cluster of every sample per algorithm), "
                   "validation_folds.csv, k_selection.csv",
                   "Cluster_Profiles/<algorithm>_profile.csv|png, Embeddings/ (PCA, t-SNE), "
                   "Metrics_Plots/ (contingency, silhouette, k selection), Explainability/ (SHAP)"],
}

METRICS = {
    "tabular": ["AUC", "Balanced Accuracy", "F-score", "Accuracy", "Sensitivity", "Specificity"],
    "image-classification": ["AUC", "Balanced Accuracy", "F-score", "Accuracy", "Sensitivity", "Specificity"],
    "object-detection": ["mAP", "AP50/AP75 (3D: AP10/AP25/AP50)", "AR", "FROC CPM", "precision, recall, F1 at the threshold",
                         "image-level sensitivity/specificity"],
    "image-segmentation": ["Dice", "IoU", "HD95 (lower is better)", "ASSD (lower is better)", "Sensitivity", "Precision"],
    "time-series-forecasting": ["MAE", "RMSE", "sMAPE", "MASE (all lower is better; vs. a seasonal naive baseline)"],
    "survival-analysis": ["C-index", "Uno C-index", "IBS (lower is better)", "AUC@<horizon>",
                          "Brier@<horizon> (lower is better)"],
    "clustering": ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin (lower is better)", "with a Target: ARI, AMI, NMI, "
                   "V-measure, Homogeneity, Completeness, FMI, Purity, Accuracy (Hungarian matching)",
                   "Stability (ARI) on the validation folds", "Clusters, Noise %"],
}


def _field(kind, default, description, **extra):
    spec = {"type": kind, "default": default, "description": description}
    spec.update(extra)
    return spec


def models(automator, dim=None):
    """The models an experiment can select: [{key, name, description, default}]."""
    if automator == "tabular":
        return [{"key": k, "name": n, "description": d, "default": dflt} for k, n, d, dflt in TABULAR_MODELS]
    if automator == "image-classification":
        if dim == 3:
            from Helpers.image3d.models import NETWORKS
            return [{"key": n.key, "name": n.name, "description": n.description, "default": n.default,
                     "pretrained_on": n.weights, "family": n.family} for n in NETWORKS]
        from Helpers.image.models import BACKBONES
        return [{"key": b.key, "name": b.name, "description": b.description, "default": b.default, "family": b.family}
                for b in BACKBONES]
    if automator == "object-detection":
        from Helpers.detection.models import DETECTORS
        return [{"key": d.key, "name": d.name, "description": d.description, "default": d.default, "light": d.light,
                 "library": d.library} for d in DETECTORS]
    if automator == "image-segmentation":
        from Helpers.segmentation.models import NETWORKS
        return [{"key": n.key, "name": n.name, "description": n.description, "default": n.default, "dim": n.dim,
                 "family": n.family, "light": n.light} for n in NETWORKS if dim is None or n.dim == dim]
    if automator == "time-series-forecasting":
        from Helpers.forecasting.models import MODELS
        return [{"key": m.key, "name": m.key, "description": m.description, "default": m.default, "slow": m.slow}
                for m in MODELS]
    if automator == "survival-analysis":
        from Helpers.survival.models import MODELS
        return [{"key": m.key, "name": m.name, "description": m.description, "default": m.default, "family": m.family}
                for m in MODELS]
    if automator == "clustering":
        from Helpers.clustering.models import ALGORITHMS
        return [{"key": a.key, "name": a.name, "description": a.description, "default": a.default, "family": a.family,
                 "takes_n_clusters": a.uses_k, "leaves_noise": a.noise, "max_rows": a.max_samples, "slow": a.slow}
                for a in ALGORITHMS]
    raise KeyError(automator)


def config_schema(automator, dim=None):
    """The configuration of an experiment: {field: {type, default, description, choices/min/max}}.
    Defaults marked "from the data" are set by inspect/create from the data summary."""
    if automator == "tabular":
        return {
            "models": _field("list[str]", [k for k, _, _, d in TABULAR_MODELS if d], "Model keys (see models).",
                             choices=[k for k, *_ in TABULAR_MODELS]),
            "k_folds": _field("int", 5, "Folds of the stratified cross-validation.", min=2, max=20),
            "threshold_metric": _field("str", "Balanced Accuracy", "Metric the decision threshold maximises on the "
                                       "validation folds (binary problems).", choices=THRESHOLD_METRICS),
            "hyperparameter_search": _field("str", "randomized", "Hyperparameter search with cross-validation.",
                                            choices=["none", "randomized", "exhaustive"]),
            "correlation_limit": _field("float", 0.7, "Features correlated above this are dropped (featurewiz).",
                                        min=0.1, max=1.0),
            "bias_feature": _field("str|null", None, "A column to assess data bias on (e.g. sex); null: no assessment."),
        }
    if automator == "image-classification" and dim == 3:
        from Helpers.image3d.volumes import CROPS, SHAPES
        return {
            "dim": _field("int", 3, "3: 3D studies; 2: one image per file.", choices=[2, 3]),
            "models": _field("list[str]", "from the registry defaults", "Network keys (see models)."),
            "mode": _field("str", "features", "features: frozen network + logistic regression (fast, CPU); finetune: "
                           "full training (GPU recommended).", choices=["features", "finetune"]),
            "k_folds": _field("int", 5, "Folds (grouped by patient).", min=2, max="patients of the smallest class"),
            "threshold_metric": _field("str", "Balanced Accuracy", "Metric of the decision threshold.", choices=THRESHOLD_METRICS),
            "positive_class": _field("str", "from the data", "Binary problems: the class to detect."),
            "channels": _field("list[str]", "from the data", "Series used as channels, the reference (grid) first."),
            "shape": _field("list[int]", list(SHAPES[0]), "Volume size (slices, rows, columns).",
                            choices=[list(s) for s in SHAPES]),
            "crop": _field("float", 1.0, "Central part of the in-plane field of view kept.", choices=list(CROPS)),
            "window": _field("str", "auto", "CT window, or auto (DICOM header).", choices=CT_WINDOWS),
            "augmentation": _field("dict", {"horizontal_flip": False, "vertical_flip": False, "rotation": True,
                                            "intensity": True}, "Fine-tuning augmentations."),
            "epochs": _field("int", 30, "Fine-tuning epochs (early stopping).", min=1, max=200),
            "learning_rate": _field("float", 1e-4, "Fine-tuning learning rate.", min=1e-6, max=1e-2),
            "patience": _field("int", 8, "Early-stopping patience (epochs).", min=1, max=50),
            "batch_size": _field("int", 4, "Batch size.", min=1, max=64),
        }
    if automator == "image-classification":
        return {
            "dim": _field("int", 2, "2: one image per file; 3: 3D studies (zips of DICOM series / NIfTI only).", choices=[2, 3]),
            "models": _field("list[str]", "from the registry defaults", "Network keys (see models)."),
            "mode": _field("str", "features", "features: frozen network + logistic regression (fast, CPU); finetune: "
                           "full training (GPU recommended).", choices=["features", "finetune"]),
            "k_folds": _field("int", 5, "Folds of the stratified cross-validation.", min=2, max="images of the smallest class"),
            "threshold_metric": _field("str", "Balanced Accuracy", "Metric of the decision threshold.", choices=THRESHOLD_METRICS),
            "positive_class": _field("str", "from the data", "Binary problems: the class to detect."),
            "window": _field("str", "auto", "CT window for CT DICOM/NIfTI, or auto (DICOM header).", choices=CT_WINDOWS),
            "volume": _field("str", "middle", "3D files classified in 2D: middle slice or maximum intensity projection.",
                             choices=["middle", "mip"]),
            "augmentation": _field("dict", {"horizontal_flip": True, "vertical_flip": False, "rotation": True,
                                            "intensity": True}, "Fine-tuning augmentations."),
            "epochs": _field("int", 20, "Fine-tuning epochs (early stopping).", min=1, max=200),
            "learning_rate": _field("float", 1e-4, "Fine-tuning learning rate.", min=1e-6, max=1e-2),
            "patience": _field("int", 5, "Early-stopping patience (epochs).", min=1, max=50),
            "batch_size": _field("int", 32, "Batch size.", min=1, max=256),
        }
    if automator == "object-detection":
        return {
            "models": _field("list[str]", "from the registry defaults", "Detector keys (see models)."),
            "validation": _field("str", "kfold", "kfold, or holdout (faster: one training per network).",
                                 choices=["kfold", "holdout"]),
            "k_folds": _field("int", 5, "Folds (grouped by patient folder).", min=2, max="patient groups"),
            "holdout_fraction": _field("float", 0.2, "Hold-out: share of Train kept for validation.", min=0.1, max=0.4),
            "epochs": _field("int", 30, "Maximum epochs (early stopping).", min=1, max=300),
            "patience": _field("int", 5, "Early-stopping patience.", min=1, max=50),
            "batch_size": _field("int", 4, "Batch size.", min=1, max=64),
            "image_size": _field("int", 640, "Images resized to a square of this side.", choices=[320, 512, 640, 800, 1024]),
            "lr_scale": _field("float", 1.0, "Multiplier of each detector's learning rate.", min=0.1, max=10),
            "augmentation": _field("dict", {"horizontal_flip": True, "vertical_flip": False, "intensity": True},
                                   "Training augmentations."),
            "window": _field("str", "auto", "CT window, or auto (DICOM header).", choices=CT_WINDOWS),
            "negative_ratio": _field("float", 1.0, "Images without boxes per image with boxes in training.", min=0, max=5),
            "drise_images": _field("int", 4, "Test images explained with D-RISE (0: none).", min=0, max=12),
            "drise_masks": _field("int", 300, "Random masks per D-RISE explanation.", min=50, max=2000),
        }
    if automator == "image-segmentation":
        return {
            "models": _field("list[str]", "from the registry defaults", "Network keys for the dimension of the data (see models)."),
            "validation": _field("str", "kfold", "kfold, or holdout (faster).", choices=["kfold", "holdout"]),
            "k_folds": _field("int", 5, "Folds (grouped by patient).", min=2, max="patient groups"),
            "holdout_fraction": _field("float", 0.2, "Hold-out: share of Train kept for validation.", min=0.1, max=0.4),
            "channels": _field("list[str]", "from the data", "3D cases with several series: the series used, the "
                               "reference (grid of input and masks) first."),
            "epochs": _field("int", 20, "Epochs of the automator's networks (fixed number of iterations each).", min=1, max=1000),
            "iterations": _field("int", 50, "Iterations per epoch.", min=1, max=1000),
            "batch_size": _field("int", "8 (2D) / 2 (3D)", "Batch size.", min=1, max=64),
            "learning_rate": _field("float", 1e-3, "Learning rate (pretrained encoders use 30% of it).", min=1e-5, max=1e-2),
            "augmentation": _field("dict", {"rotation": True, "intensity": True, "horizontal_flip": "colour photos only",
                                            "vertical_flip": False, "depth_flip": False}, "Training augmentations."),
            "normalisation": _field("str", "auto", "auto (CT for CT series, else z-score), ct or zscore.",
                                    choices=["auto", "ct", "zscore"]),
            "tta": _field("bool", True, "Test-time flips and uncertainty maps."),
            "nnunet_epochs": _field("int", 100, "nnU-Net epochs (its default is 1000).", min=1, max=1000),
            "nnunet_iterations": _field("int", 250, "nnU-Net iterations per epoch.", min=1, max=250),
        }
    if automator == "time-series-forecasting":
        return {
            "models": _field("list[str]", "from the registry defaults", "Model keys (see models)."),
            "horizon": _field("int", "from the data", "Points forecast.", min=1, max="from the data"),
            "k_folds": _field("int", "min(3, max)", "Rolling-origin windows.", min=1, max="from the data"),
            "lookback": _field("int", 0, "Input window; 0: chosen on the validation windows.", min=0),
            "trials": _field("int", 0, "Extra hyperparameter configurations tried per model.", min=0, max=50),
            "max_steps": _field("int", 500, "Training steps.", min=50, max=10000),
            "season": _field("int", "from the data", "Seasonal period (seasonal naive baseline, MASE).", min=1, max=1000),
            "future_columns": _field("list[str]", [], "Varying covariates known in advance (the others are past covariates)."),
        }
    if automator == "survival-analysis":
        return {
            "models": _field("list[str]", "from the registry defaults", "Model keys (see models), or [\"all\"]."),
            "k_folds": _field("int", 5, "Folds stratified on Event.", min=2, max="from the data (max_folds)"),
            "horizons": _field("list[float]", "quartiles of the event times", "1 to 5 times for AUC@t, Brier@t and "
                               "the predicted survival probabilities.", min=0, max="the longest follow-up"),
            "selection_metric": _field("str", "C-index", "Metric choosing the best model (validation mean).",
                                       choices=["C-index", "Uno C-index", "IBS"]),
            "ignore_columns": _field("list[str]", [], "Feature columns left out."),
            "penalty": _field("float", 0.01, "Ridge penalty of Cox and AFT models.", min=0, max=10),
            "epochs": _field("int", 200, "Maximum epochs of the networks (early stopping).", min=20, max=1000),
            "explain": _field("bool", True, "Permutation importance on Test.csv."),
            "seed": _field("int", 42, "Random seed.", min=0, max=2 ** 31 - 1),
        }
    if automator == "clustering":
        return {
            "models": _field("list[str]", "from the registry defaults", "Algorithm keys (see models), or [\"all\"]."),
            "n_clusters": _field("int|str", "\"classes\" with a Target, else \"auto\"",
                                 "For the algorithms that take k: an integer, \"classes\" (number of Target classes) or "
                                 "\"auto\" (every k from k_min to k_max tried, the best by k_criterion kept). The other "
                                 "algorithms find it themselves (their settings tuned by k_criterion)."),
            "k_min": _field("int", 2, "Smallest k tried (auto).", min=2, max="from the data"),
            "k_max": _field("int", 10, "Largest k tried (auto).", min=2, max="from the data (max_k)"),
            "k_criterion": _field("str", "silhouette", "Criterion choosing k and the settings of the self-sizing algorithms.",
                                  choices=["silhouette", "calinski_harabasz", "davies_bouldin"]),
            "validation": _field("str", "kfold", "kfold (refit on K-1 folds, held-out fold assigned) or none.",
                                 choices=["kfold", "none"]),
            "k_folds": _field("int", 5, "Folds (stratified by Target when present).", min=2, max="from the data (max_folds)"),
            "selection_metric": _field("str", "\"ARI\" with a Target, else \"Silhouette\"",
                                       "Metric choosing the best algorithm (validation mean, else test, else train).",
                                       choices=["Silhouette", "Calinski-Harabasz", "Davies-Bouldin", "Stability (ARI)",
                                                "ARI", "AMI", "NMI", "V-measure", "Homogeneity", "Completeness", "FMI",
                                                "Purity", "Accuracy"]),
            "scaling": _field("str", "standard", "Scaling of the numeric features.", choices=["standard", "robust", "minmax", "none"]),
            "reduction": _field("str", "none", "Dimensionality reduction before clustering.", choices=["none", "pca"]),
            "pca_variance": _field("float", 0.95, "Share of the variance kept by the PCA.", min=0.5, max=0.99),
            "ignore_columns": _field("list[str]", [], "Feature columns left out of the clustering."),
            "pretrain_epochs": _field("int", 100, "Deep networks: autoencoder / contrastive pretraining epochs.", min=10, max=1000),
            "epochs": _field("int", 100, "Deep networks: clustering epochs (early stop when stable).", min=10, max=1000),
            "latent_dim": _field("int", 10, "Deep networks: latent dimensions.", min=2, max=64),
            "explain": _field("bool", True, "SHAP explanations of the clusters (surrogate random forest)."),
            "tsne": _field("bool", True, "t-SNE projection figures (next to PCA)."),
            "seed": _field("int", 42, "Random seed.", min=0, max=2 ** 31 - 1),
        }
    raise KeyError(automator)


def contract(automator, dim=None):
    """Everything an agent needs to prepare data and configure an experiment of an automator."""
    if automator not in AUTOMATORS:
        raise KeyError(f"Unknown automator {automator!r}: one of {', '.join(AUTOMATORS)}.")
    dims = [2, 3] if automator in ("image-classification", "image-segmentation") and dim is None else [dim]
    entry = dict(AUTOMATORS[automator], id=automator, data=DATA_RULES[automator],
                 metrics=METRICS[automator], outputs=OUTPUTS["common"] + OUTPUTS[automator])
    if automator == "image-classification":
        entry["config"] = {f"{d}d": config_schema(automator, d) for d in dims}
        entry["models_by_dim"] = {f"{d}d": models(automator, d) for d in dims}
    elif automator == "image-segmentation":
        entry["config"] = config_schema(automator)
        entry["models_by_dim"] = {f"{d}d": models(automator, d) for d in dims}
    else:
        entry["config"] = config_schema(automator)
        entry["models"] = models(automator)
    if automator == "clustering":
        entry["note"] = ("test is optional for clustering. With a Target column the run is a supervised evaluation of "
                         "the clusters; without it, an unsupervised clustering.")
    entry["workflow"] = [
        "1. Prepare Train and Test as described in data (a zip, folder or CSV path the server can read, "
        "or upload_file).",
        "2. create_experiment(automator, train, test): checks the data and returns its summary, errors, warnings and "
        "the default configuration.",
        "3. start_experiment(experiment_id, config): any field left out keeps its default.",
        "4. get_experiment(experiment_id) until the state is completed (or failed), then get_results.",
    ]
    return entry
