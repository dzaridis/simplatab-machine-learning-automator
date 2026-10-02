"""What the web interface offers: the automators and the models of the tabular automator
(the networks of the image automator are in Helpers/image/models.py, the forecasters in
Helpers/forecasting/models.py, the detectors in Helpers/detection/models.py).

Adding an automator means adding an entry to ``AUTOMATORS``: the landing page, the
navigation and the automator pages are generated from it. An automator with
``status="coming_soon"`` gets an information page describing what is planned.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Automator:
    slug: str
    name: str
    tagline: str
    description: str
    icon: str                      # Bootstrap Icons name
    status: str                    # "available" | "coming_soon"
    endpoint: Optional[str] = None  # Flask endpoint of an available automator
    # Landing page card: what you can do with it, the data it takes, its models and explanations
    use_case: str = ""
    data: str = ""
    models_short: str = ""
    explains: str = ""
    inputs: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    @property
    def available(self):
        return self.status == "available"


AUTOMATORS = [
    Automator(
        slug="tabular",
        name="Tabular Classification",
        tagline="Predict a class from table rows",
        description=(
            "Train, validate and explain classical machine learning and deep learning "
            "classifiers on tabular data, from data bias assessment to SHAP explanations."
        ),
        icon="table",
        status="available",
        endpoint="tabular",
        use_case="e.g. a diagnosis or an outcome from clinical variables",
        data="Train.csv + Test.csv",
        models_short="11 models: XGBoost to TabPFNv2",
        explains="SHAP",
        inputs=[
            "Train.csv and Test.csv with the same feature columns",
            "A numeric Target column with classes numbered 0, 1, ..., K-1",
            "Numeric and categorical features, no missing values",
        ],
        steps=[
            "Data bias assessment on a feature of your choice",
            "Automated feature selection and preprocessing",
            "Hyperparameter search with stratified K-fold cross-validation",
            "Decision threshold optimisation on the validation folds",
            "Evaluation on the external test set",
            "SHAP explainability for every model",
        ],
        outputs=[
            "K-fold and test metrics (AUC, F-score, accuracy, sensitivity, specificity, balanced accuracy)",
            "ROC and precision-recall curves, confusion matrices",
            "SHAP feature importance plots",
            "Trained pipelines ready to reuse (.pkl)",
        ],
    ),
    Automator(
        slug="image-classification",
        name="Image Classification",
        tagline="Classify 2D images or 3D scans",
        description=(
            "Train, validate and explain ten state-of-the-art pretrained CNNs and vision transformers "
            "on DICOM, NIfTI, PNG or JPEG images, or eighteen 3D networks on studies made of one or more "
            "series (e.g. T2, ADC and DWI of an MRI), with the same validation standards as the tabular automator."
        ),
        icon="images",
        status="available",
        endpoint="image",
        use_case="e.g. benign vs. malignant on X-ray, CT or MRI",
        data="Zips with one folder per class",
        models_short="10 2D and 18 3D networks",
        explains="Grad-CAM",
        inputs=[
            "Train.zip and Test.zip with one folder per class (up to 5 GB each)",
            "DICOM (including compressed and multi-frame), NIfTI, PNG (8/16-bit), JPEG, BMP, TIFF",
            "3D: class / patient / study / series folders, a series being DICOM slices or a NIfTI file",
            "The test set is provided by you and never used for training",
        ],
        steps=[
            "Medical image preparation: DICOM rescaling, CT windows, volume slices, 16-bit normalisation",
            "Ten pretrained networks: ResNet, EfficientNet(V2), ConvNeXt(V2), ViT, DeiT III, Swin, MaxViT, DINOv2",
            "3D: series aligned in patient space; 18 networks: MedicalNet, video, SwinUNETR, 2.5D DINOv2, and MedNeXt, "
            "ConvNeXt V2, 3D UX-Net, nnU-Net ResEnc, ViT and more trained from scratch",
            "Feature extraction (fast, CPU friendly) or full fine-tuning (GPU recommended)",
            "Stratified K-fold cross-validation (grouped by patient in 3D) with decision threshold optimisation",
            "Evaluation on the external test set",
            "Grad-CAM heatmaps of the regions behind the predictions (3D Grad-CAM for volumes)",
        ],
        outputs=[
            "K-fold and test metrics (AUC, F-score, accuracy, sensitivity, specificity, balanced accuracy)",
            "ROC and precision-recall curves, confusion matrices",
            "Grad-CAM figures and per-image predictions (CSV)",
            "Trained networks ready to reuse (.pt)",
        ],
    ),
    Automator(
        slug="object-detection",
        name="Object Detection",
        tagline="Find and box objects in images or volumes",
        description=(
            "Fine-tune, validate and explain ten pretrained detectors (Faster R-CNN to RT-DETRv2 and D-FINE) "
            "on boxes drawn on medical or other images, including CT and MR volumes."
        ),
        icon="bounding-box",
        status="available",
        endpoint="detection",
        use_case="e.g. locate nodules on chest X-rays or CT",
        data="Images + boxes (COCO, YOLO, VOC, CSV)",
        models_short="10 pretrained detectors",
        explains="D-RISE maps",
        inputs=[
            "Train.zip and Test.zip: images or volumes with their boxes (up to 5 GB each)",
            "Annotations in COCO JSON, YOLO, Pascal VOC, CSV (2D or 3D boxes) or masks",
            "DICOM (also series), NIfTI, PNG (8/16-bit), JPEG, BMP, TIFF; images without boxes as negatives",
        ],
        steps=[
            "Medical image preparation; 3D volumes processed slice by slice with their neighbours (2.5D)",
            "Ten COCO-pretrained detectors: torchvision CNNs and detection transformers",
            "Grouped, stratified K-fold cross-validation or a faster hold-out validation",
            "Operating threshold tuned on the validation images, evaluation on the external test set",
            "D-RISE saliency maps: the image regions each detection depends on",
        ],
        outputs=[
            "mAP, AP50/75 (2D) or AP at 3D IoU, FROC (CPM), precision, recall and image-level sensitivity",
            "Precision-recall and FROC curves, AP per class, test images with their detections",
            "D-RISE explanations, predictions (CSV) and trained detectors ready to reuse",
        ],
    ),
    Automator(
        slug="image-segmentation",
        name="Image Segmentation",
        tagline="Outline structures pixel by pixel, 2D or 3D",
        description=(
            "Train, validate and compare segmentation networks, from the official self-configuring nnU-Net to "
            "pretrained U-Nets, SegFormer and 3D transformers, on images or volumes and their masks, with "
            "overlap and distance metrics and uncertainty maps."
        ),
        icon="bounding-box-circles",
        status="available",
        endpoint="segmentation",
        use_case="e.g. delineate organs or tumours",
        data="Images + masks",
        models_short="nnU-Net v2 and 18 networks",
        explains="Uncertainty maps",
        inputs=[
            "Train.zip and Test.zip with an images/ and a masks/ folder (or the nnU-Net layout)",
            "2D: PNG, JPEG, TIFF, DICOM, NIfTI; 3D: NIfTI, DICOM series, several series per case",
            "Masks: label images, palette or colour PNGs, NIfTI label maps; one or several classes",
        ],
        steps=[
            "Images and masks matched, read, aligned in patient space, mapped to classes",
            "The official nnU-Net v2 (2D or 3D full resolution) with a selectable schedule",
            "2D: U-Net, U-Net++, DeepLabV3+, FPN, UPerNet, SegFormer, MA-Net (ImageNet encoders), Attention U-Net",
            "3D: SwinUNETR (CT self-supervised), SwinUNETR-V2, SegResNet, DynUNet, UNETR, MedNeXt, V-Net and more",
            "K-fold cross-validation grouped by patient, or hold-out validation",
            "Sliding-window inference with test-time flips and uncertainty maps",
        ],
        outputs=[
            "Dice, IoU, HD95, ASSD, sensitivity and precision per class and per case",
            "Overlays of the reference and predicted masks with uncertainty, predicted masks",
            "Trained networks ready to reuse (TorchScript, or the nnU-Net model folder)",
        ],
    ),
    Automator(
        slug="time-series-forecasting",
        name="Time Series Forecasting",
        tagline="Forecast the next values of many time series",
        description=(
            "Train, validate and explain ten state-of-the-art deep learning forecasters (neuralforecast) "
            "on many series at once, with static, past and future covariates and rolling-origin validation."
        ),
        icon="graph-up-arrow",
        status="available",
        endpoint="forecasting",
        use_case="e.g. a patient's next glucose readings",
        data="Long-format Train.csv + Test.csv",
        models_short="10 neural forecasters",
        explains="Integrated gradients",
        inputs=[
            "Train.csv and Test.csv in long format: one row per series and time point (ID, Time, Target)",
            "Optional covariates: static (e.g. sex), past (e.g. another measurement) or known in advance (e.g. a dose)",
            "Test series continue training series (later period) or are new series (e.g. new patients)",
        ],
        steps=[
            "Regular time steps (missing points filled), categorical covariates encoded",
            "Ten networks: NHITS, NBEATSx, TiDE, KAN, DLinear, TFT, PatchTST, BiTCN, TCN, TimesNet",
            "Rolling-origin (prequential) validation, with optional hyperparameter tuning",
            "Forecasts of the last H points of every test series, compared with a seasonal naive baseline",
            "Integrated gradients: the inputs and time steps behind the forecasts",
        ],
        outputs=[
            "Validation and test errors (MAE, RMSE, sMAPE, MASE) and the error by horizon step",
            "Forecasts vs. observed values (figures and CSV), forecasts beyond the data",
            "Integrated gradients figures and trained models ready to reuse",
        ],
    ),
    Automator(
        slug="clustering",
        name="Clustering",
        tagline="Find groups in a table, with or without labels",
        description=(
            "Cluster the rows of a table with twelve classical algorithms (k-means to HDBSCAN) and six deep and "
            "neural ones (DEC, IDEC, DCN, VaDE, SCARF, self-organising maps), validate them across folds and "
            "explain what defines each cluster. With class labels, the clusters are also checked against them."
        ),
        icon="intersect",
        status="available",
        endpoint="clustering",
        use_case="e.g. discover patient subtypes",
        data="Train.csv (Target and Test.csv optional)",
        models_short="18 algorithms, classical to deep",
        explains="Profiles + SHAP",
        inputs=[
            "Train.csv (and an optional Test.csv) with one row per sample",
            "Numeric and categorical features; missing values are imputed",
            "Optional Target column of class labels, only used to evaluate the clusters",
        ],
        steps=[
            "Imputation, scaling, one-hot encoding and optional PCA",
            "Number of clusters given, set to the number of classes, or chosen by silhouette, Calinski-Harabasz or Davies-Bouldin",
            "Centroid, mixture, hierarchical, graph and density-based algorithms; DEC, IDEC, DCN, VaDE, SCARF and SOM",
            "K-fold validation: held-out samples assigned by models fitted on the other folds, stability across folds",
            "Test.csv samples assigned to the clusters of the final models",
            "Cluster profiles, 2D projections (PCA, t-SNE) and SHAP explanations of the clusters",
        ],
        outputs=[
            "Internal metrics (silhouette, Calinski-Harabasz, Davies-Bouldin) and, with labels, ARI, AMI, NMI, "
            "V-measure, purity and matched accuracy",
            "Cluster of every sample (CSV), profiles, projections, clusters vs. classes, silhouette plots",
            "Trained models that assign new samples to the clusters (.pkl)",
        ],
    ),
    Automator(
        slug="survival-analysis",
        name="Survival Analysis",
        tagline="Predict when an event will happen",
        description=(
            "Time-to-event models (Cox, AFT, XGBoost, DeepSurv, DeepHit, Logistic-Hazard) for censored follow-up "
            "data, validated with C-index, integrated Brier score and time-dependent AUC."
        ),
        icon="heart-pulse",
        status="available",
        endpoint="survival",
        use_case="e.g. overall survival or time to relapse after treatment",
        data="Train.csv + Test.csv with Time and Event",
        models_short="8 models: Cox to DeepHit",
        explains="Risk groups + importance",
        inputs=[
            "Train.csv and Test.csv, one row per patient",
            "Time (follow-up) and Event (1 event, 0 censored) columns",
            "Numeric and categorical features; missing values are imputed",
        ],
        steps=[
            "Cox PH, Weibull and log-normal AFT, XGBoost Cox and AFT, DeepSurv, DeepHit, Logistic-Hazard",
            "Stratified K-fold cross-validation, then the external test set",
            "Kaplan-Meier risk groups with a log-rank test, calibration, permutation importance",
        ],
        outputs=[
            "C-index, Uno's C-index, integrated Brier score, time-dependent AUC and Brier at your horizons",
            "Risk and survival probabilities of every test patient",
            "Trained models that predict survival curves of new patients (.pkl)",
        ],
    ),
]


def get_automator(slug):
    return next((a for a in AUTOMATORS if a.slug == slug), None)


@dataclass(frozen=True)
class ModelOption:
    field: str         # form field name
    name: str          # name in Helpers.pipelines_main.hyperparameters_models_grid
    group: str         # "classical" | "deep_learning"
    description: str
    speed: str         # "fast" | "moderate" | "slow"
    default: bool


MODELS = [
    ModelOption("logistic_regression", "Logistic Regression", "classical",
                "Linear baseline with interpretable coefficients.", "fast", True),
    ModelOption("svm", "Support Vector Machines", "classical",
                "Maximum-margin classifier with linear and kernel variants.", "moderate", True),
    ModelOption("random_forest", "Random Forest", "classical",
                "Ensemble of decision trees, robust to feature scales.", "moderate", True),
    ModelOption("sgd", "Stochastic Gradient Descent", "classical",
                "Regularised linear model trained with stochastic gradient descent.", "fast", True),
    ModelOption("neural_network", "Multi-Layer Neural Network", "classical",
                "Fully connected neural network (scikit-learn MLP).", "moderate", True),
    ModelOption("decision_trees", "Decision Trees", "classical",
                "Single tree whose decisions can be read directly.", "fast", True),
    ModelOption("xgboost", "XGBoost", "classical",
                "Gradient-boosted trees, strong on tabular data.", "moderate", True),
    ModelOption("tabpfn", "TabPFNv2", "deep_learning",
                "Pretrained tabular foundation model, no training needed. Up to 10,000 samples, "
                "500 features and 10 classes.", "slow", False),
    ModelOption("tabtransformer", "TabTransformer", "deep_learning",
                "Transformer contextualising categorical features, trained from scratch.", "slow", False),
    ModelOption("tabr", "TabR", "deep_learning",
                "Retrieval-augmented network using the nearest training samples.", "slow", False),
    ModelOption("tabicl", "TabICL", "deep_learning",
                "Pretrained tabular foundation model based on in-context learning.", "slow", False),
]

THRESHOLD_METRICS = [
    ("Balanced Accuracy", "Average of sensitivity and specificity; robust to class imbalance."),
    ("AUC", "Area under the ROC curve."),
    ("F-score", "Harmonic mean of precision and sensitivity."),
    ("Accuracy", "Share of correct predictions."),
    ("Sensitivity", "Share of positives detected (recall)."),
    ("Specificity", "Share of negatives correctly rejected."),
]
