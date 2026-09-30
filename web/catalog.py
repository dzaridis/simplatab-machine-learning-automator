"""What the web interface offers: the automators and the models of the tabular automator.

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
        tagline="Binary and multiclass classification from CSV files",
        description=(
            "Train, validate and explain classical machine learning and deep learning "
            "classifiers on tabular data, from data bias assessment to SHAP explanations."
        ),
        icon="table",
        status="available",
        endpoint="tabular",
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
        tagline="Deep learning classifiers for 2D medical and natural images",
        description=(
            "Fine-tune pretrained convolutional and transformer networks on labelled images, "
            "with the same validation, evaluation and explainability standards as the tabular automator."
        ),
        icon="images",
        status="coming_soon",
        inputs=[
            "One folder per class (or a CSV mapping each image to its label)",
            "PNG / JPEG images, with DICOM and NIfTI slices planned",
            "A held-out test set, as for tabular data",
        ],
        steps=[
            "Image quality checks, resizing and intensity normalisation",
            "Transfer learning from pretrained CNNs and vision transformers",
            "Data augmentation and stratified K-fold cross-validation",
            "Decision threshold optimisation and external test evaluation",
            "Visual explanations of the predictions (e.g. Grad-CAM)",
        ],
        outputs=[
            "Classification metrics, ROC and precision-recall curves",
            "Saliency maps highlighting the regions behind each prediction",
            "Exported trained models",
        ],
    ),
    Automator(
        slug="image-segmentation",
        name="Image Segmentation",
        tagline="Automated delineation of regions of interest in images",
        description=(
            "Train segmentation networks from images and their masks, and evaluate the "
            "delineations against expert annotations."
        ),
        icon="bounding-box-circles",
        status="coming_soon",
        inputs=[
            "Images and their masks with matching file names",
            "2D images, with 3D volumes (NIfTI) planned",
            "One or several labelled structures per mask",
        ],
        steps=[
            "Consistency checks of images and masks, resampling and normalisation",
            "U-Net family architectures with automatically configured training",
            "Cross-validation and evaluation on a held-out test set",
        ],
        outputs=[
            "Overlap and distance metrics (Dice, IoU, Hausdorff distance)",
            "Predicted masks and overlays for visual review",
            "Exported trained models",
        ],
    ),
    Automator(
        slug="longitudinal-forecasting",
        name="Longitudinal Forecasting",
        tagline="Forecasting from repeated measurements over time",
        description=(
            "Predict future values or outcomes from longitudinal data, such as repeated "
            "visits of the same subjects, with subject-aware validation."
        ),
        icon="graph-up-arrow",
        status="coming_soon",
        inputs=[
            "A long-format CSV: one row per subject and time point",
            "A subject identifier, a time column, features and the target",
            "Irregular time points and missing visits supported",
        ],
        steps=[
            "Temporal feature engineering (lags, rolling statistics, time since baseline)",
            "Statistical, gradient boosting and deep learning forecasters",
            "Subject-level cross-validation, so no subject leaks between folds",
            "Evaluation on a held-out time window or held-out subjects",
        ],
        outputs=[
            "Forecast error metrics (MAE, RMSE, MAPE) per horizon",
            "Observed versus forecast trajectories per subject",
            "Feature importance over time and exported models",
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
