"""Trained tabular models that run without the Simplatab code.

The saved pipeline is a plain scikit-learn ``Pipeline``: column selection (the features kept by
the feature selection), preprocessing (scaling, one-hot encoding) and the classifier. The
Simplatab classes of the deep learning classifiers are stored by value in the file (cloudpickle),
so loading it only needs pip packages (see ``REQUIREMENTS``).
"""
import operator

import cloudpickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Packages needed to load each model, on top of numpy, pandas and scikit-learn (versions of requirements.txt)
REQUIREMENTS = {
    "XGBoost": ["xgboost==1.7.6"],
    "TabPFNv2": ["torch==2.8.0", "tabpfn==2.0.9", "cloudpickle==3.1.2"],
    "TabICL": ["torch==2.8.0", "tabicl==2.0.2", "cloudpickle==3.1.2"],
    "TabTransformer": ["torch==2.8.0", "cloudpickle==3.1.2"],
    "TabR": ["torch==2.8.0", "cloudpickle==3.1.2"],
}
BASE_REQUIREMENTS = ["numpy==1.23.5", "pandas==2.0.3", "scikit-learn==1.3.1"]


def requirements(model_name):
    return BASE_REQUIREMENTS + REQUIREMENTS.get(model_name, [])


def standalone_pipeline(pipeline, X_train):
    """The trained pipeline with the feature-selection step (featurewiz) replaced by the
    selection of the columns it kept."""
    steps = pipeline.named_steps
    features = list(steps["FeatureWizFs"].transform(X_train).columns)
    select = FunctionTransformer(operator.itemgetter(features)).fit(X_train)
    return Pipeline([("select", select), ("preprocessor", steps["preprocessor"]), ("model", steps["model"])])


def save_pipeline(pipeline, X_train, path):
    from Helpers import dl_classifiers, dl_networks

    modules = (dl_classifiers, dl_networks)
    for module in modules:
        cloudpickle.register_pickle_by_value(module)
    try:
        with open(path, "wb") as file:
            cloudpickle.dump(standalone_pipeline(pipeline, X_train), file)
    finally:
        for module in modules:
            cloudpickle.unregister_pickle_by_value(module)
