import os
import pickle
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.datasets import make_classification
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score

from Helpers import dl_classifiers
from Helpers.dl_classifiers import (TabPFNv2Classifier, TabICLClassifier, TabTransformerClassifier,
                                    TabRClassifier, DEEP_LEARNING_CLASSIFIERS)
from Helpers import pipelines, pipelines_main, shap_module


def make_data(n_classes=2, n_samples=120, seed=0):
    X, y = make_classification(n_samples=n_samples, n_features=6, n_informative=4, n_redundant=0,
                               n_classes=n_classes, random_state=seed)
    X[:, 0] = (X[:, 0] > 0).astype(float)  # a binary (e.g. one-hot encoded) column
    return X.astype(np.float32), y


FAST = {"max_epochs": 5, "patience": 2}
TORCH_MODELS = [
    lambda **kw: TabTransformerClassifier(dim=8, depth=1, heads=2, **FAST, **kw),
    lambda **kw: TabRClassifier(d_main=16, context_size=8, **FAST, **kw),
    lambda **kw: TabRClassifier(d_main=16, context_size=8, encoder_n_blocks=1, num_embeddings="plr", **FAST, **kw),
]


class TestTorchClassifiers(unittest.TestCase):
    def test_binary_and_multiclass(self):
        for n_classes in (2, 3):
            X, y = make_data(n_classes)
            for make in TORCH_MODELS:
                model = make().fit(X, y)
                proba = model.predict_proba(X)
                self.assertEqual(proba.shape, (len(X), n_classes))
                np.testing.assert_allclose(proba.sum(axis=1), 1, rtol=1e-5)
                np.testing.assert_array_equal(model.classes_, np.arange(n_classes))
                self.assertTrue(set(model.predict(X)) <= set(model.classes_))

    def test_learns(self):
        X, y = make_data(n_samples=300)
        for model in (TabTransformerClassifier(dim=8, depth=1, heads=2), TabRClassifier(d_main=32, context_size=16)):
            proba = model.fit(X[:200], y[:200]).predict_proba(X[200:])
            self.assertGreater(roc_auc_score(y[200:], proba[:, 1]), 0.8, type(model).__name__)

    def test_string_labels(self):
        X, y = make_data()
        labels = np.array(["no", "yes"])[y]
        model = TORCH_MODELS[0]().fit(X, labels)
        self.assertEqual(list(model.classes_), ["no", "yes"])
        self.assertTrue(set(model.predict(X)) <= {"no", "yes"})

    def test_pickle_clone_sparse_and_determinism(self):
        X, y = make_data()
        for make in TORCH_MODELS:
            model = make().fit(X, y)
            proba = model.predict_proba(X)
            restored = pickle.loads(pickle.dumps(model))
            np.testing.assert_allclose(restored.predict_proba(X), proba, atol=1e-5)
            np.testing.assert_allclose(model.predict_proba(sp.csr_matrix(X)), proba, atol=1e-5)
            np.testing.assert_allclose(clone(model).fit(X, y).predict_proba(X), proba, atol=1e-5)
            self.assertEqual(clone(model).get_params(), model.get_params())

    def test_tabtransformer_categorical_columns(self):
        X, y = make_data()
        model = TORCH_MODELS[0]().fit(X, y)
        self.assertEqual(model.categorical_columns_, [0])
        self.assertEqual(model.continuous_columns_, [1, 2, 3, 4, 5])
        X_unseen = X.copy()
        X_unseen[:, 0] = 7.0  # category never seen during training
        self.assertTrue(np.isfinite(model.predict_proba(X_unseen)).all())

    def test_no_features(self):
        X, y = make_data()
        with self.assertRaises(ValueError):
            TORCH_MODELS[0]().fit(X[:, :0], y)

    def test_tiny_training_set_without_validation_split(self):
        X, y = make_data(n_samples=6)
        for make in TORCH_MODELS:
            self.assertEqual(make().fit(X, y).predict_proba(X).shape, (6, 2))

    def test_grid_search_like_model_trainer(self):
        X, y = make_data()
        search = RandomizedSearchCV(TabRClassifier(**FAST), {"d_main": [8, 16], "context_size": [4, 8]}, n_iter=2,
                                    cv=StratifiedKFold(2, shuffle=True, random_state=0), random_state=0,
                                    scoring=make_scorer(roc_auc_score, needs_threshold=True), error_score="raise")
        search.fit(X, y)
        self.assertIn(search.best_params_["d_main"], (8, 16))


class _FakeInnerClassifier(ClassifierMixin, BaseEstimator):
    """Stands in for tabpfn/tabicl (no pretrained weights needed)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_params(self, deep=True):
        return dict(self.kwargs)

    def fit(self, X, y):
        self.env_during_fit_ = os.environ.get("TABPFN_ALLOW_CPU_LARGE_DATASET")
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return np.full((len(X), len(self.classes_)), 1 / len(self.classes_))


class TestPretrainedWrappers(unittest.TestCase):
    def test_parameters_are_forwarded(self):
        X, y = make_data()
        with mock.patch.object(dl_classifiers, "_import", return_value=_FakeInnerClassifier):
            tabpfn = TabPFNv2Classifier(n_estimators=2, softmax_temperature=0.5).fit(sp.csr_matrix(X), y)
            tabicl = TabICLClassifier(n_estimators=3, device="auto").fit(X, y)
        self.assertEqual(tabpfn.model_.kwargs["n_estimators"], 2)
        self.assertEqual(tabpfn.model_.kwargs["softmax_temperature"], 0.5)
        self.assertEqual(tabpfn.model_.env_during_fit_, "1")  # TabPFN's CPU guard is lifted during fit only
        self.assertEqual(tabicl.model_.kwargs["n_estimators"], 3)
        self.assertIsNone(tabicl.model_.kwargs["device"])
        self.assertEqual(tabpfn.predict_proba(X).shape, (len(X), 2))

    def test_tabicl_pickles_its_weights(self):
        X, y = make_data()
        with mock.patch.object(dl_classifiers, "_import", return_value=_FakeInnerClassifier):
            model = TabICLClassifier().fit(X, y)
        self.assertTrue(model.model_._save_model_weights)
        restored = pickle.loads(pickle.dumps(model))
        self.assertTrue(restored.model_._save_model_weights)

    def test_missing_package_message(self):
        with self.assertRaises(ImportError) as ctx:
            dl_classifiers._import("a_package_that_does_not_exist", "X", "a_package_that_does_not_exist")
        self.assertIn("requirements.txt", str(ctx.exception))

    @unittest.skipUnless(os.environ.get("SIMPLATAB_TEST_PRETRAINED") == "1",
                         "set SIMPLATAB_TEST_PRETRAINED=1 to run TabPFNv2/TabICL with their pretrained weights")
    def test_pretrained_models(self):
        X, y = make_data(n_samples=200)
        for model in (TabPFNv2Classifier(n_estimators=1), TabICLClassifier(n_estimators=1)):
            proba = model.fit(X[:150], y[:150]).predict_proba(X[150:])
            self.assertGreater(roc_auc_score(y[150:], proba[:, 1]), 0.8, type(model).__name__)
            np.testing.assert_allclose(pickle.loads(pickle.dumps(model)).predict_proba(X[150:]), proba, atol=1e-5)


class TestIntegration(unittest.TestCase):
    NAMES = {"TabPFNv2": TabPFNv2Classifier, "TabTransformer": TabTransformerClassifier,
             "TabR": TabRClassifier, "TabICL": TabICLClassifier}

    def test_registered_in_the_model_grid(self):
        for name, cls in self.NAMES.items():
            entry = pipelines_main.hyperparameters_models_grid[name]
            self.assertIs(entry["classifier"], cls)
            self.assertTrue(is_classifier(cls))
            valid = set(cls().get_params())
            self.assertTrue(set(entry["params"]) <= valid, name)
            self.assertTrue(set(entry["default_params"]) <= valid, name)
            grid = entry["params"]
            self.assertEqual(pipelines_main.adjust_hyperparameters_for_multiclass(cls, grid, True, 3), grid)

    def test_shap_uses_the_deep_learning_explainer(self):
        for cls in DEEP_LEARNING_CLASSIFIERS:
            self.assertEqual(shap_module.ShapValues({"model": cls()}).MODEL_TYPE, 4)

    def test_ml_pipeline(self):
        X, y = make_data(n_samples=150)
        X = pd.DataFrame(X.astype(np.float64), columns=[f"f{i}" for i in range(X.shape[1])])
        X["group"] = np.where(np.arange(len(X)) % 2, "a", "b")  # categorical column -> one-hot encoded
        y = pd.Series(y, name="Target")
        pipeline = pipelines.MLPipeline(X, y, TabTransformerClassifier, dict(dim=8, depth=1, heads=2, **FAST))
        pipeline.execute_feature_selection(corr_limit=0.7)
        pipeline.execute_preprocessing()
        pipeline.train_model()
        ppln = pickle.loads(pickle.dumps(pipeline.build_pipeline()))
        self.assertEqual(ppln.predict_proba(X).shape, (len(X), 2))
        self.assertEqual(pipeline.get_best_parameters()["dim"], 8)


if __name__ == "__main__":
    unittest.main()
