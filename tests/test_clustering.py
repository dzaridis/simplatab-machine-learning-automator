"""Tests of the clustering automator: data checks, metrics, every algorithm, the pipeline end to
end (with and without labels and Test.csv), the web flow and the standalone usage code."""
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import pandas as pd
from werkzeug.test import Client

import app as appmod
from Helpers.clustering import data as cdata
from Helpers.clustering import metrics as cm
from Helpers.clustering.models import ALGORITHMS, BY_KEY, Assigner, build, fit_labels, order_mapping, raw_predict, remap
from Helpers.clustering.pipeline import run_clustering_pipeline

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO, "Examples", "clustering")
FAST = {"pretrain_epochs": 10, "epochs": 10, "k_max": 5, "tsne": False}


def blobs(n=150, seed=0, labels=True):
    """Three well separated groups in four numeric features, a categorical feature that follows the
    group, an ID and (optionally) the group as Target."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(["alpha", "beta", "gamma"], n // 3)
    centers = {"alpha": [0, 0, 0, 0], "beta": [6, 6, 0, 0], "gamma": [0, 6, 6, 6]}
    X = np.array([centers[g] for g in groups]) + rng.normal(0, 0.7, (len(groups), 4))
    frame = pd.DataFrame(X.round(3), columns=["f1", "f2", "f3", "f4"])
    frame.insert(0, "ID", [f"S{seed}_{i:03d}" for i in range(len(frame))])
    frame["Colour"] = np.where(groups == "gamma", "red", np.where(rng.random(len(groups)) < 0.5, "blue", "green"))
    if labels:
        frame["Target"] = groups
    return frame.sample(frac=1, random_state=seed).reset_index(drop=True)


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, train, test=None):
        train.to_csv(os.path.join(self.dir, "Train.csv"), index=False)
        if test is not None:
            test.to_csv(os.path.join(self.dir, "Test.csv"), index=False)
        return os.path.join(self.dir, "Train.csv"), os.path.join(self.dir, "Test.csv") if test is not None else None


class TestData(TempDir):
    def test_example_summary(self):
        summary = cdata.summarize(os.path.join(EXAMPLE, "Train.csv"), os.path.join(EXAMPLE, "Test.csv"))
        self.assertEqual(summary["errors"], [])
        self.assertEqual((summary["train_rows"], summary["test_rows"], summary["id_column"]), (600, 200, "ID"))
        self.assertEqual(summary["numeric"], ["Age", "BMI", "HbA1c", "HOMA2_B", "HOMA2_IR", "Systolic_BP"])
        self.assertEqual(summary["categorical"], ["Sex", "GADA"])
        self.assertTrue(summary["has_labels"] and summary["test_has_labels"])
        self.assertEqual(summary["classes"], ["MARD", "MOD", "SAID", "SIDD", "SIRD"])
        self.assertEqual(summary["suggested_k"], 5)
        self.assertTrue(any("missing feature value" in w for w in summary["warnings"]))

    def test_columns_left_out_and_labels(self):
        frame = blobs(labels=False)
        frame["Name"] = [f"patient {i}" for i in range(len(frame))]   # identifier-like text
        frame["Site"] = "A"                                             # constant
        frame["Lab"] = np.nan                                           # empty
        summary = cdata.summarize(self.write(frame)[0])
        self.assertEqual(summary["errors"], [])
        self.assertFalse(summary["has_labels"])
        self.assertIsNone(summary["suggested_k"])
        self.assertEqual({d["column"] for d in summary["dropped"]}, {"Name", "Site", "Lab"})
        frame["Target"] = np.linspace(0, 1, len(frame))  # a measurement, not classes
        summary = cdata.summarize(self.write(frame)[0])
        self.assertFalse(summary["has_labels"])
        self.assertTrue(any("continuous" in w for w in summary["warnings"]))

    def test_errors(self):
        train = blobs()
        summary = cdata.summarize(*self.write(train.head(5)))
        self.assertTrue(any("at least 10" in e for e in summary["errors"]))
        summary = cdata.summarize(*self.write(train, train.drop(columns=["f2"])))
        self.assertTrue(any("lacks the feature column(s) f2" in e for e in summary["errors"]))
        summary = cdata.summarize(*self.write(train[["ID", "Target"]]))
        self.assertTrue(any("no usable feature" in e for e in summary["errors"]))

    def test_prepare_features(self):
        train, test = blobs(), blobs(seed=1)
        train.loc[3, "f1"] = np.nan
        data = cdata.prepare(*self.write(train, test), {"reduction": "pca", "pca_variance": 0.9})
        self.assertEqual(data.F_train.shape, (150, 4 + 3))   # 4 numeric + 3 one-hot colours
        self.assertLess(data.X_train.shape[1], data.F_train.shape[1])
        self.assertFalse(np.isnan(data.X_train).any())
        self.assertEqual(data.ids_test[:1], [test.ID[0]])
        self.assertEqual(set(data.y_test), {"alpha", "beta", "gamma"})


class TestMetrics(unittest.TestCase):
    def test_external_metrics_ignore_the_numbering(self):
        classes = np.array(["a"] * 5 + ["b"] * 5 + ["c"] * 5, dtype=object)
        labels = np.array([2] * 5 + [0] * 5 + [1] * 5)
        scores = cm.external(classes, labels)
        for metric in ("ARI", "AMI", "NMI", "V-measure", "Purity", "Accuracy", "FMI"):
            self.assertAlmostEqual(scores[metric], 1.0, msg=metric)
        # Splitting a class in two: perfectly homogeneous, not complete; one of the two parts is unmatched
        labels = np.array([0] * 5 + [1] * 5 + [2] * 3 + [3] * 2)
        scores = cm.external(classes, labels)
        self.assertAlmostEqual(scores["Homogeneity"], 1.0)
        self.assertLess(scores["Completeness"], 1.0)
        self.assertAlmostEqual(scores["Accuracy"], 13 / 15)
        self.assertAlmostEqual(scores["Purity"], 1.0)

    def test_noise_and_degenerate_clusterings(self):
        X = np.random.default_rng(0).normal(size=(20, 2))
        scores = cm.score(X, np.zeros(20, dtype=int))
        self.assertEqual(scores["Clusters"], 1)
        self.assertTrue(np.isnan(scores["Silhouette"]))
        labels = np.array([0] * 8 + [1] * 8 + [-1] * 4)
        scores = cm.score(X, labels, np.array(["x"] * 10 + ["y"] * 10, dtype=object))
        self.assertEqual((scores["Clusters"], scores["Noise %"]), (2, 20.0))
        self.assertFalse(np.isnan(scores["Silhouette"]))
        self.assertTrue(cm.better("davies_bouldin", 0.5, 0.9) and cm.better("silhouette", 0.9, 0.5))
        self.assertFalse(cm.better("silhouette", np.nan, 0.1))

    def test_order_mapping(self):
        labels = np.array([5, 5, 2, 2, 2, -1, 7])
        self.assertEqual(remap(labels, order_mapping(labels)).tolist(), [1, 1, 0, 0, 0, -1, 2])


class TestAlgorithms(unittest.TestCase):
    def test_every_algorithm_clusters_and_assigns_new_samples(self):
        data = blobs(n=120, labels=True)
        X = np.ascontiguousarray((data[["f1", "f2", "f3", "f4"]].to_numpy() - 2) / 3, dtype=np.float32)
        new = np.ascontiguousarray((blobs(n=30, seed=3)[["f1", "f2", "f3", "f4"]].to_numpy() - 2) / 3, dtype=np.float32)
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm.key):
                estimator = build(algorithm.key, 3 if algorithm.uses_k else None, X, dict(FAST, seed=0))
                labels = fit_labels(estimator, X)
                self.assertEqual(len(labels), len(X))
                if algorithm.uses_k:
                    self.assertEqual(cm.n_clusters(labels), 3)
                    self.assertGreater(cm.external(data.Target.to_numpy(), labels)["ARI"], 0.9)
                assigner = None if algorithm.native_predict else Assigner(X, labels)
                assigned = raw_predict(algorithm, estimator, assigner, new)
                self.assertEqual(len(assigned), len(new))
                self.assertTrue(set(assigned.tolist()) <= set(labels.tolist()) | {-1})


class PipelineRun(TempDir):
    def run_pipeline(self, train, test, params):
        os.makedirs(os.path.join(self.dir, "input"))
        train.to_csv(os.path.join(self.dir, "input", "Train.csv"), index=False)
        if test is not None:
            test.to_csv(os.path.join(self.dir, "input", "Test.csv"), index=False)
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            return run_clustering_pipeline("input", params)
        finally:
            os.chdir(cwd)


class TestPipeline(PipelineRun):
    def test_supervised_evaluation_with_test_set(self):
        train, test = blobs(), blobs(n=60, seed=1)
        params = dict(FAST, models=["kmeans", "gmm", "dbscan", "agglomerative", "dec", "som"], n_clusters="classes",
                      validation="kfold", k_folds=3, selection_metric="ARI")
        self.assertEqual(self.run_pipeline(train, test, params), "Pipeline completed successfully")
        root = os.path.join(self.dir, "Materials")
        info = json.load(open(os.path.join(root, "run_info.json")))
        self.assertEqual(info["automator"], "clustering")
        self.assertTrue(info["supervised"])
        self.assertEqual(info["selection_source"], "validation")
        self.assertEqual(info["skipped"], [])
        self.assertEqual(info["clusters"]["K-Means"], 3)
        for path in ["train_results.xlsx", "test_results.xlsx", "3_fold_results.xlsx", "Clusters/train_clusters.csv",
                     "Clusters/test_clusters.csv", "Clusters/validation_folds.csv", "Models/K-Means.pkl", "Models/DEC.pkl",
                     "Cluster_Profiles/K-Means_profile.csv", "Cluster_Profiles/K-Means_profile.png",
                     "Embeddings/K-Means_clusters.png", "Embeddings/Target_classes.png", "Embeddings/projection.csv",
                     "Metrics_Plots/K-Means_contingency.png", "Metrics_Plots/K-Means_silhouette.png",
                     "Metrics_Plots/train_metrics.png", "Metrics_Plots/test_metrics.png",
                     "Explainability/K-Means_shap.png", "Explainability/K-Means_feature_importance.csv"]:
            self.assertTrue(os.path.exists(os.path.join(root, path)), path)
        test_results = pd.read_excel(os.path.join(root, "test_results.xlsx"), index_col=0)
        self.assertGreater(test_results.loc["K-Means", "ARI"], 0.95)
        self.assertGreater(test_results.loc["Gaussian Mixture", "Accuracy"], 0.95)
        kfold = pd.read_excel(os.path.join(root, "3_fold_results.xlsx"), index_col=0)
        self.assertIn("Stability (ARI)", kfold.columns)
        self.assertTrue(kfold.loc["K-Means", "Stability (ARI)"].startswith(("1.000", "0.9")))
        clusters = pd.read_csv(os.path.join(root, "Clusters", "train_clusters.csv"))
        self.assertEqual(list(clusters.columns[:2]), ["ID", "Target"])
        self.assertEqual(clusters["K-Means"].value_counts().tolist(), [50, 50, 50])
        importance = pd.read_csv(os.path.join(root, "Explainability", "K-Means_feature_importance.csv"))
        self.assertIn(importance.feature[0], {"f1", "f2", "f3", "f4", "Colour_red"})   # not the random colours
        self.assertEqual(set(importance.columns), {"feature", "total", "cluster_0", "cluster_1", "cluster_2"})
        splits = pd.read_csv(os.path.join(root, "Splits", "splits.csv"))
        self.assertEqual(list(splits.columns), ["fold", "set", "id", "row", "class"])
        self.assertEqual(len(splits[splits.set == "validation"]), 150)
        self.assertEqual(sorted(splits.fold.unique()), [1, 2, 3])

        # Results page and the standalone code, run without the repository
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            page = Client(appmod.application).get("/automl/results").get_data(as_text=True)
        finally:
            os.chdir(cwd)
        self.assertIn("Validation (Train.csv): 3 held-out folds", page)
        self.assertIn("K-Means: cluster profiles", page)
        code = html.unescape(re.sub(r"<[^>]+>", "", re.search(r'id="code-predict"><code>(.*?)</code></pre>', page, re.S).group(1)))
        self.assertNotIn("Helpers", code)
        best = info["best_model"]
        self.assertIn(f'Materials/Models/{best.replace(" ", "_")}.pkl', code)
        test.drop(columns=["Target"]).to_csv(os.path.join(self.dir, "new_samples.csv"), index=False)
        script = code + ("\nclusters.to_csv('snippet.csv')\nimport sys\n"
                         "assert not {m.split('.')[0] for m in sys.modules} & {'Helpers', 'web', 'featurewiz'}\n")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        run = subprocess.run([sys.executable, "-c", script], cwd=self.dir, capture_output=True, text=True, env=env, timeout=600)
        self.assertEqual(run.returncode, 0, run.stderr[-2000:])
        snippet = pd.read_csv(os.path.join(self.dir, "snippet.csv"))
        expected = pd.read_csv(os.path.join(root, "Clusters", "test_clusters.csv"))
        np.testing.assert_array_equal(snippet["cluster"].to_numpy(), expected[best].to_numpy())

    def test_unsupervised_without_test_set(self):
        params = dict(FAST, models=["kmeans", "hdbscan", "idec"], n_clusters="auto", k_min=2, validation="none",
                      explain=False)
        self.assertEqual(self.run_pipeline(blobs(labels=False), None, params), "Pipeline completed successfully")
        root = os.path.join(self.dir, "Materials")
        info = json.load(open(os.path.join(root, "run_info.json")))
        self.assertFalse(info["supervised"])
        self.assertEqual((info["selection_metric"], info["selection_source"]), ("Silhouette", "train"))
        self.assertEqual(info["chosen_k"]["K-Means"], 3)   # the three groups found by the silhouette
        self.assertEqual(info["chosen_k"]["IDEC"], 3)
        self.assertFalse(os.path.exists(os.path.join(root, "test_results.xlsx")))
        self.assertFalse(os.path.exists(os.path.join(root, "Splits")))
        self.assertTrue(os.path.exists(os.path.join(root, "Metrics_Plots", "k_selection.png")))
        selection = pd.read_csv(os.path.join(root, "Clusters", "k_selection.csv"))
        self.assertEqual(sorted(selection.k.unique()), [2, 3, 4, 5])
        train = pd.read_excel(os.path.join(root, "train_results.xlsx"), index_col=0)
        self.assertNotIn("ARI", train.columns)

    def test_algorithm_limited_in_rows_is_skipped(self):
        from Helpers.clustering import models
        params = dict(FAST, models=["kmeans", "affinity_propagation"], n_clusters=3, validation="none", explain=False)
        original = models.BY_KEY["affinity_propagation"]
        limited = models.Algorithm(**{**original.__dict__, "max_samples": 50})
        models.BY_KEY["affinity_propagation"] = limited
        try:
            from Helpers.clustering import pipeline
            pipeline.BY_KEY["affinity_propagation"] = limited
            self.assertEqual(self.run_pipeline(blobs(), None, params), "Pipeline completed successfully")
        finally:
            models.BY_KEY["affinity_propagation"] = original
            pipeline.BY_KEY["affinity_propagation"] = original
        info = json.load(open(os.path.join(self.dir, "Materials", "run_info.json")))
        self.assertEqual([s["model"] for s in info["skipped"]], ["Affinity Propagation"])
        self.assertIn("at most 50 samples", info["skipped"][0]["reason"])


class TestWebFlow(TempDir):
    def setUp(self):
        super().setUp()
        self.cwd = os.getcwd()
        os.chdir(self.dir)
        self.client = Client(appmod.application)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(appmod.CLUSTERING_INPUT_FOLDER, ignore_errors=True)
        super().tearDown()

    def upload(self, train, test=None):
        data = {"train_file": (open(train, "rb"), "Train.csv")}
        if test:
            data["test_file"] = (open(test, "rb"), "Test.csv")
        try:
            return self.client.post("/automl/clustering/upload", data=data)
        finally:
            for f, _ in data.values():
                f.close()

    def test_pages_and_example_files(self):
        self.assertEqual(self.client.get("/automl/clustering").status_code, 200)
        response = self.client.get("/automl/clustering/example/Train.csv")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data.startswith(b"ID,Age"))
        response.close()
        self.assertEqual(self.client.get("/automl/clustering/example/app.py").status_code, 404)
        self.assertEqual(self.client.get("/automl/automators/clustering").headers["Location"], "/automl/clustering")

    def test_invalid_upload_shows_its_message(self):
        train = blobs()
        response = self.upload(*self.write(train[["ID", "Target"]]))
        self.assertEqual(response.headers["Location"], "/automl/clustering")
        self.assertIn("no usable feature column", self.client.get("/automl/clustering").get_data(as_text=True))

    def test_upload_configure_and_run_without_test(self):
        response = self.upload(self.write(blobs())[0])
        self.assertEqual(response.headers["Location"], "/automl/clustering/parameters")
        page = self.client.get("/automl/clustering/parameters").get_data(as_text=True)
        self.assertIn("Number of classes (3)", page)
        self.assertIn('name="ignore_4"', page)   # one switch per feature column
        # A fixed k above the limit is refused
        response = self.client.post("/automl/clustering/parameters", data={"kmeans": "true", "n_clusters_mode": "fixed",
                                                                           "n_clusters": "99"})
        self.assertEqual(response.headers["Location"], "/automl/clustering/parameters")
        form = {"kmeans": "true", "som": "true", "n_clusters_mode": "classes", "validation": "kfold", "k_folds": "2",
                "selection_metric": "AMI", "scaling": "robust", "ignore_4": "true", "pretrain_epochs": "10",
                "epochs": "10", "latent_dim": "4", "explain": "true"}
        self.assertEqual(self.client.post("/automl/clustering/parameters", data=form).headers["Location"], "/automl/run")
        with open(os.path.join(appmod.CLUSTERING_INPUT_FOLDER, "params.json")) as f:
            params = json.load(f)
        self.assertEqual((params["n_clusters"], params["ignore_columns"], params["scaling"], params["tsne"]),
                         ("classes", ["Colour"], "robust", False))
        for _ in range(3000):
            status = json.loads(self.client.get("/automl/api/status").data)
            if status["state"] != "running":
                break
            time.sleep(0.1)
        self.assertEqual(status["state"], "done", status["message"])
        self.assertEqual([m["test"] for m in status["models"]], ["done", "done"])
        self.assertEqual([p[0] for p in status["phases"]], ["data", "kfold", "test", "done"])
        page = self.client.get("/automl/results").get_data(as_text=True)
        self.assertIn("Validation (Train.csv): 2 held-out folds", page)
        self.assertNotIn("Test.csv: samples assigned", page)
        self.assertIn("Self-Organizing Map: cluster profiles", page)


if __name__ == "__main__":
    unittest.main()
