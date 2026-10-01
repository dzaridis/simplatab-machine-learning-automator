"""The code shown on the results page to use the trained models must run as copied, without
the Simplatab code: real runs of both automators, then the rendered snippets executed in a
separate process that cannot import the repository."""
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from werkzeug.test import Client

sys.path.insert(0, os.path.dirname(__file__))
from image_fixtures import KINDS, make_split, write_image  # noqa: E402

import app as appmod  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Fails the snippet if it imported the Simplatab code (or its training-only dependencies)
NOT_IMPORTED = ("\nimport sys\nleaked = sorted({m.split('.')[0] for m in sys.modules} & "
                "{'Helpers', 'web', 'featurewiz', 'timm', 'torchvision'})\nassert not leaked, leaked\n")


def snippet(page, block):
    match = re.search(r'id="%s"><code>(.*?)</code></pre>' % block, page, re.S)
    return html.unescape(re.sub(r"<[^>]+>", "", match.group(1)))


def run_script(code, cwd):
    """Runs the code in a new interpreter where the repository is not importable."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, "-c", code + NOT_IMPORTED], cwd=cwd, capture_output=True,
                          text=True, env=env, timeout=600)


class TestUsageSnippets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def results_page(self):
        return Client(appmod.application).get("/automl/results").get_data(as_text=True)

    def test_tabular_snippet(self):
        data = load_breast_cancer(as_frame=True).frame.rename(columns={"target": "Target"}).sample(160, random_state=0)
        data.insert(0, "ID", [f"P{i:03d}" for i in range(len(data))])
        os.makedirs("input")
        data.iloc[:120].to_csv("input/Train.csv", index=False)
        data.iloc[120:].to_csv("input/Test.csv", index=False)
        os.makedirs("Materials/Models")
        models = ("Logistic Regression", "TabTransformer")  # a classical and a deep learning model
        params = {"BiasAssessment": False, "Feature": "None", "number_of_k_folds": 3,
                  "apply_grid_search": {"enabled": False, "type": {"Randomized": True, "Exhaustive": False}},
                  "Correlation Limit": 0.7, "Metric For Threshold Optimization": "Balanced Accuracy",
                  "Machine Learning Models": {m.name: m.name in models for m in appmod.MODELS}}
        with open("input/machine_learning_parameters.yaml", "w") as f:
            yaml.dump(params, f)
        self.assertEqual(appmod.run_pipeline("input", "output", params), "Pipeline completed successfully")

        page = self.results_page()
        install = snippet(page, "code-install")
        self.assertIn("scikit-learn==1.3.1", install)
        self.assertNotIn("git clone", install)
        code = snippet(page, "code-predict")
        self.assertIn('index_col="ID"', code)
        self.assertNotIn("Helpers", code)
        best = re.search(r'Materials/Models/(.*?)_pipeline\.pkl', code).group(1)
        # New samples: the test set, columns in another order, Target included
        new = data.iloc[120:]
        new[new.columns[::-1]].to_csv("new_samples.csv", index=False)
        test_auc = pd.read_excel("Materials/test_results.xlsx", index_col=0)["AUC"]
        for model in models:
            with self.subTest(model=model):
                result = run_script(code.replace(best, model) + "\np.dump('p.npy')", self.tmp.name)
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                self.assertIn("probability_1", result.stdout)
                # The standalone model reproduces the test set results of the pipeline
                p = np.load("p.npy", allow_pickle=True)
                self.assertAlmostEqual(roc_auc_score(new["Target"], p[:, 1]), test_auc[model], places=6)

    def test_image_snippet(self):
        make_split("input/train", 9, seed=1)
        make_split("input/test", 8, seed=2)  # every file kind
        from Helpers.image.pipeline import run_image_pipeline
        params = {"models": ["efficientnet_b0"], "mode": "features", "k_folds": 3, "metric": "Balanced Accuracy",
                  "classes": ["lesion", "normal"], "positive_class": "lesion", "window": "auto", "volume": "middle",
                  "augmentation": {}, "epochs": 1, "learning_rate": 1e-4, "patience": 1, "batch_size": 8,
                  "pretrained": False}
        self.assertEqual(run_image_pipeline(os.path.abspath("input"), params), "Pipeline completed successfully")

        page = self.results_page()
        install = snippet(page, "code-install")
        self.assertIn("pydicom", install)
        self.assertIn("nibabel", install)
        code = snippet(page, "code-predict")
        self.assertIn('torch.jit.load("Materials/Models/EfficientNet-B0.pt"', code)
        self.assertNotIn("Helpers", code)
        os.makedirs("new_images")
        for label in ("lesion", "normal"):
            for name in os.listdir(os.path.join("input/test", label)):
                os.link(os.path.join("input/test", label, name), os.path.join("new_images", name))
        result = run_script(code, self.tmp.name)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])

        # Same predictions and probabilities as the pipeline's own test predictions
        saved = pd.read_csv("Materials/Predictions/EfficientNet-B0_test_predictions.csv")
        saved.index = saved.file.map(os.path.basename)
        lines = [line.split(" ", 2) for line in result.stdout.splitlines() if line.startswith("new_images/")]
        self.assertEqual(len(lines), len(saved))
        classes = json.load(open("Materials/run_info.json"))["classes"]
        for path, predicted, probabilities in lines:
            row = saved.loc[os.path.basename(path)]
            self.assertEqual(predicted, row["predicted_class"])
            expected = [row[f"probability_{c}"] for c in classes]
            np.testing.assert_allclose(json.loads(probabilities), expected, atol=2e-3)

        # The image reading of the snippet matches Simplatab's for every format and setting
        from Helpers.image import inference, io
        rng = np.random.default_rng(3)
        paths = [write_image(os.path.join("formats", f"{kind}_{i}"), kind, i == 0, rng)
                 for kind in KINDS + ["dicom_jpeg_rgb"] for i in range(2)]
        info_path = "Materials/run_info.json"
        info = json.load(open(info_path))
        for window in ("auto", "lung", "brain"):
            for volume in ("middle", "mip"):
                with self.subTest(window=window, volume=volume):
                    info.update(window=window, volume=volume,
                                ct_window=list(io.CT_WINDOWS[window]) if window in io.CT_WINDOWS else None)
                    json.dump(info, open(info_path, "w"))
                    namespace = {}
                    exec(snippet(self.results_page(), "code-predict").replace("new_images/*", "none/*"), namespace)
                    for path in paths:
                        expected = inference.prepare(io.load_image(path, window, volume), info)
                        self.assertTrue(namespace["prepare"](namespace["load"](path)).equal(expected), path)

    def test_older_results_without_run_info(self):
        os.makedirs("Materials/Models")
        open("Materials/Models/SVM_pipeline.pkl", "wb").close()
        page = self.results_page()
        self.assertIn("SVM_pipeline.pkl", snippet(page, "code-predict"))
        self.assertIn("threshold", snippet(page, "code-predict"))


if __name__ == "__main__":
    unittest.main()
