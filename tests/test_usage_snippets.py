"""The code shown on the results page to use the trained models must run as copied: real
runs of both automators, then the rendered snippets executed in a separate process."""
import html
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
from werkzeug.test import Client

sys.path.insert(0, os.path.dirname(__file__))
from image_fixtures import make_split  # noqa: E402

import app as appmod  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def snippet(page, block):
    match = re.search(r'id="%s"><code>(.*?)</code></pre>' % block, page, re.S)
    return html.unescape(re.sub(r"<[^>]+>", "", match.group(1)))


def run_script(code, cwd):
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=REPO), timeout=600)


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
        os.makedirs("input")
        data.iloc[:120].to_csv("input/Train.csv", index=False)
        data.iloc[120:].to_csv("input/Test.csv", index=False)
        os.makedirs("Materials/Models")
        params = {"BiasAssessment": False, "Feature": "None", "number_of_k_folds": 3,
                  "apply_grid_search": {"enabled": False, "type": {"Randomized": True, "Exhaustive": False}},
                  "Correlation Limit": 0.7, "Metric For Threshold Optimization": "Balanced Accuracy",
                  "Machine Learning Models": {m.name: m.name == "Logistic Regression" for m in appmod.MODELS}}
        with open("input/machine_learning_parameters.yaml", "w") as f:
            yaml.dump(params, f)
        self.assertEqual(appmod.run_pipeline("input", "output", params), "Pipeline completed successfully")
        self.assertIn("Logistic Regression", pd.read_json("Materials/Models/thresholds.json", typ="series"))

        page = self.results_page()
        self.assertIn("git clone --branch", snippet(page, "code-install"))
        code = snippet(page, "code-predict")
        self.assertIn('"Materials/Models/Logistic Regression_pipeline.pkl"', code)
        data.iloc[120:].to_csv("new_samples.csv", index=False)
        result = run_script(code, self.tmp.name)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("probability_1", result.stdout)

    def test_image_snippet(self):
        make_split("input/train", 9, seed=1)
        make_split("input/test", 3, seed=2)
        from Helpers.image.pipeline import run_image_pipeline
        params = {"models": ["efficientnet_b0"], "mode": "features", "k_folds": 3, "metric": "Balanced Accuracy",
                  "classes": ["lesion", "normal"], "positive_class": "lesion", "window": "auto", "volume": "middle",
                  "augmentation": {}, "epochs": 1, "learning_rate": 1e-4, "patience": 1, "batch_size": 8,
                  "pretrained": False}
        self.assertEqual(run_image_pipeline(os.path.abspath("input"), params), "Pipeline completed successfully")

        code = snippet(self.results_page(), "code-predict")
        self.assertIn('load_model("Materials/Models/EfficientNet-B0.pt")', code)
        os.makedirs("new_images")
        for label in ("lesion", "normal"):
            for name in os.listdir(os.path.join("input/test", label)):
                os.link(os.path.join("input/test", label, name), os.path.join("new_images", name))
        result = run_script(code, self.tmp.name)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])

        # Same predictions as the pipeline's own test predictions
        saved = pd.read_csv("Materials/Predictions/EfficientNet-B0_test_predictions.csv")
        saved.index = saved.file.map(os.path.basename)
        lines = [line.split(" ", 2) for line in result.stdout.splitlines() if line.startswith("new_images/")]
        self.assertEqual(len(lines), len(saved))
        for path, predicted, _ in lines:
            self.assertEqual(predicted, saved.loc[os.path.basename(path), "predicted_class"])
        self.assertTrue(np.all(saved.predicted_class.isin(["lesion", "normal"])))


if __name__ == "__main__":
    unittest.main()
