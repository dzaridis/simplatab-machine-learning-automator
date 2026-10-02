"""Tests of the 3D image classification automator: study and series discovery in every layout,
alignment of the series, the 10 networks, the pipeline (both modes), the standalone code of the
results page and the web flow. Networks are randomly initialised (no download)."""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image3d_fixtures import CLASSES, make_studies  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from Helpers.image3d import volumes  # noqa: E402
from Helpers.image3d.dataset import summarize  # noqa: E402
from Helpers.image3d.explain import grad_cam, strongest_slices  # noqa: E402
from Helpers.image3d.inference import export_model, load_model  # noqa: E402
from Helpers.image3d.models import BY_KEY, NETWORKS, VolumeClassifier, adapt_input  # noqa: E402


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()


class TestStudies(TempDir):
    def test_layouts(self):
        expected = {"full": ["t2", "adc"], "no_study": ["t2", "adc"], "pacs": ["t2_tse", "adc"], "single": ["volume"]}
        for layout, channels in expected.items():
            with self.subTest(layout=layout):
                root = os.path.join(self.dir, layout)
                truth = make_studies(root, per_class=2, seed=1, layout=layout)
                studies, ignored = volumes.scan_split(root)
                self.assertEqual(ignored, [])
                self.assertEqual(sorted(s["id"] for s in studies), sorted(truth))
                self.assertEqual({s["class"] for s in studies}, set(CLASSES))
                self.assertEqual(volumes.default_channels(studies), channels)
                for study in studies:
                    self.assertTrue(study["patient"].startswith("patient_"))
                    for series in study["series"]:
                        if series["kind"] == "dicom":
                            self.assertEqual(series["slices"], 12)

    def test_series_missing_in_some_studies(self):
        make_studies(self.dir, per_class=2, seed=1, layout="full", extra_series=True)  # dwi in half of the studies
        studies, _ = volumes.scan_split(self.dir)
        self.assertEqual(volumes.default_channels(studies), ["t2", "adc"])
        self.assertEqual(sum(volumes.series_of(s, ["t2", "dwi"]) is not None for s in studies), 2)
        self.assertEqual(volumes.series_name("T2 TSE.nii.gz"), "t2_tse")

    def test_series_are_aligned_in_patient_space(self):
        """The ADC series has a coarser grid and a different field of view: after resampling onto
        the T2 grid, the lesion is at the same voxels in both channels."""
        import scipy.ndimage as ndimage
        truth = make_studies(self.dir, per_class=2, seed=3, layout="full")
        studies, _ = volumes.scan_split(self.dir)
        for study in studies:
            if truth[study["id"]] is None:
                continue
            array, meta = volumes.load_study(volumes.series_of(study, ["t2", "adc"]), (12, 40, 40))
            self.assertEqual(array.shape, (2, 12, 40, 40))
            self.assertEqual(meta["spacing"], [3.0, 0.8, 0.8])
            self.assertTrue(0 <= array.min() and array.max() <= 1)
            t2, adc = ndimage.center_of_mass(array[0] > 0.8), ndimage.center_of_mass(array[1] < 0.2)
            np.testing.assert_allclose(t2, adc, atol=0.5)

    def test_crop_and_summary(self):
        make_studies(os.path.join(self.dir, "train"), per_class=3, seed=1, layout="full")
        make_studies(os.path.join(self.dir, "test"), per_class=2, seed=2, layout="pacs")
        summary = summarize(os.path.join(self.dir, "train"), os.path.join(self.dir, "test"))
        self.assertEqual(summary["errors"], [])
        self.assertTrue(summary["volumetric"])
        self.assertEqual((summary["train_studies"], summary["test_studies"], summary["min_class_patients"]), (6, 4, 3))
        self.assertEqual(summary["default_channels"], ["t2", "adc"])
        self.assertTrue(any("do not have the series t2, adc" in w for w in summary["warnings"]))  # test names t2_tse
        cropped = volumes.resize(np.random.rand(1, 4, 40, 40).astype(np.float32), (4, 20, 20), crop=0.5)
        self.assertEqual(cropped.shape, (1, 4, 20, 20))


class TestNetworks(unittest.TestCase):
    def test_every_network(self):
        shape = (32, 64, 64)
        x = torch.rand(2, 2, *shape)
        for spec in NETWORKS:
            with self.subTest(network=spec.name):
                model = VolumeClassifier(spec, 2, shape, 3, pretrained=False)
                logits = model(x)
                self.assertEqual(tuple(logits.shape), (2, 3))
                self.assertEqual(model.features(x).shape[1], model.encoder.num_features)
                logits.sum().backward()
                model.eval()  # no stochastic depth
                with torch.no_grad():
                    torch.testing.assert_close(model.from_early(model.early(x)), model(x))

    def test_input_adaptation_keeps_the_response_to_a_repeated_volume(self):
        conv = torch.nn.Conv3d(3, 8, 3)  # pretrained on RGB, used with 2 series
        adapted = adapt_input(conv, 2)
        volume = torch.rand(1, 1, 5, 6, 6)
        with torch.no_grad():
            torch.testing.assert_close(adapted(volume.repeat(1, 2, 1, 1, 1)), conv(volume.repeat(1, 3, 1, 1, 1)),
                                       atol=1e-5, rtol=1e-4)
        self.assertIs(adapt_input(conv, 3), conv)

    def test_export_and_grad_cam(self):
        shape = (32, 64, 64)
        with tempfile.TemporaryDirectory() as folder:
            for key in ("medicalnet_resnet18", "swin3d_t", "dinov2_25d"):  # shortcut A, channels-last, 2.5D
                with self.subTest(network=key):
                    model = VolumeClassifier(BY_KEY[key], 2, shape, 2, pretrained=False).eval()
                    path = os.path.join(folder, f"{key}.pt")
                    export_model(model, path, ["a", "b"], 0.4, "features",
                                 {"channels": ["t2", "adc"], "shape": shape, "crop": 1.0, "window": "auto"})
                    loaded, info = load_model(path)
                    self.assertEqual((info["channels"], info["shape"], info["threshold"]), (["t2", "adc"], list(shape), 0.4))
                    for _ in range(2):  # traced, not frozen on the example input
                        x = torch.rand(1, 2, *shape)
                        with torch.no_grad():
                            torch.testing.assert_close(loaded(x), torch.softmax(model(x), dim=1), atol=1e-5, rtol=1e-4)
                    cam = grad_cam(model, x, 1)
                    self.assertEqual(cam.shape, shape)
                    self.assertTrue(0 <= cam.min() and cam.max() <= 1)

    def test_strongest_slices(self):
        cam = np.zeros((10, 4, 4))
        cam[7] = 1
        cam[2] = 0.5
        self.assertEqual(strongest_slices(cam, 2), [2, 7])
        self.assertEqual(strongest_slices(np.zeros((10, 4, 4)), 3), [2, 4, 7])  # flat: evenly spaced


class TestPipeline(TempDir):
    def run_pipeline(self, mode, models, layout="full", per_class=6):
        from Helpers.image3d.pipeline import run_image3d_pipeline
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            make_studies("input/train", per_class=per_class, seed=1, layout=layout)
            make_studies("input/test", per_class=3, seed=2, layout=layout)
            summary = summarize("input/train", "input/test")
            params = {"models": models, "mode": mode, "k_folds": 2, "metric": "Balanced Accuracy",
                      "classes": summary["classes"], "positive_class": "malignant",
                      "channels": summary["default_channels"], "shape": [32, 64, 64], "crop": 0.75, "window": "auto",
                      "augmentation": {"horizontal_flip": True, "rotation": True, "intensity": True}, "epochs": 2,
                      "learning_rate": 1e-4, "patience": 1, "batch_size": 4, "pretrained": False}
            return run_image3d_pipeline("input", params), summary
        finally:
            os.chdir(cwd)

    def test_features_outputs_and_standalone_code(self):
        result, summary = self.run_pipeline("features", ["medicalnet_resnet10", "mc3_18"])
        self.assertEqual(result, "Pipeline completed successfully")
        root = os.path.join(self.dir, "Materials")
        for path in ("2_fold_results.xlsx", "test_results.xlsx", "classes.csv", "ROC_Curves/ROC_CURVES.png",
                     "Models/MedicalNet_ResNet-10.pt", "Models/MC3-18.pt", "Predictions/MC3-18_test_predictions.csv",
                     "GradCAM/MedicalNet_ResNet-10/MedicalNet_ResNet-10_gradcam_malignant.png"):
            self.assertTrue(os.path.exists(os.path.join(root, path)), path)
        info = json.load(open(os.path.join(root, "run_info.json")))
        self.assertEqual((info["dim"], info["channels"], info["shape"], info["crop"]), (3, ["t2", "adc"], [32, 64, 64], 0.75))
        self.assertEqual(info["classes"], ["benign", "malignant"])

        # The code of the results page, run without the repository, gives the pipeline's predictions
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader(os.path.join(REPO, "templates")), autoescape=True)
        code = str(env.get_template("_usage.html").module.image3d_code("MedicalNet_ResNet-10.pt", info))
        self.assertIn("SimpleITK", code)
        predictions = os.path.join(root, "Predictions", "MedicalNet_ResNet-10_test_predictions.csv")
        script = code.split("\nprint(predict(")[0] + f"""
import pandas as pd
reference = pd.read_csv({predictions!r})
for _, row in reference.iterrows():
    folder = os.path.join({os.path.join(self.dir, 'input', 'test')!r}, row.study)
    label, p = predict([os.path.join(folder, "t2"), os.path.join(folder, "adc.nii.gz")])
    assert label == row.predicted_class, (row.study, label, row.predicted_class)
    assert abs(p[1] - row.probability_malignant) < 1e-3, (row.study, p, row.probability_malignant)
import sys
assert not {{m.split('.')[0] for m in sys.modules}} & {{'Helpers', 'web'}}
"""
        env_vars = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        run = subprocess.run([sys.executable, "-c", script], cwd=self.dir, capture_output=True, text=True, env=env_vars,
                             timeout=600)
        self.assertEqual(run.returncode, 0, run.stderr[-3000:])

    def test_finetune_single_series(self):
        result, summary = self.run_pipeline("finetune", ["r3d_18"], layout="single", per_class=5)
        self.assertEqual(result, "Pipeline completed successfully")
        self.assertEqual(summary["default_channels"], ["volume"])
        self.assertTrue(os.path.exists(os.path.join(self.dir, "Materials", "Models", "R3D-18.pt")))

    def test_too_few_patients_for_the_folds(self):
        from Helpers.image3d.pipeline import run_image3d_pipeline
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            make_studies("input/train", per_class=2, seed=1, layout="no_study")
            make_studies("input/test", per_class=1, seed=2, layout="no_study")
            params = {"models": ["mc3_18"], "mode": "features", "k_folds": 3, "metric": "AUC", "classes": list(CLASSES),
                      "positive_class": "malignant", "channels": ["t2"], "shape": [32, 64, 64], "crop": 1.0,
                      "window": "auto", "augmentation": {}, "epochs": 1, "learning_rate": 1e-4, "patience": 1,
                      "batch_size": 2, "pretrained": False}
            self.assertIn("at least 3 readable training patients", run_image3d_pipeline("input", params))
        finally:
            os.chdir(cwd)


class TestWebFlow(TempDir):
    def setUp(self):
        super().setUp()
        import app as appmod
        from werkzeug.test import Client
        from web.jobs import PipelineJob
        self.app = appmod
        self.cwd = os.getcwd()
        os.chdir(self.dir)
        self.client = Client(appmod.application)
        appmod.job = PipelineJob()
        self.patch = mock.patch.object(appmod, "IMAGE_INPUT_FOLDER", os.path.join(self.dir, "images"))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        os.chdir(self.cwd)
        super().tearDown()

    def upload(self, layout="full"):
        data = {}
        for split, seed in (("train", 1), ("test", 2)):
            folder = os.path.join(self.dir, "src", split)
            make_studies(folder, per_class=3, seed=seed, layout=layout, extra_series=True)
            archive = shutil.make_archive(os.path.join(self.dir, split), "zip", folder)
            with open(archive, "rb") as f:
                data[f"{split}_zip"] = (io.BytesIO(f.read()), f"{split.capitalize()}.zip")
        return self.client.post("/automl/image/upload", data=data, headers={"X-Requested-With": "XMLHttpRequest"},
                                content_type="multipart/form-data")

    def test_upload_parameters_run_and_2d_switch(self):
        self.assertEqual(self.upload().get_json(), {"redirect": "/automl/image/parameters"})
        page = self.client.get("/automl/image/parameters").get_data(as_text=True)
        for network in NETWORKS:
            self.assertIn(f'name="{network.key}"', page)
        for text in ("Reference series", 'value="dwi"', "Volume size", "grouped by patient", "6</div><div class=\"small-muted\">train studies"):
            self.assertIn(text, page)
        self.assertIn("2D mode", self.client.get("/automl/image/parameters?dim=2").get_data(as_text=True))

        received = {}

        def fake(folder, params):
            received.update(params)
            print("------------- \n", "Preparing images \n", "-------------")
            return "Pipeline completed successfully"

        form = {"dim": "3", "medicalnet_resnet10": "true", "swinvit_ssl": "true", "channels": ["t2", "adc", "dwi"],
                "reference": "adc", "mode": "features", "k_folds": "3", "shape": "64x128x128", "crop": "0.75",
                "window": "auto", "positive_class": "malignant", "optimization_metric": "AUC"}
        with mock.patch("Helpers.image3d.pipeline.run_image3d_pipeline", fake):
            response = self.client.post("/automl/image/parameters", data=form)
            self.assertEqual(response.headers["Location"], "/automl/run")
            for _ in range(100):
                if not self.app.job.running:
                    break
                time.sleep(0.05)
        self.assertEqual(received["models"], ["medicalnet_resnet10", "swinvit_ssl"])
        self.assertEqual(received["channels"], ["adc", "t2", "dwi"])  # the reference first
        self.assertEqual((received["shape"], received["crop"], received["k_folds"]), ([64, 128, 128], 0.75, 3))
        self.assertEqual(self.client.get("/automl/api/status").get_json()["state"], "done")

        # Too many folds for 3 patients per class, no series
        self.assertEqual(self.client.post("/automl/image/parameters", data=dict(form, k_folds="4")).headers["Location"],
                         "/automl/image/parameters")
        self.assertEqual(self.client.post("/automl/image/parameters", data=dict(form, channels=[])).headers["Location"],
                         "/automl/image/parameters")

    def test_results_page(self):
        materials = os.path.join(self.dir, "Materials")
        for folder in ("GradCAM/MedicalNet_ResNet-10", "Models", "Predictions"):
            os.makedirs(os.path.join(materials, folder), exist_ok=True)
        import pandas as pd
        metrics = ["Sensitivity", "Specificity", "AUC", "F-score", "Accuracy", "Balanced Accuracy"]
        pd.DataFrame([[0.8, 0.9, 0.93, 0.85, 0.88, 0.86]], index=["MedicalNet ResNet-10"], columns=metrics).to_excel(
            os.path.join(materials, "test_results.xlsx"))
        pd.DataFrame({"index": [0, 1], "class": ["benign", "malignant"], "train_images": [6, 6], "test_images": [3, 3]}
                     ).to_csv(os.path.join(materials, "classes.csv"), index=False)
        for path in ("GradCAM/MedicalNet_ResNet-10/MedicalNet_ResNet-10_gradcam_malignant.png",
                     "Models/MedicalNet_ResNet-10.pt", "Predictions/MedicalNet_ResNet-10_test_predictions.csv"):
            with open(os.path.join(materials, path), "wb") as f:
                f.write(b"x")
        with open(os.path.join(materials, "run_info.json"), "w") as f:
            json.dump({"automator": "image-classification", "dim": 3, "mode": "features", "k_folds": 3, "metric": "AUC",
                       "device": "CPU", "channels": ["t2", "adc"], "shape": [32, 128, 128], "crop": 1.0,
                       "window": "auto", "ct_window": None, "classes": ["benign", "malignant"]}, f)
        page = self.client.get("/automl/results").get_data(as_text=True)
        for text in ("3D Grad-CAM", "series t2, adc", "volume 32 × 128 × 128", "grouped by patient", "Training studies",
                     "MedicalNet ResNet-10", "SimpleITK", "new_study/t2", "Class malignant"):
            self.assertIn(text, page)


if __name__ == "__main__":
    unittest.main()
