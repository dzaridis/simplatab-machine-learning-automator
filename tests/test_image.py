"""Tests of the image classification automator. Networks are created without pretrained
weights (no download): the pipeline, the metrics and the exports are the same."""
import io
import os
import sys
import tempfile
import time
import unittest
import zipfile
from unittest import mock

import numpy as np
import torch
from werkzeug.test import Client

sys.path.insert(0, os.path.dirname(__file__))
from image_fixtures import KINDS, make_split, write_image, zip_folder  # noqa: E402

import app as appmod  # noqa: E402
from Helpers.image import dataset, io as mio  # noqa: E402
from Helpers.image.explain import grad_cam  # noqa: E402
from Helpers.image.inference import load_model, predict  # noqa: E402
from Helpers.image.models import BACKBONES, BY_KEY, INPUT_SIZE, ImageClassifier  # noqa: E402
from Helpers.image.pipeline import run_image_pipeline  # noqa: E402
from Helpers.image.training import fit_linear_head, linear_head_weights  # noqa: E402
from web.jobs import PipelineJob  # noqa: E402


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()


class TestMedicalImageReading(TempDirTestCase):
    def test_every_format_is_read_with_the_lesion_bright(self):
        for kind in KINDS + ["dicom_jpeg_rgb"]:
            with self.subTest(kind=kind):
                lesion = mio.load_image(write_image(os.path.join(self.dir, kind, "a"), kind, True, np.random.default_rng(3)))
                normal = mio.load_image(write_image(os.path.join(self.dir, kind, "b"), kind, False, np.random.default_rng(3)))
                self.assertEqual(lesion.dtype, np.uint8)
                self.assertIn(lesion.ndim, (2, 3))
                gray = lambda a: a if a.ndim == 2 else a.mean(axis=-1)  # noqa: E731
                self.assertGreater(np.percentile(gray(lesion), 99), np.percentile(gray(normal), 99))

    def test_dicom_details(self):
        rng = np.random.default_rng(0)
        # Extensionless DICOM files are recognised
        path = write_image(os.path.join(self.dir, "noext"), "dicom_noext", True, rng)
        self.assertEqual(mio.file_kind(path), "dicom")
        # CT soft-tissue window from the header: -400 HU background is black
        ct = mio.load_image(write_image(os.path.join(self.dir, "ct"), "dicom_ct", False, rng))
        self.assertEqual(np.median(ct), 0)
        # A lung window keeps the background visible
        lung = mio.load_image(os.path.join(self.dir, "ct.dcm"), window="lung")
        self.assertGreater(np.median(lung), 0)
        # Compressed colour DICOM (YBR JPEG) comes out in RGB: red > green > blue as generated
        rgb = mio.load_image(write_image(os.path.join(self.dir, "rgb"), "dicom_jpeg_rgb", True, rng))
        red, green, blue = rgb.reshape(-1, 3).mean(axis=0)
        self.assertGreater(red, green)
        self.assertGreater(green, blue)
        # Volumes: the middle slice misses a lesion of the first slice, the maximum intensity projection shows it
        import nibabel as nib
        volume = 100 + 10 * rng.standard_normal((40, 40, 5))
        volume[10:20, 10:20, 0] = 1000
        path = os.path.join(self.dir, "vol.nii.gz")
        nib.save(nib.Nifti1Image(volume.astype(np.float32), np.eye(4)), path)
        bright = lambda a: np.mean(a >= 250)  # noqa: E731
        self.assertGreater(bright(mio.load_image(path, volume="mip")), 0.05)
        self.assertLess(bright(mio.load_image(path, volume="middle")), 0.02)
        self.assertTrue(mio.image_info(path)["volume"])

    def test_letterbox_keeps_proportions(self):
        image = mio.letterbox(np.full((20, 80), 255, np.uint8), 64)
        array = np.asarray(image)
        self.assertEqual(array.shape, (64, 64))
        self.assertEqual(array[0, 32], 0)     # padding above
        self.assertEqual(array[32, 32], 255)  # image in the middle


class TestDataset(TempDirTestCase):
    def test_zip_extraction_and_summary(self):
        make_split(os.path.join(self.dir, "src", "train"), 8, seed=1)  # every file kind
        make_split(os.path.join(self.dir, "src", "test"), 3, seed=2)
        for split in ("train", "test"):
            archive = zip_folder(os.path.join(self.dir, "src", split), os.path.join(self.dir, f"{split}.zip"), prefix="Wrapper")
            with zipfile.ZipFile(archive, "a") as z:  # macOS metadata and junk are ignored
                z.writestr("__MACOSX/Wrapper/._x", b"x")
                z.writestr("Wrapper/lesion/notes.txt", b"not an image")
            dataset.extract_zip(archive, os.path.join(self.dir, split))
        summary = dataset.summarize(os.path.join(self.dir, "train"), os.path.join(self.dir, "test"))
        self.assertEqual(summary["errors"], [])
        self.assertEqual(summary["classes"], ["lesion", "normal"])
        self.assertEqual(summary["positive_class"], "lesion")  # "normal" is the usual negative class
        self.assertEqual((summary["train_images"], summary["test_images"]), (16, 6))
        self.assertTrue(summary["has_ct"] and summary["has_volumes"])
        self.assertEqual(summary["ignored"], 2)

    def test_upload_errors(self):
        make_split(os.path.join(self.dir, "train"), 3, kinds=["png8"], classes=("lesion",))
        make_split(os.path.join(self.dir, "test"), 2, kinds=["png8"], classes=("lesion", "other"))
        errors = dataset.summarize(os.path.join(self.dir, "train"), os.path.join(self.dir, "test"))["errors"]
        self.assertTrue(any("single class" in e for e in errors))
        self.assertTrue(any("not in Train.zip: other" in e for e in errors))

    def test_duplicates_between_splits_are_reported(self):
        make_split(os.path.join(self.dir, "train"), 4, kinds=["png8"], seed=1)
        make_split(os.path.join(self.dir, "test"), 2, kinds=["png8"], seed=1)  # same seed: the first lesions repeat
        summary = dataset.summarize(os.path.join(self.dir, "train"), os.path.join(self.dir, "test"))
        self.assertEqual(summary["duplicates"], 2)
        self.assertTrue(any("identical to training images" in w for w in summary["warnings"]))

    def test_unsafe_zips_are_refused(self):
        for name in ("../evil.png", "/abs.png"):
            archive = os.path.join(self.dir, "evil.zip")
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr(name, b"x")
            with self.assertRaises(dataset.DatasetError):
                dataset.extract_zip(archive, os.path.join(self.dir, "out"))
        with open(os.path.join(self.dir, "bad.zip"), "wb") as f:
            f.write(b"not a zip")
        with self.assertRaises(dataset.DatasetError):
            dataset.extract_zip(os.path.join(self.dir, "bad.zip"), os.path.join(self.dir, "out"))

    def test_default_positive_class(self):
        self.assertEqual(dataset.default_positive_class(["abnormal", "normal"]), "abnormal")
        self.assertEqual(dataset.default_positive_class(["benign", "malignant"]), "malignant")
        self.assertEqual(dataset.default_positive_class(["cat", "dog"]), "dog")
        self.assertIsNone(dataset.default_positive_class(["a", "b", "c"]))


class TestNetworks(unittest.TestCase):
    def test_ten_networks(self):
        self.assertEqual(len(BACKBONES), 10)
        self.assertEqual(len({b.key for b in BACKBONES}), 10)
        self.assertFalse(any("/" in b.name for b in BACKBONES))  # used in file names

    def test_linear_head_equals_logistic_regression(self):
        rng = np.random.default_rng(0)
        for classes in (2, 3):
            features = rng.standard_normal((60, 8)) + np.repeat(np.eye(classes, 8) * 2, 60 // classes, axis=0)
            labels = np.repeat(np.arange(classes), 60 // classes)
            head = fit_linear_head(features, labels)
            weight, bias = linear_head_weights(head, classes)
            logits = features @ weight.T + bias
            softmax = np.exp(logits - logits.max(1, keepdims=True))
            softmax /= softmax.sum(1, keepdims=True)
            np.testing.assert_allclose(softmax, head.predict_proba(features), atol=1e-5)

    def test_grad_cam_for_each_feature_layout(self):
        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        for key in ("efficientnet_b0", "swin_tiny", "vit_small"):  # nchw, nhwc, tokens
            with self.subTest(key=key):
                model = ImageClassifier(BY_KEY[key], 2, pretrained=False)
                cam = grad_cam(model, x, 1)
                self.assertEqual(cam.shape, (INPUT_SIZE, INPUT_SIZE))
                self.assertTrue(0 <= cam.min() and cam.max() <= 1)


class TestPipeline(TempDirTestCase):
    def make_input(self, classes=("lesion", "normal"), per_class=12):
        inputs = os.path.join(self.dir, "input")
        make_split(os.path.join(inputs, "train"), per_class, seed=1, classes=classes)
        make_split(os.path.join(inputs, "test"), 4, seed=2, classes=classes)
        return inputs, dataset.summarize(os.path.join(inputs, "train"), os.path.join(inputs, "test"))

    def params(self, summary, **overrides):
        params = {"models": ["efficientnet_b0"], "mode": "features", "k_folds": 3, "metric": "Balanced Accuracy",
                  "classes": summary["classes"], "positive_class": summary["positive_class"], "window": "auto",
                  "volume": "middle", "augmentation": {"rotation": True, "intensity": True},
                  "epochs": 2, "learning_rate": 1e-4, "patience": 1, "batch_size": 8, "pretrained": False}
        params.update(overrides)
        return params

    def run_in(self, inputs, params):
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            return run_image_pipeline(inputs, params)
        finally:
            os.chdir(cwd)

    def test_feature_extraction_run_and_exported_model(self):
        inputs, summary = self.make_input()
        self.assertEqual(self.run_in(inputs, self.params(summary)), "Pipeline completed successfully")
        materials = os.path.join(self.dir, "Materials")
        for path in ("test_results.xlsx", "3_fold_results.xlsx", "classes.csv", "run_info.json",
                     "ROC_Curves/ROC_CURVES.png", "Models/EfficientNet-B0.pt",
                     "Predictions/EfficientNet-B0_test_predictions.csv",
                     "GradCAM/EfficientNet-B0/EfficientNet-B0_gradcam_lesion.png",
                     "ConfusionMatrices/EfficientNet-B0_Test_confusion_matrix.png"):
            self.assertTrue(os.path.exists(os.path.join(materials, path)), path)

        # The exported network reproduces the test predictions from the original files
        import pandas as pd
        model, info = load_model(os.path.join(materials, "Models", "EfficientNet-B0.pt"))
        self.assertEqual(info["classes"], ["normal", "lesion"])  # the positive class is class 1
        table = pd.read_csv(os.path.join(materials, "Predictions", "EfficientNet-B0_test_predictions.csv"))
        rows = predict(model, info, [os.path.join(inputs, "test", f) for f in table.file])
        np.testing.assert_allclose([r["probabilities"]["lesion"] for r in rows], table.probability_lesion, atol=1e-4)
        self.assertEqual([r["predicted_class"] for r in rows], list(table.predicted_class))

    def test_fine_tuning_multiclass_run(self):
        inputs, summary = self.make_input(classes=("a_lesion", "b_normal", "c_other"), per_class=9)
        result = self.run_in(inputs, self.params(summary, mode="finetune", models=["efficientnet_b0", "vit_small"]))
        self.assertEqual(result, "Pipeline completed successfully")
        import pandas as pd
        test = pd.read_excel(os.path.join(self.dir, "Materials", "test_results.xlsx"), index_col=0)
        self.assertEqual(sorted(test.index), ["EfficientNet-B0", "ViT-Small"])
        self.assertEqual(load_model(os.path.join(self.dir, "Materials", "Models", "ViT-Small.pt"))[1]["threshold"], None)

    def test_too_few_images_for_the_folds(self):
        inputs, summary = self.make_input(per_class=4)
        self.assertIn("at least 5 readable training images", self.run_in(inputs, self.params(summary, k_folds=5)))


class TestImageWebFlow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.client = Client(appmod.application)
        appmod.job = PipelineJob()
        self.patch = mock.patch.object(appmod, "IMAGE_INPUT_FOLDER", os.path.join(self.tmp.name, "images"))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def zips(self, train_classes=("lesion", "normal"), test_classes=("lesion", "normal")):
        src = os.path.join(self.tmp.name, "src")
        make_split(os.path.join(src, "train"), 8, seed=1, classes=train_classes)  # every file kind
        make_split(os.path.join(src, "test"), 3, seed=2, classes=test_classes)
        data = {}
        for split in ("train", "test"):
            with open(zip_folder(os.path.join(src, split), os.path.join(self.tmp.name, f"{split}.zip")), "rb") as f:
                data[f"{split}_zip"] = (io.BytesIO(f.read()), f"{split.capitalize()}.zip")
        return data

    def upload(self, data, xhr=True):
        headers = {"X-Requested-With": "XMLHttpRequest"} if xhr else {}
        return self.client.post("/automl/image/upload", data=data, headers=headers, content_type="multipart/form-data")

    def test_pages(self):
        page = self.client.get("/automl/image").get_data(as_text=True)
        self.assertIn("Train.zip", page)
        self.assertIn("5 GB", page)
        self.assertEqual(self.client.get("/automl/image/parameters").headers["Location"], "/automl/image")

    def test_upload_then_parameters(self):
        response = self.upload(self.zips())
        self.assertEqual(response.get_json(), {"redirect": "/automl/image/parameters"})
        page = self.client.get("/automl/image/parameters").get_data(as_text=True)
        for backbone in BACKBONES:
            self.assertIn(f'name="{backbone.key}"', page)
        self.assertIn("Positive class", page)
        self.assertIn("Intensity window", page)  # DICOM files
        self.assertIn("3D volumes", page)        # NIfTI and multi-frame DICOM

    def test_upload_errors(self):
        response = self.upload(self.zips(train_classes=("lesion", "normal"), test_classes=("lesion", "other")))
        self.assertEqual(response.status_code, 400)
        self.assertIn("not in Train.zip: other", response.get_json()["error"])
        data = self.zips()
        data["train_zip"] = (io.BytesIO(b"x"), "Train.txt")
        self.assertIn("zip", self.upload(data).get_json()["error"])
        # Without JavaScript: redirect with the message
        data = self.zips()
        data["test_zip"] = (io.BytesIO(b"not a zip"), "Test.zip")
        response = self.upload(data, xhr=False)
        self.assertEqual(response.headers["Location"], "/automl/image")
        self.assertIn("not a valid zip file", self.client.get("/automl/image").get_data(as_text=True))

    def test_run_in_background(self):
        self.upload(self.zips())
        form = {"efficientnet_b0": "true", "dinov2_small": "true", "mode": "features", "k_folds": "3",
                "optimization_metric": "AUC", "positive_class": "lesion", "window": "lung", "volume": "mip"}
        received = {}

        def fake(folder, params):
            received.update(params)
            print("------------- \n", "Preparing images \n", "-------------")
            print("------------- \n", "Training on K-Fold cross validation \n", "-------------")
            for name in ("EfficientNet-B0", "DINOv2-Small"):
                print("-------------------- \n", f"{name} is starting \n", "--------------------")
                print("-------------------- \n", f"{name} is completed successfully \n", "--------------------")
            return "Pipeline completed successfully"

        with mock.patch("Helpers.image.pipeline.run_image_pipeline", fake):
            response = self.client.post("/automl/image/parameters", data=form)
            self.assertEqual(response.headers["Location"], "/automl/run")
            for _ in range(100):
                if not appmod.job.running:
                    break
                time.sleep(0.05)
        status = self.client.get("/automl/api/status").get_json()
        self.assertEqual(status["automator"], "image-classification")
        self.assertEqual([p[0] for p in status["phases"]], ["prep", "kfold", "test", "done"])
        self.assertEqual(status["state"], "done")
        self.assertEqual(received["models"], ["efficientnet_b0", "dinov2_small"])
        self.assertEqual((received["window"], received["volume"], received["metric"]), ("lung", "mip", "AUC"))
        self.assertEqual(received["augmentation"]["horizontal_flip"], False)
        page = self.client.get("/automl/run").get_data(as_text=True)
        self.assertIn("Preparing images", page)
        self.assertIn("Image Classification", page)

    def test_invalid_settings(self):
        self.upload(self.zips())
        response = self.client.post("/automl/image/parameters", data={"mode": "features", "k_folds": "3"})
        self.assertEqual(response.headers["Location"], "/automl/image/parameters")
        response = self.client.post("/automl/image/parameters", data={"efficientnet_b0": "true", "k_folds": "50"})
        self.assertEqual(response.headers["Location"], "/automl/image/parameters")
        self.assertEqual(appmod.job.state, "idle")

    def test_results_page(self):
        materials = os.path.join(self.tmp.name, "Materials")
        for folder in ("GradCAM/EfficientNet-B0", "Models", "Predictions"):
            os.makedirs(os.path.join(materials, folder), exist_ok=True)
        import pandas as pd
        metrics = ["Sensitivity", "Specificity", "AUC", "F-score", "Accuracy", "Balanced Accuracy"]
        pd.DataFrame([[0.8, 0.9, 0.93, 0.85, 0.88, 0.86]], index=["EfficientNet-B0"], columns=metrics).to_excel(
            os.path.join(materials, "test_results.xlsx"))
        pd.DataFrame({"index": [0, 1], "class": ["normal", "lesion"], "train_images": [6, 6], "test_images": [3, 3]}).to_csv(
            os.path.join(materials, "classes.csv"), index=False)
        for path in ("GradCAM/EfficientNet-B0/EfficientNet-B0_gradcam_lesion.png", "Models/EfficientNet-B0.pt",
                     "Predictions/EfficientNet-B0_test_predictions.csv"):
            with open(os.path.join(materials, path), "wb") as f:
                f.write(b"x")
        with open(os.path.join(materials, "run_info.json"), "w") as f:
            f.write('{"automator": "image-classification", "mode": "features", "k_folds": 3, "metric": "AUC", "device": "CPU"}')
        page = self.client.get("/automl/results").get_data(as_text=True)
        for text in ("Grad-CAM", "Class lesion", "Test predictions", "Trained networks", "External test set (Test.zip)",
                     "class 1 is the positive class", "Feature extraction", "/automl/image"):
            self.assertIn(text, page)


if __name__ == "__main__":
    unittest.main()
