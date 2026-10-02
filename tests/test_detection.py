"""Tests of the object detection automator: annotation formats (2D and 3D), metrics (checked
against pycocotools), 3D merging, splits, the ten networks, D-RISE, the pipeline end to end,
the exported networks and the code of the results page, and the web flow. Networks are built
without pretrained weights (no download)."""
import contextlib
import html
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import pandas as pd
import torch
from werkzeug.test import Client

sys.path.insert(0, os.path.dirname(__file__))
from detection_fixtures import CLASSES, make_2d, make_3d  # noqa: E402
from image_fixtures import zip_folder  # noqa: E402

import app as appmod  # noqa: E402
from Helpers.detection import dataset as dd  # noqa: E402
from Helpers.detection import metrics as M  # noqa: E402
from Helpers.detection import volumes  # noqa: E402
from Helpers.detection.annotations import AnnotationError, load_split  # noqa: E402
from Helpers.detection.explain import drise  # noqa: E402
from Helpers.detection.merge import merge_slices  # noqa: E402
from Helpers.detection.models import DETECTORS, build, input_size  # noqa: E402
from Helpers.detection.pipeline import run_detection_pipeline  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()


def _same(truth, annotated):
    """The parsed boxes are those written by the fixtures."""
    by_name = {s.name: s for s in annotated.samples}
    for name, objects in truth.items():
        sample = by_name[name]
        expected = sorted((c, tuple(float(v) for v in b)) for c, b in objects)
        got = sorted((label, tuple(float(v) for v in np.round(box, 3))) for label, box in zip(sample.labels, sample.boxes))
        if expected != got:
            return False
    return len(by_name) == len(truth)


class TestAnnotations(TempDir):
    def test_every_2d_format(self):
        for fmt in ("coco", "yolo", "voc", "csv", "mask"):
            with self.subTest(fmt=fmt):
                folder = os.path.join(self.dir, fmt)
                truth = make_2d(folder, 4, fmt, nested=fmt == "coco")
                annotated = load_split(folder)
                self.assertEqual((annotated.format, annotated.dim), (fmt, 2))
                self.assertEqual(annotated.classes, list(CLASSES))
                self.assertTrue(_same(truth, annotated))
                self.assertEqual(sum(not s.labels for s in annotated.samples), 1)  # the negative image
        groups = {s.group for s in load_split(os.path.join(self.dir, "coco")).samples}
        self.assertEqual(groups, {"patient_00", "patient_01", "patient_02"})  # one sub-folder per patient

    def test_3d_boxes_masks_and_dicom_series(self):
        for fmt in ("csv", "mask"):
            with self.subTest(fmt=fmt):
                folder = os.path.join(self.dir, fmt)
                truth = make_3d(folder, 3, fmt, series=1 if fmt == "csv" else 0)
                annotated = load_split(folder)
                self.assertEqual((annotated.format, annotated.dim), (fmt, 3))
                self.assertTrue(_same(truth, annotated))
        # The DICOM series (shuffled files) is read in slice order, like the NIfTI volumes
        series = [s for s in load_split(os.path.join(self.dir, "csv")).samples if os.path.isdir(s.path)][0]
        volume = volumes.read_volume(series.path)
        self.assertEqual(volume.shape, (16, 48, 48))
        x1, y1, z1, x2, y2, z2 = series.boxes[0].astype(int)
        self.assertGreater(volume[z1:z2, y1:y2, x1:x2].mean(), volume.mean() + 50)

    def test_errors(self):
        os.makedirs(os.path.join(self.dir, "empty"))
        make_2d(os.path.join(self.dir, "empty"), 2, "voc")
        for f in os.listdir(os.path.join(self.dir, "empty")):
            if f.endswith(".xml"):
                os.remove(os.path.join(self.dir, "empty", f))
        with self.assertRaisesRegex(AnnotationError, "No annotations found"):
            load_split(os.path.join(self.dir, "empty"))
        both = os.path.join(self.dir, "both")
        make_2d(both, 2, "coco")
        make_2d(os.path.join(both, "more"), 1, "voc")
        with self.assertRaisesRegex(AnnotationError, "Several annotation formats"):
            load_split(both)


class TestMetricsAndMerging(unittest.TestCase):
    def test_average_precision_matches_pycocotools(self):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        rng = np.random.default_rng(0)
        truth, detections, images, annotations, results = {}, {}, [], [], []
        for i in range(15):
            n = int(rng.integers(0, 4))
            xy = rng.uniform(0, 80, (n, 2))
            boxes = np.hstack([xy, xy + rng.uniform(5, 40, (n, 2))])
            labels = rng.integers(0, 3, n)
            truth[i] = (boxes, labels)
            images.append({"id": i, "width": 128, "height": 128})
            for b, label in zip(boxes, labels):
                annotations.append({"id": len(annotations) + 1, "image_id": i, "category_id": int(label), "iscrowd": 0,
                                    "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "area": float(np.prod(b[2:] - b[:2]))})
            noisy = [b + rng.normal(0, 3, 4) for b in boxes] + [np.r_[q, q + 20] for q in rng.uniform(0, 80, (5, 2))]
            det_boxes, det_scores = np.array(noisy).reshape(-1, 4), rng.uniform(0, 1, n + 5)
            det_labels = np.r_[labels, rng.integers(0, 3, 5)]
            detections[i] = (det_boxes, det_scores, det_labels)
            results += [{"image_id": i, "category_id": int(label), "score": float(s), "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]}
                        for b, s, label in zip(det_boxes, det_scores, det_labels)]
        with contextlib.redirect_stdout(io.StringIO()):
            coco = COCO()
            coco.dataset = {"images": images, "annotations": annotations, "categories": [{"id": c} for c in range(3)]}
            coco.createIndex()
            evaluation = COCOeval(coco, coco.loadRes(results), "bbox")
            evaluation.evaluate()
            evaluation.accumulate()
            evaluation.summarize()
        ap, ar = M.average_precision(truth, detections, 3, M.IOU_THRESHOLDS[2])
        np.testing.assert_allclose([np.nanmean(ap), np.nanmean(ap[:, 0]), np.nanmean(ap[:, 5]), np.nanmean(ar)],
                                   [evaluation.stats[0], evaluation.stats[1], evaluation.stats[2], evaluation.stats[8]], atol=1e-9)

    def test_3d_iou_froc_and_operating_point(self):
        a = np.array([[0, 0, 0, 10, 10, 10]])
        self.assertAlmostEqual(M.iou_matrix(a, np.array([[0, 0, 5, 10, 10, 15]]))[0, 0], 500 / 1500)
        truth = {0: (np.array([[0, 0, 10, 10]]), np.array([0])), 1: (np.zeros((0, 4)), np.zeros(0, int))}
        detections = {0: (np.array([[0, 0, 10, 10], [50, 50, 60, 60]]), np.array([0.9, 0.3]), np.array([0, 0])),
                      1: (np.array([[5, 5, 9, 9]]), np.array([0.6]), np.array([0]))}
        _, _, cpm = M.froc(truth, detections, 0.5)
        self.assertEqual(cpm, 1.0)  # the box is found before any false positive
        self.assertAlmostEqual(M.best_threshold(truth, detections, 0.5), 0.9)
        point = M.at_threshold(truth, detections, 0.9, 0.5)
        self.assertEqual((point["Precision"], point["Recall"], point["sensitivity"], point["specificity"]), (1, 1, 1, 1))
        point = M.at_threshold(truth, detections, 0.5, 0.5)
        self.assertEqual((point["Precision"], point["specificity"]), (0.5, 0))

    def test_merge_slices(self):
        slices = {z: (np.array([[10, 10, 20, 20], [40, 40, 50, 50]]), np.array([0.9, 0.2]), np.array([0, 1])) for z in range(3, 7)}
        slices[8] = (np.array([[10, 10, 20, 20]]), np.array([0.8]), np.array([0]))  # gap: a new object
        boxes, scores, labels = merge_slices(slices)
        self.assertEqual(boxes.tolist()[0], [10, 10, 3, 20, 20, 7])
        self.assertEqual(sorted(map(tuple, boxes.tolist())), [(10, 10, 3, 20, 20, 7), (10, 10, 8, 20, 20, 9), (40, 40, 3, 50, 50, 7)])
        self.assertEqual(scores.tolist(), [0.9, 0.8, 0.2])


class TestSplitsAndNetworks(TempDir):
    def test_units_and_grouped_folds(self):
        make_2d(os.path.join(self.dir, "2d"), 10, "coco", nested=True, negatives=2)
        annotated = load_split(os.path.join(self.dir, "2d"))
        items, _ = dd.cache_split(annotated, os.path.join(self.dir, "cache"), annotated.classes)
        for fit, val in dd.kfold_splits(items, range(len(items)), 3):
            self.assertFalse({items[i].group for i in fit} & {items[i].group for i in val})  # patients not split
        fit, val = dd.holdout_split(items, range(len(items)), 0.25)
        self.assertEqual(len(fit) + len(val), len(items))
        make_3d(os.path.join(self.dir, "3d"), 2, "csv", negatives=1)
        annotated = load_split(os.path.join(self.dir, "3d"))
        items, _ = dd.cache_split(annotated, os.path.join(self.dir, "cache3d"), annotated.classes)
        units = dd.units(items, range(len(items)), negative_ratio=1.0)
        for k, item in enumerate(items):
            mine = [u for u in units if u.item == k]
            positive = {z for b in item.boxes for z in range(int(b[2]), int(b[5]))}
            self.assertEqual({u.z for u in mine if len(u.boxes)}, positive)  # every slice crossing a box
            negatives = [u for u in mine if not len(u.boxes)]
            self.assertEqual(len(negatives), min(item.shape[0] - len(positive), max(2, len(positive))))
            for u in mine:  # slice boxes are the cross-sections of the 3D boxes
                expected, _ = dd.slice_boxes(item, u.z)
                np.testing.assert_array_equal(u.boxes, expected)

    def test_every_network_trains_and_predicts(self):
        for spec in DETECTORS:
            with self.subTest(network=spec.key):
                size = input_size(spec, 160)
                model = build(spec, 2, size, pretrained=False)
                images = torch.rand(2, 3, size, size)
                targets = [{"boxes": torch.tensor([[10.0, 20.0, 60.0, 70.0]]), "labels": torch.tensor([1])},
                           {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)}]
                model.train()
                loss = model.loss(images, targets)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                model.eval()
                boxes, scores, labels = model.predict(images)[0]
                self.assertLessEqual(len(boxes), 100)
                self.assertTrue(((labels >= 0) & (labels < 2)).all())

    def test_drise_highlights_what_the_detection_needs(self):
        class Fake(torch.nn.Module):
            """Detects a box at the top-left corner, with a score = how visible that region is."""
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1))

            def predict(self, images):
                return [(torch.tensor([[0.0, 0.0, 16.0, 16.0]]), image[:, :16, :16].mean().view(1), torch.tensor([0])) for image in images]
        saliency = drise(Fake(), torch.ones(3, 64, 64), np.array([0, 0, 16, 16]), 0, n_masks=200, grid=8)
        self.assertEqual(saliency.shape, (64, 64))
        self.assertGreater(saliency[:16, :16].mean(), saliency[32:, 32:].mean() + 0.2)


class TestPipeline(TempDir):
    def run_pipeline(self, models, dim, validation, size):
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            if dim == 2:
                make_2d("input/train", 12, "coco", seed=1, negatives=2, nested=True)
                make_2d("input/test", 4, "voc", seed=2, negatives=1)
            else:
                make_3d("input/train", 5, "csv", seed=1, size=40, depth=12, negatives=1, series=1)
                make_3d("input/test", 2, "mask", seed=2, size=40, depth=12)
            return run_detection_pipeline("input", {
                "models": models, "validation": validation, "k_folds": 2, "holdout_fraction": 0.25, "epochs": 1,
                "patience": 1, "batch_size": 4, "image_size": size, "augmentation": {"horizontal_flip": True},
                "negative_ratio": 1.0, "drise_images": 1, "drise_masks": 16, "pretrained": False, "window": "auto"})
        finally:
            os.chdir(cwd)

    def test_2d_kfold_outputs_exports_and_standalone_code(self):
        self.assertEqual(self.run_pipeline(["ssdlite", "rtdetr_v2"], 2, "kfold", 160), "Pipeline completed successfully")
        root = os.path.join(self.dir, "Materials")
        for path in ["2_fold_results.xlsx", "test_results.xlsx", "run_info.json", "Detection_Curves/PR_CURVES.png",
                     "Detection_Curves/FROC_CURVES.png", "Detection_Curves/ap_per_class.png", "Models/SSDLite_MobileNetV3.pt",
                     "Models/RT-DETRv2_R50.zip", "Predictions/SSDLite_MobileNetV3_test_predictions.csv"]:
            self.assertTrue(os.path.exists(os.path.join(root, path)), path)
        test = pd.read_excel(os.path.join(root, "test_results.xlsx"), index_col=0)
        self.assertEqual(list(test.columns), M.metric_names(2))
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            page = Client(appmod.application).get("/automl/results").get_data(as_text=True)
        finally:
            os.chdir(cwd)
        self.assertIn("Internal 2-fold cross-validation", page)
        # The code of the results page, run without the repository, gives the pipeline's detections
        for network in ("SSDLite_MobileNetV3.pt", "RT-DETRv2_R50.zip"):
            with self.subTest(network=network):
                info = json.load(open(os.path.join(root, "run_info.json")))
                from jinja2 import Environment, FileSystemLoader
                env = Environment(loader=FileSystemLoader(os.path.join(REPO, "templates")), autoescape=True)
                macro = env.get_template("_usage.html").module.detection_code
                library = "torchvision" if network.endswith(".pt") else "transformers"
                code = str(macro({"file": network, "library": library, "name": network}, info))
                code = code.split("\nboxes, scores, labels = detect(")[0]
                predictions = os.path.join(root, "Predictions", network.rsplit(".", 1)[0] + "_test_predictions.csv")
                script = code + f"""
import numpy as np, pandas as pd, sys
reference = pd.read_csv({predictions!r})
for name in sorted(reference.image.unique()):
    boxes, scores, labels = detect(load({os.path.join(self.dir, 'input', 'test')!r} + '/' + name))
    expected = reference[reference.image == name]
    assert len(scores) == len(expected), (name, len(scores), len(expected))
    # order-free comparison (an untrained network gives tied scores): each expected detection has its
    # twin, except the ties at the top-100 cut-off, where either of the tied boxes may be kept
    got = np.c_[scores * 100, boxes]
    want = np.c_[expected.score.values * 100, expected[['x_min', 'y_min', 'x_max', 'y_max']].values]
    want = want[want[:, 0] > want[:, 0].min() + 0.02]
    assert len(want) == 0 or np.abs(want[:, None] - got[None]).max(axis=2).min(axis=1).max() < 0.1
assert not {{m.split('.')[0] for m in sys.modules}} & {{'Helpers', 'web'}}
"""
                env_vars = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
                run = subprocess.run([sys.executable, "-c", script], cwd=self.dir, capture_output=True, text=True, env=env_vars, timeout=600)
                self.assertEqual(run.returncode, 0, run.stderr[-3000:])

    def test_3d_holdout(self):
        self.assertEqual(self.run_pipeline(["fasterrcnn_mobilenet"], 3, "holdout", 64), "Pipeline completed successfully")
        root = os.path.join(self.dir, "Materials")
        self.assertTrue(os.path.exists(os.path.join(root, "holdout_results.xlsx")))
        splits = pd.read_csv(os.path.join(root, "Splits", "splits.csv"))
        self.assertEqual(set(splits.set), {"train", "validation"})
        self.assertEqual(set(splits.fold), {1})
        self.assertFalse(set(splits[splits.set == "train"].patient) & set(splits[splits.set == "validation"].patient))
        test = pd.read_excel(os.path.join(root, "test_results.xlsx"), index_col=0)
        self.assertEqual(list(test.columns), M.metric_names(3))
        predictions = pd.read_csv(os.path.join(root, "Predictions", "Faster_R-CNN_MobileNetV3_test_predictions.csv"))
        self.assertEqual(list(predictions.columns)[3:9], ["x_min", "y_min", "z_min", "x_max", "y_max", "z_max"])
        info = json.load(open(os.path.join(root, "run_info.json")))
        self.assertEqual((info["dim"], info["validation"]), (3, "holdout"))


class TestWebFlow(TempDir):
    def setUp(self):
        super().setUp()
        self.cwd = os.getcwd()
        os.chdir(self.dir)
        os.environ["SIMPLATAB_PRETRAINED"] = "0"
        self.client = Client(appmod.application)

    def tearDown(self):
        os.environ.pop("SIMPLATAB_PRETRAINED", None)
        os.chdir(self.cwd)
        super().tearDown()

    def upload(self, train_folder, test_folder):
        zip_folder(train_folder, "Train.zip", "Train/")
        zip_folder(test_folder, "Test.zip")
        with open("Train.zip", "rb") as a, open("Test.zip", "rb") as b:
            return self.client.post("/automl/detection/upload", data={"train_zip": (a, "Train.zip"), "test_zip": (b, "Test.zip")},
                                    headers={"X-Requested-With": "XMLHttpRequest"})

    def test_pages_and_examples(self):
        self.assertEqual(self.client.get("/automl/detection").status_code, 200)
        self.assertEqual(self.client.get("/automl/automators/object-detection").headers["Location"], "/automl/detection")
        response = self.client.get("/automl/detection/example/Train3D.zip")
        self.assertEqual(response.status_code, 200)
        response.close()
        self.assertEqual(self.client.get("/automl/detection/example/app.py").status_code, 404)

    def test_invalid_upload(self):
        make_2d("train", 3, "coco")
        make_3d("test", 2, "csv")
        response = self.upload("train", "test")
        self.assertEqual(response.status_code, 400)
        self.assertIn("both hold 2D images or both 3D volumes", response.get_json()["error"])

    def test_upload_configure_run_results(self):
        make_2d("train", 8, "yolo", seed=1, nested=True)
        make_2d("test", 3, "csv", seed=2)
        response = self.upload("train", "test")
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        page = self.client.get("/automl/detection/parameters").get_data(as_text=True)
        self.assertIn("Hold-out (fast)", page)
        self.assertIn("YOLO", page)
        form = {"fasterrcnn_mobilenet": "true", "validation": "holdout", "holdout_percent": "25", "image_size": "320", "epochs": "1",
                "patience": "1", "batch_size": "4", "drise_images": "0", "drise_masks": "100"}
        self.assertEqual(self.client.post("/automl/detection/parameters", data=form).headers["Location"], "/automl/run")
        for _ in range(2400):
            status = json.loads(self.client.get("/automl/api/status").data)
            if status["state"] != "running":
                break
            time.sleep(0.25)
        self.assertEqual(status["state"], "done", status["message"])
        self.assertEqual(status["models"][0]["test"], "done")
        page = self.client.get("/automl/results").get_data(as_text=True)
        self.assertIn("Hold-out validation (Train.zip)", page)
        code = html.unescape(re.sub(r"<[^>]+>", "", re.search(r'id="code-predict"><code>(.*?)</code></pre>', page, re.S).group(1)))
        self.assertIn('torch.jit.load("Materials/Models/Faster_R-CNN_MobileNetV3.pt"', code)
        self.assertNotIn("Helpers", code)


if __name__ == "__main__":
    unittest.main()
