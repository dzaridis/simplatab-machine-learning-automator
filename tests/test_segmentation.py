"""Tests of the image segmentation automator: the data layouts and labels, the metrics, the 18
networks, the pipeline in 2D (K-fold) and 3D (hold-out, two series) with the official nnU-Net, the
standalone code of the results page and the web flow. Networks are randomly initialised (no
download); nnU-Net trains for a few iterations."""
import io
import json
import os
import re
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
from segmentation_fixtures import make_2d, make_3d  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from Helpers.segmentation import data, metrics  # noqa: E402
from Helpers.segmentation.inference import export_model, load_model  # noqa: E402
from Helpers.segmentation.models import NETWORKS, build, divisor, fit_patch, logits  # noqa: E402


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()


class TestData(TempDir):
    def test_2d_layouts_and_labels(self):
        for mask, nested, suffix in (("palette", False, ""), ("rgb", True, "_mask"), ("gray", False, "_seg")):
            with self.subTest(mask=mask):
                root = os.path.join(self.dir, mask)
                make_2d(os.path.join(root, "train"), 6, seed=1, mask=mask, nested=nested, suffix=suffix)
                make_2d(os.path.join(root, "test"), 3, seed=2, mask=mask, nested=nested, suffix=suffix)
                cases, lonely, fmt = data.scan_split(os.path.join(root, "train"))
                self.assertEqual((len(cases), fmt), (6, "folders"))
                self.assertEqual(lonely, {"masks_without_image": [], "images_without_mask": []})
                summary = data.summarize(os.path.join(root, "train"), os.path.join(root, "test"))
                self.assertEqual(summary["errors"], [])
                self.assertEqual((summary["dim"], summary["rgb"], summary["train_cases"]), (2, True, 6))
                self.assertEqual(len(summary["classes"]), 3)
                self.assertEqual(summary["groups"], 3 if nested else 6)  # patient folders group the cases
                if mask == "gray":
                    self.assertEqual(summary["mapping"]["values"], [0, 100, 200])
                    self.assertEqual(summary["mapping"]["classes"], ["background", "disc", "square"])
                elif mask == "palette":  # palette PNG: the palette indices
                    self.assertEqual(summary["mapping"]["values"], [0, 1, 2])
                else:
                    self.assertEqual(summary["mapping"]["values"][1:], [[40, 160, 60], [220, 40, 40]])
                # Cached as nnU-Net cases: one file per channel, labels 0..K
                metas, failed = data.cache_split(cases, os.path.join(root, "cache"), summary["mapping"])
                self.assertEqual(failed, [])
                image, labels = data.load_cached(os.path.join(root, "cache"), metas[0], 3)
                self.assertEqual((image.shape, labels.shape), ((3, 1, 64, 64), (1, 64, 64)))
                self.assertEqual(sorted(np.unique(labels)), [0, 1, 2])

    def test_binary_masks_255(self):
        mapping = data.label_mapping({0: 900, 255: 100}, {})
        self.assertEqual((mapping["values"], mapping["classes"]), ([0, 255], ["background", "foreground"]))
        self.assertEqual(data.map_labels(np.array([[0, 255]]), mapping).tolist(), [[0, 1]])

    def test_3d_layouts(self):
        for layout in ("nifti", "folder", "nnunet"):
            with self.subTest(layout=layout):
                root = os.path.join(self.dir, layout)
                make_3d(os.path.join(root, "train"), 4, seed=1, layout=layout)
                make_3d(os.path.join(root, "test"), 2, seed=2, layout=layout)
                summary = data.summarize(os.path.join(root, "train"), os.path.join(root, "test"))
                self.assertEqual(summary["errors"], [])
                self.assertEqual((summary["dim"], summary["format"]), (3, "nnunet" if layout == "nnunet" else "folders"))
                self.assertEqual(summary["channels"], ["t2", "adc"] if layout == "folder" else [])
                self.assertEqual(summary["size_range"], [40, 40])  # of the reference (T2) series
                cases, _, _ = data.scan_split(os.path.join(root, "train"))
                self.assertEqual(len(cases[0].sources), 1 if layout == "nifti" else 2)
                metas, failed = data.cache_split(cases, os.path.join(root, "cache"), summary["mapping"],
                                                 summary["channels"] or None)
                self.assertEqual(failed, [])
                self.assertEqual((metas[0]["dim"], metas[0]["shape"]), (3, [12, 40, 40]))
                np.testing.assert_allclose(metas[0]["spacing"], [2.5, 0.8, 0.8])
                if layout == "nnunet":
                    self.assertEqual(summary["classes"][1]["name"], "lesion")  # from dataset.json


class TestMetrics(unittest.TestCase):
    def test_overlap_and_distances(self):
        reference = np.zeros((1, 20, 20), np.uint8)
        reference[0, 5:15, 5:15] = 1
        perfect = metrics.case_metrics(reference, reference, 2, [1.0, 1.0, 1.0])[1]
        self.assertEqual((perfect["Dice"], perfect["IoU"], perfect["HD95"], perfect["ASSD"]), (1.0, 1.0, 0.0, 0.0))
        shifted = np.roll(reference, 2, axis=2)
        m = metrics.case_metrics(shifted, reference, 2, [1.0, 0.5, 0.5])[1]
        self.assertAlmostEqual(m["Dice"], 0.8)
        self.assertAlmostEqual(m["IoU"], 80 / 120)
        self.assertAlmostEqual(m["HD95"], 1.0)  # 2 pixels of 0.5 mm
        empty = metrics.case_metrics(np.zeros_like(reference), np.zeros_like(reference), 2, [1, 1, 1])[1]
        self.assertTrue(np.isnan(empty["Dice"]))
        missed = metrics.case_metrics(np.zeros_like(reference), reference, 2, [1, 1, 1])[1]
        self.assertEqual(missed["Dice"], 0.0)
        self.assertTrue(np.isnan(missed["HD95"]))
        summary = metrics.summarise([{1: perfect}, {1: missed}, {1: empty}], ["background", "lesion"])
        self.assertAlmostEqual(summary["values"]["Dice"], 0.5)
        self.assertEqual(summary["case_dice"][:2], [1.0, 0.0])


class TestNetworks(unittest.TestCase):
    def test_every_network(self):
        torch.manual_seed(0)
        for spec in NETWORKS:
            if spec.family == "nnunet":
                continue
            with self.subTest(network=spec.name):
                patch = fit_patch(spec, [64, 64] if spec.dim == 2 else [32, 32, 32])  # SwinUNETR: [64, 32, 32]
                self.assertEqual([p % divisor(spec, patch) for p in patch], [0] * spec.dim)
                channels = 3 if spec.dim == 2 else 2
                model = build(spec, channels, 3, patch, pretrained=False, rgb=spec.dim == 2)
                model.train()
                x = torch.rand(2, channels, *patch)
                out = logits(model(x))
                self.assertEqual(tuple(out.shape), (2, 3, *patch))
                out.mean().backward()
                model.eval()
                with torch.no_grad():
                    self.assertEqual(tuple(logits(model(x[:1])).shape), (1, 3, *patch))

    def test_export_round_trip(self):
        spec = next(n for n in NETWORKS if n.key == "segresnet")
        model = build(spec, 2, 3, [32, 32, 32], pretrained=False).eval()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "m.pt")
            export_model(model, path, {"channels": 2, "patch": [32, 32, 32], "classes": ["background", "a", "b"]})
            loaded, info = load_model(path)
            x = torch.rand(1, 2, 32, 32, 32)
            with torch.no_grad():
                torch.testing.assert_close(loaded(x), model(x), atol=1e-4, rtol=1e-4)
            self.assertEqual((info["format"], info["classes"][1]), ("simplatab-segmenter", "a"))


def _run_snippet(test, materials, info, model, inputs, outputs):
    """Runs the code of the results page for ``model`` (outside the repository) on each input and
    checks it gives the pipeline's predicted masks."""
    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader(os.path.join(REPO, "templates")), autoescape=True)
    code = str(env.get_template("_usage.html").module.segmentation_code(model, info))
    is3d = info["dim"] == 3
    body = code.split("\nimage, ")[0]
    reads = []
    for source, reference in zip(inputs, outputs):
        if is3d:
            reads.append(f"""
image, reference = load({source!r})
classes = segment(image, list(reference.GetSpacing())[::-1])
expected = sitk.GetArrayFromImage(sitk.ReadImage({reference!r}))
assert classes.shape == expected.shape, (classes.shape, expected.shape)
agreement = (np.array(MASK_VALUES)[classes] == expected).mean()""")
        else:
            reads.append(f"""
image, spacing = load({source!r})
classes = segment(image, spacing)[0]
expected = np.array(Image.open({reference!r}))
mask = np.array(MASK_VALUES, np.uint8)[classes]
agreement = (mask == expected).all(-1).mean() if expected.ndim == 3 else (mask == expected).mean()""")
        reads.append("assert agreement > 0.999, agreement")
    script = body + "\n" + "\n".join(reads) + """
import sys
assert not {m.split('.')[0] for m in sys.modules} & {'Helpers', 'web'}
"""
    work = tempfile.mkdtemp(dir=test.dir)
    os.symlink(materials, os.path.join(work, "Materials"))
    env_vars = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    run = subprocess.run([sys.executable, "-c", script], cwd=work, capture_output=True, text=True, env=env_vars,
                         timeout=900)
    test.assertEqual(run.returncode, 0, model["name"] + "\n" + run.stderr[-3000:])


class TestPipeline(TempDir):
    def run_pipeline(self, dim, models, validation, **extra):
        from Helpers.segmentation.pipeline import run_segmentation_pipeline
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            if dim == 2:
                make_2d("input/train", 8, seed=1, mask="rgb", nested=True)
                make_2d("input/test", 3, seed=2, mask="rgb")
            else:
                make_3d("input/train", 6, seed=1, layout="folder")
                make_3d("input/test", 2, seed=2, layout="folder")
            summary = data.summarize("input/train", "input/test")
            params = {"models": models, "dim": dim, "mapping": summary["mapping"], "channels": summary["channels"] or None,
                      "validation": validation, "k_folds": 2, "holdout_fraction": 0.25, "epochs": 2, "iterations": 3,
                      "batch_size": 2, "learning_rate": 1e-3, "normalisation": "auto", "tta": True,
                      "augmentation": {"rotation": True, "intensity": True, "horizontal_flip": True},
                      "nnunet_epochs": 1, "nnunet_iterations": 3, "pretrained": False, **extra}
            return run_segmentation_pipeline("input", params), summary
        finally:
            os.chdir(cwd)

    def results_page(self):
        import app as appmod
        from werkzeug.test import Client
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            return Client(appmod.application).get("/automl/results").get_data(as_text=True)
        finally:
            os.chdir(cwd)

    def test_2d_kfold_with_nnunet(self):
        result, summary = self.run_pipeline(2, ["nnunet_2d", "unet_resnet34"], "kfold")
        self.assertEqual(result, "Pipeline completed successfully")
        root = os.path.join(self.dir, "Materials")
        for path in ("2_fold_results.xlsx", "test_results.xlsx", "Models/nnU-Net_2D.zip", "Models/U-Net_ResNet-34.pt",
                     "Segmentation_Metrics/test_per_class.csv", "Segmentation_Metrics/test_per_case.csv",
                     "Segmentation_Plots/test_overlap_metrics.png", "Segmentation_Plots/test_dice_per_class.png",
                     "Predictions/nnU-Net_2D/img_000.png", "Predictions/U-Net_ResNet-34/img_002.png"):
            self.assertTrue(os.path.exists(os.path.join(root, path)), path)
        self.assertTrue(any(f.startswith("U-Net_ResNet-34_best_") for f in os.listdir(os.path.join(root, "Overlays", "U-Net_ResNet-34"))))
        info = json.load(open(os.path.join(root, "run_info.json")))
        self.assertEqual((info["automator"], info["dim"], info["channels"], info["unit"]), ("image-segmentation", 2, 3, "px"))
        self.assertEqual(info["models"]["nnU-Net 2D"]["fold"], "all")
        import pandas as pd
        splits = pd.read_csv(os.path.join(root, "Splits", "splits.csv"))
        validation = splits[splits.set == "validation"]
        self.assertEqual(sorted(validation.id), sorted(splits.id.unique()))
        self.assertEqual(validation.groupby("patient").fold.nunique().max(), 1)
        nnunet_splits = json.load(open(os.path.join(self.dir, "input", "work", "nnunet", "preprocessed",
                                                    "Dataset501_Simplatab", "splits_final.json")))
        self.assertEqual([len(f["val"]) for f in nnunet_splits], [len(validation[validation.fold == k]) for k in (1, 2)])
        import zipfile
        names = zipfile.ZipFile(os.path.join(root, "Models", "nnU-Net_2D.zip")).namelist()
        self.assertIn("fold_all/checkpoint_final.pth", names)
        self.assertIn("plans.json", names)

        inputs = [os.path.join(self.dir, "input", "test", "images", n) for n in ("img_000.jpg", "img_002.png")]
        for name, exported in info["models"].items():
            safe = exported["file"].rsplit(".", 1)[0]
            outputs = [os.path.join(root, "Predictions", safe, n) for n in ("img_000.png", "img_002.png")]
            _run_snippet(self, root, info, {"name": name, **exported}, [[p] for p in inputs], outputs)

        page = self.results_page()
        for text in ("2-fold cross-validation", "nnU-Net 2D", "U-Net ResNet-34", "HD95", "Masks &amp; uncertainty",
                     "nnunetv2==2.4.2", "Test Dice per class", "Predicted test masks"):
            self.assertIn(text, page)

    def test_3d_holdout_two_series(self):
        result, summary = self.run_pipeline(3, ["nnunet_3d", "segresnet"], "holdout")
        self.assertEqual(result, "Pipeline completed successfully")
        root = os.path.join(self.dir, "Materials")
        info = json.load(open(os.path.join(root, "run_info.json")))
        self.assertEqual((info["dim"], info["channel_names"], info["unit"]), (3, ["t2", "adc"], "mm"))
        self.assertEqual(info["models"]["nnU-Net 3D full resolution"]["fold"], 0)
        self.assertTrue(os.path.exists(os.path.join(root, "holdout_results.xlsx")))
        test = os.path.join(self.dir, "input", "test", "images")
        inputs = [[os.path.join(test, c, "t2"), os.path.join(test, c, "adc.nii.gz")] for c in ("case_000", "case_001")]
        for name, exported in info["models"].items():
            safe = exported["file"].rsplit(".", 1)[0]
            outputs = [os.path.join(root, "Predictions", safe, f"{c}.nii.gz") for c in ("case_000", "case_001")]
            _run_snippet(self, root, info, {"name": name, **exported}, inputs, outputs)
        page = self.results_page()
        for text in ("hold-out validation", "2 channels (t2, adc)", "monai==1.3.2", "new_case/t2", "mm, ↓"):
            self.assertIn(text, page)

    def test_too_few_cases_for_the_folds(self):
        result, _ = self.run_pipeline(2, ["attention_unet_2d"], "kfold", k_folds=10)
        self.assertIn("at least 10 training patients", result)


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
        self.patch = mock.patch.object(appmod, "SEGMENTATION_INPUT_FOLDER", os.path.join(self.dir, "seg"))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        os.chdir(self.cwd)
        super().tearDown()

    def upload(self, dim):
        files = {}
        for split, seed in (("train", 1), ("test", 2)):
            folder = os.path.join(self.dir, "src", split)
            if dim == 2:
                make_2d(folder, 6, seed=seed, mask="gray")
            else:
                make_3d(folder, 4, seed=seed, layout="folder")
            archive = shutil.make_archive(os.path.join(self.dir, f"{split}{dim}"), "zip", folder)
            shutil.rmtree(folder)
            with open(archive, "rb") as f:
                files[f"{split}_zip"] = (io.BytesIO(f.read()), f"{split.capitalize()}.zip")
        return self.client.post("/automl/segmentation/upload", data=files, headers={"X-Requested-With": "XMLHttpRequest"},
                                content_type="multipart/form-data")

    def test_upload_parameters_and_run(self):
        self.assertEqual(self.upload(3).get_json(), {"redirect": "/automl/segmentation/parameters"})
        page = self.client.get("/automl/segmentation/parameters").get_data(as_text=True)
        for network in NETWORKS:
            self.assertEqual(f'name="{network.key}"' in page, network.dim == 3, network.name)
        for text in ("Reference series", 'value="adc"', "nnU-Net schedule", "Intensity normalisation"):
            self.assertIn(text, page)

        received = {}

        def fake(folder, params):
            received.update(params)
            print("------------- \n", "Preparing images \n", "-------------")
            return "Pipeline completed successfully"

        form = {"nnunet_3d": "true", "segresnet": "true", "channels": ["t2", "adc"], "reference": "adc",
                "validation": "kfold", "k_folds": "2", "epochs": "5", "iterations": "7", "nnunet_epochs": "10",
                "tta": "true", "rotation": "true"}
        with mock.patch("Helpers.segmentation.pipeline.run_segmentation_pipeline", fake):
            response = self.client.post("/automl/segmentation/parameters", data=form)
            self.assertEqual(response.headers["Location"], "/automl/run")
            for _ in range(100):
                if not self.app.job.running:
                    break
                time.sleep(0.05)
        self.assertEqual(received["models"], ["nnunet_3d", "segresnet"])
        self.assertEqual(received["channels"], ["adc", "t2"])  # the reference first
        self.assertEqual((received["k_folds"], received["epochs"], received["iterations"], received["nnunet_epochs"]),
                         (2, 5, 7, 10))
        self.assertEqual(received["mapping"]["classes"], ["background", "foreground"])
        self.assertTrue(received["tta"] and received["augmentation"]["rotation"])
        self.assertEqual(self.client.get("/automl/api/status").get_json()["state"], "done")
        # No network, too many folds, no series
        for change in ({"nnunet_3d": "", "segresnet": ""}, {"k_folds": "5"}, {"channels": []}):
            self.assertEqual(self.client.post("/automl/segmentation/parameters", data=dict(form, **change)).headers["Location"],
                             "/automl/segmentation/parameters", change)

    def test_2d_upload_names_and_examples(self):
        self.upload(2)
        page = self.client.get("/automl/segmentation/parameters").get_data(as_text=True)
        for text in ("disc", "square", "U-Net ResNet-34", "SegFormer-B2", "nnU-Net 2D"):
            self.assertIn(text, page)
        self.assertNotIn("Reference series", page)
        for name in ("Train.zip", "Test.zip", "Train3D.zip", "Test3D.zip"):
            response = self.client.get(f"/automl/segmentation/example/{name}")
            self.assertEqual(response.status_code, 200, name)
            self.assertGreater(len(response.get_data()), 10000)

    def test_bad_upload(self):
        folder = os.path.join(self.dir, "empty")
        os.makedirs(os.path.join(folder, "images"))
        with open(os.path.join(folder, "images", "readme.txt"), "w") as f:
            f.write("no image")
        archive = shutil.make_archive(os.path.join(self.dir, "bad"), "zip", folder)
        data_ = {k: (io.BytesIO(open(archive, "rb").read()), f"{k.split('_')[0].capitalize()}.zip") for k in ("train_zip", "test_zip")}
        response = self.client.post("/automl/segmentation/upload", data=data_, headers={"X-Requested-With": "XMLHttpRequest"},
                                    content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertTrue(re.search(r"mask|image", response.get_json()["error"]), response.get_json())


if __name__ == "__main__":
    unittest.main()
