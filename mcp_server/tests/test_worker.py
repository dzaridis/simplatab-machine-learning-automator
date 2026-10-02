"""Tests of the worker half of the MCP server (Python environment of Simplatab): data contracts,
configurations, data checks and a tabular run with its results.

    cd mcp_server && SIMPLATAB_PRETRAINED=0 python -m unittest discover tests
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("SIMPLATAB_PRETRAINED", "0")

from simplatab_mcp import configs, contracts, datasets  # noqa: E402
from simplatab_mcp.paths import simplatab_root  # noqa: E402

sys.path.insert(0, str(simplatab_root()))
AUTOMATORS = list(contracts.AUTOMATORS)


class Workspace(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.env = mock_env = {"SIMPLATAB_WORKSPACE": str(self.dir / "ws"), "SIMPLATAB_DATA_DIR": str(self.dir / "data")}
        self._saved = {k: os.environ.get(k) for k in mock_env}
        os.environ.update(mock_env)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def worker(self, *args):
        output = self.dir / "out.json"
        env = dict(os.environ, PYTHONPATH=str(HERE.parent))
        subprocess.run([sys.executable, "-m", "simplatab_mcp.worker", *args, "--output", str(output)], env=env,
                       cwd=str(self.dir), capture_output=True, text=True, timeout=1800)
        return json.loads(output.read_text())

    def experiment(self, automator, train, test):
        folder = self.dir / "ws" / "experiments" / "e1"
        folder.mkdir(parents=True)
        (folder / "experiment.json").write_text(json.dumps({"id": "e1", "automator": automator, "train": str(train),
                                                            "test": str(test)}))
        return folder


class TestContracts(unittest.TestCase):
    def test_every_automator(self):
        for automator in AUTOMATORS:
            with self.subTest(automator=automator):
                contract = contracts.contract(automator)
                for key in ("data", "config", "metrics", "outputs", "workflow"):
                    self.assertIn(key, contract)
                models = contract.get("models") or [m for ms in contract["models_by_dim"].values() for m in ms]
                self.assertTrue(models and all({"key", "name", "description"} <= set(m) for m in models))
                self.assertTrue(any("Splits/splits.csv" in o for o in contract["outputs"]))

    def test_example_configurations_use_known_models(self):
        from simplatab_mcp.examples import QUICK
        self.assertEqual(set(QUICK), set(AUTOMATORS))
        for automator, models in (("object-detection", ["fasterrcnn_mobilenet"]), ("time-series-forecasting", ["NHITS", "DLinear"]),
                                  ("image-classification", ["efficientnet_b0"])):
            keys = {m["key"] for m in contracts.models(automator, 2)}
            self.assertTrue(set(models) <= keys, automator)
        self.assertIn("medicalnet_resnet10", {m["key"] for m in contracts.models("image-classification", 3)})
        self.assertTrue({"nnunet_2d", "unet_resnet34"} <= {m["key"] for m in contracts.models("image-segmentation", 2)})
        self.assertTrue({"nnunet_3d", "segresnet"} <= {m["key"] for m in contracts.models("image-segmentation", 3)})


class TestConfigs(unittest.TestCase):
    SUMMARY = {"min_class_count": 30, "features": ["age", "sex"]}

    def test_tabular_defaults_and_errors(self):
        params, names = configs.build("tabular", {}, self.SUMMARY)
        self.assertEqual(params["number_of_k_folds"], 5)
        self.assertTrue(params["apply_grid_search"]["enabled"])
        self.assertIn("XGBoost", names)
        self.assertNotIn("TabPFNv2", names)
        params, names = configs.build("tabular", {"models": ["tabpfn"], "hyperparameter_search": "none", "k_folds": 3,
                                                  "bias_feature": "sex"}, self.SUMMARY)
        self.assertEqual((names, params["number_of_k_folds"], params["Feature"]), (["TabPFNv2"], 3, "sex"))
        self.assertFalse(params["apply_grid_search"]["enabled"])
        for bad, message in (({"k_fold": 3}, "Unknown configuration field"), ({"models": ["gpt"]}, "Unknown model"),
                             ({"k_folds": 50}, "between 2 and 20"), ({"k_folds": 2.5}, "integer"),
                             ({"threshold_metric": "R2"}, "must be one of"), ({"bias_feature": "x"}, "bias_feature")):
            with self.subTest(config=bad):
                with self.assertRaisesRegex(configs.ConfigError, message):
                    configs.build("tabular", bad, self.SUMMARY)

    def test_folds_limited_by_the_data(self):
        with self.assertRaisesRegex(configs.ConfigError, "between 2 and 4"):
            configs.build("tabular", {"k_folds": 5}, {"min_class_count": 4, "features": []})

    def test_defaults_in_the_agent_vocabulary(self):
        defaults = configs.defaults("tabular", self.SUMMARY)
        self.assertEqual(defaults["hyperparameter_search"], "randomized")
        params, _ = configs.build("tabular", defaults, self.SUMMARY)  # the defaults are a valid configuration
        self.assertEqual(params["number_of_k_folds"], 5)


class TestData(Workspace):
    def test_tabular_checks(self):
        import pandas as pd
        data = self.dir / "data"
        data.mkdir()
        pd.DataFrame({"ID": ["a", "b", "c", "d"], "x": [1, 2, 3, 4], "Target": [1, 2, 1, 2]}).to_csv(data / "Train.csv", index=False)
        pd.DataFrame({"ID": ["e"], "y": [1], "Target": [1]}).to_csv(data / "Test.csv", index=False)
        datasets.prepare_inputs("tabular", "Train.csv", "Test.csv", self.dir / "input")  # relative to the data folder
        summary, errors, warnings = datasets.check("tabular", self.dir / "input")
        self.assertTrue(any("numbered" in e for e in errors))  # classes 1, 2 instead of 0, 1
        self.assertTrue(any("lacks the columns x" in e for e in errors))
        self.assertTrue(any("not in Train.csv" in w for w in warnings))
        with self.assertRaisesRegex(FileNotFoundError, "was not found"):
            datasets.prepare_inputs("tabular", "missing.csv", "Test.csv", self.dir / "input")
        with self.assertRaisesRegex(FileNotFoundError, "was not found"):  # no way out of the data folders
            datasets.prepare_inputs("tabular", "../../../etc/passwd", "Test.csv", self.dir / "input")
        with self.assertRaisesRegex(PermissionError, "outside the data folders"):
            datasets.prepare_inputs("tabular", str(simplatab_root() / "Examples" / "time-series-forecasting" / "Train.csv"),
                                    "Test.csv", self.dir / "input")
        with self.assertRaisesRegex(datasets.DataError, "zip file or a folder"):
            datasets.prepare_inputs("object-detection", str(data / "Train.csv"), str(data / "Test.csv"), self.dir / "input2")

    def test_segmentation_zip_and_folder(self):
        os.environ["SIMPLATAB_ALLOW_ANY_PATH"] = "1"  # the examples of the repository, outside the data folders
        self.addCleanup(os.environ.pop, "SIMPLATAB_ALLOW_ANY_PATH", None)
        examples = simplatab_root() / "Examples" / "image-segmentation"
        unpacked = self.dir / "test_folder"
        shutil.unpack_archive(str(examples / "Test.zip"), str(unpacked))
        datasets.prepare_inputs("image-segmentation", str(examples / "Train.zip"), str(unpacked), self.dir / "input")
        self.assertTrue((self.dir / "input" / "test").is_symlink())
        summary, errors, _ = datasets.check("image-segmentation", self.dir / "input")
        self.assertEqual(errors, [])
        self.assertEqual([c["name"] for c in summary["classes"]], ["background", "building", "road"])
        defaults = configs.defaults("image-segmentation", summary)
        self.assertIn("nnunet_2d", defaults["models"])
        self.assertTrue(defaults["augmentation"]["horizontal_flip"])  # colour photos


class TestRun(Workspace):
    def test_tabular_run_results_and_splits(self):
        example = self.worker("example", "--automator", "tabular")
        folder = self.experiment("tabular", example["train"], example["test"])
        prepared = self.worker("prepare", "--experiment", str(folder))
        self.assertEqual(prepared["errors"], [])
        self.assertEqual(prepared["summary"]["id_column"], "ID")
        configured = self.worker("configure", "--experiment", str(folder), "--config",
                                 json.dumps({"models": ["logistic_regression"], "k_folds": 2, "hyperparameter_search": "none"}))
        self.assertEqual(configured["model_names"], ["Logistic Regression"])
        meta = json.loads((folder / "experiment.json").read_text())
        meta.update(params=configured["params"], model_names=configured["model_names"])
        (folder / "experiment.json").write_text(json.dumps(meta))
        result = self.worker("run", "--experiment", str(folder))
        self.assertEqual(result["state"], "completed", result)
        status = json.loads((folder / "status.json").read_text())
        self.assertEqual((status["progress"], status["models"][0]["test"]), (100, "done"))
        results = json.loads((folder / "results.json").read_text())
        self.assertEqual(results["best_model"], "Logistic Regression")
        self.assertGreater(results["test_metrics"][0]["AUC"], 0.9)
        self.assertEqual([f["fold"] for f in results["splits"]["folds"]], [1, 2])
        self.assertIn("Models/Logistic Regression_pipeline.pkl", results["models"])


if __name__ == "__main__":
    unittest.main()
