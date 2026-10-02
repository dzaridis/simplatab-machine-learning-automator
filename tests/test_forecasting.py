"""Tests of the time series forecasting automator: data checks, rolling-origin windows,
metrics, the pipeline end to end, the web flow and the standalone usage code."""
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
from Helpers.forecasting import data as fdata
from Helpers.forecasting import metrics as fm
from Helpers.forecasting.models import BY_KEY, MODELS, build, search_space
from Helpers.forecasting.pipeline import make_folds, run_forecasting_pipeline

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO, "Examples", "time-series-forecasting")


def synthetic(n_series=6, length=48, start=0, new=2, horizon=4, seed=0, gaps=False):
    """Integer-step series with a static, a future and a past covariate. Test.csv continues the
    training series (``horizon`` points) and adds ``new`` unseen series (history + horizon)."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for i in range(n_series + new):
        steps = np.arange(start, start + length + horizon)
        dose = rng.integers(0, 2, len(steps))
        y = 10 + i + 3 * np.sin(steps / 3) + 2 * dose + rng.normal(0, 0.3, len(steps))
        frame = pd.DataFrame({"ID": f"S{i}", "Time": steps, "Target": y.round(3), "Dose": dose,
                              "Lab": (y + rng.normal(0, 1, len(steps))).round(3), "Group": "A" if i % 2 else "B"})
        if i < n_series:
            part = frame.iloc[:length]
            if gaps and i == 0:
                part = part.drop(index=[10, 11])
            train.append(part)
            test.append(frame.iloc[length:])
        else:
            test.append(frame.iloc[-(20 + horizon):])
    return pd.concat(train), pd.concat(test)


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, train, test):
        train.to_csv(os.path.join(self.dir, "Train.csv"), index=False)
        test.to_csv(os.path.join(self.dir, "Test.csv"), index=False)
        return os.path.join(self.dir, "Train.csv"), os.path.join(self.dir, "Test.csv")


class TestData(TempDir):
    def test_example_summary(self):
        summary = fdata.summarize(os.path.join(EXAMPLE, "Train.csv"), os.path.join(EXAMPLE, "Test.csv"))
        self.assertEqual(summary["errors"], [])
        self.assertEqual((summary["kind"], summary["freq"], summary["season"]), ("date", "D", 7))
        self.assertEqual((summary["train_series"], summary["continuing_series"], summary["new_series"]), (30, 30, 10))
        self.assertEqual(summary["static"], ["Age", "Sex", "BMI"])
        self.assertEqual(summary["dynamic"], ["Insulin_units", "Weekend", "Steps"])
        self.assertEqual(summary["max_horizon"], 14)

    def test_prepare_covariates_and_test_series(self):
        train, test = synthetic(gaps=True)
        prepared = fdata.prepare(*self.write(train, test), horizon=4, future_columns=["Dose"])
        self.assertEqual(prepared.futr, ["Dose"])
        self.assertEqual(prepared.hist, ["Lab"])
        self.assertEqual(prepared.stat, ["Group_A", "Group_B"])  # one-hot encoded static feature
        self.assertEqual(prepared.added_points, 2)  # the gap of S0 is filled in
        self.assertEqual(len(prepared.train[prepared.train.unique_id == "S0"]), 48)
        self.assertEqual(sorted(prepared.continuing), [f"S{i}" for i in range(6)])
        self.assertEqual(sorted(prepared.new), ["S6", "S7"])
        # A continued series: its Train.csv points are its history; a new one: its own first points
        self.assertEqual(len(prepared.test_history[prepared.test_history.unique_id == "S1"]), 48)
        self.assertEqual(len(prepared.test_history[prepared.test_history.unique_id == "S6"]), 20)
        self.assertTrue((prepared.test_horizon.groupby("unique_id").size() == 4).all())
        self.assertAlmostEqual(prepared.static["Group_A"].mean(), 0, places=6)  # standardised

    def test_missing_target_values(self):
        train, test = synthetic()
        series = test[test.ID == "S7"].copy()
        series.iloc[0, 2] = np.nan    # first point: dropped from the history
        series.iloc[19, 2] = np.nan   # last history point: carried forward, not interpolated from the horizon
        series.iloc[21, 2] = np.nan   # a horizon point: not scored
        test = pd.concat([test[test.ID != "S7"], series])
        prepared = fdata.prepare(*self.write(train, test), horizon=4)
        history = prepared.test_history[prepared.test_history.unique_id == "S7"]
        self.assertEqual(len(history), 19)
        self.assertEqual(history.y.iloc[-1], series.Target.iloc[18])
        horizon = prepared.test_horizon[prepared.test_horizon.unique_id == "S7"]
        self.assertEqual(horizon.observed.tolist(), [True, False, True, True])

    def test_errors(self):
        train, test = synthetic()
        self.assertIn("ID / Time", fdata.summarize(*self.write(train.drop(columns=["ID", "Time"]), test))["errors"][0])
        self.assertIn("several rows", fdata.summarize(*self.write(pd.concat([train, train.iloc[:1]]), test))["errors"][0])
        overlap = pd.concat([train[train.ID == "S0"].tail(3), test])
        self.assertIn("repeats time points", fdata.summarize(*self.write(train, overlap))["errors"][0])
        text = train.copy()
        text.loc[text.index[0], "Target"] = "high"
        self.assertIn("must contain numbers", fdata.summarize(*self.write(text, test))["errors"][0])

    def test_frequencies(self):
        dates = pd.DataFrame({"ID": ["a"] * 4 + ["b"] * 3, "Time": pd.to_datetime(
            ["2024-01-01", "2024-02-01", "2024-04-01", "2024-05-01", "2024-01-01", "2024-03-01", "2024-04-01"])})
        self.assertEqual(fdata.infer_frequency([dates], "date"), "MS")  # monthly, with missing months
        steps = pd.DataFrame({"ID": ["a"] * 3, "Time": [0, 2, 6]})
        self.assertEqual(fdata.infer_frequency([steps], "integer"), 2)
        irregular = pd.DataFrame({"ID": ["a"] * 3, "Time": pd.to_datetime(["2024-01-01", "2024-01-04", "2024-01-09"])})
        with self.assertRaises(fdata.DataError):
            fdata.infer_frequency([irregular], "date")


class TestValidationAndMetrics(unittest.TestCase):
    def test_rolling_origin_windows(self):
        frame = pd.DataFrame({"unique_id": ["a"] * 20 + ["b"] * 9, "ds": list(range(20)) + list(range(9)), "y": 0.0})
        windows = make_folds(frame, 3, 4)
        # Fold k trains up to (K - k + 1) horizons before the end and is scored on the next horizon
        self.assertEqual([len(fit[fit.unique_id == "a"]) for fit, _ in windows], [8, 12, 16])
        self.assertEqual([list(score[score.unique_id == "a"].ds) for _, score in windows][0], [8, 9, 10, 11])
        # Series b (9 points) only has enough history for the last window
        self.assertEqual([("b" in set(fit.unique_id)) for fit, _ in windows], [False, False, True])
        with self.assertRaises(fdata.DataError):
            make_folds(frame, 5, 4)

    def test_metrics(self):
        self.assertEqual(list(fm.seasonal_naive([1, 2, 3, 4, 5, 6, 7], 5, 3)), [5, 6, 7, 5, 6])
        self.assertEqual(fm.naive_scale([1, 3, 5, 7], 1), 2.0)
        scores = fm.score([10, 20], [12, 18], ["a", "a"], {"a": 4.0})
        self.assertEqual(scores["MAE"], 2.0)
        self.assertEqual(scores["RMSE"], 2.0)
        self.assertAlmostEqual(scores["MASE"], 0.5)
        self.assertAlmostEqual(scores["sMAPE"], (200 * 2 / 22 + 200 * 2 / 38) / 2)

    def test_models_and_search_space(self):
        self.assertEqual(len(MODELS), 10)
        configs = search_space(BY_KEY["TFT"], horizon=4, longest=30, trials=3)
        self.assertEqual(configs[0], {"input_size": 8, "learning_rate": 1e-3})
        self.assertEqual(len(configs), 4)
        self.assertEqual(search_space(BY_KEY["NHITS"], 4, 30, lookback=10)[0]["input_size"], 10)
        # Each network gets only the covariates it supports
        nhits = build(BY_KEY["NHITS"], 4, configs[0], ["f"], ["p"], ["s"], 10)
        patch = build(BY_KEY["PatchTST"], 4, configs[0], ["f"], ["p"], ["s"], 10)
        self.assertEqual((nhits.futr_exog_list, nhits.hist_exog_list, nhits.stat_exog_list), (["f"], ["p"], ["s"]))
        self.assertEqual((list(patch.futr_exog_list), list(patch.hist_exog_list)), ([], []))


class TestPipelineAndUsage(TempDir):
    def run_pipeline(self, future_columns):
        train, test = synthetic()
        os.makedirs(os.path.join(self.dir, "input"))
        train.to_csv(os.path.join(self.dir, "input", "Train.csv"), index=False)
        test.to_csv(os.path.join(self.dir, "input", "Test.csv"), index=False)
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            result = run_forecasting_pipeline("input", {
                "models": ["NHITS", "DLinear"], "horizon": 4, "k_folds": 2, "lookback": 0, "trials": 1,
                "max_steps": 20, "future_columns": future_columns})
        finally:
            os.chdir(cwd)
        return result, train, test

    def test_pipeline_outputs_and_standalone_code(self):
        result, train, test = self.run_pipeline(["Dose"])
        self.assertEqual(result, "Pipeline completed successfully")
        root = os.path.join(self.dir, "Materials")
        for path in ["2_fold_results.xlsx", "test_results.xlsx", "run_info.json", "Forecasts/test_forecasts.csv",
                     "Metrics_Plots/test_metrics.png", "Metrics_Plots/error_by_horizon.png", "Models/NHITS.zip",
                     "Forecast_Plots/NHITS_test_forecasts.png", "Explainability/NHITS_integrated_gradients.png"]:
            self.assertTrue(os.path.exists(os.path.join(root, path)), path)
        self.assertFalse(os.path.exists(os.path.join(root, "Forecasts", "future_forecasts.csv")))  # needs future covariates
        splits = pd.read_csv(os.path.join(root, "Splits", "splits.csv"))
        self.assertEqual(list(splits.columns), ["fold", "set", "id", "start", "end", "points"])
        validation = splits[splits.set == "validation"]
        self.assertTrue((validation.points == 4).all())  # one horizon per window and series
        self.assertEqual(sorted(validation.fold.unique()), [1, 2])
        tests = pd.read_excel(os.path.join(root, "test_results.xlsx"), index_col=0)
        self.assertEqual(list(tests.index), ["NHITS", "DLinear", "Seasonal naive"])
        forecasts = pd.read_csv(os.path.join(root, "Forecasts", "test_forecasts.csv"))
        self.assertEqual(len(forecasts), 8 * 4)
        self.assertEqual(list(forecasts.columns), ["ID", "Time", "step", "Target", "NHITS", "DLinear", "Seasonal naive"])
        importance = pd.read_csv(os.path.join(root, "Explainability", "NHITS_feature_importance.csv"))
        self.assertAlmostEqual(importance.share_of_attribution_percent.sum(), 100)
        self.assertEqual(set(importance.input), {"Target (history)", "Lab (past)", "Dose (future)",
                                                 "Group_A (static)", "Group_B (static)"})

        # Results page and the standalone code, run without the repository
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            page = Client(appmod.application).get("/automl/results").get_data(as_text=True)
        finally:
            os.chdir(cwd)
        self.assertIn("Rolling-origin validation", page)
        code = html.unescape(re.sub(r"<[^>]+>", "", re.search(r'id="code-predict"><code>(.*?)</code></pre>', page, re.S).group(1)))
        self.assertIn('NeuralForecast.load("models/NHITS")', code)
        self.assertNotIn("Helpers", code)
        history, future = [], []
        for uid, part in test.groupby("ID", sort=False):
            history.append(pd.concat([train[train.ID == uid], part.iloc[:-4]]))
            future.append(part.iloc[-4:][["ID", "Time", "Dose"]])
        pd.concat(history).to_csv(os.path.join(self.dir, "history.csv"), index=False)
        pd.concat(future).to_csv(os.path.join(self.dir, "future_covariates.csv"), index=False)
        script = code + ("\nforecast.to_csv('snippet.csv', index=False)\nimport sys\n"
                         "assert not {m.split('.')[0] for m in sys.modules} & {'Helpers', 'web', 'featurewiz'}\n")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        run = subprocess.run([sys.executable, "-c", script], cwd=self.dir, capture_output=True, text=True, env=env, timeout=600)
        self.assertEqual(run.returncode, 0, run.stderr[-2000:])
        snippet = pd.read_csv(os.path.join(self.dir, "snippet.csv")).rename(columns={"unique_id": "ID", "ds": "Time"})
        merged = forecasts.merge(snippet, on=["ID", "Time"], suffixes=("", "_snippet"))
        self.assertEqual(len(merged), len(forecasts))
        np.testing.assert_allclose(merged["NHITS_snippet"], merged["NHITS"], atol=1e-4)

    def test_forecasts_beyond_the_data_without_future_covariates(self):
        result, _, _ = self.run_pipeline([])
        self.assertEqual(result, "Pipeline completed successfully")
        future = pd.read_csv(os.path.join(self.dir, "Materials", "Forecasts", "future_forecasts.csv"))
        self.assertEqual(len(future), 8 * 4)
        self.assertEqual(future[future.ID == "S0"].Time.tolist(), [52, 53, 54, 55])  # after the last test point
        self.assertTrue(os.path.exists(os.path.join(self.dir, "Materials", "Forecast_Plots", "DLinear_future_forecasts.png")))


class TestWebFlow(TempDir):
    def setUp(self):
        super().setUp()
        self.cwd = os.getcwd()
        os.chdir(self.dir)
        self.client = Client(appmod.application)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(appmod.FORECAST_INPUT_FOLDER, ignore_errors=True)
        super().tearDown()

    def upload(self, train, test):
        with open(train, "rb") as train_file, open(test, "rb") as test_file:
            return self.client.post("/automl/forecasting/upload", data={
                "train_file": (train_file, "Train.csv"), "test_file": (test_file, "Test.csv")})

    def test_pages_and_example_files(self):
        self.assertEqual(self.client.get("/automl/forecasting").status_code, 200)
        response = self.client.get("/automl/forecasting/example/Train.csv")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data.startswith(b"ID,Time,Target"))
        response.close()
        self.assertEqual(self.client.get("/automl/forecasting/example/app.py").status_code, 404)
        self.assertEqual(self.client.get("/automl/automators/time-series-forecasting").headers["Location"], "/automl/forecasting")

    def test_invalid_upload_shows_its_message(self):
        train, test = synthetic()
        response = self.upload(*self.write(train.drop(columns=["Time"]), test))
        self.assertEqual(response.headers["Location"], "/automl/forecasting")
        self.assertIn("no Time column", self.client.get("/automl/forecasting").get_data(as_text=True))

    def test_upload_configure_and_run(self):
        train, test = synthetic()
        response = self.upload(*self.write(train, test))
        self.assertEqual(response.headers["Location"], "/automl/forecasting/parameters")
        page = self.client.get("/automl/forecasting/parameters").get_data(as_text=True)
        self.assertIn('name="role_0"', page)  # one role per time-varying covariate
        self.assertIn("Slow on CPU", page)
        # Too long a horizon is refused
        response = self.client.post("/automl/forecasting/parameters", data={"NHITS": "true", "horizon": "40"})
        self.assertEqual(response.headers["Location"], "/automl/forecasting/parameters")
        form = {"DLinear": "true", "horizon": "4", "k_folds": "2", "trials": "0", "max_steps": "50", "season": "1",
                "lookback_mode": "fixed", "lookback": "8", "role_0": "future", "role_1": "past"}
        self.assertEqual(self.client.post("/automl/forecasting/parameters", data=form).headers["Location"], "/automl/run")
        with open(os.path.join(appmod.FORECAST_INPUT_FOLDER, "params.json")) as f:
            params = json.load(f)
        self.assertEqual((params["future_columns"], params["lookback"], params["models"]), (["Dose"], 8, ["DLinear"]))
        for _ in range(1200):
            status = json.loads(self.client.get("/automl/api/status").data)
            if status["state"] != "running":
                break
            time.sleep(0.1)
        self.assertEqual(status["state"], "done", status["message"])
        self.assertEqual(status["models"][0]["test"], "done")
        page = self.client.get("/automl/results").get_data(as_text=True)
        self.assertIn("Test series (Test.csv)", page)
        self.assertIn("DLinear: test forecasts vs. observed values", page)


if __name__ == "__main__":
    unittest.main()
