"""Tests of the survival analysis automator: data checks, metrics against direct computations,
every model on simulated data, the pipeline end to end, the web flow and the standalone usage code."""
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
from Helpers.survival import data as sdata
from Helpers.survival import metrics as sm
from Helpers.survival.models import MODELS, CoxPH, build
from Helpers.survival.pipeline import run_survival_pipeline

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO, "Examples", "survival")


def cohort(n=300, seed=0, beta=(0.8, -0.5)):
    """Exponential event times with hazard exp(0.8 x1 - 0.5 x2), a categorical feature with an
    effect, a noise feature, and uniform censoring (~35%)."""
    rng = np.random.default_rng(seed)
    x1, x2, noise = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    group = rng.choice(["A", "B"], n)
    eta = beta[0] * x1 + beta[1] * x2 + 0.5 * (group == "B")
    t = rng.exponential(10 * np.exp(-eta))
    c = rng.uniform(0, 30, n)
    return pd.DataFrame({"ID": [f"P{seed}_{i}" for i in range(n)], "Time": np.round(np.maximum(np.minimum(t, c), 0.01), 3),
                         "Event": (t <= c).astype(int), "x1": x1.round(4), "x2": x2.round(4), "noise": noise.round(4),
                         "Group": group})


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, train, test):
        paths = os.path.join(self.dir, "Train.csv"), os.path.join(self.dir, "Test.csv")
        train.to_csv(paths[0], index=False)
        test.to_csv(paths[1], index=False)
        return paths


class TestData(TempDir):
    def test_example_summary(self):
        summary = sdata.summarize(os.path.join(EXAMPLE, "Train.csv"), os.path.join(EXAMPLE, "Test.csv"))
        self.assertEqual(summary["errors"], [])
        self.assertEqual((summary["train_rows"], summary["test_rows"], summary["id_column"]), (800, 300, "ID"))
        self.assertEqual(summary["categorical"], ["Sex", "Stage", "Adjuvant_chemo"])
        self.assertGreater(summary["events"], 300)
        self.assertEqual(len(summary["suggested_horizons"]), 3)
        self.assertTrue(summary["suggested_horizons"] == sorted(summary["suggested_horizons"]))

    def test_errors(self):
        train, test = cohort(), cohort(seed=1)
        for broken, message in ((train.drop(columns=["Event"]), "no Event column"),
                                (train.assign(Time=-1.0), "Time must be positive"),
                                (train.assign(Event=2), "Event must be 1"),
                                (train.assign(Event=0), "survival models need at least 10"),
                                (train[["ID", "Time", "Event"]], "no usable feature")):
            with self.subTest(message=message):
                summary = sdata.summarize(*self.write(broken, test))
                self.assertTrue(any(message in e for e in summary["errors"]), summary["errors"])
        summary = sdata.summarize(*self.write(train, test.drop(columns=["x2"])))
        self.assertTrue(any("lacks the feature column(s) x2" in e for e in summary["errors"]))
        summary = sdata.summarize(*self.write(train.assign(Event=train.Event.map({1: "yes", 0: "no"})), test))
        self.assertEqual(summary["errors"], [])   # yes/no events are accepted


class TestMetrics(unittest.TestCase):
    def test_kaplan_meier_and_c_index_by_hand(self):
        time = np.array([1, 2, 2, 3, 4, 5.0])
        event = np.array([1, 1, 0, 1, 0, 1])
        times, surv = sm.kaplan_meier(time, event)
        np.testing.assert_allclose(surv, [5 / 6, 5 / 6 * 4 / 5, 5 / 6 * 4 / 5 * 2 / 3, 5 / 6 * 4 / 5 * 2 / 3, 0.0])
        # Brute force: pairs (i event, j later) ordered by risk
        risk = np.array([5, 3, 4, 2, 1, 0.0])
        pairs = [(i, j) for i in range(6) for j in range(6) if event[i] and time[j] > time[i]]
        expected = np.mean([1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0 for i, j in pairs])
        self.assertAlmostEqual(sm.harrell_c(time, event, risk), expected)
        self.assertAlmostEqual(sm.harrell_c(time, event, -time), 1.0)
        # Without censoring, Uno's C-index is Harrell's and the Brier score is the plain squared error
        full = np.ones(6, int)
        censoring = sm.Censoring(time, full)
        self.assertAlmostEqual(sm.uno_c(time, full, risk, censoring), sm.harrell_c(time, full, risk))
        surv_t = np.linspace(0.1, 0.9, 6)
        self.assertAlmostEqual(sm.brier(2.5, surv_t, time, full, censoring), np.mean(((time > 2.5) - surv_t) ** 2))

    def test_logrank_and_auc(self):
        data = cohort(n=400)
        x1 = data.x1.to_numpy()
        _, dof, p_separated = sm.logrank(data.Time, data.Event, np.where(x1 > 0, "high", "low"))
        rng = np.random.default_rng(1)
        _, _, p_random = sm.logrank(data.Time, data.Event, rng.choice(["a", "b"], len(data)))
        self.assertEqual(dof, 1)
        self.assertLess(p_separated, 1e-6)
        self.assertGreater(p_random, 0.01)
        censoring = sm.Censoring(data.Time, data.Event)
        self.assertGreater(sm.td_auc(5.0, x1, data.Time, data.Event, censoring), 0.65)
        self.assertAlmostEqual(sm.td_auc(5.0, np.zeros(len(data)), data.Time, data.Event, censoring), 0.5)


class TestModels(unittest.TestCase):
    def test_cox_recovers_the_coefficients(self):
        data = cohort(n=2000, seed=3)
        model = CoxPH(penalty=0.0).fit(data[["x1", "x2"]].to_numpy(), data.Time.to_numpy(), data.Event.to_numpy())
        np.testing.assert_allclose(model.coef_, [0.8, -0.5], atol=0.1)

    def test_every_model_ranks_and_predicts_survival(self):
        train, test = cohort(n=400, seed=4), cohort(n=200, seed=5)
        features = ["x1", "x2", "noise"]
        X, Xt = train[features].to_numpy(), test[features].to_numpy()
        times = np.array([1.0, 5.0, 10.0])
        for spec in MODELS:
            with self.subTest(model=spec.key):
                model = build(spec.key, {"epochs": 60, "seed": 0}).fit(X, train.Time.to_numpy(), train.Event.to_numpy())
                self.assertGreater(sm.harrell_c(test.Time, test.Event, model.predict_risk(Xt)), 0.65)
                S = model.predict_survival(Xt, times)
                self.assertEqual(S.shape, (len(test), 3))
                self.assertTrue(((S >= -1e-9) & (S <= 1 + 1e-9)).all())
                self.assertTrue((np.diff(S, axis=1) <= 1e-9).all())   # non-increasing in time


class PipelineRun(TempDir):
    def run_pipeline(self, train, test, params):
        os.makedirs(os.path.join(self.dir, "input"))
        train.to_csv(os.path.join(self.dir, "input", "Train.csv"), index=False)
        test.to_csv(os.path.join(self.dir, "input", "Test.csv"), index=False)
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            return run_survival_pipeline("input", params)
        finally:
            os.chdir(cwd)


class TestPipeline(PipelineRun):
    def test_pipeline_outputs_page_and_standalone_code(self):
        train, test = cohort(n=300, seed=6), cohort(n=150, seed=7)
        params = {"models": ["coxph", "xgb_cox", "deepsurv"], "k_folds": 3, "horizons": [3, 8], "epochs": 60}
        self.assertEqual(self.run_pipeline(train, test, params), "Pipeline completed successfully")
        root = os.path.join(self.dir, "Materials")
        info = json.load(open(os.path.join(root, "run_info.json")))
        self.assertEqual((info["automator"], info["horizons"], info["skipped"]), ("survival-analysis", [3.0, 8.0], []))
        for path in ["3_fold_results.xlsx", "test_results.xlsx", "Predictions/test_predictions.csv", "Models/Cox_PH.pkl",
                     "Models/DeepSurv.pkl", "Survival_Plots/Cox_PH_risk_groups.png", "Survival_Plots/Cox_PH_patients.png",
                     "Metrics_Plots/Cox_PH_calibration.png", "Metrics_Plots/auc_over_time.png", "Metrics_Plots/brier_over_time.png",
                     "Explainability/Cox_PH_importance.png", "Explainability/Cox_PH_coefficients.csv",
                     "Explainability/XGBoost_Cox_permutation_importance.csv"]:
            self.assertTrue(os.path.exists(os.path.join(root, path)), path)
        tests = pd.read_excel(os.path.join(root, "test_results.xlsx"), index_col=0)
        self.assertEqual(list(tests.index), ["Cox PH", "XGBoost Cox", "DeepSurv", "Kaplan-Meier (no covariates)"])
        self.assertEqual(list(tests.columns), ["C-index", "Uno C-index", "IBS", "AUC@3", "Brier@3", "AUC@8", "Brier@8"])
        self.assertGreater(tests.loc["Cox PH", "C-index"], 0.7)
        self.assertLess(tests.loc["Cox PH", "IBS"], tests.loc["Kaplan-Meier (no covariates)", "IBS"])
        importance = pd.read_csv(os.path.join(root, "Explainability", "Cox_PH_permutation_importance.csv"))
        self.assertEqual(importance.feature[0], "x1")
        self.assertIn("noise", list(importance.feature.iloc[-2:]))   # the noise feature matters least
        coefficients = pd.read_csv(os.path.join(root, "Explainability", "Cox_PH_coefficients.csv"))
        self.assertGreater(coefficients.set_index("feature").loc["x1", "hazard_ratio"], 1.5)
        splits = pd.read_csv(os.path.join(root, "Splits", "splits.csv"))
        self.assertEqual(list(splits.columns), ["fold", "set", "id", "row", "event"])
        self.assertEqual(len(splits[splits.set == "validation"]), 300)

        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            page = Client(appmod.application).get("/automl/results").get_data(as_text=True)
        finally:
            os.chdir(cwd)
        self.assertIn("Cross-validation (Train.csv): 3 folds", page)
        self.assertIn("Cox PH: risk groups", page)
        code = html.unescape(re.sub(r"<[^>]+>", "", re.search(r'id="code-predict"><code>(.*?)</code></pre>', page, re.S).group(1)))
        self.assertNotIn("Helpers", code)
        test.drop(columns=["Time", "Event"]).to_csv(os.path.join(self.dir, "new_patients.csv"), index=False)
        script = code + ("\nsurvival.join(risk).to_csv('snippet.csv')\nimport sys\n"
                         "assert not {m.split('.')[0] for m in sys.modules} & {'Helpers', 'web', 'featurewiz'}\n")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        run = subprocess.run([sys.executable, "-c", script], cwd=self.dir, capture_output=True, text=True, env=env, timeout=600)
        self.assertEqual(run.returncode, 0, run.stderr[-2000:])
        snippet = pd.read_csv(os.path.join(self.dir, "snippet.csv"))
        expected = pd.read_csv(os.path.join(root, "Predictions", "test_predictions.csv"))
        best = info["best_model"]
        np.testing.assert_allclose(snippet["S(3)"], expected[f"{best} S(3)"], atol=1e-3)
        np.testing.assert_allclose(snippet["risk"], expected[f"{best} risk"], atol=1e-3)


class TestWebFlow(TempDir):
    def setUp(self):
        super().setUp()
        self.cwd = os.getcwd()
        os.chdir(self.dir)
        self.client = Client(appmod.application)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(appmod.SURVIVAL_INPUT_FOLDER, ignore_errors=True)
        super().tearDown()

    def upload(self, train, test):
        with open(train, "rb") as a, open(test, "rb") as b:
            return self.client.post("/automl/survival/upload", data={"train_file": (a, "Train.csv"), "test_file": (b, "Test.csv")})

    def test_pages_and_example_files(self):
        self.assertEqual(self.client.get("/automl/survival").status_code, 200)
        response = self.client.get("/automl/survival/example/Train.csv")
        self.assertTrue(response.data.startswith(b"ID,Time,Event"))
        response.close()
        self.assertEqual(self.client.get("/automl/survival/example/app.py").status_code, 404)
        self.assertEqual(self.client.get("/automl/automators/survival-analysis").headers["Location"], "/automl/survival")

    def test_invalid_upload_shows_its_message(self):
        response = self.upload(*self.write(cohort().drop(columns=["Event"]), cohort(seed=1)))
        self.assertEqual(response.headers["Location"], "/automl/survival")
        self.assertIn("no Event column", self.client.get("/automl/survival").get_data(as_text=True))

    def test_upload_configure_and_run(self):
        response = self.upload(*self.write(cohort(n=200), cohort(n=100, seed=1)))
        self.assertEqual(response.headers["Location"], "/automl/survival/parameters")
        page = self.client.get("/automl/survival/parameters").get_data(as_text=True)
        self.assertIn('name="ignore_3"', page)
        response = self.client.post("/automl/survival/parameters", data={"coxph": "true", "horizons": "5, 9999"})
        self.assertEqual(response.headers["Location"], "/automl/survival/parameters")   # horizon past the follow-up
        form = {"coxph": "true", "weibull_aft": "true", "k_folds": "2", "horizons": "2, 6", "selection_metric": "IBS",
                "ignore_2": "true", "penalty": "0.1", "epochs": "50"}
        self.assertEqual(self.client.post("/automl/survival/parameters", data=form).headers["Location"], "/automl/run")
        with open(os.path.join(appmod.SURVIVAL_INPUT_FOLDER, "params.json")) as f:
            params = json.load(f)
        self.assertEqual((params["horizons"], params["ignore_columns"], params["explain"]), ([2.0, 6.0], ["noise"], False))
        for _ in range(3000):
            status = json.loads(self.client.get("/automl/api/status").data)
            if status["state"] != "running":
                break
            time.sleep(0.1)
        self.assertEqual(status["state"], "done", status["message"])
        self.assertEqual([m["test"] for m in status["models"]], ["done", "done"])
        page = self.client.get("/automl/results").get_data(as_text=True)
        self.assertIn("Best model (validation IBS)", page)
        self.assertIn("Weibull AFT: risk groups", page)


if __name__ == "__main__":
    unittest.main()
