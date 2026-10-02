import io
import os
import tempfile
import time
import unittest
from unittest import mock

import pandas as pd
from werkzeug.test import Client

import app as appmod
from web.catalog import AUTOMATORS, MODELS
from web.jobs import PipelineJob

TRAIN = "a,Sex,Target\n1,M,1\n2,F,2\n3,M,1\n"
TEST = "a,Sex,Target\n1,M,1\n"
GOOD_TRAIN = "a,Sex,Target\n1,M,0\n2,F,1\n3,M,0\n4,F,1\n"


def fake_pipeline(models, fail=None):
    """Prints the progress lines of Helpers.pipelines_main, like a real run."""
    def run():
        print("------------- \n", "Loading Data \n", "-------------")
        for phase in ("Training on K-Fold cross validation \n", "Evaluating algorithms on Test.csv \n"):
            print("------------- \n", phase, "-------------")
            for name in models:
                if name == fail:
                    if "K-Fold" in phase:
                        print("-------------------- \n", f"{name} failed and was skipped: no weights \n", "--------------------")
                    continue
                print("-------------------- \n", f"{name} is starting \n", "--------------------")
                print("-------------------- \n", f"{name} is completed successfully \n", "--------------------")
        print("Pipeline completed successfully.")
        return "Pipeline completed successfully"
    return run


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.client = Client(appmod.application)
        self.workdir = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.workdir.name)  # the pipeline outputs go to ./Materials
        appmod.job = PipelineJob()

    def tearDown(self):
        os.chdir(self.cwd)
        self.workdir.cleanup()

    def upload(self, train, test, train_name="Train.csv"):
        return self.client.post("/automl/upload", content_type="multipart/form-data", data={
            "train_file": (io.BytesIO(train.encode()), train_name),
            "test_file": (io.BytesIO(test.encode()), "Test.csv"),
        })

    def wait_for_job(self):
        for _ in range(100):
            if not appmod.job.running:
                return
            time.sleep(0.05)
        self.fail("the job did not finish")


class TestPages(AppTestCase):
    def test_landing_lists_the_automators(self):
        page = self.client.get("/automl/").get_data(as_text=True)
        for automator in AUTOMATORS:
            self.assertIn(automator.name, page)
        if not all(automator.available for automator in AUTOMATORS):
            self.assertIn("Coming soon", page)

    def test_automator_pages(self):
        for automator in AUTOMATORS:
            response = self.client.get(f"/automl/automators/{automator.slug}")
            if automator.available:
                self.assertEqual(response.status_code, 302)
                self.assertEqual(response.headers["Location"], f"/automl/{automator.endpoint}")
            else:
                self.assertEqual(response.status_code, 200)
                self.assertIn(automator.steps[0], response.get_data(as_text=True))
        self.assertEqual(self.client.get("/automl/automators/unknown").status_code, 404)

    def test_tabular_flow_requires_uploaded_data(self):
        self.assertEqual(self.client.get("/automl/tabular").status_code, 200)
        self.assertEqual(self.client.get("/automl/run").headers["Location"], "/automl/")

    def test_static_assets_are_served_locally(self):
        for path in ("css/app.css", "js/app.js", "vendor/bootstrap-5.3.8/css/bootstrap.min.css",
                     "vendor/bootstrap-icons-1.13.1/fonts/bootstrap-icons.woff2"):
            response = self.client.get(f"/automl/static/{path}")
            self.assertEqual(response.status_code, 200, path)
            response.close()


class TestUploadAndParameters(AppTestCase):
    def test_invalid_upload_shows_its_message(self):
        response = self.upload("a,Target\n1,0\n", "a,Target\n1,0\n", train_name="Train.txt")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/automl/tabular")
        page = self.client.get("/automl/tabular").get_data(as_text=True)
        self.assertIn("Invalid file type", page)

    def test_parameters_warns_about_target_labels(self):
        self.upload(TRAIN, TEST)
        page = self.client.get("/automl/parameters").get_data(as_text=True)
        self.assertIn("Check the Target column", page)

        self.upload(GOOD_TRAIN, "a,Sex,Target\n1,M,1\n")
        page = self.client.get("/automl/parameters").get_data(as_text=True)
        self.assertNotIn("Check the Target column", page)
        self.assertIn('<option value="Sex">Sex</option>', page)  # bias assessment candidates
        for model in MODELS:
            self.assertIn(f'name="{model.field}"', page)

    def test_run_requires_a_model(self):
        self.upload(GOOD_TRAIN, "a,Sex,Target\n1,M,1\n")
        response = self.client.post("/automl/parameters", data={"k_folds": "2", "correlation_limit": "0.7",
                                                                 "optimization_metric": "AUC"})
        self.assertEqual(response.headers["Location"], "/automl/parameters")
        self.assertEqual(appmod.job.state, "idle")

    def test_run_in_background_with_live_status(self):
        self.upload(GOOD_TRAIN, "a,Sex,Target\n1,M,1\n")
        form = {"k_folds": "2", "correlation_limit": "0.7", "optimization_metric": "AUC", "grid_search": "true",
                "grid_search_type": "randomized", "logistic_regression": "true", "tabpfn": "true"}
        with mock.patch.object(appmod, "run_pipeline",
                               lambda *args: fake_pipeline(["Logistic Regression", "TabPFNv2"], fail="TabPFNv2")()):
            response = self.client.post("/automl/parameters", data=form)
            self.assertEqual(response.headers["Location"], "/automl/run")
            self.wait_for_job()
        status = self.client.get("/automl/api/status").get_json()
        self.assertEqual(status["state"], "done")
        self.assertEqual(status["progress"], 100)
        models = {m["name"]: m for m in status["models"]}
        self.assertEqual((models["Logistic Regression"]["kfold"], models["Logistic Regression"]["test"]), ("done", "done"))
        self.assertEqual((models["TabPFNv2"]["kfold"], models["TabPFNv2"]["test"]), ("skipped", "skipped"))
        self.assertEqual(models["TabPFNv2"]["note"], "no weights")
        self.assertEqual(self.client.get("/automl/run").status_code, 200)

        with open(os.path.join(appmod.TEMP_INPUT_FOLDER, "machine_learning_parameters.yaml")) as f:
            saved = f.read()
        self.assertIn("Logistic Regression: true", saved)
        self.assertIn("XGBoost: false", saved)

    def test_new_run_replaces_previous_results(self):
        self.upload(GOOD_TRAIN, "a,Sex,Target\n1,M,1\n")
        os.makedirs("Materials", exist_ok=True)
        with open(os.path.join("Materials", "5_fold_results.xlsx"), "wb") as f:
            f.write(b"old run")
        page = self.client.get("/automl/parameters").get_data(as_text=True)
        self.assertIn("Starting this run replaces the previous results", page)

        form = {"k_folds": "2", "correlation_limit": "0.7", "optimization_metric": "AUC", "xgboost": "true"}
        with mock.patch.object(appmod, "run_pipeline", lambda *args: fake_pipeline(["XGBoost"])()):
            self.client.post("/automl/parameters", data=form)
            self.wait_for_job()
        self.assertEqual(os.listdir("Materials"), ["Models"])


class TestJobs(unittest.TestCase):
    def test_error_and_single_run(self):
        job = PipelineJob()
        self.assertTrue(job.start(lambda: (time.sleep(0.2), "Error: bad data")[1], ["XGBoost"]))
        self.assertFalse(job.start(lambda: None, ["XGBoost"]))  # one run at a time
        while job.running:
            time.sleep(0.02)
        self.assertEqual(job.snapshot()["state"], "error")
        self.assertEqual(job.snapshot()["message"], "Error: bad data")


class TestResults(AppTestCase):
    def make_materials(self):
        root = os.path.join(self.workdir.name, "Materials")
        for folder in ("ROC_Curves", "ConfusionMatrices", "Shap_Features/XGBoost", "Models"):
            os.makedirs(os.path.join(root, folder), exist_ok=True)
        metrics = ["Sensitivity", "Specificity", "AUC", "F-score", "Accuracy", "Balanced Accuracy"]
        pd.DataFrame([[0.8, 0.9, 0.95, 0.85, 0.88, 0.86], [0.7, 0.8, 0.90, 0.75, 0.78, 0.76]],
                     index=["XGBoost", "Decision Trees"], columns=metrics).to_excel(os.path.join(root, "test_results.xlsx"))
        pd.DataFrame([["0.9 ± 0.1"] * 6], index=["XGBoost"], columns=metrics).to_excel(os.path.join(root, "5_fold_results.xlsx"))
        for path in ("ROC_Curves/ROC_CURVES.png", "ConfusionMatrices/XGBoost_Test_confusion_matrix.png",
                     "ConfusionMatrices/XGBoost_Internal_5_fold_confusion_matrix.png",
                     "Shap_Features/XGBoost/bar_plot_XGBoost.png", "Models/XGBoost_pipeline.pkl"):
            with open(os.path.join(root, path), "wb") as f:
                f.write(b"data")
        with open(os.path.join(root, "error_log.log"), "w") as f:
            f.write("2026-01-01 10:00:00,000:ERROR:TabICL failed and was skipped: no weights\n")

    def test_empty_results(self):
        page = self.client.get("/automl/results").get_data(as_text=True)
        self.assertIn("No results yet", page)

    def test_results_dashboard(self):
        self.make_materials()
        page = self.client.get("/automl/results").get_data(as_text=True)
        self.assertIn("Best model on the test set", page)
        self.assertIn("XGBoost", page)
        self.assertIn("0.950", page)
        self.assertIn("Internal 5-fold cross-validation", page)
        self.assertIn("TabICL", page)  # skipped model
        self.assertIn("Shap_Features/XGBoost/bar_plot_XGBoost.png", page)

        results = appmod.collect_results(os.path.abspath("Materials"))
        self.assertEqual(results["best"]["model"], "XGBoost")
        self.assertEqual([i["title"] for i in results["confusion"]["XGBoost"]],
                         ["Internal K-fold (mean)", "External test"])
        self.assertEqual(results["models"][0]["name"], "XGBoost")

    def test_files_are_served_from_materials_only(self):
        self.make_materials()
        response = self.client.get("/automl/materials/ROC_Curves/ROC_CURVES.png")
        self.assertEqual(response.status_code, 200)
        response.close()
        response = self.client.get("/automl/download/Models/XGBoost_pipeline.pkl")
        self.assertIn("attachment", response.headers["Content-Disposition"])
        response.close()
        self.assertEqual(self.client.get("/automl/materials/../app.py").status_code, 404)
        self.assertEqual(self.client.get("/automl/download/%2e%2e/%2e%2e/etc/passwd").status_code, 404)

    def test_clear_files(self):
        self.make_materials()
        response = self.client.post("/automl/clear_files")
        self.assertEqual(response.headers["Location"], "/automl/results")
        self.assertEqual(os.listdir("Materials"), ["Models"])


if __name__ == "__main__":
    unittest.main()
