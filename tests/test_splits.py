"""Tests of the validation splits written with the results (Materials/Splits): the shared writer,
the tabular K-fold splits (ID column and line of Train.csv) and the card of the results page."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from Helpers.data_checks import DataChecker  # noqa: E402
from Helpers.splits import index_rows, write_splits  # noqa: E402


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.cwd = os.getcwd()
        os.chdir(self.dir)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()


class TestWriter(TempDir):
    def test_csv_and_json(self):
        folds = [([0, 1], [2]), ([1, 2], [0]), ([0, 2], [1])]
        rows = index_rows(folds, ["a", "b", "c"], {"patient": np.array(["p1", "p1", "p2"])})
        write_splits(rows, kind="kfold", description="three folds")
        table = pd.read_csv("Materials/Splits/splits.csv")
        self.assertEqual(list(table.columns), ["fold", "set", "id", "patient"])
        self.assertEqual(len(table), 9)
        meta = json.load(open("Materials/Splits/splits.json"))
        self.assertEqual(meta["kind"], "kfold")
        self.assertEqual(meta["folds"][0], {"fold": 1, "train": ["a", "b"], "validation": ["c"]})

    def test_three_sets(self):
        rows = index_rows([([0], [1], [2])], [10, 11, 12], sets=("train", "early_stopping", "validation"))
        write_splits(rows, kind="kfold")
        meta = json.load(open("Materials/Splits/splits.json"))
        self.assertEqual(meta["folds"][0], {"fold": 1, "train": [10], "early_stopping": [11], "validation": [12]})


class TestTabular(TempDir):
    def write(self, with_ids):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"a": rng.normal(size=20), "b": rng.normal(size=20), "Target": [0, 1] * 10})
        frame.loc[[3, 7], "a"] = np.nan  # removed before the folds
        if with_ids:
            frame.insert(0, "ID", [f"P{i:02d}" for i in range(20)])
        os.makedirs("input", exist_ok=True)
        frame.to_csv("input/Train.csv", index=False)
        frame.to_csv("input/Test.csv", index=False)
        os.makedirs("Materials", exist_ok=True)

    def splits(self, with_ids):
        from Helpers.pipelines_main import write_kfold_splits
        self.write(with_ids)
        checker = DataChecker("input")
        train, _ = checker.process_data()
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=10)
        write_kfold_splits(train.drop("Target", axis=1), train["Target"], skf, checker.train_rows, checker.train_has_ids)
        return pd.read_csv("Materials/Splits/splits.csv")

    def test_ids_and_lines_of_train_csv(self):
        table = self.splits(with_ids=True)
        validation = table[table.set == "validation"]
        self.assertEqual(len(validation), 18)  # 20 rows, 2 with missing values
        self.assertNotIn(4, set(table.row))  # line 4 is row index 3 (missing value)
        source = pd.read_csv("input/Train.csv")
        for _, row in validation.iterrows():
            self.assertEqual(source.loc[row.row - 1, "ID"], row.id)
            self.assertEqual(source.loc[row.row - 1, "Target"], row["class"])

    def test_without_id_column_the_line_is_the_id(self):
        table = self.splits(with_ids=False)
        self.assertTrue((table.id == table.row).all())


class TestResultsPage(TempDir):
    def test_card(self):
        import app as appmod
        from werkzeug.test import Client
        os.makedirs("Materials", exist_ok=True)
        write_splits(index_rows([([0, 1], [2]), ([0, 2], [1])], ["x.png", "y.png", "z.png"]), kind="kfold",
                     description="Two folds of the images.")
        with open("Materials/run_info.json", "w") as f:
            json.dump({"automator": "tabular", "classes": [0, 1]}, f)
        page = Client(appmod.application).get("/automl/results").get_data(as_text=True)
        for text in ("Validation splits", "Two folds of the images.", "Splits/splits.csv", "splits.json"):
            self.assertIn(text, page)


if __name__ == "__main__":
    unittest.main()
