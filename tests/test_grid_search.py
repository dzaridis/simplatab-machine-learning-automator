import unittest

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from xgboost import XGBClassifier

from Helpers import pipelines
from Helpers.pipelines_main import adjust_hyperparameters_for_multiclass, hyperparameters_models_grid


class TestHyperparameterGrids(unittest.TestCase):
    def test_grid_values_are_valid(self):
        for name, entry in hyperparameters_models_grid.items():
            cls = entry["classifier"]
            if not hasattr(cls, "_parameter_constraints"):  # e.g. XGBoost: no scikit-learn validation
                continue
            for param, values in entry["params"].items():
                for value in values:
                    cls(**{param: value})._validate_params()  # raises InvalidParameterError

    def test_adjusted_multiclass_grids_only_contain_lists(self):
        for name, entry in hyperparameters_models_grid.items():
            grid = adjust_hyperparameters_for_multiclass(entry["classifier"], entry["params"], True, 3,
                                                         is_param_grid=True)
            ParameterGrid(grid)  # raises TypeError on non-list values
            for param, values in grid.items():
                self.assertIsInstance(values, list, f"{name}: {param}")

    def test_xgboost_multiclass_objective(self):
        fixed = XGBClassifier().get_params()
        adjusted = adjust_hyperparameters_for_multiclass(XGBClassifier, fixed, True, 3)
        self.assertEqual(adjusted["objective"], "multi:softprob")
        self.assertEqual(adjusted["num_class"], 3)
        grid = adjust_hyperparameters_for_multiclass(XGBClassifier, hyperparameters_models_grid["XGBoost"]["params"],
                                                     True, 3, is_param_grid=True)
        self.assertEqual(grid["objective"], ["multi:softprob"])
        self.assertEqual(grid["num_class"], [3])


class TestMulticlassGridSearch(unittest.TestCase):
    def test_candidates_are_scored(self):
        data = load_iris(as_frame=True)
        X, y = data.data, data.target.rename("Target")
        for name in ("Support Vector Machines", "XGBoost", "Decision Trees"):
            entry = hyperparameters_models_grid[name]
            grid = adjust_hyperparameters_for_multiclass(entry["classifier"], entry["params"], True, 3,
                                                         is_param_grid=True)
            pipeline = pipelines.MLPipeline(X, y, entry["classifier"], {})
            pipeline.execute_feature_selection(corr_limit=0.7)
            pipeline.execute_preprocessing()
            pipeline.train_model(perform_grid_search=True, param_grid=grid, hp_type="Randomized",
                                 cv=StratifiedKFold(3, shuffle=True, random_state=10))
            # Before the fix every candidate failed on multiclass targets and scored 0
            self.assertGreater(pipeline.model_trainer.grid_search.best_score_, 0.8, name)
            proba = pipeline.build_pipeline().predict_proba(X)
            self.assertEqual(proba.shape, (len(X), 3))


if __name__ == "__main__":
    unittest.main()
