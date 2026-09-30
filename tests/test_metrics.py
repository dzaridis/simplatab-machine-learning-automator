import unittest
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, roc_curve, auc
from Helpers.behave_metrics import Metrics  # Replace with the actual module name

class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Example ground truth and predicted probabilities
        self.y_test = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
        self.y_pred_proba = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.7, 0.3],
            [0.4, 0.6],
            [0.9, 0.1],
            [0.6, 0.4],
            [0.8, 0.2]
        ])
        self.threshold = 0.5

    def test_proba_convert(self):
        metrics = Metrics(y_pred=self.y_pred_proba, y_test=self.y_test)
        metrics._proba_convert(threshold=self.threshold)
        expected_y_pred = np.array([0, 1, 1, 0, 1, 0, 1, 0, 0, 0])
        np.testing.assert_array_equal(metrics.y_pred, expected_y_pred)

    def test_perf_measure(self):
        metrics = Metrics(y_pred=self.y_pred_proba, y_test=self.y_test)
        metrics._proba_convert(threshold=self.threshold)
        metrics._perf_measure()
        expected_mts = {"TP": 4, "FP": 0, "TN": 5, "FN": 1}
        self.assertEqual(metrics.mts, expected_mts)

    def test_sensitivity(self):
        metrics = Metrics(y_pred=self.y_pred_proba, y_test=self.y_test)
        metrics._proba_convert(threshold=self.threshold)
        metrics._perf_measure()
        metrics._sensitivity()
        expected_sensitivity = 4 / (4 + 1)
        self.assertAlmostEqual(metrics.sens, expected_sensitivity)

    def test_specificity(self):
        metrics = Metrics(y_pred=self.y_pred_proba, y_test=self.y_test)
        metrics._proba_convert(threshold=self.threshold)
        metrics._perf_measure()
        metrics._specificity()
        expected_specificity = 5 / (5 + 0)
        self.assertAlmostEqual(metrics.spec, expected_specificity)

    def test_roc_auc(self):
        metrics = Metrics(y_pred=self.y_pred_proba, y_test=self.y_test)
        y_scores = self.y_pred_proba[:, 1]
        expected_roc_auc = roc_auc_score(self.y_test, y_scores)
        metrics._roc_auc()
        self.assertAlmostEqual(metrics.roc, expected_roc_auc)

    def test_f_score(self):
        metrics = Metrics(y_pred=self.y_pred_proba, y_test=self.y_test)
        metrics._proba_convert(threshold=self.threshold)
        metrics._perf_measure()
        metrics._f_score()
        expected_precision = 4 / (4 + 0)
        expected_recall = 4 / (4 + 1)
        expected_f_score = 2 * (expected_precision * expected_recall) / (expected_precision + expected_recall)
        self.assertAlmostEqual(metrics.fsc, expected_f_score)

    def test_accuracy(self):
        metrics = Metrics(y_pred=self.y_pred_proba, y_test=self.y_test)
        metrics._proba_convert(threshold=self.threshold)
        metrics._perf_measure()
        metrics._accuracy()
        expected_accuracy = (4 + 5) / (4 + 5 + 0 + 1)
        self.assertAlmostEqual(metrics.acc, expected_accuracy)

    def test_balanced_accuracy(self):
        metrics = Metrics(y_pred=self.y_pred_proba, y_test=self.y_test)
        metrics._proba_convert(threshold=self.threshold)
        metrics._perf_measure()
        metrics._balanced_accuracy()
        expected_sensitivity = 4 / (4 + 1)
        expected_specificity = 5 / (5 + 0)
        expected_balanced_accuracy = (expected_sensitivity + expected_specificity) / 2
        self.assertAlmostEqual(metrics.balanced_acc, expected_balanced_accuracy)

    def test_compute_metrics(self):
        metrics = Metrics(y_pred=self.y_pred_proba, y_test=self.y_test)
        metrics.compute_metrics(threshold=self.threshold)

        y_scores = self.y_pred_proba[:, 1]
        expected_metrics = {
            "Sensitivity": recall_score(self.y_test, metrics.y_pred),
            "Specificity": 5 / (5 + 0),
            "AUC": roc_auc_score(self.y_test, y_scores),
            "F-score": f1_score(self.y_test, metrics.y_pred),
            "Accuracy": accuracy_score(self.y_test, metrics.y_pred),
            "Balanced Accuracy": (recall_score(self.y_test, metrics.y_pred) + 5 / (5 + 0)) / 2,
            "Confusion Matrix": {"TP": 4, "FP": 0, "TN": 5, "FN": 1}
        }

        actual_metrics = metrics.get_scores()
        for key in expected_metrics:
            if isinstance(expected_metrics[key], dict):
                self.assertEqual(actual_metrics[key], expected_metrics[key])
            else:
                self.assertAlmostEqual(actual_metrics[key], expected_metrics[key])

if __name__ == '__main__':
    unittest.main()