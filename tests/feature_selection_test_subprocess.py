import unittest
import pandas as pd
from sklearn.datasets import make_classification
from unittest.mock import patch
from Helpers.pipelines import FeatureSelection


class TestFeatureSelection(unittest.TestCase):

    def setUp(self):
        self.X_train, self.y_train = make_classification(n_samples=100,
                                                         n_features=10,
                                                         random_state=42)
        self.X_train = pd.DataFrame(self.X_train,
                                    columns=[
                                        f'feature_{i}' for i in range(10)
                                            ])
        self.y_train = pd.Series(self.y_train)
        self.feature_selector = FeatureSelection()

    @patch('featurewiz.FeatureWiz')
    def test_featurewiz_selection(self, MockFeatureWiz):
        mock_featwiz = MockFeatureWiz.return_value
        mock_featwiz.fit.return_value = None
        mock_featwiz.transform.return_value = self.X_train

        self.feature_selector.featurewiz_selection(self.X_train, self.y_train)

        mock_featwiz.fit.assert_called_once_with(self.X_train, self.y_train)
        self.assertIsNotNone(self.feature_selector.selected_features)

    def test_get_features(self):
        self.feature_selector.selected_features = ['feature_1', 'feature_2']
        features, feat_sel = self.feature_selector.get_features()

        self.assertEqual(features, ['feature_1', 'feature_2'])

    def test_rfe_selection(self):
        self.feature_selector.rfe_selection(self.X_train, self.y_train,
                                            n_features_to_select=5)
        self.assertEqual(len(self.feature_selector.selected_features), 5)

    def test_lasso_selection(self):
        self.feature_selector.lasso_selection(self.X_train, self.y_train,
                                              alpha=0.001)
        self.assertIsNotNone(self.feature_selector.selected_features)

    def test_random_forest_selection(self):
        self.feature_selector.random_forest_selection(self.X_train,
                                                      self.y_train,
                                                      threshold=0.01)
        self.assertIsNotNone(self.feature_selector.selected_features)

    def test_xgboost_selection(self):
        self.feature_selector.xgboost_selection(self.X_train, self.y_train,
                                                threshold=0.01)
        self.assertIsNotNone(self.feature_selector.selected_features)


if __name__ == '__main__':
    unittest.main()
