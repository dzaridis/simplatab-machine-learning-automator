from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, is_classifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import featurewiz
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.feature_selection import RFE, SelectFromModel, mutual_info_classif
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

class FeatureSelection:
    def __init__(self) -> None:
        self.selected_features = None

    def featurewiz_selection(self, X_train, y_train, corr_limit=0.9):
        # Assume featurewiz is correctly imported and used here.
        self.selected_features, _ =featurewiz.featurewiz(X_train.join(y_train), target='Target', 
                                            corr_limit=corr_limit, verbose=0, feature_engg='',
                                            dask_xgboost_flag=False, nrows=None, skip_sulov=False, skip_xgboost=False)

    def correlation_filter(self, X_train, y_train, threshold=0.8):
        corr_matrix = X_train.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        self.selected_features = [feature for feature in X_train.columns if feature not in to_drop]

    def rfe_selection(self, X_train, y_train, n_features_to_select=5):
        model = LogisticRegression()
        rfe = RFE(model, n_features_to_select=n_features_to_select)
        rfe.fit(X_train, y_train)
        self.selected_features = X_train.columns[rfe.support_]

    def lasso_selection(self, X_train, y_train, alpha=0.001):
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        self.selected_features = X_train.columns[model.coef_ != 0]

    def random_forest_selection(self, X_train, y_train, threshold=0.01):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        self.selected_features = X_train.columns[importances > threshold]

    def xgboost_selection(self, X_train, y_train, threshold=0.01):
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        self.selected_features = X_train.columns[importances > threshold]

    def mutual_info_selection(self, X_train, y_train, threshold=0.01):
        mi = mutual_info_classif(X_train, y_train)
        self.selected_features = X_train.columns[mi > threshold]

    def get_features(self):
        return self.selected_features
    
class DataPreprocessor:
    def __init__(self, X_train):
        self.X_train = X_train
        self.preprocessor = None

    def auto_select_transformers(self):
        numerical_features = self.X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.X_train.select_dtypes(include=['object']).columns
        
        num_transformer = StandardScaler()
        cat_transformer = OneHotEncoder()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, numerical_features),
                ('cat', cat_transformer, categorical_features)
            ])
        return self.preprocessor

class FeatureSelector:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def select_features(self, method='featurewiz', **kwargs):
        fs = FeatureSelection()
        getattr(fs, f"{method}_selection")(self.X_train, self.y_train, **kwargs)
        selected_features = fs.get_features()
        self.X_train = self.X_train.loc[:, selected_features]
        return selected_features

class ModelTrainer:
    def __init__(self, X_train, y_train, classifier, classifier_hyperparameters):
        self.X_train = X_train
        self.y_train = y_train
        self.classifier = classifier
        self.classifier_hyperparameters = classifier_hyperparameters
        self.clas = None

    def fit(self):
        if not is_classifier(self.classifier):
            raise ValueError("The provided classifier is not a valid scikit-learn classifier.")

        # Adjusting class weights if applicable
        adjusted_hyperparameters = self.__adjust_class_weights(self.classifier, self.classifier_hyperparameters)

        # Create and fit the pipeline
        self.clas = self.classifier(**adjusted_hyperparameters).fit(self.X_train, self.y_train)
        return self.clas
    
    def perform_grid_search(self, param_grid, cv=5, scoring='recall'):
        self.grid_search = GridSearchCV(self.classifier(), param_grid, cv=cv, scoring=scoring)
        self.grid_search.fit(self.X_train, self.y_train)
        if self.grid_search.best_estimator_ is None:
            raise ValueError("Grid search did not yield a best estimator.")
        
        self.clas = self.grid_search.best_estimator_

        # For debugging:
        print(f"Best estimator from grid search: {self.clas}")
        print(f"Best score:{self.grid_search.best_score_}")
        return self.clas

    def __adjust_class_weights(self, classifier, hyperparameters):
        """Adjusts class weights for the classifier if applicable.

        Args:
            classifier (BaseEstimator): The classifier.
            hyperparameters (dict): The classifier hyperparameters.

        Returns:
            dict: Adjusted hyperparameters.
        """
        # For classifiers that don't support 'class_weight' directly, handle separately
        if classifier == XGBClassifier and "class_weight" in hyperparameters:
            scale_pos_weight = self.__calculate_scale_pos_weight(self.y_train)
            adjusted_hyperparameters = {**hyperparameters, 'scale_pos_weight': scale_pos_weight}
        else:
            adjusted_hyperparameters = hyperparameters

        # Removing 'class_weight' if not applicable
        adjusted_hyperparameters.pop("class_weight", None)

        return adjusted_hyperparameters

    def __calculate_scale_pos_weight(self, y:np.array):
        from collections import Counter
        counter = Counter(y)
        if len(counter) == 2:
            return counter[0] / counter[1]
        else:
            return 1
        
    def predict_proba(self, X_test: np.array):
        #if hasattr(self.clas, 'predict_proba'):
        return self.clas.predict_proba(X_test)
        #else:
            #print("The classifier does not support probability estimation.")
            #return None

class MLPipeline:
    def __init__(self, X_train, y_train, classifier, classifier_hyperparameters):
        self.X_train = X_train
        self.y_train = y_train
        self.classifier = classifier
        self.classifier_hyperparameters = classifier_hyperparameters
        self.selected_features = None

        # Initialize components of the pipeline
        self.preprocessor_pipe = DataPreprocessor(self.X_train)
        self.feature_selector = FeatureSelector(self.X_train, self.y_train)
        self.model_trainer = None  # To be initialized after feature selection

    def execute_feature_selection(self, method = "featurewiz", **kwargs):
        self.feature_selector.X_train,self.feature_selector.y_train = self.X_train, self.y_train
        self.selected_features = self.feature_selector.select_features(method, **kwargs)
        self.X_train = self.X_train.loc[:, self.selected_features]

    def execute_preprocessing(self):
        self.preprocessor_pipe.X_train = self.X_train
        self.preprocessor_pipe.auto_select_transformers()
        self.X_train = self.preprocessor_pipe.preprocessor.fit_transform(self.X_train)

    def train_model(self, perform_grid_search=False, param_grid=None, cv=2):
        self.model_trainer = ModelTrainer(self.X_train, self.y_train, self.classifier, self.classifier_hyperparameters)
        auc_scorer = make_scorer(roc_auc_score, needs_threshold=True)
        if perform_grid_search:
            self.best_model = self.model_trainer.perform_grid_search(param_grid, cv=cv, scoring= auc_scorer)
        else:
            self.best_model = self.model_trainer.fit()
        if self.best_model is None:
            raise ValueError("Failed to train the model.")
        else:
            print(f"Model trained successfully: {self.best_model}")

    def predict(self, X_test):
        if self.best_model:
            return self.best_model.predict_proba(X_test)
        else:
            raise NotFittedError("Model not trained yet. Call train_model() before prediction.")

class ModelEvaluator:
    def __init__(self, pipeline, X_test):
        self.pipeline = pipeline
        self.X_test = X_test

    def apply_transformations(self, X):
        
        # put a try/except to check whether a feature selector is applied
        selected_features = self.pipeline.selected_features
        X_transformed = X.loc[:, selected_features]
        # Apply preprocessing
        X_transformed = self.pipeline.preprocessor_pipe.preprocessor.transform(X_transformed)
        return X_transformed

    def evaluate(self):
        # Apply transformations to validation and test sets
        self.X_test_transformed = self.apply_transformations(self.X_test)

        # Evaluate on test set
        y_test_pred = self.pipeline.predict(self.X_test_transformed)
        # Add evaluation metrics as needed for test set

        # Return evaluation results
        return {"x_test": self.X_test_transformed, "y_test":y_test_pred}
