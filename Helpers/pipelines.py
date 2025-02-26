from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, is_classifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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
        self.feat_sel = None

    def featurewiz_selection(self, X_train, y_train, corr_limit=0.9):
        self.feat_sel = featurewiz.FeatureWiz(corr_limit=corr_limit, verbose=0)
        self.feat_sel.fit(X_train, y_train)
        self.selected_features = self.feat_sel.transform(X_train).columns

    def rfe_selection(self, X_train, y_train, n_features_to_select=5):
        model = LogisticRegression()
        self.feat_sel = RFE(model, n_features_to_select=n_features_to_select)
        self.feat_sel.fit(X_train, y_train)
        self.selected_features = X_train.columns[self.feat_sel.support_]

    def lasso_selection(self, X_train, y_train, alpha=0.001):
        self.feat_sel = Lasso(alpha=alpha)
        self.feat_sel.fit(X_train, y_train)
        self.selected_features = X_train.columns[self.feat_sel.coef_ != 0]

    def random_forest_selection(self, X_train, y_train, threshold=0.01):
        self.feat_sel = RandomForestClassifier()
        self.feat_sel.fit(X_train, y_train)
        importances = self.feat_sel.feature_importances_
        self.selected_features = X_train.columns[importances > threshold]

    def xgboost_selection(self, X_train, y_train, threshold=0.01):
        self.feat_sel = xgb.XGBClassifier()
        self.feat_sel.fit(X_train, y_train)
        importances = self.feat_sel.feature_importances_
        self.selected_features = X_train.columns[importances > threshold]
    
    
    def get_features(self):
        return self.selected_features, self.feat_sel


class DataPreprocessor:
    def __init__(self, X_train):
        self.X_train = X_train
        self.preprocessor = None

    def auto_select_transformers(self):
        numerical_features = self.X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.X_train.select_dtypes(include=['object']).columns
        
        num_transformer = StandardScaler()
        
        if len(categorical_features) > 0:
            cat_transformer = OneHotEncoder()
            transformers = [
                ('num', num_transformer, numerical_features),
                ('cat', cat_transformer, categorical_features)
            ]
        else:
            transformers = [
                ('num', num_transformer, numerical_features)
            ]

        self.preprocessor = ColumnTransformer(transformers=transformers)
        return self.preprocessor

class FeatureSelector:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def select_features(self, method='featurewiz', **kwargs):
        fs = FeatureSelection()
        getattr(fs, f"{method}_selection")(self.X_train, self.y_train, **kwargs)
        self.features,  self.feat_sel = fs.get_features()
        self.X_train = self.feat_sel.transform(self.X_train)
        return self.features, self.feat_sel

class ModelTrainer:
    def __init__(self, X_train, y_train, classifier, classifier_hyperparameters):
        self.X_train = X_train
        self.y_train = y_train
        self.classifier = classifier
        self.classifier_hyperparameters = classifier_hyperparameters
        self.clas = None
        self.best_params = None

    def fit(self):
        if not is_classifier(self.classifier):
            raise ValueError("The provided classifier is not a valid scikit-learn classifier.")

        # Adjusting class weights if applicable
        #adjusted_hyperparameters = self.__adjust_class_weights(self.classifier, self.classifier_hyperparameters)

        # Create and fit the pipeline
        self.clas = self.classifier(**self.classifier_hyperparameters).fit(self.X_train, self.y_train)
        return self.clas
    
    def perform_grid_search(self, param_grid, cv=5, scoring='recall', hp_type = None):
        if hp_type=="Exhaustive":
            self.grid_search = GridSearchCV(self.classifier(), param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        else:
            self.grid_search = RandomizedSearchCV(self.classifier(), param_grid, n_iter=40, cv=cv, scoring=scoring, verbose=1, random_state=42, n_jobs=-1, error_score=0)
        self.grid_search.fit(self.X_train, self.y_train)
        if self.grid_search.best_estimator_ is None:
            raise ValueError("Grid search did not yield a best estimator.")
        
        self.clas = self.grid_search.best_estimator_
        self.best_params = self.grid_search.best_params_

        # For debugging:
        print(f"Best estimator from grid search: {self.clas}")
        print(f"Best score:{self.grid_search.best_score_}")
        return self.clas, self.best_params

    def __adjust_class_weights(self, classifier, hyperparameters):
        """Adjusts class weights for the classifier if applicable.

        Args:
            classifier (BaseEstimator): The classifier.
            hyperparameters (dict): The classifier hyperparameters.

        Returns:
            dict: Adjusted hyperparameters.
        """
        # For classifiers that don't support 'class_weight' directly, handle separately
        try:
            if classifier == XGBClassifier and "class_weight" in hyperparameters:
                scale_pos_weight = self.__calculate_scale_pos_weight(self.y_train)
                adjusted_hyperparameters = {**hyperparameters, 'scale_pos_weight': scale_pos_weight}
            else:
                adjusted_hyperparameters = hyperparameters
            adjusted_hyperparameters.pop("class_weight", None)
        except:
                adjusted_hyperparameters = {}

        # Removing 'class_weight' if not applicable
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
        self.best_params = None

        # Initialize components of the pipeline
        self.preprocessor_pipe = DataPreprocessor(self.X_train)
        self.feature_selector = FeatureSelector(self.X_train, self.y_train)
        self.model_trainer = None  # To be initialized after feature selection

    def execute_feature_selection(self, method = "featurewiz", **kwargs):
        self.feature_selector.X_train,self.feature_selector.y_train = self.X_train, self.y_train
        self.selected_features, self.feat_sel = self.feature_selector.select_features(method, **kwargs)
        self.X_train = self.feat_sel.transform(self.X_train)

    def execute_preprocessing(self):
        self.preprocessor_pipe.X_train = self.X_train
        self.preprocessor_pipe.auto_select_transformers()
        self.X_train = self.preprocessor_pipe.preprocessor.fit_transform(self.X_train)

    def train_model(self, perform_grid_search=False, param_grid=None, cv=3, hp_type = None):
        self.model_trainer = ModelTrainer(self.X_train, self.y_train, self.classifier, self.classifier_hyperparameters)
        auc_scorer = make_scorer(roc_auc_score, needs_threshold=True)
        if perform_grid_search:
            self.best_model, self.best_params = self.model_trainer.perform_grid_search(param_grid, cv=cv, scoring= auc_scorer, hp_type=hp_type)
        else:
            self.best_model = self.model_trainer.fit()
            self.best_params = self.best_model.get_params()
        if self.best_model is None:
            raise ValueError("Failed to train the model.")
        else:
            print(f"Model trained successfully: {self.best_model}")

    def build_pipeline(self):
        from sklearn.pipeline import Pipeline
        ppln = Pipeline([
            ('FeatureWizFs',self.feat_sel),
            ('preprocessor', self.preprocessor_pipe.preprocessor),
            ('model', self.best_model)
        ])
        return ppln
    
    def get_best_parameters(self):
        return self.best_params 
    
    def predict(self, X_test):
        if self.best_model:
            return self.best_model.predict_proba(X_test)
        else:
            raise NotFittedError("Model not trained yet. Call train_model() before prediction.")

        self.X_test = X_test

    def evaluate(self):
        
        y_test_pred = self.pipeline.predict_proba(self.X_test)

        return {"y_test":y_test_pred}
