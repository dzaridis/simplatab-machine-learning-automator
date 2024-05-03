from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, is_classifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import featurewiz
from sklearn.metrics import make_scorer

class FeatureSelection:
    def __init__(self) -> None:
        self.selected_features = None

    def featurewiz_selection(self, X_train, y_train, corr_limit=0.9):
        self.selected_features, trainm  = featurewiz.featurewiz(X_train.join(y_train), target = "Target", corr_limit=corr_limit, 
                            verbose=0, 
                            feature_engg='', 
                            dask_xgboost_flag=False, 
                            nrows=None, 
                            skip_sulov=False, 
                            skip_xgboost=False)

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

    def select_features(self, corr_limit=0.9):
        fw = FeatureSelection()
        fw.featurewiz_selection(self.X_train, self.y_train, corr_limit=corr_limit)
        selected_features = fw.get_features()
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

    def execute_feature_selection(self, corr_limit=0.9):
        self.feature_selector.X_train,self.feature_selector.y_train = self.X_train, self.y_train
        self.selected_features = self.feature_selector.select_features(corr_limit)
        self.X_train = self.X_train.loc[:, self.selected_features]

    def execute_preprocessing(self):
        self.preprocessor_pipe.X_train = self.X_train
        self.preprocessor_pipe.auto_select_transformers()
        self.X_train = self.preprocessor_pipe.preprocessor.fit_transform(self.X_train)

    def train_model(self, perform_grid_search=False, param_grid=None, cv=2, scoring = balanced_FAR_FRR):
        self.model_trainer = ModelTrainer(self.X_train, self.y_train, self.classifier, self.classifier_hyperparameters)
        if perform_grid_search:
            balanced_scorer = make_scorer(scoring, greater_is_better=False)
            self.best_model = self.model_trainer.perform_grid_search(param_grid, cv=cv, scoring= balanced_scorer)
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
    def __init__(self, pipeline, X_val, X_test):
        self.pipeline = pipeline
        self.X_val = X_val
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
        self.X_val_transformed = self.apply_transformations(self.X_val)
        self.X_test_transformed = self.apply_transformations(self.X_test)

        # Evaluate on validation set
        y_val_pred = self.pipeline.predict(self.X_val_transformed)
        # Add evaluation metrics as needed, e.g., accuracy, ROC AUC, etc. for validation set

        # Evaluate on test set
        y_test_pred = self.pipeline.predict(self.X_test_transformed)
        # Add evaluation metrics as needed for test set

        # Return evaluation results
        return {"validation_results": y_val_pred, "test_results": y_test_pred}
