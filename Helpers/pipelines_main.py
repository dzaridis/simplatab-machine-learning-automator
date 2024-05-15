import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import warnings
#from Helpers.radiomics_setup import RadiomicsClean, MultiParamProcessor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import os
from Helpers import pipelines
from Helpers import behave_metrics
from Helpers import shap_module
from Helpers import MetricsReport
lr_hyperparameters = {'C': 1.0, 'penalty': 'l2'}
svm_hyperparameters = {'C': 1.0, 'kernel': 'rbf', 'probability':True}
xgb_hyperparameters = {'n_estimators': 100}
ada_hyperparameters = {'n_estimators': 50}
rf_hyperparameters = {'n_estimators': 100}
dt_hyperparameters = {'criterion': 'gini'}

names = ["Logistic Regression", "Support Vector Machines", "Random Forest", "AdaBoost", "Decision Trees", "XGBoost"]
classifiers = [LogisticRegression, SVC, RandomForestClassifier, AdaBoostClassifier, DecisionTreeClassifier, XGBClassifier]
hypers = [lr_hyperparameters , svm_hyperparameters, rf_hyperparameters, ada_hyperparameters, dt_hyperparameters, xgb_hyperparameters]
lr_hyperparameters = {
'C': [0.001, 0.01, 0.1, 1, 10, 100],
'penalty': [None, 'l2'],
'solver': ['lbfgs','liblinear', 'saga','newton-cholesky'],
'max_iter': [100, 200, 300],
'class_weight': [None, 'balanced'],
}

svm_hyperparameters = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto'],
    'probability':[True],
}

rf_hyperparameters = {
    'n_estimators': [2,4,6,10,20,50],
    'max_depth': [None, 1,2,3, 4,6,8, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

ada_hyperparameters = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1],
}

dt_hyperparameters = {
    'max_depth': [None, 2,3, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

xgb_hyperparameters = {
    'n_estimators': [3,5,10, 20],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [None, 1,2, 3, 6, 10, 20],
    'subsample': [0.5, 0.7, 1],
    'colsample_bytree': [0.5, 0.7, 1],
}
hypers = [lr_hyperparameters , svm_hyperparameters, rf_hyperparameters, ada_hyperparameters, dt_hyperparameters, xgb_hyperparameters]

def train_k_fold(X_train, y_train, k_folds=3):

    pipeline_dict ={}
    scores_storage = {}
    params_dict = {}
    thresholds = {}
    # to add the automated k-fold selector based on the number of y
    skf = StratifiedKFold(n_splits=k_folds, shuffle = True, random_state=10)
    for cls, hp, nm in zip(classifiers, hypers, names):
        # find optimal parameters
        pipeline = pipelines.MLPipeline(X_train, y_train, cls, hp)
        pipeline.execute_feature_selection(corr_limit=0.6)
        pipeline.execute_preprocessing()
        pipeline.train_model(perform_grid_search=True, param_grid=hp, cv=skf)
        ppln = pipeline.build_pipeline()
        params = pipeline.get_best_parameters()
        params_dict.update({nm:params})

        hpers = params # best parameters based on cv grid search

        scores_storage_algo = {}
        thresholds_algo = {}
        for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
            xtrain = X_train.iloc[train_index,:]
            ytrain = y_train.iloc[train_index]
            xval = X_train.iloc[test_index,:]
            yval = y_train.iloc[test_index]

            pipeline = pipelines.MLPipeline(xtrain, ytrain, cls, hpers)
            pipeline.execute_feature_selection(corr_limit=0.6)
            pipeline.execute_preprocessing()
            pipeline.train_model()
            ppln = pipeline.build_pipeline()
            
            me = behave_metrics.ModelEvaluator(ppln,xval)
            scores = me.evaluate()["y_test"]

            # Find optimal threshold
            tho = behave_metrics.ThresholdOptimizer(ppln, xval, yval)
            thresh = tho.find_optimal_threshold(metric_to_track="AUC")
            thresholds_algo.update({f"fold_{i+1}":thresh})

            # Compute metrics on that threshold
            mr = behave_metrics.Metrics(scores, yval)
            mr.compute_metrics(threshold=thresh)
            scores_dict = mr.get_scores() # the scores on the fold based on the best hyperparameters
            scores_storage_algo.update({f"fold_{i+1}":scores_dict})
        scores_storage.update({nm:scores_storage_algo})
        thresholds.update({nm:thresholds_algo})
    MetricsReport.summary_results_excel(scores_storage, file = "k_fold_results", conf_matrix_name="Internal_K_fold")
    return params_dict, scores_storage, thresholds

def external_test(X_train, y_train, X_test, y_test, params_dict, thresholds):
    pipeline_dict_inf = {}
    params_inf= {}
    scores_inf = {}
    for cls, hp, nm in zip(classifiers, hypers, names):
        hpers = params_dict[nm]
        pipeline = pipelines.MLPipeline(X_train, y_train, cls, hpers)
        pipeline.execute_feature_selection(corr_limit=0.6)
        pipeline.execute_preprocessing()
        pipeline.train_model(perform_grid_search=False)
        ppln = pipeline.build_pipeline()
        pipeline_dict_inf.update({nm:ppln})
        params = pipeline.get_best_parameters()
        params_inf.update({nm:params})

        me = behave_metrics.ModelEvaluator(ppln,X_test)
        scores = me.evaluate()["y_test"]

        # Set optimal threshold as the average across Folds
        cnt = 0
        meas = 0
        for fold,val in thresholds[nm].items():
            cnt += 1
            meas += val
        average_threshold = meas/cnt if cnt!=0 else 0.5

        # Compute metrics on that threshold
        mr = behave_metrics.Metrics(scores, y_test)
        mr.compute_metrics(threshold=average_threshold)
        scores_dict = mr.get_scores() # the scores on the fold based on the best hyperparameters
        scores_inf.update({nm:scores_dict})

        # Perform Shap
        sp = shap_module.ShapAnalysis(X_val = X_test, pipeline_module = ppln, features=pipeline.selected_features) 
        shap_values,features = sp.perform_shap(), pipeline.selected_features
        try:
            os.mkdir(os.path.join("Materials", "Shap_Features"))
        except OSError:
            pass

        path_shap = os.path.join("Materials", "Shap_Features")
        sp.plot_shap_values(model_name=nm, path= path_shap)

    MetricsReport.external_summary(scores_inf, file = "test_results", conf_matrix_name="Test")
    # display roc curves
    roc = behave_metrics.ROCCurveEvaluator(pipeline_dict_inf,X_test=X_test, y_true=y_test)
    roc.evaluate_models()
    roc.plot_roc_curves(save_path=r"./Materials")