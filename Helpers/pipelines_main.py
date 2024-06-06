import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import warnings
#from Helpers.radiomics_setup import RadiomicsClean, MultiParamProcessor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import os
from Helpers import pipelines
from Helpers import behave_metrics
from Helpers import shap_module
from Helpers import MetricsReport
import pickle
import yaml

import logging
import traceback
# Define the hyperparameters for each model
hyperparameters_models_grid = {
    "Logistic Regression": {
        "classifier": LogisticRegression,
        "params": {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
            'max_iter': [300, 500, 1000, 2000],
            'l1_ratio': [0, 0.1, 0.2, 0.5, 0.7, 0.9, 1]
        },
        "default_params": {}
    },
    "Support Vector Machines": {
        "classifier": SVC,
        "params": {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4, 5],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10, 100],
            'coef0': [0, 0.1, 0.5, 1, 2],
            'shrinking': [True, False],
            'tol': [1e-4, 1e-3, 1e-2, 1e-1],
            'max_iter': [1000, 2000, 3000],
            'probability': [True]
        },
        "default_params": {}
    },
    "Random Forest": {
        "classifier": RandomForestClassifier,
        "params": {
            'n_estimators': [20, 50, 100, 200, 300, 400, 500],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [None, 2, 4, 6, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 10],
            'max_features': [None, 'auto', 'sqrt', 'log2'],
            'bootstrap': [True, False]
        },
        "default_params": {}
    },
    "Stochastic Gradient Descent": {
        "classifier": SGDClassifier,
        "params": {
            'loss': ['log', 'modified_huber'],
            'penalty': ['none', 'l2', 'l1', 'elasticnet'],
            'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
            'l1_ratio': [0.0, 0.15, 0.5, 0.85, 1.0],
            'fit_intercept': [True, False],
            'max_iter': [1000, 2000, 3000],
            'tol': [1e-3, 1e-4, 1e-5],
            'shuffle': [True, False],
            'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
            'eta0': [1e-4, 1e-3, 1e-2, 1e-1],
            'power_t': [0.25, 0.5, 0.75],
            'early_stopping': [True, False],
            'validation_fraction': [0.1, 0.2, 0.3],
            'n_iter_no_change': [5, 10, 20],
            'average': [False, True]
        },
        "default_params": {}
    },
    "Multi-Layer Neural Network": {
        "classifier": MLPClassifier,
        "params": {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'max_iter': [200, 300, 500, 1000, 2000, 3000],
            'shuffle': [True, False],
            'tol': [1e-4, 1e-3, 1e-2],
            'momentum': [0.9, 0.95, 0.99],
            'nesterovs_momentum': [True, False],
            'early_stopping': [True, False],
            'beta_1': [0.9, 0.95, 0.99],
            'beta_2': [0.999, 0.9995, 0.9999],
            'epsilon': [1e-8, 1e-7, 1e-6]
        },
        "default_params": {}
    },
    "Decision Trees": {
        "classifier": DecisionTreeClassifier,
        "params": {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 2, 3, 5, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 10],
            'max_features': [None, 'auto', 'sqrt', 'log2'],
            'max_leaf_nodes': [None, 2, 4, 10, 20, 30, 40, 50]
        },
        "default_params": {}
    },
    "XGBoost": {
        "classifier": XGBClassifier,
        "params": {
            'n_estimators': [3, 5, 10, 50, 100, 200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7, 8, 10],
            'learning_rate': [0.001, 0.01, 0.1, 0.3, 0.5],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'reg_alpha': [0, 0.01, 0.1, 1, 10, 100],
            'reg_lambda': [0.01, 0.1, 1, 10, 100]
        },
        "default_params": {}
    }
}
def read_yaml():
    global CORRELATION_LIMIT, NUMBER_OF_FOLDS, HP_TYPE, METRIC_TO_TRACK, GRID_SEARCH_ENABLING, names, classifiers, hypers
    yaml_file = os.path.join("./input_data", "machine_learning_parameters.yaml")
    with open(yaml_file, 'r') as file:
        params = yaml.safe_load(file)

    CORRELATION_LIMIT = params["Correlation Limit"]

    NUMBER_OF_FOLDS = params["number_of_k_folds"]

    if params["apply_grid_search"]["type"]["Randomized"]:
        HP_TYPE = "Randomized"
    else:
        HP_TYPE = "Exhaustive"

    METRIC_TO_TRACK = params["Metric For Threshold Optimization"]
    GRID_SEARCH_ENABLING = params["apply_grid_search"]["enabled"]
    names = []
    classifiers = []
    hypers = []
    for model_name, is_enabled in params["Machine Learning Models"].items():
        if is_enabled:
            model_info = hyperparameters_models_grid.get(model_name)
            if GRID_SEARCH_ENABLING:
                names.append(model_name)
                classifiers.append(model_info["classifier"])
                hypers.append(model_info["params"])
            else:
                names.append(model_name)
                classifiers.append(model_info["classifier"])
                hypers.append(model_info["default_params"])

def train_k_fold(X_train, y_train):
    log_file = './Materials/error_log.log'
    logging.basicConfig(filename=log_file, level=logging.ERROR, 
                    format='%(asctime)s:%(levelname)s:%(message)s')
    k_folds = NUMBER_OF_FOLDS
    scores_storage = {}
    params_dict = {}
    thresholds = {}
    base_models_folds = {}
    # to add the automated k-fold selector based on the number of y
    skf = StratifiedKFold(n_splits=k_folds, shuffle = True, random_state=10)
    for cls, hp, nm in zip(classifiers, hypers, names):
        print("-------------------- \n", f"{nm} is starting \n", "--------------------")
        logging.info(f"{nm} is starting")
        # find optimal parameters
        pipeline = pipelines.MLPipeline(X_train, y_train, cls, hp)
        pipeline.execute_feature_selection(corr_limit=CORRELATION_LIMIT)
        pipeline.execute_preprocessing()
        pipeline.train_model(perform_grid_search=GRID_SEARCH_ENABLING, param_grid=hp, cv=skf, hp_type=HP_TYPE)
        ppln = pipeline.build_pipeline()
        params = pipeline.get_best_parameters()
        params_dict.update({nm:params})
        logging.info(f"{nm} training completed with parameters: {params}")
        hpers = params # best parameters based on cv grid search

        base_models = {}
        scores_storage_algo = {}
        thresholds_algo = {}
        for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
            xtrain = X_train.iloc[train_index,:]
            ytrain = y_train.iloc[train_index]
            xval = X_train.iloc[test_index,:]
            yval = y_train.iloc[test_index]
            try:
                pipeline = pipelines.MLPipeline(xtrain, ytrain, cls, hpers)
                pipeline.execute_feature_selection(corr_limit=CORRELATION_LIMIT)
                pipeline.execute_preprocessing()
                pipeline.train_model()
                ppln = pipeline.build_pipeline()
                base_models.update({f"fold_{i+1}":ppln})
            except Exception as e:
                error_message = f"An error occurred: {e}"
                logging.error(error_message)
                logging.error(traceback.format_exc())
            
            me = behave_metrics.ModelEvaluator(ppln,xval)
            scores = me.evaluate()["y_test"]

            # Find optimal threshold
            tho = behave_metrics.ThresholdOptimizer(ppln, xval, yval)
            thresh = tho.find_optimal_threshold(metric_to_track=METRIC_TO_TRACK)
            thresholds_algo.update({f"fold_{i+1}":thresh})

            # Compute metrics on that threshold
            mr = behave_metrics.Metrics(scores, yval)
            mr.compute_metrics(threshold=thresh)
            scores_dict = mr.get_scores() # the scores on the fold based on the best hyperparameters
            scores_storage_algo.update({f"fold_{i+1}":scores_dict})
        scores_storage.update({nm:scores_storage_algo})
        thresholds.update({nm:thresholds_algo})
        base_models_folds.update({nm:base_models})
        print("-------------------- \n", f"{nm} is completed successfully \n", "--------------------")
    MetricsReport.summary_results_excel(scores_storage, file = f"{NUMBER_OF_FOLDS}_fold_results", conf_matrix_name=f"Internal_{NUMBER_OF_FOLDS}_fold")
    return params_dict, scores_storage, thresholds, base_models_folds

# def train_k_fold_stacking(X_train, y_train, base_models_folds):
#     k_folds = NUMBER_OF_FOLDS
#     thresholds_algo = {}
#     skf = StratifiedKFold(n_splits=k_folds, shuffle = True, random_state=10)
#     for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
#         xtrain = X_train.iloc[train_index,:]
#         ytrain = y_train.iloc[train_index]
#         xval = X_train.iloc[test_index,:]
#         yval = y_train.iloc[test_index]

#         base_models_fitted = []
#         for nm, base_models in base_models_folds.items():
#             base_models_fitted.append((nm,base_models[f"fold_{i+1}"]))
#         # Train the stacking model
#         meta_learner = LogisticRegression(random_state=42)
        
#         # Create the stacking classifier
#         stacking_clf = StackingClassifier(
#             estimators=base_models_fitted,
#             final_estimator=meta_learner,
#             passthrough=True,  # Include original features along with predictions for meta-learner
#             cv='prefit'  # 5-fold cross-validation
#             )
        
#         stacking_clf.fit(xtrain, ytrain)
        

#         tho = behave_metrics.ThresholdOptimizer(stacking_clf, xval, yval)
#         thresh = tho.find_optimal_threshold(metric_to_track=METRIC_TO_TRACK)
#         thresholds_algo.update({f"fold_{i+1}":thresh})
            

def external_test(X_train, y_train, X_test, y_test, params_dict, thresholds):
    pipeline_dict_inf = {}
    params_inf= {}
    scores_inf = {}
    for cls, hp, nm in zip(classifiers, hypers, names):
        print("-------------------- \n", f"{nm} is starting \n", "--------------------")
        hpers = params_dict[nm]
        pipeline = pipelines.MLPipeline(X_train, y_train, cls, hpers)
        pipeline.execute_feature_selection(corr_limit=CORRELATION_LIMIT)
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

        shap_module.ShapAnalysis(ppln = ppln,
                                X_test = X_test, 
                                y_test=y_test, 
                                nm=nm)

        print("-------------------- \n", f"{nm} is completed successfully \n", "--------------------")
    MetricsReport.external_summary(scores_inf, file = "test_results", conf_matrix_name="Test")
    # display roc curves
    roc = behave_metrics.ROCCurveEvaluator(pipeline_dict_inf,X_test=X_test, y_true=y_test)
    roc.evaluate_models()
    roc.plot_roc_curves(save_path=r"./Materials")
    try:
        roc.plot_pr_curves(save_path=r"./Materials")
    except:
        pass

    save_path_for_models = os.path.join("./Materials", "Models")
    os.makedirs(save_path_for_models, exist_ok=True)

    try:
        for name, pipeline in pipeline_dict_inf.items():
            filename = os.path.join(save_path_for_models, f"{name}_pipeline.pkl")
            with open(filename, "wb") as file:
                pickle.dump(pipeline, file)
            print(f"Saved {name} pipeline to {filename}")
    except Exception as e:
        error_message = f"Error here: {e}\n"
        with open(os.path.join("Materials", "Model_save_error_log.txt"), "a") as file:
            file.write(error_message)