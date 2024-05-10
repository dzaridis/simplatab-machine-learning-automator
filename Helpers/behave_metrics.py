import numpy as np
from Helpers.pipelines import * 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class Metrics:
    def __init__(self, y_pred: np.array, y_test: np.array):
        self.y_pred = y_pred
        self.y_test = y_test
        self.mts = {}
        self.sens = None
        self.spec = None
        self.roc = None
        self.acc = None
        self.fsc = None

    def proba_convert(self, threshold: float):
        """Convert probabilities into binary outcomes based on threshold."""
        self.y_pred = np.where(self.y_pred[:, 1] > threshold, 1, 0)

    def perf_measure(self):
        """Compute confusion matrix components."""
        self.mts["TP"] = np.sum((self.y_pred == 1) & (self.y_test == 1))
        self.mts["FP"] = np.sum((self.y_pred == 1) & (self.y_test == 0))
        self.mts["TN"] = np.sum((self.y_pred == 0) & (self.y_test == 0))
        self.mts["FN"] = np.sum((self.y_pred == 0) & (self.y_test == 1))

    def sensitivity(self):
        """Calculate recall (sensitivity), which is the same as sensitivity."""
        self.sens = self.mts["TP"] / (self.mts["TP"] + self.mts["FN"]) if (self.mts["TP"] + self.mts["FN"]) > 0 else 0

    def specificity(self):
        """Calculate specificity, the proportion of actual negatives that are correctly identified."""
        self.spec = self.mts["TN"] / (self.mts["TN"] + self.mts["FP"]) if (self.mts["TN"] + self.mts["FP"]) > 0 else 0 

    def roc_auc(self):
        """Calculate ROC AUC score, ensuring y_pred is binary and y_test has the same shape."""
        if self.y_test.shape == self.y_pred.shape:
            self.roc = roc_auc_score(self.y_test, self.y_pred)

    def f_score(self):
        """Calculate the F1 score."""
        precision = self.mts["TP"] / (self.mts["TP"] + self.mts["FP"]) if (self.mts["TP"] + self.mts["FP"]) > 0 else 0
        recall = self.sensitivity()
        self.fsc = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def accuracy(self):
        """Calculate accuracy."""
        total = self.mts["TP"] + self.mts["TN"] + self.mts["FP"] + self.mts["FN"]
        self.acc = (self.mts["TP"] + self.mts["TN"]) / total if total > 0 else 0

    def get_scores(self):
        """Calculate balanced accuracy (mean of FAR and FRR)."""
        return {"Sensitivity": self.sens, 
                "Specificity": self.spec, 
                "AUC": self.roc,
                "F-score": self.fsc, 
                "Accuracy": self.acc}
    
def compute_sensitivity_analysis(pipeline, X_val:np.array, y_val:np.array):
    """Computes the FAR FRR and Balanced for thresholds

    Args:
        pipeline (sklearn pipeline): a fitted classifier
        X_val (np.array): data to predict on
        y_val (np.array): labels to assess on

    Returns:
        thresholds_far (dict): keys are the thresholds, values are the far for that specific theshold
        thresholds_frr (dict): keys are the thresholds, values are the frr for that specific theshold
        thresholds_bal (dict): keys are the thresholds, values are the balanced for that specific theshold
    """
    thresholds_sens, thresholds_spec, thresholds_fscore = {}, {}, {}
    for trh in np.arange(0,1.01,0.1):
        trh = round(trh, 2)
        mtr = Metrics(pipeline.predict_proba(X_val), y_val.to_numpy())
        mtr.proba_convert(threshold=trh)
        mtr.perf_measure()
        mtr.sensitivity()
        mtr.specificity()
        mtr.f_score()
        scores = mtr.get_scores()
        thresholds_sens.update({trh:scores["Sensitivity"]})
        thresholds_spec.update({trh:scores["Specificity"]})
        thresholds_fscore.update({trh:scores["F-score"]})
    return thresholds_sens, thresholds_spec, thresholds_fscore

class ThresholdOptimizer:
    def __init__(self, pipeline, X_val, y_val, X_test, y_test):
        self.pipeline = pipeline
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.optimal_threshold = None

    def find_optimal_threshold(self):
        # Calculate probabilities on validation data
        probas_val = self.pipeline.predict(self.X_val)

        # Initialize the best score to a high value (as we are minimizing the score)
        best_score = float(0)
        best_threshold = None

        for threshold in np.arange(0, 1.01, 0.05):
            # Convert probabilities to binary predictions based on the current threshold
            mtr = Metrics(probas_val, self.y_val)
            mtr.proba_convert(threshold=threshold)
            mtr.perf_measure()
            mtr.f_score()
            scores = mtr.get_scores()

            # Calculate the balanced score for the current threshold
            score = scores["F-score"]

            if score > best_score:
                best_score = score
                best_threshold = threshold

        self.optimal_threshold = best_threshold
        return best_threshold

    def apply_optimal_threshold(self, custom_threshold = False, custom_threshold_value = 0.1):
        if not custom_threshold:
            if self.optimal_threshold is None:
                raise ValueError("Optimal threshold not identified. Please run find_optimal_threshold() first.")

            probas_test = self.pipeline.predict(self.X_test)
            optimized_preds_test = (probas_test[:, 1] >= self.optimal_threshold).astype(int)
        else:
            probas_test = self.pipeline.predict(self.X_test)
            optimized_preds_test = (probas_test[:, 1] >= custom_threshold_value).astype(int)
        return optimized_preds_test

class ROCCurveEvaluator:
    def __init__(self, pipelines:dict, X_test, y_true):
        self.pipelines = pipelines
        self.X_test = X_test
        self.y_true = y_true

    def evaluate_models(self):
        # Dictionary to store ROC data for each model
        roc_data = {}
        for nm, pipeline in self.pipelines.items():
            # Get transformed test data using current pipeline
            me = ModelEvaluator(pipeline,self.X_test)
            y_scores = me.evaluate()["y_test"][:, 1]

            # Compute ROC curve and ROC area for each class
            fpr, tpr, _ = roc_curve(self.y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            # Store ROC data
            roc_data[nm] = (fpr, tpr, roc_auc)

        return roc_data

    def plot_roc_curves(self):
        roc_data = self.evaluate_models()
        plt.figure(figsize=(10, 8))

        for model_name, (fpr, tpr, roc_auc) in roc_data.items():
            plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()