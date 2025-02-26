import numpy as np
from Helpers.pipelines import * 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc,  precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Metrics:
    def __init__(self, y_pred: np.array, y_test: np.array):
        self.y_pred = y_pred
        self.y_scores = y_pred
        self.y_test = y_test
        self.mts = {}
        self.sens = None
        self.spec = None
        self.roc = None
        self.acc = None
        self.fsc = None
        self.balanced_acc = None

    def _proba_convert(self, threshold: float):
        """Convert probabilities into binary outcomes based on threshold."""
        try:
            self.y_pred = np.where(self.y_pred[:, 1] > threshold, 1, 0)
        except IndexError:
            self.y_pred = np.where(self.y_pred > threshold, 1, 0)

    def _perf_measure(self):
        """Compute confusion matrix components."""
        self.mts["TP"] = np.sum((self.y_pred == 1) & (self.y_test == 1))
        self.mts["FP"] = np.sum((self.y_pred == 1) & (self.y_test == 0))
        self.mts["TN"] = np.sum((self.y_pred == 0) & (self.y_test == 0))
        self.mts["FN"] = np.sum((self.y_pred == 0) & (self.y_test == 1))

    def _sensitivity(self):
        """Calculate recall (sensitivity), which is the same as sensitivity."""
        self.sens = self.mts["TP"] / (self.mts["TP"] + self.mts["FN"]) if (self.mts["TP"] + self.mts["FN"]) > 0 else 0

    def _specificity(self):
        """Calculate specificity, the proportion of actual negatives that are correctly identified."""
        self.spec = self.mts["TN"] / (self.mts["TN"] + self.mts["FP"]) if (self.mts["TN"] + self.mts["FP"]) > 0 else 0 

    def _roc_auc(self):
        """Calculate ROC AUC score, ensuring y_pred is binary and y_test has the same shape."""
        y_scores = self.y_scores[:,1]
        
        fpr, tpr, _ = roc_curve(self.y_test, y_scores)
        self.roc = auc(fpr, tpr)

    def _f_score(self):
        """Calculate the F1 score."""
        precision = self.mts["TP"] / (self.mts["TP"] + self.mts["FP"]) if (self.mts["TP"] + self.mts["FP"]) > 0 else 0
        self._sensitivity()
        recall = self.sens
        self.fsc = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def _accuracy(self):
        """Calculate accuracy."""
        total = self.mts["TP"] + self.mts["TN"] + self.mts["FP"] + self.mts["FN"]
        self.acc = (self.mts["TP"] + self.mts["TN"]) / total if total > 0 else 0
    
    def _balanced_accuracy(self):
        sensitivity = self.mts["TP"] / (self.mts["TP"] + self.mts["FN"]) if (self.mts["TP"] + self.mts["FN"]) > 0 else 0
        specificity = self.mts["TN"] / (self.mts["TN"] + self.mts["FP"]) if (self.mts["TN"] + self.mts["FP"]) > 0 else 0
        self.balanced_acc = (sensitivity + specificity) / 2


    def compute_metrics(self, threshold:float):
        self._proba_convert(threshold=threshold)
        self._perf_measure()
        self._sensitivity()
        self._specificity()
        self._roc_auc()
        self._f_score()
        self._accuracy()
        self._balanced_accuracy()

    def get_scores(self):
        """Calculate balanced accuracy (mean of FAR and FRR)."""
        return {"Sensitivity": self.sens, 
                "Specificity": self.spec, 
                "AUC": self.roc,
                "F-score": self.fsc, 
                "Accuracy": self.acc,
                "Balanced Accuracy": self.balanced_acc,
                "Confusion Matrix": self.mts}
    
def compute_sensitivity_analysis(pipeline, X_val, y_val):
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
        thresholds_fscore.update({trh:scores["AUC"]})
    return thresholds_sens, thresholds_spec, thresholds_fscore

class ThresholdOptimizer:
    def __init__(self, pipeline, X_val, y_val):
        self.pipeline = pipeline
        self.X_val = X_val
        self.y_val = y_val
        self.optimal_threshold = None

    def find_optimal_threshold(self, metric_to_track:str):
        # Calculate probabilities on validation data
        probas_val = self.pipeline.predict_proba(self.X_val)
        # Initialize the best score to a high value (as we are minimizing the score)
        best_score = float(0)
        best_threshold = None

        for threshold in np.arange(0.02, 1.01, 0.05):
            # Convert probabilities to binary predictions based on the current threshold
            mr = Metrics(probas_val, self.y_val)
            mr.compute_metrics(threshold=threshold)
            scores_dict = mr.get_scores()

            # Calculate the balanced score for the current threshold
            score = scores_dict[metric_to_track]

            if score > best_score:
                best_score = score
                best_threshold = threshold

        self.optimal_threshold = best_threshold

        return self.optimal_threshold


class ROCCurveEvaluator:
    def __init__(self, pipeline_dict: dict, X_test, y_true):
        self.piplns = pipeline_dict
        self.X_test = X_test
        self.y_true = y_true

    def evaluate_models(self):
        # Dictionary to store ROC data for each model
        roc_data = {}
        pr_data = {}
        for nm, pipeline in self.piplns.items():
            # Get transformed test data using current pipeline
            me = ModelEvaluator(pipeline, self.X_test)
            y_scores = me.evaluate()["y_test"][:, 1]  # Ensure correct key is used from evaluate() output

            # Compute ROC curve and ROC area for each class
            fpr, tpr, _ = roc_curve(self.y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            # Store ROC data
            roc_data[nm] = (fpr, tpr, roc_auc)

            # Compute Precision-Recall curve and average precision for each model
            precision, recall, _ = precision_recall_curve(self.y_true, y_scores)
            avg_precision = average_precision_score(self.y_true, y_scores)
            pr_data[nm] = (precision, recall, avg_precision)

        return roc_data, pr_data

    def plot_roc_curves(self, save_path = None):
        roc_data, _ = self.evaluate_models()
        
        # Initialize the plot
        plt.figure(figsize=(10, 8))
        sns.set()  # Set Seaborn style

        for model_name, (fpr, tpr, roc_auc) in roc_data.items():
            plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})', linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', label='Chance (area = 0.50)', linewidth=2)  # Dashed diagonal
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        sns.despine()  # Remove the top and right spines
        if save_path:
            plt.savefig(os.path.join(save_path,"ROC CURVES.png"), format='png', dpi=600)  # Save the figure
        #plt.show()
    
    def plot_pr_curves(self, save_path=None):
        _, pr_data = self.evaluate_models()
        
        # Initialize the plot
        plt.figure(figsize=(10, 8))
        sns.set()  # Set Seaborn style

        for model_name, (precision, recall, avg_precision) in pr_data.items():
            plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.2f})', linewidth=2)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        sns.despine()  # Remove the top and right spines
        if save_path:
            plt.savefig(os.path.join(save_path, "PR_CURVES.png"), format='png', dpi=600)  # Save the figure
        plt.show()


class ModelEvaluator:
    def __init__(self, pipeline, X_test):
        self.pipeline = pipeline
        self.X_test = X_test

    def evaluate(self):
        
        y_test_pred = self.pipeline.predict_proba(self.X_test)

        return {"y_test":y_test_pred}