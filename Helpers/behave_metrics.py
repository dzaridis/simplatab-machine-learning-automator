import numpy as np
from Helpers.pipelines import * 
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import cycle


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
        self.num_classes = len(np.unique(y_test))
        self.is_multiclass = self.num_classes > 2

    def _proba_convert(self, threshold: float):
        """Convert probabilities into class outcomes based on threshold."""
        if self.is_multiclass:
            # For multiclass, get the highest probability class
            self.y_pred = np.argmax(self.y_scores, axis=1)
        else:
            try:
                self.y_pred = np.where(self.y_pred[:, 1] > threshold, 1, 0)
            except IndexError:
                self.y_pred = np.where(self.y_pred > threshold, 1, 0)

    def _perf_measure(self):
        """Compute confusion matrix components."""
        if self.is_multiclass:
            # Calculate full confusion matrix for multiclass
            cm = confusion_matrix(self.y_test, self.y_pred)
            self.mts["confusion_matrix"] = cm
            
            # Extract TP, FP, TN, FN for each class
            self.mts["TP"] = np.diag(cm)
            self.mts["FP"] = np.sum(cm, axis=0) - np.diag(cm)
            self.mts["FN"] = np.sum(cm, axis=1) - np.diag(cm)
            self.mts["TN"] = np.sum(cm) - (self.mts["TP"] + self.mts["FP"] + self.mts["FN"])
        else:
            # Binary case remains the same
            self.mts["TP"] = np.sum((self.y_pred == 1) & (self.y_test == 1))
            self.mts["FP"] = np.sum((self.y_pred == 1) & (self.y_test == 0))
            self.mts["TN"] = np.sum((self.y_pred == 0) & (self.y_test == 0))
            self.mts["FN"] = np.sum((self.y_pred == 0) & (self.y_test == 1))

    def _sensitivity(self):
        """Calculate recall (sensitivity) for each class."""
        if self.is_multiclass:
            # Calculate sensitivity for each class
            tp = self.mts["TP"]
            fn = self.mts["FN"]
            self.sens = np.zeros(self.num_classes)
            for i in range(self.num_classes):
                self.sens[i] = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0
            # Average sensitivity across classes
            self.sens_avg = np.mean(self.sens)
        else:
            self.sens = self.mts["TP"] / (self.mts["TP"] + self.mts["FN"]) if (self.mts["TP"] + self.mts["FN"]) > 0 else 0

    def _specificity(self):
        """Calculate specificity for each class."""
        if self.is_multiclass:
            # Calculate specificity for each class
            tn = self.mts["TN"]
            fp = self.mts["FP"]
            self.spec = np.zeros(self.num_classes)
            for i in range(self.num_classes):
                self.spec[i] = tn[i] / (tn[i] + fp[i]) if (tn[i] + fp[i]) > 0 else 0
            # Average specificity across classes
            self.spec_avg = np.mean(self.spec)
        else:
            self.spec = self.mts["TN"] / (self.mts["TN"] + self.mts["FP"]) if (self.mts["TN"] + self.mts["FP"]) > 0 else 0

    def _roc_auc(self):
        """Calculate ROC AUC score for multiclass using one-vs-rest approach."""
        if self.is_multiclass:
            # Binarize the labels for multiclass ROC
            y_bin = label_binarize(self.y_test, classes=range(self.num_classes))
            
            # Calculate ROC AUC for each class
            self.roc_per_class = np.zeros(self.num_classes)
            for i in range(self.num_classes):
                try:
                    self.roc_per_class[i] = roc_auc_score(y_bin[:, i], self.y_scores[:, i])
                except:
                    self.roc_per_class[i] = 0.5  # Default value if calculation fails
                    
            # Average ROC AUC across all classes
            self.roc = np.mean(self.roc_per_class)
        else:
            y_scores = self.y_scores[:,1]
            fpr, tpr, _ = roc_curve(self.y_test, y_scores)
            self.roc = auc(fpr, tpr)

    def _f_score(self):
        """Calculate the F1 score for multiclass."""
        if self.is_multiclass:
            # Calculate precision and recall for each class
            tp = self.mts["TP"]
            fp = self.mts["FP"]
            fn = self.mts["FN"]
            
            precision = np.zeros(self.num_classes)
            recall = np.zeros(self.num_classes)
            self.fsc_per_class = np.zeros(self.num_classes)
            
            for i in range(self.num_classes):
                precision[i] = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0
                recall[i] = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0
                self.fsc_per_class[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
            
            # Average F1 score across all classes (macro averaging)
            self.fsc = np.mean(self.fsc_per_class)
        else:
            precision = self.mts["TP"] / (self.mts["TP"] + self.mts["FP"]) if (self.mts["TP"] + self.mts["FP"]) > 0 else 0
            self._sensitivity()
            recall = self.sens
            self.fsc = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def _accuracy(self):
        """Calculate accuracy."""
        if self.is_multiclass:
            # For multiclass, accuracy is the sum of TP divided by total samples
            self.acc = np.sum(self.mts["TP"]) / np.sum(self.mts["confusion_matrix"])
        else:
            total = self.mts["TP"] + self.mts["TN"] + self.mts["FP"] + self.mts["FN"]
            self.acc = (self.mts["TP"] + self.mts["TN"]) / total if total > 0 else 0

    def _balanced_accuracy(self):
        """Calculate balanced accuracy."""
        if self.is_multiclass:
            # For multiclass, balanced accuracy is the average of per-class recall values
            self._sensitivity()
            self.balanced_acc = np.mean(self.sens)
        else:
            sensitivity = self.mts["TP"] / (self.mts["TP"] + self.mts["FN"]) if (self.mts["TP"] + self.mts["FN"]) > 0 else 0
            specificity = self.mts["TN"] / (self.mts["TN"] + self.mts["FP"]) if (self.mts["TN"] + self.mts["FP"]) > 0 else 0
            self.balanced_acc = (sensitivity + specificity) / 2

    def compute_metrics(self, threshold=0.5):
        self._proba_convert(threshold=threshold)
        self._perf_measure()
        self._sensitivity()
        self._specificity()
        self._roc_auc()
        self._f_score()
        self._accuracy()
        self._balanced_accuracy()

    def get_scores(self):
        """Return metrics including multiclass-specific metrics if applicable."""
        if self.is_multiclass:
            return {
                "Sensitivity": self.sens_avg,
                "Specificity": self.spec_avg,
                "AUC": self.roc,
                "F-score": self.fsc,
                "Accuracy": self.acc,
                "Balanced Accuracy": self.balanced_acc,
                "Confusion Matrix": self.mts["confusion_matrix"],
                "Class-wise Metrics": {
                    "Sensitivity": self.sens,
                    "Specificity": self.spec,
                    "AUC": self.roc_per_class,
                    "F-score": self.fsc_per_class
                },
                "Number of Classes": self.num_classes
            }
        else:
            return {
                "Sensitivity": self.sens,
                "Specificity": self.spec,
                "AUC": self.roc,
                "F-score": self.fsc,
                "Accuracy": self.acc,
                "Balanced Accuracy": self.balanced_acc,
                "Confusion Matrix": self.mts
            }


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
        self.num_classes = len(np.unique(y_val))
        self.is_multiclass = self.num_classes > 2
        self.optimal_threshold = None

    def find_optimal_threshold(self, metric_to_track:str):
        # Calculate probabilities on validation data
        probas_val = self.pipeline.predict_proba(self.X_val)
        
        if self.is_multiclass:
            # For multiclass, use the default decision function (argmax)
            # No thresholding is typically used, but we can return 0.5 as a placeholder
            self.optimal_threshold = 0.5
            
            # We could alternatively optimize for a specific metric 
            # using class-specific thresholds, but that would require a more complex approach
            return self.optimal_threshold
        else:
            # Binary classification case - use original implementation
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
        self.num_classes = len(np.unique(y_true))
        self.is_multiclass = self.num_classes > 2

    def evaluate_models(self):
        # Dictionary to store ROC data for each model
        roc_data = {}
        pr_data = {}
        for nm, pipeline in self.piplns.items():
            # Get transformed test data using current pipeline
            me = ModelEvaluator(pipeline, self.X_test)
            y_scores = me.evaluate()["y_test"]
            
            if self.is_multiclass:
                # Binarize the output for multiclass ROC curves
                y_bin = label_binarize(self.y_true, classes=range(self.num_classes))
                
                # Compute ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(self.num_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_scores[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Compute micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_scores.reshape(-1, self.num_classes).ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                
                # Compute macro-average ROC curve and ROC area
                # First aggregate all false positive rates
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.num_classes)]))
                
                # Then interpolate all ROC curves at these points
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(self.num_classes):
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                
                # Finally average it and compute AUC
                mean_tpr /= self.num_classes
                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
                
                # Store ROC data
                roc_data[nm] = (fpr, tpr, roc_auc)
                
                # Compute Precision-Recall curve and average precision for each class
                precision = dict()
                recall = dict()
                average_precision = dict()
                
                for i in range(self.num_classes):
                    precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], y_scores[:, i])
                    average_precision[i] = average_precision_score(y_bin[:, i], y_scores[:, i])
                
                # A "micro-average": quantifying score on all classes jointly
                precision["micro"], recall["micro"], _ = precision_recall_curve(y_bin.ravel(), y_scores.reshape(-1, self.num_classes).ravel())
                average_precision["micro"] = average_precision_score(y_bin.ravel(), y_scores.reshape(-1, self.num_classes).ravel())
                
                pr_data[nm] = (precision, recall, average_precision)
                
            else:
                # Binary case - use original implementation
                y_scores_binary = y_scores[:, 1]  # Probability of positive class
                fpr, tpr, _ = roc_curve(self.y_true, y_scores_binary)
                roc_auc = auc(fpr, tpr)
                roc_data[nm] = (fpr, tpr, roc_auc)
                
                precision, recall, _ = precision_recall_curve(self.y_true, y_scores_binary)
                avg_precision = average_precision_score(self.y_true, y_scores_binary)
                pr_data[nm] = (precision, recall, avg_precision)

        return roc_data, pr_data

    def plot_roc_curves(self, save_path=None):
        roc_data, _ = self.evaluate_models()
        
        if self.is_multiclass:
            # Set up the figure
            plt.figure(figsize=(15, 10))
            sns.set()  # Set Seaborn style
            
            # Plot micro-average ROC curve
            for model_name, (fpr, tpr, roc_auc) in roc_data.items():
                plt.plot(
                    fpr["micro"], 
                    tpr["micro"],
                    label=f'{model_name} (micro-avg area = {roc_auc["micro"]:.2f})',
                    linewidth=2
                )
            
            # Plot macro-average ROC curve
            for model_name, (fpr, tpr, roc_auc) in roc_data.items():
                plt.plot(
                    fpr["macro"], 
                    tpr["macro"],
                    label=f'{model_name} (macro-avg area = {roc_auc["macro"]:.2f})',
                    linewidth=2,
                    linestyle='--'
                )
        else:
            # Binary case - use original implementation
            plt.figure(figsize=(10, 8))
            sns.set()  # Set Seaborn style
            
            for model_name, (fpr, tpr, roc_auc) in roc_data.items():
                plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})', linewidth=2)
        
        # Common plotting elements
        plt.plot([0, 1], [0, 1], 'k--', label='Chance (area = 0.50)', linewidth=2)  # Dashed diagonal
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        sns.despine()  # Remove the top and right spines
        
        if save_path:
            plt.savefig(os.path.join(save_path, "ROC_CURVES.png"), format='png', dpi=600)  # Save the figure
            
            # For multiclass, also save per-class ROC curves
            if self.is_multiclass:
                colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
                
                for model_name, (fpr, tpr, roc_auc) in roc_data.items():
                    plt.figure(figsize=(15, 10))
                    
                    for i, color in zip(range(self.num_classes), colors):
                        plt.plot(
                            fpr[i], 
                            tpr[i],
                            color=color,
                            label=f'Class {i} (AUC = {roc_auc[i]:.2f})',
                            linewidth=2
                        )
                    
                    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)  # Dashed diagonal
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'{model_name} - ROC Curve per class')
                    plt.legend(loc="lower right")
                    sns.despine()
                    plt.savefig(os.path.join(save_path, f"{model_name}_class_ROC_CURVES.png"), format='png', dpi=600)
                    plt.close()
        
        plt.show()
    
    def plot_pr_curves(self, save_path=None):
        _, pr_data = self.evaluate_models()
        
        if self.is_multiclass:
            # Set up the figure
            plt.figure(figsize=(15, 10))
            sns.set()  # Set Seaborn style
            
            # Plot micro-average PR curve
            for model_name, (precision, recall, avg_precision) in pr_data.items():
                plt.plot(
                    recall["micro"], 
                    precision["micro"],
                    label=f'{model_name} (micro-avg AP = {avg_precision["micro"]:.2f})',
                    linewidth=2
                )
        else:
            # Binary case
            plt.figure(figsize=(10, 8))
            sns.set()  # Set Seaborn style
            
            for model_name, (precision, recall, avg_precision) in pr_data.items():
                plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.2f})', linewidth=2)
        
        # Common plotting elements
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        sns.despine()  # Remove the top and right spines
        
        if save_path:
            plt.savefig(os.path.join(save_path, "PR_CURVES.png"), format='png', dpi=600)  # Save the figure
            
            # For multiclass, also save per-class PR curves
            if self.is_multiclass:
                colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
                
                for model_name, (precision, recall, avg_precision) in pr_data.items():
                    plt.figure(figsize=(15, 10))
                    
                    for i, color in zip(range(self.num_classes), colors):
                        plt.plot(
                            recall[i], 
                            precision[i],
                            color=color,
                            label=f'Class {i} (AP = {avg_precision[i]:.2f})',
                            linewidth=2
                        )
                    
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title(f'{model_name} - Precision-Recall Curve per class')
                    plt.legend(loc="lower left")
                    sns.despine()
                    plt.savefig(os.path.join(save_path, f"{model_name}_class_PR_CURVES.png"), format='png', dpi=600)
                    plt.close()
        
        plt.show()


class ModelEvaluator:
    def __init__(self, pipeline, X_test):
        self.pipeline = pipeline
        self.X_test = X_test

    def evaluate(self):
        
        y_test_pred = self.pipeline.predict_proba(self.X_test)

        return {"y_test":y_test_pred}