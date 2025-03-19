import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def summary_results_excel(results, file:str, conf_matrix_name:str):
    summary_results = {}

    # Iterate over each model and its folds
    summary_results = {}
    confusion_matrix_results = {}

    # Iterate over each model and its folds
    for model, folds in results.items():
        # Initialize a dictionary to store the metrics for the current model
        model_metrics = {
            'Sensitivity': [],
            'Specificity': [],
            'AUC': [],
            'F-score': [],
            'Accuracy': [],
            'Balanced Accuracy': []
        }
        
        # Initialize a dictionary to store confusion matrix values
        confusion_matrix_metrics = {
            'TP': [],
            'FP': [],
            'TN': [],
            'FN': []
        }
        
        # Collect metrics from each fold
        for fold, metrics in folds.items():
            for metric, value in metrics.items():
                if metric != 'Confusion Matrix':  # Skip confusion matrix
                    model_metrics[metric].append(value)
                else:
                    confusion_matrix_metrics['TP'].append(value['TP'])
                    confusion_matrix_metrics['FP'].append(value['FP'])
                    confusion_matrix_metrics['TN'].append(value['TN'])
                    confusion_matrix_metrics['FN'].append(value['FN'])
        
        # Calculate mean and std for each metric
        summary_results[model] = {}
        for metric, values in model_metrics.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            summary_results[model][metric] = f"{mean_value:.4f} ± {std_value:.4f}"
        
        # Calculate mean confusion matrix values
        mean_tp = np.mean(confusion_matrix_metrics['TP'])
        mean_fp = np.mean(confusion_matrix_metrics['FP'])
        mean_tn = np.mean(confusion_matrix_metrics['TN'])
        mean_fn = np.mean(confusion_matrix_metrics['FN'])
        
        # Store mean confusion matrix values
        confusion_matrix_results[model] = {
            'TP': mean_tp,
            'FP': mean_fp,
            'TN': mean_tn,
            'FN': mean_fn
        }

        try: 
            os.mkdir(os.path.join("Materials", "ConfusionMatrices"))
        except OSError:
            pass
        path_conf = os.path.join("Materials", "ConfusionMatrices")

        # Plot the confusion matrix using seaborn
        cm_array = np.array([[mean_tp, mean_fp], [mean_fn, mean_tn]])
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm_array, annot=True, fmt=".2f", cmap="Reds", xticklabels=['Predicted Positive', 'Predicted Negative'], yticklabels=['Actual Positive', 'Actual Negative'])
        plt.title(f'{model} Mean Confusion Matrix')
        plt.savefig(os.path.join(path_conf, f"{model}_{conf_matrix_name}_confusion_matrix.png"), dpi=1000)
        plt.close()
    df_summary = pd.DataFrame(summary_results).T
    df_summary.to_excel(os.path.join("Materials",f"{file}.xlsx"))

def external_summary(results, file:str, conf_matrix_name:str):
    summary_results = {}
    confusion_matrix_results = {}

    # Process the external test set results
    for model, metrics in results.items():
        # Extract confusion matrix values
        tp = metrics['Confusion Matrix']['TP']
        fp = metrics['Confusion Matrix']['FP']
        tn = metrics['Confusion Matrix']['TN']
        fn = metrics['Confusion Matrix']['FN']
        
        # Store metrics
        summary_results[model] = {
            'Sensitivity': metrics['Sensitivity'],
            'Specificity': metrics['Specificity'],
            'AUC': metrics['AUC'],
            'F-score': metrics['F-score'],
            'Accuracy': metrics['Accuracy'],
            'Balanced Accuracy': metrics['Balanced Accuracy']
        }
        
        # Store confusion matrix values
        confusion_matrix_results[model] = {
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn
        }
        try: 
            os.mkdir(os.path.join("Materials", "ConfusionMatrices"))
        except OSError:
            pass

        path_conf = os.path.join("Materials", "ConfusionMatrices")
        # Plot the confusion matrix using seaborn
        cm_array = np.array([[tp, fp], [fn, tn]])
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm_array, annot=True, fmt=".2f", cmap="Reds", xticklabels=['Predicted Positive', 'Predicted Negative'], yticklabels=['Actual Positive', 'Actual Negative'])
        plt.title(f'{model} Confusion Matrix')
        plt.savefig(os.path.join(path_conf, f"{model}_{conf_matrix_name}_confusion_matrix.png"), dpi=1000)
        plt.close()

    # Convert summary results to DataFrame
    df_summary = pd.DataFrame(summary_results).T
    df_summary.to_excel(os.path.join("Materials",f"{file}.xlsx"))

