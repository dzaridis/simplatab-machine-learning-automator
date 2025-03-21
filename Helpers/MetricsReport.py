import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


def summary_results_excel(results, file:str, conf_matrix_name:str):
    summary_results = {}
    confusion_matrix_results = {}

    # Check if we're dealing with multiclass data
    is_multiclass = False
    num_classes = 2
    
    # Detect if we have multiclass data by checking first model's first fold
    for model in results:
        first_fold = list(results[model].keys())[0]
        if "Number of Classes" in results[model][first_fold]:
            is_multiclass = results[model][first_fold]["Number of Classes"] > 2
            num_classes = results[model][first_fold]["Number of Classes"]
        break

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
        
        # Initialize data structure for confusion matrices
        if is_multiclass:
            # For multiclass, we'll store the full confusion matrix
            confusion_matrices = []
        else:
            # For binary, use the original approach
            confusion_matrix_metrics = {
                'TP': [],
                'FP': [],
                'TN': [],
                'FN': []
            }
        
        # Collect metrics from each fold
        for fold, metrics in folds.items():
            for metric, value in metrics.items():
                if metric not in ['Confusion Matrix', 'Class-wise Metrics', 'Number of Classes']:
                    model_metrics[metric].append(value)
            
            # Handle confusion matrix based on whether we have multiclass or binary data
            if is_multiclass:
                confusion_matrices.append(metrics['Confusion Matrix'])
            else:
                cm_data = metrics['Confusion Matrix']
                confusion_matrix_metrics['TP'].append(cm_data['TP'])
                confusion_matrix_metrics['FP'].append(cm_data['FP'])
                confusion_matrix_metrics['TN'].append(cm_data['TN'])
                confusion_matrix_metrics['FN'].append(cm_data['FN'])
        
        # Calculate mean and std for each metric
        summary_results[model] = {}
        for metric, values in model_metrics.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            summary_results[model][metric] = f"{mean_value:.4f} ± {std_value:.4f}"
        
        try: 
            os.mkdir(os.path.join("Materials", "ConfusionMatrices"))
        except OSError:
            pass
        path_conf = os.path.join("Materials", "ConfusionMatrices")

        # Plot the confusion matrix differently based on multiclass or binary
        if is_multiclass:
            # Average all confusion matrices from folds
            avg_cm = np.mean(confusion_matrices, axis=0)
            
            plt.figure(figsize=(max(10, num_classes * 2), max(8, num_classes * 1.5)))
            sns.heatmap(
                avg_cm, 
                annot=True, 
                fmt=".2f", 
                cmap="Reds",
                xticklabels=[f'Predicted {i}' for i in range(num_classes)],
                yticklabels=[f'Actual {i}' for i in range(num_classes)]
            )
            plt.title(f'{model} Mean Confusion Matrix')
            plt.savefig(os.path.join(path_conf, f"{model}_{conf_matrix_name}_confusion_matrix.png"), dpi=1000)
            plt.close()
            
            # Store the averaged confusion matrix
            confusion_matrix_results[model] = avg_cm
        else:
            # Binary case - use original implementation
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
            
            # Plot the binary confusion matrix
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

    # Determine if we're dealing with multiclass
    is_multiclass = False
    num_classes = 2
    
    # Detect multiclass by checking the first model
    for model in results:
        if "Number of Classes" in results[model]:
            is_multiclass = results[model]["Number of Classes"] > 2
            num_classes = results[model]["Number of Classes"]
        break

    # Process the external test set results
    for model, metrics in results.items():
        # Store metrics
        summary_results[model] = {
            'Sensitivity': metrics['Sensitivity'],
            'Specificity': metrics['Specificity'],
            'AUC': metrics['AUC'],
            'F-score': metrics['F-score'],
            'Accuracy': metrics['Accuracy'],
            'Balanced Accuracy': metrics['Balanced Accuracy']
        }
        
        try: 
            os.mkdir(os.path.join("Materials", "ConfusionMatrices"))
        except OSError:
            pass

        path_conf = os.path.join("Materials", "ConfusionMatrices")
        
        # Handle confusion matrix differently based on multiclass or binary
        if is_multiclass:
            # For multiclass, plot the full confusion matrix
            cm = metrics['Confusion Matrix']
            confusion_matrix_results[model] = cm
            
            plt.figure(figsize=(max(10, num_classes * 2), max(8, num_classes * 1.5)))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt=".2f", 
                cmap="Reds",
                xticklabels=[f'Predicted {i}' for i in range(num_classes)],
                yticklabels=[f'Actual {i}' for i in range(num_classes)]
            )
            plt.title(f'{model} Confusion Matrix')
            plt.savefig(os.path.join(path_conf, f"{model}_{conf_matrix_name}_confusion_matrix.png"), dpi=1000)
            plt.close()
        else:
            # Binary case - extract TP, FP, TN, FN
            tp = metrics['Confusion Matrix']['TP']
            fp = metrics['Confusion Matrix']['FP']
            tn = metrics['Confusion Matrix']['TN']
            fn = metrics['Confusion Matrix']['FN']
            
            # Store confusion matrix values
            confusion_matrix_results[model] = {
                'TP': tp,
                'FP': fp,
                'TN': tn,
                'FN': fn
            }
            
            # Plot the binary confusion
                        # Plot the binary confusion matrix
            cm_array = np.array([[tp, fp], [fn, tn]])
            plt.figure(figsize=(12, 8))
            sns.heatmap(cm_array, annot=True, fmt=".2f", cmap="Reds", xticklabels=['Predicted Positive', 'Predicted Negative'], yticklabels=['Actual Positive', 'Actual Negative'])
            plt.title(f'{model} Confusion Matrix')
            plt.savefig(os.path.join(path_conf, f"{model}_{conf_matrix_name}_confusion_matrix.png"), dpi=1000)
            plt.close()

    # Convert summary results to DataFrame
    df_summary = pd.DataFrame(summary_results).T
    df_summary.to_excel(os.path.join("Materials",f"{file}.xlsx"))

