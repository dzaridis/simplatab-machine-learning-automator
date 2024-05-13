import shap
import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, is_classifier
from sklearn.pipeline import Pipeline


def save_shap_values(shap_values:np.array, features:list, path:str, model_name:str):
    shap_values_class_0 = shap_values[0]
    shap_values_class_1 = shap_values[1]
    
    # Convert to DataFrame
    shap_df_class_0 = pd.DataFrame(shap_values_class_0, columns=features)
    shap_df_class_1 = pd.DataFrame(shap_values_class_1, columns=features)

    # Melt the DataFrames
    shap_melted_df_class_0 = shap_df_class_0.melt(value_vars=features)
    shap_melted_df_class_1 = shap_df_class_1.melt(value_vars=features)

    # Initialize a matplotlib figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot for class 0
    sns.barplot(x=shap_melted_df_class_0['value'], y=shap_melted_df_class_0['variable'], ax=axes[0])
    axes[0].set_title('Class Non-user SHAP values')
    
    # Plot for class 1
    sns.barplot(x=shap_melted_df_class_1['value'], y=shap_melted_df_class_1['variable'], ax=axes[1])
    axes[1].set_title('Class User SHAP values')

    # Adjust the layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(path,model_name+"_shap_class.png"), dpi=400, bbox_inches='tight')
    plt.clf()

class ShapAnalysis:
    def __init__(self, X_val: np.array, pipeline_module: Pipeline, features: list) -> None:
        self.xvl = X_val
        self.explainer = None
        self.model = pipeline_module["model"]
        self.features = features
        self.X_test_transformed = None
        self.ppln = pipeline_module
    
    def perform_shap(self):
        
        if int(len(self.xvl))>100:
            sample_size=100
        else:
            sample_size = int(len(self.xvl))
        sampled_data = shap.sample(self.xvl, sample_size)
        print(f"Sample size: {sample_size}, Sampled data shape: {sampled_data.shape}")
        for item in self.ppln.named_steps.keys():
            if item != "model":
                self.X_test_transformed = self.ppln.named_steps[item].transform(sampled_data)

        # Step 2: Apply the preprocessor transformation

        if isinstance(self.model, (XGBClassifier, RandomForestClassifier, DecisionTreeClassifier, AdaBoostClassifier)):
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.KernelExplainer(self.model.predict_proba, self.X_test_transformed)
        
        shap_values = self.explainer.shap_values(self.X_test_transformed)
        print(f"SHAP values shape: {np.array(shap_values).shape}")
        return shap_values
    
    def plot_shap_values(self, model_name: str, path: str):
        plt.clf()
        shap_values = self.explainer.shap_values(self.X_test_transformed)
        print(f"SHAP values for plotting shape: {np.array(shap_values).shape}")
        
        if isinstance(shap_values, list):
            # Assuming binary classification and interested in SHAP values for class 1
            values_to_plot = shap_values[1]
        else:
            # For regression or binary classification models where a single array is returned
            values_to_plot = shap_values
        
        shap.summary_plot(values_to_plot, self.X_test_transformed, feature_names=self.features,max_display=10, show=False, plot_type="dot")
        
        if path:
            plot_path = os.path.join(path, model_name + "_ShapFeatures.png")
        else:
            plot_path = model_name + "_ShapFeatures.png"
        plt.savefig(plot_path, dpi=400, bbox_inches='tight')
        plt.close()