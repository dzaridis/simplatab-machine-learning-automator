from sklearn.base import BaseEstimator, is_classifier
import pandas as pd
import numpy as np
import seaborn as sns
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import os


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
    def __init__(self, X_train:np.array, X_val:np.array, model:BaseEstimator, features:list) -> None:
        self.xtr = X_train
        self.xvl = X_val
        self.explainer = None
        self.model = model
        self.features = features
    def perform_shap(self):

        if isinstance(self.model, (XGBClassifier, RandomForestClassifier, DecisionTreeClassifier, AdaBoostClassifier)):
            self.explainer = shap.TreeExplainer(self.model, shap.sample(self.xtr, int(0.1*len(self.xtr))))
        else:
            self.explainer = shap.KernelExplainer(self.model.predict_proba, shap.sample(self.xtr, int(0.1*len(self.xtr))))
        return self.explainer.shap_values(self.xvl)
    
    def plot_shap_values(self, model_name:str, path:str):
        plt.clf()
        shap.summary_plot(self.explainer.shap_values(self.xvl)[1],
                        self.xvl,
                        feature_names= self.features, show=False)
        # Save the plot to a file
        plt.savefig(os.path.join(path,model_name+"_ShapFeatures.png"), dpi=400, bbox_inches='tight')
        # Close the plot to free up memory
        #save_shap_values(self.explainer.shap_values(self.model.named_steps['preprocessor'].transform(self.xvl)), pipeline=self.model, path = path)
        #plt.clf()