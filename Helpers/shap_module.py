import traceback
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


class ShapValues:
    """Calculate SHAP values for a given model and dataset
    Extended to handle multiclass classification
    """
    def __init__(self, ppln:Pipeline) -> None:
        """ Initialize the ShapValues object with a given pipeline

        Args:
            ppln (Pipeline): A pipeline object that contains the model and the preprocessing steps
        """
        self.ppln = ppln
        self.model = ppln["model"]
        if isinstance(self.model, (XGBClassifier, RandomForestClassifier, AdaBoostClassifier)):
            self.MODEL_TYPE = 1
        elif isinstance(self.model, DecisionTreeClassifier):
            self.MODEL_TYPE = 2
        elif isinstance(self.model, LogisticRegression):
            self.MODEL_TYPE = 3
        else:
            self.MODEL_TYPE = 0
        
        # Detect if multiclass by checking classes_ attribute if available
        self.is_multiclass = False
        if hasattr(self.model, 'classes_'):
            self.is_multiclass = len(self.model.classes_) > 2
            self.num_classes = len(self.model.classes_)
        else:
            # Try to infer from the number of output columns in coef_ if it exists
            if hasattr(self.model, 'coef_'):
                if len(self.model.coef_.shape) > 1 and self.model.coef_.shape[0] > 1:
                    self.is_multiclass = True
                    self.num_classes = self.model.coef_.shape[0]

    def __sample_data(self, x_val: pd.DataFrame, y_val: pd.Series, sample_size=100) -> tuple:
        """This function samples the data to get an equal number of samples from each class.
        It samples a percentage from each class to get a balanced dataset for SHAP values calculation
        and also mitigate the slow kernel explanations for large datasets.

        Args:
            x_val (pd.DataFrame): (samples, features) DataFrame of input data
            y_val (pd.Series): (samples,) Series of target data (0 or 1)

        Returns:
            tuple: (sampled_x, sampled_y) tuple of sampled data
        """
        positive_indices = y_val[y_val == 1].index
        control_indices = y_val[y_val == 0].index

        num_positive = len(positive_indices)
        num_control = len(control_indices)

        # Determine the number of samples to take from each class
        total_samples = num_positive + num_control
        if total_samples <= sample_size:
            return x_val, y_val

        # Calculate the percentage to sample from each class
        sample_percentage = min(1, sample_size / total_samples)

        num_positive_samples = int(num_positive * sample_percentage)
        num_control_samples = int(num_control * sample_percentage)

        # Sample instances from each class
        sampled_positive = x_val.loc[positive_indices].sample(n=num_positive_samples, random_state=42)
        sampled_control = x_val.loc[control_indices].sample(n=num_control_samples, random_state=42)

        # Combine the samples
        sampled_data = pd.concat([sampled_positive, sampled_control])
        sampled_y = y_val.loc[sampled_data.index]

        return sampled_data, sampled_y

    def check_size(self, x: pd.DataFrame, y: pd.Series, sample_size=100) -> tuple:
        """Sample data to a reasonable size for SHAP analysis.
        
        Args:
            x (pd.DataFrame): Features DataFrame
            y (pd.Series): Target Series
            sample_size (int): Maximum number of samples to use
            
        Returns:
            tuple: (sampled_x, sampled_y) tuple of sampled data
        """
        if x.shape[0] > sample_size:
            # Stratified sampling to maintain class distribution
            unique_classes = y.unique()
            is_multiclass = len(unique_classes) > 2
            
            if is_multiclass:
                # For multiclass, sample from each class to maintain distribution
                sampled_indices = []
                for cls in unique_classes:
                    cls_indices = y[y == cls].index
                    n_samples = min(int(sample_size * len(cls_indices) / len(y)), len(cls_indices))
                    sampled_indices.extend(np.random.choice(cls_indices, size=n_samples, replace=False))
                
                # Shuffle the sampled indices
                np.random.shuffle(sampled_indices)
                
                # Limit to sample_size if we got more
                if len(sampled_indices) > sample_size:
                    sampled_indices = sampled_indices[:sample_size]
                    
                sampled_x = x.loc[sampled_indices]
                sampled_y = y.loc[sampled_indices]
                
            else:
                # For binary, use the __sample_data method which already handles binary
                return self.__sample_data(x, y, sample_size=sample_size)
                
            return sampled_x, sampled_y
        else:
            return x, y
    
    @staticmethod
    def __data_transform(ppln, x_val:pd.DataFrame, y_val:pd.Series) -> tuple:
        """_summary_

        Args:
            x_val (pd.DataFrame): features DataFrame
            y_val (pd.Series): target Series

        Returns:
            tuple: (transformed_data, columns_names) tuple of transformed data and column names
        """
        data_test = pd.concat([x_val, y_val], axis=1)
        transformed_1 = ppln["FeatureWizFs"].transform(data_test)
        transformed_2 = ppln["preprocessor"].transform(transformed_1)
        
        # Check for the presence of categorical features
        if 'cat' in ppln.named_steps["preprocessor"].named_transformers_.keys():
            try:
                categorical_features = ppln.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(ppln.named_steps["preprocessor"].transformers_[1][2]).tolist()
            except NotFittedError:
                categorical_features = []
        else:
            categorical_features = []
        
        numeric_features = list(ppln.named_steps["preprocessor"].transformers_[0][2])
        columns_names = numeric_features + categorical_features
        
        return transformed_2, columns_names

    def calculate_shap_values(self, x_val: pd.DataFrame, y_val: pd.Series) -> np.array:
        """Calculate SHAP values, with special handling for multiclass models.

        Args:
            x_val (pd.DataFrame): Features DataFrame
            y_val (pd.Series): Target Series

        Returns:
            np.ndarray or dict: SHAP values, either as array for binary 
                               or as dict with class-specific values for multiclass
        """
        x_val, y_val = self.check_size(x_val, y_val, sample_size=100)
        transformed_data, columns_names = self.__data_transform(self.ppln, x_val, y_val)
        
        if self.MODEL_TYPE == 1:  # Tree-based models
            explainer = shap.Explainer(self.model, transformed_data)
            shap_values = explainer(transformed_data, check_additivity=False)
            shap_values.feature_names = columns_names
            
        elif self.MODEL_TYPE == 2:  # Decision Trees
            explainer = shap.TreeExplainer(self.model, transformed_data)
            shap_values = explainer(transformed_data, check_additivity=False)
            shap_values.feature_names = columns_names
            
        elif self.MODEL_TYPE == 3:  # Linear models
            explainer = shap.LinearExplainer(self.model, transformed_data)
            shap_values = explainer(transformed_data)
            shap_values.feature_names = columns_names
            
        else:  # Kernel explainer for other models
            x_val, y_val = self.check_size(x_val, y_val, sample_size=14)  # Use fewer samples for kernel explainer
            transformed_data, columns_names = self.__data_transform(self.ppln, x_val, y_val)
            explainer = shap.KernelExplainer(self.model.predict_proba, transformed_data)
            shap_values = explainer(transformed_data)
            shap_values.feature_names = columns_names

        return shap_values

    def __get_shap_values_for_class(self, shap_values):
        """_summary_

        Args:
            shap_values (np.ndarray): (samples, features) array of SHAP values

        Returns:
            np.ndarray: (samples, features) array of SHAP values for the positive class
        """
        if len(shap_values.shape) == 3:
            return shap_values[:, :, 1]
        else:
            return shap_values


class ShapPlots:
    def __init__(self, shap_values: np.ndarray) -> None:
        self.shap_values = shap_values
        
        # Auto-detect if multiclass by checking shap_values structure
        self.is_multiclass = False
        if hasattr(self.shap_values, 'values'):
            if len(self.shap_values.values.shape) == 3:
                self.is_multiclass = True
                self.num_classes = self.shap_values.values.shape[2]

    def _get_class_specific_values(self, class_idx=None):
        """For multiclass SHAP values, extract values for a specific class."""
        if not self.is_multiclass:
            return self.shap_values
            
        if hasattr(self.shap_values, 'values'):
            if class_idx is None:
                # Return values for the class with highest overall impact
                class_impacts = np.abs(self.shap_values.values).mean(axis=(0, 1))
                class_idx = np.argmax(class_impacts)
                
            # Extract values for the selected class
            class_values = self.shap_values.values[:, :, class_idx]
            
            # Create a new shap.Explanation object
            import copy
            class_shap_values = copy.deepcopy(self.shap_values)
            class_shap_values.values = class_values
            class_shap_values.base_values = self.shap_values.base_values[:, class_idx]
            
            return class_shap_values
        
        return self.shap_values

    def summary(self, save: bool = False, filename: str = "summary_plot.png"):
        plt.clf()
        if self.is_multiclass:
            # For multiclass, plot for each class separately
            for c in range(self.num_classes):
                class_values = self._get_class_specific_values(c)
                plt.figure(figsize=(12, 8))
                shap.plots.beeswarm(class_values, max_display=10, show=False, order=class_values.abs.mean(0))
                plt.title(f"SHAP Values for Class {c}")
                if save:
                    class_filename = filename.replace('.png', f'_class_{c}.png')
                    plt.savefig(class_filename, dpi=600, bbox_inches='tight')
                plt.close()
            
            # Also create an aggregated plot showing overall feature importance
            plt.figure(figsize=(12, 8))
            # Use absolute SHAP values across all classes to rank features
            overall_importance = np.abs(self.shap_values.values).mean(axis=(0, 2))
            feature_idx = np.argsort(-overall_importance)
            
            # Select top features
            top_features = feature_idx[:10]
            
            # For each top feature, show class-specific importance
            feature_names = self.shap_values.feature_names
            if feature_names is None:
                feature_names = [f"Feature {i}" for i in range(self.shap_values.values.shape[1])]
            
            # Create a bar plot for multiclass feature importance
            importances = np.abs(self.shap_values.values).mean(axis=0)  # Average across samples
            plt.figure(figsize=(12, 8))
            
            bar_data = []
            for i in top_features:
                for c in range(self.num_classes):
                    bar_data.append({
                        'Feature': feature_names[i],
                        'Class': f'Class {c}',
                        'Importance': importances[i, c]
                    })
            
            bar_df = pd.DataFrame(bar_data)
            
            # Plot using seaborn
            sns.barplot(x='Importance', y='Feature', hue='Class', data=bar_df)
            plt.title('Feature Importance by Class')
            if save:
                multi_filename = filename.replace('.png', '_multiclass_importance.png')
                plt.savefig(multi_filename, dpi=600, bbox_inches='tight')
            plt.close()
        else:
            # Binary case
            shap.plots.beeswarm(self.shap_values, max_display=10, show=False, order=self.shap_values.abs.mean(0))
            plt.title("SHAP Values Beeswarm Plot (Max of SHAP Values)")
            if save:
                plt.savefig(filename, dpi=600, bbox_inches='tight')
            plt.close()

    def bar(self, save: bool = False, filename: str = "bar_plot.png"):
        plt.clf()
        if self.is_multiclass:
            # For multiclass, create a bar plot for each class
            for c in range(self.num_classes):
                class_values = self._get_class_specific_values(c)
                plt.figure(figsize=(12, 8))
                shap.plots.bar(class_values, max_display=10, show=False)
                plt.title(f"Feature Importance for Class {c}")
                if save:
                    class_filename = filename.replace('.png', f'_class_{c}.png')
                    plt.savefig(class_filename, dpi=600, bbox_inches="tight")
                plt.close()
                
            # Also create an aggregated bar plot
            plt.figure(figsize=(12, 8))
            # Average absolute SHAP values across all classes
            importances = np.abs(self.shap_values.values).mean(axis=(0, 2))
            feature_names = self.shap_values.feature_names
            if feature_names is None:
                feature_names = [f"Feature {i}" for i in range(len(importances))]
                
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
            plt.title('Overall Feature Importance (Averaged Across Classes)')
            if save:
                agg_filename = filename.replace('.png', '_aggregated.png')
                plt.savefig(agg_filename, dpi=600, bbox_inches="tight")
            plt.close()
        else:
            # Binary case
            shap.plots.bar(self.shap_values, max_display=10, show=False)
            plt.title("Aggregated feature importance based on mean SHAP values")
            if save:
                plt.savefig(filename, dpi=600, bbox_inches="tight")
            plt.close()

    def heatmap(self, save: bool = False, filename: str = "heatmap_plot.png"):
        plt.clf()
        if self.is_multiclass:
            # For multiclass, create a heatmap for each class
            for c in range(self.num_classes):
                class_values = self._get_class_specific_values(c)
                plt.figure(figsize=(12, 8))
                shap.plots.heatmap(class_values, max_display=10, show=False)
                plt.title(f"Feature Impact Heatmap for Class {c}")
                if save:
                    class_filename = filename.replace('.png', f'_class_{c}.png')
                    plt.savefig(class_filename, dpi=600, bbox_inches='tight')
                plt.close()
        else:
            # Binary case
            shap.plots.heatmap(self.shap_values, max_display=10, show=False)
            plt.title("Overall Feature Impact on Model Outcome")
            if save:
                plt.savefig(filename, dpi=600, bbox_inches='tight')
            plt.close()

    def beeswarm(self, save: bool = False, filename: str = "beeswarm_plot.png"):
        plt.clf()
        if self.is_multiclass:
            # For multiclass, create a beeswarm plot for each class
            for c in range(self.num_classes):
                class_values = self._get_class_specific_values(c)
                plt.figure(figsize=(12, 8))
                shap.plots.beeswarm(class_values, show=False)
                plt.title(f"SHAP Values Beeswarm Plot for Class {c}")
                if save:
                    class_filename = filename.replace('.png', f'_class_{c}.png')
                    plt.savefig(class_filename, dpi=600, bbox_inches='tight')
                plt.close()
        else:
            # Binary case
            shap.plots.beeswarm(self.shap_values, show=False)
            plt.title("SHAP Values Beeswarm Plot (Mean Density of SHAP Values)")
            if save:
                plt.savefig(filename, dpi=600, bbox_inches='tight')
            plt.close()


def ShapAnalysis(ppln:Pipeline, X_test:pd.DataFrame, y_test:pd.Series, nm:str):
    try:
        sv = ShapValues(ppln)
        shap_values = sv.calculate_shap_values(X_test, y_test)
        print("-----------------------------------------------------------\n")
        print("------------Shap values calculated successfully------------\n")
        print("-----------------------------------------------------------\n")
        
        # Create directory structure
        try:
            os.mkdir(os.path.join("./Materials", "Shap_Features"))
        except OSError:
            pass
        shap_path = os.path.join("./Materials", "Shap_Features")
        try:
            os.mkdir(os.path.join(shap_path, f"{nm}"))
        except OSError:
            pass
        shap_md_path = os.path.join(shap_path, f"{nm}")
        
        # Create plots
        plts = ShapPlots(shap_values=shap_values)
        
        # Auto-detect if multiclass
        is_multiclass = plts.is_multiclass
        
        try:
            plts.bar(save=True, filename=os.path.join(shap_md_path, f"bar_plot_{nm}.png"))
            if is_multiclass:
                print(f"Created class-specific bar plots for {nm}")
        except Exception as e:
            error_message = f"Bar plot failed: {e}\n"
            with open(os.path.join("Materials", "Shap_error_log.txt"), "a") as file:
                file.write(error_message)
        
        try:
            plts.summary(save=True, filename=os.path.join(shap_md_path, f"summary_plot_{nm}.png"))
            if is_multiclass:
                print(f"Created class-specific summary plots for {nm}")
        except Exception as e:
            error_message = f"Summary plot failed: {e}\n"
            with open(os.path.join("Materials", "Shap_error_log.txt"), "a") as file:
                file.write(error_message)

        try:
            plts.beeswarm(save=True, filename=os.path.join(shap_md_path, f"beeswarm_plot_{nm}.png"))
            if is_multiclass:
                print(f"Created class-specific beeswarm plots for {nm}")
        except Exception as e:
            error_message = f"Beeswarm plot failed: {e}\n"
            with open(os.path.join("Materials", "Shap_error_log.txt"), "a") as file:
                file.write(error_message)

        try:
            plts.heatmap(save=True, filename=os.path.join(shap_md_path, f"heatmap_plot_{nm}.png"))
            if is_multiclass:
                print(f"Created class-specific heatmap plots for {nm}")
        except Exception as e:
            error_message = f"Heatmap plot failed: {e}\n"
            with open(os.path.join("Materials", "Shap_error_log.txt"), "a") as file:
                file.write(error_message)

    except Exception as e:
        error_message = f"Error in SHAP analysis: {e}\n"
        traceback_msg = traceback.format_exc()
        with open(os.path.join("./Materials", "Shap_error_log.txt"), "a") as file:
            file.write(error_message)
            file.write(traceback_msg)
