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

class ShapValues:
    """Calculate SHAP values for a given model and dataset
    the input model should be a classifier model from sklearn or xgboost, and the input dataset should be a pandas DataFrame
    usage Example:
    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    ppln = Pipeline([
        ("preprocessor", StandardScaler()),
        ("model", LogisticRegression())
        ])
    ppln.fit(X_train, y_train)

    ShapObject = ShapValues(ppln)
    ShapObject.calculate_shap_values(X_test, y_test)
    ```
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

    def __sample_data(self, x_val: pd.DataFrame, y_val: pd.Series,sample_size=100) -> tuple:
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
        if x.shape[0] > sample_size:
            sampled_x, sampled_y = self.__sample_data(x, y, sample_size=sample_size)
            return sampled_x, sampled_y
        else:
            return x, y
    
    @staticmethod
    def __get_shap_values_for_class(shap_values):
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

    def calculate_shap_values(self, x_val: pd.DataFrame, y_val: pd.Series) -> np.array:
        """_summary_

        Args:
            x_val (pd.DataFrame): (samples, features) DataFrame of input data
            y_val (pd.Series): (samples,) Series of target data

        Returns:
            np.ndarray: (samples, features) array of SHAP values
        """

        if self.MODEL_TYPE == 1:
            x_val, y_val = self.check_size(x_val, y_val, sample_size=100)
            transformed_2, columns_names = self.__data_transform(self.ppln, x_val, y_val)
            explainer = shap.Explainer(self.model, transformed_2)
            shap_values = explainer(transformed_2, check_additivity=False)
            shap_values.feature_names = columns_names
        elif self.MODEL_TYPE == 2:
            x_val, y_val = self.check_size(x_val, y_val, sample_size=100)
            transformed_2, columns_names = self.__data_transform(self.ppln, x_val, y_val)
            explainer = shap.TreeExplainer(self.model, transformed_2)
            shap_values = explainer(transformed_2, check_additivity=False)
            shap_values.feature_names = columns_names
        elif self.MODEL_TYPE == 3:
            x_val, y_val = self.check_size(x_val, y_val, sample_size=100)
            transformed_2, columns_names = self.__data_transform(self.ppln, x_val, y_val)
            explainer = shap.LinearExplainer(self.model, transformed_2)
            shap_values = explainer(transformed_2)
            shap_values.feature_names = columns_names
        else:
            sampled_x, sampled_y = self.__sample_data(x_val, y_val, sample_size=14)
            transformed_2, columns_names = self.__data_transform(self.ppln, sampled_x, sampled_y)
            explainer = shap.KernelExplainer(self.model.predict_proba, transformed_2)
            shap_values = explainer(transformed_2)
            shap_values.feature_names = columns_names
        
        shap_values = self.__get_shap_values_for_class(shap_values)

        return shap_values


class ShapPlots:
    def __init__(self, shap_values: np.ndarray) -> None:
        self.shap_values = shap_values

    def summary(self, save: bool = False, filename: str = "summary_plot.png"):
        plt.clf()
        shap.plots.beeswarm(self.shap_values, max_display=10, show=False, order=self.shap_values.abs.max(0))
        plt.title("SHAP Values Beeswarm Plot (Max of SHAP Values)")
        if save:
            plt.savefig(filename, dpi=600, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def bar(self, save: bool = False, filename: str = "bar_plot.png"):
        plt.clf()
        #fig, ax = plt.subplots(figsize=(12, 8))  # Create a figure and an axes object
        shap.plots.bar(self.shap_values, max_display=10, show=False)
        plt.title("Aggregated feature importance based on mean SHAP values")
        #plt.subplots_adjust(bottom=0.25)  # Adjust bottom margin to prevent x-axis clipping
        if save:
            plt.savefig(filename, dpi=600, bbox_inches="tight")  # Save the figure
        else:
            plt.show()
        plt.close()

    def heatmap(self, save: bool = False, filename: str = "heatmap_plot.png"):
        plt.clf()
        shap.plots.heatmap(self.shap_values, max_display=10, show=False)
        plt.title("Overall Feature Impact on Model Outcome")
        if save:
            plt.savefig(filename, dpi=600, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def beeswarm(self, save: bool = False, filename: str = "beeswarm_plot.png"):
        plt.clf()
        shap.plots.beeswarm(self.shap_values, show=False)
        plt.title("SHAP Values Beeswarm Plot (Mean Density of SHAP Values)")
        if save:
            plt.savefig(filename, dpi=600, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        plt.show()
    
def ShapAnalysis(ppln:Pipeline, 
                X_test:pd.DataFrame, 
                y_test:pd.Series, 
                nm:str):
    try:
        sv = ShapValues(ppln)
        shap_values = sv.calculate_shap_values(X_test, y_test)
        print("-----------------------------------------------------------\n")
        print("------------Shap values calculated successfully------------\n")
        print("-----------------------------------------------------------\n")
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
        
        plts = ShapPlots(shap_values=shap_values)
        try:
            plts.bar(save=True, filename=os.path.join(shap_md_path, f"bar_plot_{nm}.png"))
        except IndexError as e:
            error_message = f"bar plot failed. Utilize the other shap figures to identify interpretabilty {e}\n"
            with open(os.path.join("Materials", "Shap_error_log.txt"), "a") as file:
                file.write(error_message)
            pass
        try:
            plts.beeswarm(save=True, filename=os.path.join(shap_md_path, f"beeswarm_plot_MaxShapValues_{nm}.png"))
        except Exception as e:
            print(e)
            with open(os.path.join("Materials", "Shap_error_log.txt"), "a") as file:
                file.write(error_message)
            pass
        try:
            plts.heatmap(save=True, filename=os.path.join(shap_md_path, f"heatmap_plot_{nm}.png"))
        except Exception as e:
            print(e)
            error_message = f"Error here: Heatmap  failed. Utilize the other shap figures to identify interpretabilty {e}\n"
            with open(os.path.join("Materials", "Shap_error_log.txt"), "a") as file:
                file.write(error_message)
            pass
    except Exception as e:
        error_message = f"Error here: {e}\n"
        with open(os.path.join("./Materials", "Shap_error_log.txt"), "a") as file:
            file.write(error_message)
        pass