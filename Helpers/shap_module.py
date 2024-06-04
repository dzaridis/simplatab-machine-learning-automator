import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
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
        if isinstance(self.model, (XGBClassifier, RandomForestClassifier, DecisionTreeClassifier, AdaBoostClassifier, SGDClassifier)):
            self.MODEL_TYPE = 1
        else:
            self.MODEL_TYPE = 0
    
    def __data_transform(self, x_val:pd.DataFrame, y_val:pd.Series) -> tuple:
        """_summary_

        Args:
            x_val (pd.DataFrame): features DataFrame
            y_val (pd.Series): target Series

        Returns:
            tuple: (transformed_data, columns_names) tuple of transformed data and column names
        """
        data_test = pd.concat([x_val, y_val], axis = 1)
        transformed_1 = self.ppln.named_steps["FeatureWizFs"].transform(data_test)
        transformed_2 = self.ppln.named_steps["preprocessor"].transform(transformed_1)
        return transformed_2, transformed_1.columns.tolist()

    def __sample_data(self, x_val: pd.DataFrame, y_val: pd.Series) -> tuple:
        """This function samples the data to get equal number of samples from each class.
        It samples 10 samples from each class in order to get a balanced dataset for SHAP values calculation
        and also mitigate the slow kernel explanations for large datasets.

        Args:
            x_val (pd.DataFrame): (samples, features) DataFrame of input data
            y_val (pd.Series): (samples,) Series of target data (0 or 1)

        Returns:
            tuple: (sampled_x, sampled_y) tuple of sampled data
        """
        positive_indices = y_val[y_val == 1].index
        control_indices = y_val[y_val == 0].index
        num_samples = min(len(positive_indices), len(control_indices), 10)

        # Sample instances from each class
        sampled_positive = x_val.loc[positive_indices].sample(n=num_samples, random_state=42)
        sampled_control = x_val.loc[control_indices].sample(n=num_samples, random_state=42)

        # Combine the samples
        sampled_data = pd.concat([sampled_positive, sampled_control])
        sampled_y = y_val.loc[sampled_data.index]

        return sampled_data, sampled_y
    
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
            transformed_data, columns_names = self.__data_transform(x_val, y_val)
            explainer = shap.Explainer(self.model, transformed_data)
            shap_values = explainer(transformed_data)
            shap_values.feature_names = columns_names
        else:
            sampled_x, sampled_y = self.__sample_data(x_val, y_val)
            transformed_data, columns_names = self.__data_transform(sampled_x, sampled_y)
            explainer = shap.KernelExplainer(self.model.predict_proba, transformed_data)
            shap_values = explainer(transformed_data)
            shap_values.feature_names = columns_names
        
        shap_values = self.__get_shap_values_for_class(shap_values)

        return shap_values

class ShapPlots:

    def __init__(self, 
                shap_values:np.ndarray) -> None:
        
        self.shap_values = shap_values

    def summary(self, 
                save: bool = False, 
                filename: str = "summary_plot.png"):
        
        fig = shap.summary_plot(self.shap_values, max_display=10, show=False, plot_type="dot")
        if save:
            plt.savefig(filename, dpi=400)
        else:
            plt.show()
    
    def bar(self, 
            save: bool = False, 
            filename: str = "bar_plot.png"):
        
        fig = shap.summary_plot(self.shap_values, max_display=10, show=False, plot_type="bar")
        if save:
            plt.savefig(filename, dpi=400)
        else:
            plt.show()
    
    def heatmap(self, 
                save: bool = False, 
                filename: str = "heatmap_plot.png"):
        
        fig = shap.plots.heatmap(self.shap_values, max_display=10, show=False)
        if save:
            plt.savefig(filename, dpi=400)
        else:
            plt.show()
    
    def beeswarm(self, 
                save: bool = False,
                filename: str = "beeswarm_plot.png"):
        
        fig = shap.plots.beeswarm(self.shap_values, show=False) 
        if save:
            plt.savefig(filename, dpi=400)
        else:
            plt.show()
    
def ShapAnalysis(ppln:Pipeline, 
                X_test:pd.DataFrame, 
                y_test:pd.Series, 
                nm:str):
    try:
        sv = ShapValues(ppln)
        shap_values = sv.calculate_shap_values(X_test, y_test)
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
        
        plts = ShapPlots(shap_values = shap_values)
        plts.summary(save=True, filename=os.path.join(shap_md_path,f"summary_plot_{nm}.png"))
        plts.bar(save=True, filename=os.path.join(shap_md_path,f"bar_plot_{nm}.png"))
        plts.beeswarm(save=True, filename=os.path.join(shap_md_path,f"beeswarm_plot_{nm}.png"))
        plts.heatmap(save=True, filename=os.path.join(shap_md_path,f"heatmap_plot_{nm}.png"))
    except Exception as e:
        error_message = f"Error here: {e}\n"
        with open(os.path.join("Materials", "Shap_error_log.txt"), "a") as file:
            file.write(error_message)
        pass