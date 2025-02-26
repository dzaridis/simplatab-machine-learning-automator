import os
import subprocess

def activate_giskard_env():
    activate_giskard_env_path = os.path.join("/giskard_env/bin/activate_this.py")
    if os.path.exists(activate_giskard_env_path):
        try:
            with open(activate_giskard_env_path) as f:
                exec(f.read(), {'__file__': activate_giskard_env_path})
            print("Giskard environment activated successfully.")
        except Exception as e:
            print(f"Failed to activate Giskard environment: {e}")
    else:
        print(f"Giskard environment activation script not found at {activate_giskard_env_path}")

    # Check installed packages
    try:
        result = subprocess.run(
            ["/giskard_env/bin/pip", "list"],
            capture_output=True,
            text=True
        )
        print("Installed packages:\n", result.stdout)
    except Exception as e:
        print(f"Failed to list installed packages: {e}")


activate_giskard_env()
import numpy as np
import pandas as pd

from giskard import Model, Dataset, scan


import pickle


class VulnerabilityDetection:
    def __init__(self, df: pd.DataFrame, model_instance):
        self.model_instance = model_instance
        self.df = df
    
    def gisk_dataset(self):
        CATEGORICAL_COLUMNS = list(self.df[self.df.columns[self.df.dtypes == 'object']].columns)
        giskard_dataset = Dataset(
                df=self.df,
                target="Target",
                name="",
                cat_columns=CATEGORICAL_COLUMNS,
                )
        return giskard_dataset
    
    def gisk_model(self):
        model_inst = self.model_instance

        def prediction_function(df: pd.DataFrame) -> np.ndarray:
            return model_inst.predict_proba(df)
        
        giskard_model = Model(
            model=prediction_function,
            model_type="classification",
            name="Vulnerability Detection Model",
            classification_labels=model_inst.classes_,
            feature_names=self.df.columns
        )
        return giskard_model


def execute_vulnerabilities_detection(df_test: pd.DataFrame):
    READ_FOLDER = "./Materials"
    Models_paths = os.path.join(READ_FOLDER, "Models")
    models = {}
    for model in os.listdir(Models_paths):
        model_path = os.path.join(Models_paths, model)
        with open(model_path, 'rb') as f:
            mdl = pickle.load(f)
        models.update({model.split("_")[0]: mdl})
    
    for model_name, model_instance in models.items():
        vulnerability_detection = VulnerabilityDetection(df_test, model_instance)
        dataset = vulnerability_detection.gisk_dataset()
        model = vulnerability_detection.gisk_model()
        results = scan(model, dataset)
        try:
            os.mkdir("./Materials/Model_Vulnerabilities")
        except FileExistsError:
            pass
        pth = os.path.join("./Materials/Model_Vulnerabilities", f"{model_name}_vulnerabilities.html")
        results.to_html(pth)