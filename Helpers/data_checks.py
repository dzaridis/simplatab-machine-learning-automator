import os
import pandas as pd


class DataChecker:
    def __init__(self, input_folder):
        self.input_folder = input_folder

    def load_data(self):
        train_path = os.path.join(self.input_folder, "Train.csv")
        test_path = os.path.join(self.input_folder, "Test.csv")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Train.csv or Test.csv not found in the specified input folder.")

        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        
        return train, test

    def check_target_column(self, df, target_col="Target"):
        if target_col not in df.columns:
            raise ValueError(f"The target column '{target_col}' is not present in the dataframe.")
        
        if not df[target_col].isin([0, 1]).all():
            raise ValueError(f"The target column '{target_col}' does not contain binary values 0 and 1.")

    def set_index_column(self, df, index_col="ID"):
        if index_col not in df.columns and "patient_id" not in df.columns:
            raise ValueError(f"The index column '{index_col}' is not present in the dataframe.")
        try:
            df.set_index(index_col, inplace=True)
        except KeyError:
            df.set_index("patient_id", inplace=True)

    def remove_nan_rows(self, df):
        df.dropna(inplace=True)

    def process_data(self):
        train, test = self.load_data()
        
        # Check target column in train
        self.check_target_column(train)

        # Set index column
        self.set_index_column(train)
        self.set_index_column(test)

        # Remove rows with NaN values
        self.remove_nan_rows(train)
        self.remove_nan_rows(test)

        return train, test