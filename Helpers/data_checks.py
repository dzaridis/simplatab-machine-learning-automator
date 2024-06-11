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

    @staticmethod
    def check_target_column(df, target_col="Target"):
        if target_col not in df.columns:
            raise ValueError(f"The target column '{target_col}' is not present in the dataframe.")
        
        if not df[target_col].isin([0, 1]).all():
            raise ValueError(f"The target column '{target_col}' does not contain binary values 0 and 1.")

    @staticmethod
    def set_index_column(df, index_col="ID"):
        
        if index_col in df.columns.to_list():
            df.set_index(index_col, inplace=True)
            return df
        elif "patient_id" in df.columns.to_list():
            df.set_index("patient_id", inplace=True)
            return df
        else:
            raise ValueError(f"Neither '{index_col}' nor 'patient_id' is present in the dataframe.")
        
        
    @staticmethod
    def remove_nan_rows(df):
        df.dropna(inplace=True)
        return df
    
    @staticmethod
    def check_categorical_features(train, test):
        categorical_cols = train.select_dtypes(include=['object', 'category']).columns
        if categorical_cols.empty:
            # No categorical columns to process
            return train, test, []

        cols_to_drop = []

        for col in categorical_cols:
            if col in test.columns:
                train_unique_values = set(train[col].dropna().unique())
                test_unique_values = set(test[col].dropna().unique())
                if train_unique_values != test_unique_values:
                    cols_to_drop.append(col)

        train.drop(columns=cols_to_drop, inplace=True)
        test.drop(columns=cols_to_drop, inplace=True)

        return train, test, cols_to_drop

    def process_data(self):
        train, test = self.load_data()
        
        # Check target column in train
        self.check_target_column(train)

        # Set index column
        train = self.set_index_column(train)
        test= self.set_index_column(test)

        # Remove rows with NaN values
        train = self.remove_nan_rows(train)
        test = self.remove_nan_rows(test)
        train, test, cols_to_drop = self.check_categorical_features(train, test)

        return train, test