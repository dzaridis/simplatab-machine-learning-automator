import os
import pandas as pd
import uuid
import numpy as np


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

        # For multiclass, we only need to check if values are numeric
        # Not restricting to binary values anymore
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            raise ValueError(f"The target column '{target_col}' should contain numeric values.")

        # Print class information
        unique_values = df[target_col].unique()
        num_classes = len(unique_values)
        print(f"Target column '{target_col}' has {num_classes} unique classes: {sorted(unique_values)}")

        # Check for class imbalance
        class_counts = df[target_col].value_counts()
        print("Class distribution:")
        for cls, count in class_counts.items():
            print(f"  Class {cls}: {count} samples ({count/len(df)*100:.2f}%)")

    @staticmethod
    def target_label_issue(train, test=None, target_col="Target"):
        """Describe how the Target labels differ from what the metrics expect, if they do.

        The metrics, ROC curves and confusion matrices assume the classes are numbered
        0, 1, ..., K-1 (0/1 for binary classification, 1 being the positive class).

        Returns:
            str or None: a message for the user, or None when the labels are as expected.
        """
        train_labels = sorted(pd.unique(train[target_col]))
        expected = list(range(len(train_labels)))
        if train_labels != expected:
            return (f"The Target column of Train.csv contains {train_labels}, but the classes must be numbered "
                    f"{expected} (0 and 1 for binary classification, 1 being the positive class). "
                    f"Recode the Target column, otherwise the reported metrics will be wrong.")
        if test is not None and target_col in test.columns:
            unknown = sorted(set(pd.unique(test[target_col])) - set(train_labels))
            if unknown:
                return (f"The Target column of Test.csv contains {unknown}, which are not classes of Train.csv "
                        f"{train_labels}. Both files must use the same classes.")
        return None

    @staticmethod
    def set_index_column(df, index_col="ID"):
        # Check if index_col exists
        if index_col in df.columns.to_list():
            df.set_index(index_col, inplace=True)
            return df
        # Check if patient_id exists as fallback
        elif "patient_id" in df.columns.to_list():
            df.set_index("patient_id", inplace=True)
            return df
        # Create a new ID column with random unique values
        else:
            print(f"Creating new '{index_col}' column with unique identifiers")
            # Generate random unique IDs (using UUID for guaranteed uniqueness)
            random_ids = [str(uuid.uuid4())[:8] for _ in range(len(df))]
            # Add the new column to the dataframe
            df[index_col] = random_ids
            # Set it as index
            df.set_index(index_col, inplace=True)
            return df
        
        
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