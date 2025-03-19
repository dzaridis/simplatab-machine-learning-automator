import unittest
import pandas as pd
from io import StringIO
from Helpers.data_checks import DataChecker  # Replace with the actual module name

class TestDataCheckerMethods(unittest.TestCase):
    def setUp(self):
        self.data_checker = DataChecker(input_folder="dummy_folder")  # The folder is not used here
        
        self.train_csv = """patient_id,ID,Target,feature1,feature2,categorical
1,1,0,10,20,A
2,2,1,30,40,B
3,3,0,50,60,C
4,4,1,70,80,A
"""
        self.test_csv = """patient_id,ID,feature1,feature2,categorical
5,5,90,100,A
6,6,110,120,B
7,7,130,140,D
8,8,150,160,A
"""
        self.train_df = pd.read_csv(StringIO(self.train_csv))
        self.test_df = pd.read_csv(StringIO(self.test_csv))

    def test_check_target_column(self):
        # Test with valid data
        self.data_checker.check_target_column(self.train_df)

        # Test with invalid data
        invalid_df = self.train_df.copy()
        invalid_df.loc[0, 'Target'] = 2
        with self.assertRaises(ValueError):
            self.data_checker.check_target_column(invalid_df)

    def test_set_index_column(self):
        # Test with 'ID' column
        df_with_id = self.train_df.copy()
        df_with_id = self.data_checker.set_index_column(df_with_id)
        self.assertEqual(df_with_id.index.name, "ID")

        # Test with 'patient_id' column
        df_with_patient_id = self.train_df.drop(columns=['ID']).copy()
        df_with_patient_id = self.data_checker.set_index_column(df_with_patient_id)
        self.assertEqual(df_with_patient_id.index.name, "patient_id")

        # Test with neither 'ID' nor 'patient_id' column
        df_invalid = self.train_df.drop(columns=['ID', 'patient_id']).copy()
        with self.assertRaises(ValueError):
            self.data_checker.set_index_column(df_invalid)

    def test_remove_nan_rows(self):
        # Add NaN value to the dataframe
        df_with_nan = self.train_df.copy()
        df_with_nan.loc[0, 'feature1'] = None
        df_with_nan = self.data_checker.remove_nan_rows(df_with_nan)
        self.assertFalse(df_with_nan.isna().any().any())

    def test_check_categorical_features(self):
        train, test, cols_to_drop = self.data_checker.check_categorical_features(self.train_df, self.test_df)
        self.assertEqual(cols_to_drop, ['categorical'])
        self.assertNotIn('categorical', train.columns)
        self.assertNotIn('categorical', test.columns)

    def test_process_data(self):
        # Assuming data is already loaded
        train_df = self.train_df.copy()
        test_df = self.test_df.copy()

        with unittest.mock.patch.object(DataChecker, 'load_data', return_value=(train_df, test_df)):
            train, test = self.data_checker.process_data()

        # Ensure the data is processed correctly
        self.assertEqual(train.index.name, "ID")
        self.assertEqual(test.index.name, "ID")
        self.assertFalse(train.isna().any().any())
        self.assertFalse(test.isna().any().any())
        self.assertIn("Target", train.columns)
        self.assertNotIn("Target", test.columns)
        self.assertNotIn("categorical", train.columns)
        self.assertNotIn("categorical", test.columns)

if __name__ == '__main__':
    unittest.main()