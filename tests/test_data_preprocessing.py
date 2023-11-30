import unittest
import pandas as pd
from data_preprocessing import load_data, preprocess_data


class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        # Sample data similar to the actual dataset
        self.sample_data = pd.DataFrame({
            'Time': [1, 2, 3],
            'Amount': [100, 200, 300],
            'Class': [0, 1, 0],
            'V1': [0.1, 0.2, 0.3],
            'V2': [0.1, 0.2, 0.3]
        })

    def test_load_data(self):
        # Testing load_data function with sample data
        df = load_data('path_to_sample_data.csv')
        self.assertIsInstance(df, pd.DataFrame)

    def test_preprocess_data(self):
        # Testing preprocess_data function
        X_train, X_test, y_train, y_test = preprocess_data(self.sample_data)
        self.assertEqual(X_train.shape[1], 4)  # Check if 'Time' column is dropped
        self.assertTrue('Amount' in X_train.columns)  # Check if 'Amount' column exists


if __name__ == '__main__':
    unittest.main()
