import unittest
import pandas as pd
import os
import sys

sys.path.append('../src/')
from utils import save_dataframe_to_csv, load_dataframe_from_csv, setup_logging, log_message


class TestUtils(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        self.test_csv = 'test.csv'

    def test_save_dataframe_to_csv(self):
        # Test saving a DataFrame to CSV
        save_dataframe_to_csv(self.df, self.test_csv)
        self.assertTrue(os.path.exists(self.test_csv))

    def test_load_dataframe_from_csv(self):
        # Test loading a DataFrame from CSV
        df = load_dataframe_from_csv(self.test_csv)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)

    def test_logging(self):
        # Test logging setup and message logging
        setup_logging('test.log')
        log_message('Testing logging functionality.')
        self.assertTrue(os.path.exists('test.log'))

    def tearDown(self):
        # Clean up created files
        os.remove(self.test_csv)
        os.remove('test.log')


if __name__ == '__main__':
    unittest.main()
