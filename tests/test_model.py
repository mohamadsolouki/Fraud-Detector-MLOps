import unittest
from sklearn.datasets import make_classification
from model import train_model, evaluate_model, save_model

from sklearn.ensemble import RandomForestClassifier


class TestModel(unittest.TestCase):
    def setUp(self):
        # Generate a sample dataset for testing
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        self.X_train, self.X_test = X[:80], X[80:]
        self.y_train, self.y_test = y[:80], y[80:]

    def test_train_model(self):
        # Testing train_model function
        model = train_model(self.X_train, self.y_train)
        self.assertIsInstance(model, RandomForestClassifier)

    def test_evaluate_model(self):
        # Testing evaluate_model function
        model = train_model(self.X_train, self.y_train)
        # Since evaluate_model prints results, we're just checking for errors here
        try:
            evaluate_model(model, self.X_test, self.y_test)
        except Exception as e:
            self.fail(f"evaluate_model raised an exception {e}")

    def test_save_model(self):
        # Testing save_model function
        model = train_model(self.X_train, self.y_train)
        try:
            save_model(model, 'test_model.joblib')
        except Exception as e:
            self.fail(f"save_model raised an exception {e}")


if __name__ == '__main__':
    unittest.main()
