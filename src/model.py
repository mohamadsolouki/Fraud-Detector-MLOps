import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def train_model(X_train, y_train):
    """
    Trains the Random Forest model on the provided training data.

    :param X_train: DataFrame, training feature data
    :param y_train: Series, training target data
    :return: trained model
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test data.

    :param model: the trained model
    :param X_test: DataFrame, testing feature data
    :param y_test: Series, testing target data
    :return: None, prints the evaluation metrics
    """
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def save_model(model, filename):
    """
    Saves the trained model to a file.

    :param model: the trained model
    :param filename: str, the path to save the model
    :return: None
    """
    joblib.dump(model, filename)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = ...  # Load your dataset here
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, 'random_forest_model.joblib')
