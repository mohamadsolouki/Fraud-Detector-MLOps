import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """
    Loads the dataset from the specified file path.

    :param file_path: str, path to the dataset file
    :return: DataFrame, the loaded dataset
    """
    return pd.read_csv(file_path)


def preprocess_data(data):
    """
    Performs preprocessing on the dataset including scaling and splitting.

    :param data: DataFrame, the dataset to preprocess
    :return: tuple, containing the split and scaled data (X_train, X_test, y_train, y_test)
    """
    # Drop the 'Time' column and scale the 'Amount' column
    data = data.drop(['Time'], axis=1)
    scaler = StandardScaler()
    data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

    # Splitting the data into features and target
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Example usage
    file_path = '../data/dataset.csv'
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
