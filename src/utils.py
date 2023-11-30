import logging
import os
import pandas as pd


def setup_logging(log_file='project.log'):
    """
    Sets up the logging configuration.

    :param log_file: str, the name of the log file
    :return: None
    """
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def log_message(message, level='info'):
    """
    Logs a message.

    :param message: str, the message to log
    :param level: str, the logging level ('info', 'warning', 'error')
    :return: None
    """
    if level == 'info':
        logging.info(message)
    elif level == 'warning':
        logging.warning(message)
    elif level == 'error':
        logging.error(message)


def save_dataframe_to_csv(df, filename, index=False):
    """
    Saves a DataFrame to a CSV file.

    :param df: DataFrame, the DataFrame to save
    :param filename: str, the filename to save the DataFrame to
    :param index: bool, whether to include the DataFrame's index in the CSV
    :return: None
    """
    df.to_csv(filename, index=index)
    log_message(f'DataFrame saved to {filename}')


def load_dataframe_from_csv(filename):
    """
    Loads a DataFrame from a CSV file.

    :param filename: str, the filename to load the DataFrame from
    :return: DataFrame
    """
    if not os.path.exists(filename):
        log_message(f'File {filename} not found', 'error')
        return None
    return pd.read_csv(filename)


if __name__ == "__main__":
    setup_logging()
    log_message('Testing logging functionality.')
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    save_dataframe_to_csv(df, 'test.csv')
    loaded_df = load_dataframe_from_csv('test.csv')
