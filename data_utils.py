"""Data loading and preprocessing utilities."""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split


def load_and_preprocess_iris(test_size=0.2, random_state=42):
    """
    Load and preprocess the Iris dataset.
    
    Args:
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, encoder)
    """
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # One-hot encode the target labels
    encoder = LabelBinarizer()
    y = encoder.fit_transform(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, scaler, encoder


def reshape_for_rnn(X_train, X_test):
    """
    Reshape data for RNN/LSTM models (add time dimension).
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        tuple: (X_train_rnn, X_test_rnn)
    """
    X_train_rnn = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_rnn = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    return X_train_rnn, X_test_rnn
