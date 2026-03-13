"""Data loading and preprocessing utilities."""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_dataset(dataset_name='iris', test_size=0.2, random_state=42):
    """
    Load and preprocess either Iris or CDC Diabetes dataset.
    
    Args:
        dataset_name: 'iris' or 'diabetes'
        test_size: Fraction for test split
        random_state: Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, encoder, full_X, full_y)
    """
    UCI_DATASETS = {
        'iris': ('iris', None),
        'heart': (45, 'num'),  # Binary/multi-class
        'breast': (15, 'binary'), 
        'youtube': (380, 'binary')
    }
    
    if dataset_name not in UCI_DATASETS:
        raise ValueError(f"dataset_name must be one of {list(UCI_DATASETS.keys())}")
    
    dataset_id, target_type = UCI_DATASETS[dataset_name]
    
    if dataset_name == 'iris':
        iris = load_iris()
        X = iris.data
        y = iris.target
        
    else:
        dataset = fetch_ucirepo(id=dataset_id)
        X = dataset.data.features.values
        if target_type == 'binary':
            y = dataset.data.targets.iloc[:, 0].values
        else:
            y = dataset.data.targets.values.argmax(axis=1) if dataset.data.targets.shape[1] > 1 else dataset.data.targets.iloc[:, 0].values
        
        # Numeric features only
        X_df = pd.DataFrame(X)
        X = X_df.select_dtypes(include=[np.number]).fillna(0).values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    encoder = LabelBinarizer()
    y = encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=np.argmax(y, axis=1) if y.ndim > 1 else y
    )
    
    print(f"{dataset_name.upper()} - Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, encoder

def reshape_for_rnn(X_train, X_test):
    """Reshape data for RNN/LSTM (add time dimension)."""
    X_train_rnn = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_rnn = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    return X_train_rnn, X_test_rnn

