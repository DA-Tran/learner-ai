"""Data loading and preprocessing utilities."""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

def load_and_preprocess_dataset(dataset_name='iris', test_size=0.2, random_state=42):
    # Dataset configs: (UCI_ID, target_mode)
    UCI_IDS = {'iris': None, 'heart': 45, 'breast': 15, 'wine': 186}
    
    if dataset_name not in UCI_IDS:
        raise ValueError(f"Dataset must be one of {list(UCI_IDS.keys())}")
    
    if dataset_name == 'iris':
        iris = load_iris()
        X = iris.data
        y = iris.target
    else:
        dataset = fetch_ucirepo(id=UCI_IDS[dataset_name])
        X = dataset.data.features 
        y = dataset.data.targets 
        
        # Wine: quality → binary good/bad
        if dataset_name == 'wine':
            y_numeric = y['quality'].fillna(y['quality'].median()).values
            y = (y_numeric >= 6).astype(int)
        
        # Binary target (first col >0)
        elif UCI_IDS[dataset_name] in [45,186]:
            y = (y.iloc[:, 0].fillna(0) > 0).astype(int).values
        else:
            y = y.values.ravel()
        
        # Features: categorical → codes, numeric only
        X_processed = X.copy()
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object':
                X_processed[col] = pd.Categorical(X_processed[col].fillna('missing')).codes
        
        X_numeric = X_processed.select_dtypes([np.number]).fillna(0).values
        if X_numeric.shape[1] == 0:
            raise ValueError("No numeric features!")
        
        X = X_numeric
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Encode labels
    unique_classes = np.unique(y)
    is_binary = len(unique_classes) == 2
    
    if is_binary:
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
    else:
        encoder = LabelBinarizer()
        y = encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Silent
    
    return X_train, X_test, y_train, y_test, scaler, encoder, is_binary

def reshape_for_rnn(X_train, X_test):
    X_train_rnn = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_rnn = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    return X_train_rnn, X_test_rnn

