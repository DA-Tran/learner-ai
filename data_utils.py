"""Data loading and preprocessing utilities."""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

def load_and_preprocess_dataset(dataset_name='iris', test_size=0.2, random_state=42):
    # Dataset configs: (UCI_ID, target_mode)
    UCI_IDS = {'iris': None, 'heart': 45, 'breast': 15, 'wine': 186, 'phishing': 327, 'mushroom': 73, 'gendername': 591}
    
    if dataset_name not in UCI_IDS:
        raise ValueError(f"Dataset must be one of {list(UCI_IDS.keys())}")
    
    if dataset_name == 'iris':
        iris = load_iris()
        X = iris.data
        y = iris.target
    else:
        dataset = fetch_ucirepo(id=UCI_IDS[dataset_name])
        if dataset is None:
            raise ValueError(f"UCI repo {UCI_IDS[dataset_name]} not found")
        X = dataset.data.features 
        y = dataset.data.targets 
        
        # Wine: quality → binary good/bad
        if dataset_name == 'wine':
            y_numeric = y['quality'].fillna(y['quality'].median()).values
            y = (y_numeric >= 6).astype(int)
        
        # Binary target (first col =1 legit vs phishing, gender)
        if UCI_IDS[dataset_name] in [45, 327, 591]:
            y = y.iloc[:, 0].fillna(0).eq(1).astype(int).values if hasattr(y, 'iloc') else np.zeros(len(X))
        elif UCI_IDS[dataset_name] == 186:
            if hasattr(y, 'iloc'):
                y = (y.iloc[:, 0].fillna(0) > 0).astype(int).values
            else:
                y = (y > 0).astype(int)
        elif UCI_IDS[dataset_name] == 73:  # Mushroom
            if 'class' in y.columns:
                y = (y['class'].map({'e': 0, 'p': 1}).fillna(0)).values
            else:
                y = y.iloc[:, 0].map({'e': 0, 'p': 1}).fillna(0).values if hasattr(y, 'iloc') else y.values.ravel()
        else:
            y = y.values.ravel()
        
        # Features: categorical → codes, numeric only
        X_processed = X.copy()
        for col in X_processed.columns:
            col_series = X_processed[col].fillna('missing')
            if col_series.dtype == 'object':
                X_processed[col] = pd.Categorical(col_series).codes.astype(float)
            else:
                X_processed[col] = pd.to_numeric(col_series, errors='coerce').fillna(0).astype(float)
        
        X_numeric = X_processed.select_dtypes([np.number]).fillna(0).values
        if X_numeric.shape[1] == 0:
            # All categorical - use encoded
            print(f"Warning: No numeric features for {dataset_name}, using all encoded")
            X_numeric = X_processed.astype(float).fillna(0).values
            if X_numeric.shape[1] == 0:
                raise ValueError("No features available!")
        
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
    
    return X_train, X_test, y_train, y_test, scaler, encoder, is_binary

def reshape_for_rnn(X_train, X_test):
    X_train_rnn = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_rnn = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    return X_train_rnn, X_test_rnn

