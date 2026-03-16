"""Data loading and preprocessing utilities."""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

def load_and_preprocess_dataset(dataset_name='iris', test_size=0.2, random_state=42):
    UCI_DATASETS = {
        'iris': ('iris', None),
'heart': (45, 'binary'),
        'breast': (15, None), 
        'wine': (186, 'binary')  # Wine Quality UCI 186
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
        X = dataset.data.features 
        y = dataset.data.targets 
        
        print(f"Raw features shape: {X.shape}")
        print(f"Raw target shape: {y.shape}")
        
        # Handle wine quality (multi-class → binary)
        if dataset_name == 'wine':
            y_numeric = y['quality'].fillna(y['quality'].median()).values
            y = (y_numeric >= 6).astype(int)  # Good (6+) vs Bad
        
        elif target_type == 'binary':
            y = (y.iloc[:, 0].fillna(0) > 0).astype(int).values
        else:
            y = y.values.ravel()
        
        # Process features
        X_processed = X.copy()
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object':
                X_processed[col] = pd.Categorical(X_processed[col].fillna('missing')).codes
        
        X_numeric = X_processed.select_dtypes([np.number]).fillna(0).values
        
        if X_numeric.shape[1] == 0:
            raise ValueError(f"No numeric features!")
        
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
    
    print(f"{dataset_name.upper()} - Train: {X_train.shape}, Test: {X_test.shape}, is_binary: {is_binary}")
    
    return X_train, X_test, y_train, y_test, scaler, encoder, is_binary

def reshape_for_rnn(X_train, X_test):
    X_train_rnn = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_rnn = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    return X_train_rnn, X_test_rnn

