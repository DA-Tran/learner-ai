"""LightGBM (GBM) model for tabular classification."""
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class LightGBMModel:
    """LightGBM classifier with binary/multi-class support."""
    
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=6):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None
    
    def build(self, input_dim, num_classes=3, is_binary=False):
        """Build model."""
        self.is_binary = is_binary
        self.num_classes = 2 if is_binary else num_classes  # LightGBM uses num_class
        params = {
            'objective': 'binary' if is_binary else 'multiclass',
            'metric': 'multi_logloss' if not is_binary else 'binary_logloss',
            'learning_rate': self.learning_rate,
            'num_leaves': 31,
            'max_depth': self.max_depth,
            'verbose': -1
        }
        if not is_binary:
            params['num_class'] = self.num_classes
        
        self.model = lgb.LGBMClassifier(**params, n_estimators=self.n_estimators, random_state=42)
        return self
    
    def train(self, X_train, y_train):
        """Train model. Input: flat arrays (not RNN-shaped)."""
        # Flatten y if onehot
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            y_train_flat = np.argmax(y_train, axis=1)
        else:
            y_train_flat = y_train.flatten()
        
        self.model.fit(X_train, y_train_flat)
        return self
    
    def predict(self, X_test):
        """Predict probabilities."""
        return self.model.predict_proba(X_test)
    
    def evaluate(self, X_test, y_test):
        """Full metrics."""
        y_pred_proba = self.predict(X_test)
        y_pred = self.model.predict(X_test)
        
        if self.is_binary:
            y_test_labels = y_test.flatten()
        else:
            y_test_labels = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
        
        metrics = {
            'accuracy': accuracy_score(y_test_labels, y_pred),
            'precision': precision_score(y_test_labels, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test_labels, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test_labels, y_pred, average='weighted', zero_division=0)
        }
        return metrics
