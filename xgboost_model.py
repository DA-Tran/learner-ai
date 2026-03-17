"""XGBoost model for tabular classification."""
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class XGBoostModel:
    """XGBoost classifier with binary/multi-class support."""
    
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=6):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None
    
    def build(self, input_dim, num_classes=3, is_binary=False):
        """Build model."""
        self.is_binary = is_binary
        self.num_classes = 2 if is_binary else num_classes
        
        params = {
            'objective': 'binary:logistic' if is_binary else 'multi:softprob',
            'eval_metric': 'logloss',
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'random_state': 42
        }
        if not is_binary:
            params['num_class'] = self.num_classes
        
        self.model = xgb.XGBClassifier(**params, n_estimators=self.n_estimators)
        return self
    
    def train(self, X_train, y_train):
        """Train. Flat arrays."""
        y_train_flat = np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train.flatten()
        self.model.fit(X_train, y_train_flat)
        return self
    
    def predict(self, X_test):
        """Proba."""
        return self.model.predict_proba(X_test)
    
    def evaluate(self, X_test, y_test):
        """Metrics."""
        y_pred = self.model.predict(X_test)
        y_test_labels = y_test.flatten() if self.is_binary else np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
        
        metrics = {
            'accuracy': accuracy_score(y_test_labels, y_pred),
            'precision': precision_score(y_test_labels, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test_labels, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test_labels, y_pred, average='weighted', zero_division=0)
        }
        return metrics
