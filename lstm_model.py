"""LSTM model implementation with expanded functionality."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class LSTMModel:
    """LSTM model for classification."""
    
    def __init__(self, hidden_units=50, dropout=0.0, learning_rate=0.001):
        """
        Initialize LSTM model.
        
        Args:
            hidden_units: Number of LSTM units
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
        """
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build(self, input_shape, num_classes=3):
        """
        Build the LSTM model.
        
        Args:
            input_shape: Input shape (time_steps, features)
            num_classes: Number of output classes
            
        Returns:
            self
        """
        self.model = Sequential([
            tf.keras.layers.Input(shape=input_shape),
            LSTM(self.hidden_units, activation='relu'),
        ])
        
        if self.dropout > 0:
            self.model.add(Dropout(self.dropout))
            
        self.model.add(Dense(num_classes, activation='softmax'))
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self
    
    def train(self, X_train, y_train, epochs=50, batch_size=10, validation_split=0.2, verbose=0):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            verbose: Verbosity level
            
        Returns:
            history: Training history
        """
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        return self.history
    
    def predict(self, X_test, verbose=0):
        """Predict on test data."""
        return self.model.predict(X_test, verbose=verbose)
    
    def evaluate(self, X_test, y_test, verbose=0):
        """
        Evaluate model performance with multiple metrics.
        
        Args:
            X_test: Test features
            y_test: Test labels (one-hot encoded)
            verbose: Verbosity level
            
        Returns:
            dict: Dictionary with accuracy, precision, recall, f1
        """
        y_pred = self.predict(X_test, verbose=verbose)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
        
        metrics = {
            'accuracy': accuracy_score(y_test_labels, y_pred_labels),
            'precision': precision_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0),
            'recall': recall_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0),
            'f1': f1_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def save(self, filepath):
        """Save the model."""
        if self.model:
            self.model.save(filepath)
    
    def summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()
