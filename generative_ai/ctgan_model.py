"""CTGAN wrapper for tabular generation - requires pip install ctgan[sds]"""
import tensorflow as tf
import numpy as np
try:
    from ctgan import CTGAN
    HAS_CTGAN = True
except ImportError:
    HAS_CTGAN = False
    print("Install ctgan: pip install ctgan[sds]")

class CTGANModel:
    def __init__(self):
        self.model = None
        self.classifier = None

    def build(self, input_dim, num_classes, is_binary=False):
        if not HAS_CTGAN:
            raise ImportError("ctgan not installed")
        self.model = CTGAN(epochs=50, batch_size=500)
        # Placeholder classifier
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        self.classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, epochs=30, verbose=0):
        self.model.fit(X_train, epochs=epochs)
        # Train classifier on generated + real
        generated = self.model.sample(len(X_train))
        X_combined = np.vstack([X_train, generated])
        y_combined = np.zeros(len(X_combined))  # Dummy labels for unsupervised
        self.classifier.fit(X_combined, y_combined[:len(X_combined)], epochs=epochs, verbose=verbose)

    def evaluate(self, X_test, y_test):
        pred = self.classifier.predict(X_test)
        y_pred_class = np.argmax(pred, axis=1)
        y_test_class = np.argmax(y_test, axis=1) if len(y_test.shape)>1 else y_test
        acc = np.mean(y_pred_class == y_test_class)
        return {'accuracy': float(acc)}

    def generate_samples(self, n_samples):
        return self.model.sample(n_samples)

