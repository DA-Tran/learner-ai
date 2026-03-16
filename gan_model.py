"""GAN for data augmentation + classifier."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class GAN:
    """GAN generator + discriminator."""

    def __init__(self, input_dim=4, generator_units=8, discriminator_units=8, learning_rate=0.0002):
        self.input_dim = input_dim
        self.generator_units = generator_units
        self.discriminator_units = discriminator_units
        self.learning_rate = learning_rate
        
        self.generator = None
        self.discriminator = None
        self.gan = None
        self.d_loss_hist = []
        self.g_loss_hist = []
        
    def build_generator(self):
        self.generator = Sequential([
            tf.keras.layers.Input(shape=(self.input_dim,)),
            Dense(self.generator_units, activation='relu'),
            Dense(self.input_dim, activation='tanh')
        ])
        return self
    
    def build_discriminator(self):
        self.discriminator = Sequential([
            tf.keras.layers.Input(shape=(self.input_dim,)),
            Dense(self.discriminator_units, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.discriminator.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return self
    
    def build_gan(self):
        self.discriminator.trainable = False
        
        self.gan = Sequential([
            self.generator,
            self.discriminator
        ])
        
        # Skip fit() call since discriminator.trainable=False
        self.gan.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy'
        )
        return self
    
    def train(self, X_train, epochs=100, batch_size=32):
        half_batch = batch_size // 2
        
        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            real_samples = X_train[idx]
            
            noise = np.random.normal(0, 1, (half_batch, self.input_dim))
            generated_samples = self.generator.predict(noise, verbose=0)
            
            real_labels = np.ones((half_batch, 1))
            fake_labels = np.zeros((half_batch, 1))
            
            d_loss_real = self.discriminator.train_on_batch(real_samples, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(generated_samples, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            valid_labels = np.ones((batch_size, 1))
            g_loss = self.gan.train_on_batch(noise, valid_labels)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch} [D loss: {d_loss[0]:.4f}] [G loss: {g_loss:.4f}]")
        
        return self.d_loss_hist, self.g_loss_hist
    
    def generate_samples(self, num_samples):
        noise = np.random.normal(0, 1, (num_samples, self.input_dim))
        return self.generator.predict(noise, verbose=0)


class GANClassifier:
    """Classifier trained on GAN-augmented data."""
    
    def __init__(self, hidden_units=10, learning_rate=0.001):
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.model = None
    
    def build(self, input_dim, num_classes=3):
        self.model = Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(self.hidden_units, activation='relu'),
        ])
        
        if num_classes == 1:
            self.model.add(Dense(1, activation='sigmoid'))
            self.model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            self.model.add(Dense(num_classes, activation='softmax'))
            self.model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        return self
    
    def train(self, X_train, y_train, epochs=50, batch_size=10, validation_split=0.2, verbose=0):
        return self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
    
    def predict(self, X_test, verbose=0):
        return self.model.predict(X_test, verbose=verbose)
    
    def evaluate(self, X_test, y_test, verbose=0):
        """
        Evaluate with binary/multi-class support.
        """
        y_pred = self.predict(X_test, verbose=verbose)
        
        if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
            # Binary
            y_pred_labels = (y_pred.flatten() > 0.5).astype(int)
            y_test_labels = (y_test.flatten() > 0.5).astype(int) if len(y_test.shape) > 1 else y_test
        else:
            # Multi-class
            y_pred_labels = np.argmax(y_pred, axis=1)
            y_test_labels = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
            
        metrics = {
            'accuracy': accuracy_score(y_test_labels, y_pred_labels),
            'precision': precision_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0),
            'recall': recall_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0),
            'f1': f1_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0)
        }
        
        return metrics

