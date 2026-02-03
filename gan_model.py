"""GAN model implementation with expanded functionality."""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class GAN:
    """Generative Adversarial Network for data augmentation."""
    
    def __init__(self, input_dim=4, generator_units=8, discriminator_units=8, learning_rate=0.0002):
        """
        Initialize GAN.
        
        Args:
            input_dim: Input dimension
            generator_units: Units in generator layers
            discriminator_units: Units in discriminator layers
            learning_rate: Learning rate for optimizer
        """
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
        """Build the generator network."""
        self.generator = Sequential([
            Dense(self.generator_units, activation='relu', input_dim=self.input_dim),
            Dense(self.input_dim, activation='linear')
        ])
        return self
    
    def build_discriminator(self):
        """Build the discriminator network."""
        self.discriminator = Sequential([
            Dense(self.discriminator_units, activation='relu', input_dim=self.input_dim),
            Dense(1, activation='sigmoid')
        ])
        
        self.discriminator.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return self
    
    def build_gan(self):
        """Build the combined GAN model."""
        self.discriminator.trainable = False
        
        self.gan = Sequential([
            self.generator,
            self.discriminator
        ])
        
        self.gan.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy'
        )
        return self
    
    def train(self, X_train, epochs=100, batch_size=32, verbose_interval=1000):
        """
        Train the GAN.
        
        Args:
            X_train: Training data
            epochs: Number of training epochs
            batch_size: Batch size
            verbose_interval: Print progress every N epochs
            
        Returns:
            tuple: (d_loss_hist, g_loss_hist)
        """
        self.d_loss_hist = []
        self.g_loss_hist = []
        
        half_batch = batch_size // 2
        
        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            real_samples = X_train[idx]
            real_labels = np.ones((half_batch, 1))
            
            # Generate fake samples
            noise = np.random.normal(0, 1, (half_batch, self.input_dim))
            generated_samples = self.generator.predict(noise, verbose=0)
            fake_labels = np.zeros((half_batch, 1))
            
            # Train discriminator on real and fake data
            d_loss_real = self.discriminator.train_on_batch(real_samples, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(generated_samples, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            valid_labels = np.ones((batch_size, 1))
            g_loss = self.gan.train_on_batch(noise, valid_labels)
            
            # Store losses
            if isinstance(d_loss, (list, tuple, np.ndarray)):
                self.d_loss_hist.append(d_loss[0])
            else:
                self.d_loss_hist.append(d_loss)
            self.g_loss_hist.append(g_loss)
            
            # Print progress
            if epoch % verbose_interval == 0:
                d_loss_val = d_loss[0] if isinstance(d_loss, (list, tuple, np.ndarray)) else d_loss
                print(f"Epoch {epoch} [D loss: {d_loss_val:.4f}] [G loss: {g_loss:.4f}]")
        
        return self.d_loss_hist, self.g_loss_hist
    
    def generate_samples(self, num_samples):
        """
        Generate fake samples.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        noise = np.random.normal(0, 1, (num_samples, self.input_dim))
        return self.generator.predict(noise, verbose=0)
    
    def augment_data(self, X_train, num_generated=None):
        """
        Augment training data with generated samples.
        
        Args:
            X_train: Original training data
            num_generated: Number of samples to generate (default: same as X_train)
            
        Returns:
            Augmented data
        """
        if num_generated is None:
            num_generated = X_train.shape[0]
        
        generated = self.generate_samples(num_generated)
        return np.concatenate((X_train, generated), axis=0)


class GANClassifier:
    """Classifier trained on GAN-augmented data."""
    
    def __init__(self, hidden_units=10, learning_rate=0.001):
        """Initialize classifier."""
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.model = None
    
    def build(self, input_dim, num_classes=3):
        """Build classifier model."""
        self.model = Sequential([
            Dense(self.hidden_units, activation='relu', input_dim=input_dim),
            Dense(num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return self
    
    def train(self, X_train, y_train, epochs=50, batch_size=10, verbose=0):
        """Train the classifier."""
        return self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
    
    def predict(self, X_test, verbose=0):
        """Predict on test data."""
        return self.model.predict(X_test, verbose=verbose)
    
    def evaluate(self, X_test, y_test, verbose=0):
        """
        Evaluate classifier performance.
        
        Args:
            X_test: Test features
            y_test: Test labels (one-hot encoded)
            verbose: Verbosity level
            
        Returns:
            dict: Dictionary with metrics
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
