import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model

class DCGAN:
    def __init__(self, input_dim, latent_dim=100):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.generator = None
        self.discriminator = None
        self.gan = None

    def build_generator(self):
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.latent_dim,)),
            layers.Reshape((128, 1)),
            layers.Conv1DTranspose(64, 5, strides=2, padding='same', activation='relu'),
            layers.Conv1DTranspose(32, 5, strides=2, padding='same', activation='relu'),
            layers.Flatten(),
            layers.Dense(self.input_dim, activation='tanh')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam')
        self.generator = model

    def build_discriminator(self):
        model = tf.keras.Sequential([
            layers.Flatten(input_shape=(self.input_dim,)),
            layers.Dense(128),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam')
        self.discriminator = model

    def build_gan(self):
        self.discriminator.trainable = False
        gan_input = layers.Input(shape=(self.latent_dim,))
        gan_output = self.discriminator(self.generator(gan_input))
        self.gan = Model(gan_input, gan_output)
        self.gan.compile(loss='binary_crossentropy', optimizer='adam')

    def train(self, X_train, epochs=50, batch_size=32):
        batch_size = min(batch_size, len(X_train) // 10)
        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_samples = X_train[idx]
            noise = tf.random.normal((batch_size, self.latent_dim))
            fake_samples = self.generator.predict(noise, verbose=0)
            d_loss_real = self.discriminator.train_on_batch(real_samples, np.ones((batch_size, 1)))
            d_loss_fake = self.discriminator.train_on_batch(fake_samples, np.zeros((batch_size, 1)))
            noise = tf.random.normal((batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, np.ones((batch_size, 1)))

    def generate_samples(self, n_samples):
        noise = tf.random.normal((n_samples, self.latent_dim))
        return self.generator.predict(noise, verbose=0)

class DCGANClassifier:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = None

    def build(self, input_dim, num_classes):
        self.model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X, y, epochs=30, verbose=0):
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y.ravel()[:min_len]
        self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=verbose)

    def evaluate(self, X, y):
        y_sparse = np.argmax(y, axis=1) if len(y.shape) > 1 else y.ravel()
        loss, acc = self.model.evaluate(X, y_sparse, verbose=0)
        return {'accuracy': acc}

