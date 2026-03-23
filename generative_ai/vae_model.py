import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
import warnings
warnings.filterwarnings('ignore')

class VAEClassifier:
    def __init__(self, latent_dim=16, hidden_units=64):
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.encoder = None
        self.decoder = None
        self.classifier = None
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def build_encoder(self, input_dim, num_classes):
        inputs = layers.Input(shape=(input_dim,))
        x = layers.Dense(self.hidden_units, activation='relu')(inputs)
        x = layers.Dense(self.hidden_units//2, activation='relu')(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        encoder = Model(inputs, [z_mean, z_log_var], name="encoder")
        encoder.compile(optimizer='adam')
        self.encoder = encoder
        return encoder

    def build_decoder(self):
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(self.hidden_units//2, activation='relu')(latent_inputs)
        x = layers.Dense(self.hidden_units, activation='relu')(x)
        outputs = layers.Dense(self.input_dim, activation='sigmoid')(x)  # Assume self.input_dim set
        decoder = Model(latent_inputs, outputs, name="decoder")
        self.decoder = decoder
        return decoder

    def build_classifier(self, input_dim, num_classes):
        inputs = layers.Input(shape=(input_dim,))
        x = layers.Dense(32, activation='relu')(inputs)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        classifier = Model(inputs, outputs, name="classifier")
        classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.classifier = classifier
        return classifier

    def build(self, input_dim, num_classes, is_binary=False):
        self.input_dim = input_dim
        self.num_classes = 2 if is_binary else num_classes
        self.build_encoder(input_dim, num_classes)
        self.build_decoder()
        self.build_classifier(input_dim, self.num_classes)

    @tf.function
    def encode(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        return z_mean, z_log_var

    def decode(self, z):
        return self.decoder(z)

    @tf.function
    def reparameterize(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        eps = tf.random.normal(shape=(batch, self.latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

    def train_step(self, x):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encode(x)
            z = self.reparameterize(z_mean, z_log_var)
            reconstructed = self.decode(z)
# Reconstruction loss (tabular 1D)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(x, reconstructed)
            )
            # KL loss
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + 0.1 * kl_loss
        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def train(self, X_train, epochs=30, batch_size=32, verbose=0):
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(10000).batch(batch_size)
        self.encoder.trainable = True
        self.decoder.trainable = True
        self.classifier.trainable = False
        for epoch in range(epochs):
            for step, x_batch_train in enumerate(dataset):
                self.train_step(x_batch_train)
        self.classifier.trainable = True

    def evaluate(self, X_test, y_test):
        z_mean, z_log_var = self.encode(X_test)
        z = self.reparameterize(z_mean, z_log_var)
        generated = self.decode(z)
        pred = self.classifier(generated)
        y_test_sparse = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test.ravel()
        loss, acc = self.classifier.evaluate(X_test, y_test_sparse, verbose=0)
        return {'accuracy': acc}

