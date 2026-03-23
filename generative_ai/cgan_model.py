import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Embedding, Concatenate, Dense
from tensorflow.keras.models import Model

class CGAN:
    def __init__(self, input_dim, num_classes, latent_dim=100):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.generator = None
        self.discriminator = None

    def build_generator(self):
        noise_input = Input(shape=(self.latent_dim,))
        label_input = Input(shape=(1,))
        label_embedding = Embedding(self.num_classes, self.latent_dim)(label_input)
        label_embedding = tf.keras.layers.Reshape((self.latent_dim,))(label_embedding)
        noise_label = Concatenate()([noise_input, label_embedding])
        x = Dense(128, activation='relu')(noise_label)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(self.input_dim, activation='tanh')(x)
        self.generator = Model([noise_input, label_input], outputs)
        self.generator.compile(optimizer='adam', loss='binary_crossentropy')

    def build_discriminator(self):
        feature_input = Input(shape=(self.input_dim,))
        label_input = Input(shape=(1,))
        label_embedding = Embedding(self.num_classes, self.input_dim)(label_input)
        label_embedding = tf.keras.layers.Flatten()(label_embedding)
        x = Concatenate()([feature_input, label_embedding])
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        self.discriminator = Model([feature_input, label_input], outputs)
        self.discriminator.compile(optimizer='adam', loss='binary_crossentropy')

    def build(self):
        self.build_generator()
        self.build_discriminator()
        from tensorflow.keras.layers import Input
        from tensorflow.keras.models import Model
        self.discriminator.trainable = False
        noise_input = Input(shape=(self.latent_dim,))
        label_input = Input(shape=(1,))
        gan_output = self.discriminator([self.generator([noise_input, label_input]), label_input])
        self.gan = Model([noise_input, label_input], gan_output)
        self.gan.compile(optimizer='adam', loss='binary_crossentropy')

    def train(self, X_train, y_train, epochs=50):
        # Simplified training loop (similar to GAN)
        batch_size = 32
        for epoch in range(epochs):
            idx = np.random.randint(0, len(X_train), batch_size)
            real_X, real_y = X_train[idx], y_train[idx]
            noise = tf.random.normal((batch_size, self.latent_dim))
            fake_y = tf.random.uniform((batch_size,), 0, self.num_classes, dtype=tf.int32)
            fake_X = self.generator([noise, fake_y], training=True)
            self.discriminator.train_on_batch([real_X, real_y], tf.ones((batch_size, 1)))
            self.discriminator.train_on_batch([fake_X, fake_y], tf.zeros((batch_size, 1)))
            self.gan.train_on_batch([noise, fake_y], tf.ones((batch_size, 1)))

    def generate_samples(self, n_samples, class_label):
        noise = tf.random.normal((n_samples, self.latent_dim))
        labels = tf.fill((n_samples,), class_label)
        return self.generator([noise, labels], training=False)

class CGANClassifier:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = None

    def build(self, input_dim, num_classes):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X, y, epochs=30, verbose=0):
        self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=verbose)

    def evaluate(self, X, y):
        y_sparse = np.argmax(y, axis=1) if len(y.shape) > 1 else y.ravel()
        loss, acc = self.model.evaluate(X, y_sparse, verbose=0)
        return {'accuracy': acc}

# Usage same as DCGAN / GAN

