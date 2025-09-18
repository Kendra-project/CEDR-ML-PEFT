import tensorflow as tf
from tensorflow.keras import layers, losses

class AutoEncoder(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        x = tf.cast(x, tf.float32)
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

def create_AE(input_shape):
    encoder = tf.keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Dropout(0.05),
        layers.Dense(8, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.05),
    ])

    decoder = tf.keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=[8]),
        layers.BatchNormalization(),
        layers.Dropout(0.05),
        layers.Dense(input_shape[0])
    ])

    autoencoder = AutoEncoder(encoder, decoder)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    return autoencoder

def detect_anomaly(model, data, reconstruction_threshold=0.6):
    reconstructed = model(data)
    reconstruction_error = tf.reduce_mean(tf.square(reconstructed - data), axis=-1)
    is_anomaly = tf.cast(reconstruction_error > reconstruction_threshold, tf.int32)
    return is_anomaly, reconstruction_error
