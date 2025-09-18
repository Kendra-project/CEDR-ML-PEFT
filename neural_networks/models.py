import numpy as np
import csv
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import re
from numpy import random
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LeakyReLU

def build_vae_encoder(input_dim, intermediate_dim, latent_dim):
    inputs = layers.Input(shape=(input_dim,), name='encoder_input')
    x = layers.Dense(intermediate_dim, activation='relu')(inputs)

    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    def sample(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sample, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    return Model(inputs, [z_mean, z_log_var, z], name='encoder')

def build_vae_decoder(latent_dim, intermediate_dim, original_dim):
    latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = layers.Dense(original_dim, activation='sigmoid')(x)
    return Model(latent_inputs, outputs, name='decoder')

def create_vae(input_dim, intermediate_dim, latent_dim):
    encoder = build_vae_encoder(input_dim, intermediate_dim, latent_dim)
    decoder = build_vae_decoder(latent_dim, intermediate_dim, input_dim)

    x = layers.Input(shape=(input_dim,), name='encoder_input')
    z_mean, z_log_var, z = encoder(x)
    x_decoded_mean = decoder(z)

    reconstruction_loss = K.sum(K.square(x - x_decoded_mean))
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)

    vae = Model(x, x_decoded_mean)
    vae.add_loss(vae_loss)
   
    return vae



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

def detect_anomaly(model, data,reconstruction_threshold = 0.6):
    reconstructed = model(data)

    reconstruction_error = tf.reduce_mean(tf.square(reconstructed - data),axis=-1) 
    
    is_anomaly = tf.cast(reconstruction_error > reconstruction_threshold, tf.int32)
    return is_anomaly,reconstruction_error

def create_AE(input_shape):
    encoder = tf.keras.Sequential([
       # tf.keras.layers.Dense(32, input_shape=input_shape),
       # tf.keras.layers.BatchNormalization(),
       # tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(16, activation='relu',input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.05),       
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.05),
        #tf.keras.layers.Dense(4, activation='relu'),
        #tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.Dropout(0.05),
        #tf.keras.layers.Dense(4, activation='relu'),       
    ])

    decoder = tf.keras.Sequential([
        #tf.keras.layers.Dense(8, activation='relu', input_shape=[4]),
        #tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(16, activation='relu',input_shape=[8]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.05),
        #tf.keras.layers.Dense(32, activation='relu'),
        #tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.Dropout(0.05),       
        tf.keras.layers.Dense(input_shape[0])
    ])
    autoencoder = AutoEncoder(encoder, decoder)

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    #tf.keras.utils.plot_model(decoder, "decoder.svg", show_shapes=True, show_layer_names=True, expand_nested=True,show_layer_activations=True)
    #tf.keras.utils.plot_model(encoder, "encoder.svg", show_shapes=True, show_layer_names=True, expand_nested=True,show_layer_activations=True)
    return autoencoder


def create_AE_HLS4ML(in_layer):
    input_layer = tf.keras.Input(shape=in_layer)
    x = layers.Dense(16, activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.05)(x)
    x = layers.Dense(8, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.05)(x)

    # Decoder

    x = layers.Dense(16, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.05)(x)
    x = layers.Dense(in_layer[0])(x)
    out_layer = x
    # Create Model
    autoencoder = Model(inputs=in_layer, outputs=out_layer)

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    #tf.keras.utils.plot_model(decoder, "decoder.svg", show_shapes=True, show_layer_names=True, expand_nested=True,show_layer_activations=True)
    #tf.keras.utils.plot_model(encoder, "encoder.svg", show_shapes=True, show_layer_names=True, expand_nested=True,show_layer_activations=True)
    return autoencoder

# expected results for zcu102

