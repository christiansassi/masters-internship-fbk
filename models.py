from constants import (
    WIDE_DEEP_NETWORKS,
    THRESHOLD_NETWORKS,
    WINDOW_PAST
)

import config

from os import listdir
from os.path import join

import tensorflow as tf
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.models import load_model # type: ignore

class FeatureExtractor(tf.keras.Model):
    def __init__(self, window_size_in, window_size_out, n_devices_in, kernel_size):
        super().__init__()
        proj_dim = window_size_in * 3 if window_size_in >= n_devices_in else n_devices_in * 3

        self.proj = layers.Dense(proj_dim)
        self.conv1 = layers.Conv1D(64, kernel_size, activation=tf.nn.leaky_relu)
        self.pool1 = layers.MaxPooling1D(2)
        self.conv2 = layers.Conv1D(128, kernel_size, activation=tf.nn.leaky_relu)
        self.pool2 = layers.MaxPooling1D(2)
        self.conv_out = layers.Dense(window_size_out)

        self.wide_fc = layers.Dense(window_size_out)
        self.combine_fc = layers.Dense(80, activation=tf.nn.leaky_relu)
        self.dropout = layers.Dropout(0.4)
        self.window_size_out = window_size_out

    def call(self, x, training=False):
        wide = self.wide_fc(x)
        wide = tf.nn.leaky_relu(wide)
        wide = self.dropout(wide, training=training)

        x_proj = self.proj(x)
        x_conv = self.pool1(self.conv1(x_proj))
        x_conv = self.pool2(self.conv2(x_conv))
        x_conv = self.dropout(x_conv, training=training)
        x_conv = self.conv_out(x_conv)

        x_conv = tf.expand_dims(x_conv, -1)
        x_conv = tf.image.resize(x_conv, [tf.shape(wide)[1], self.window_size_out])
        x_conv = tf.squeeze(x_conv, -1)

        merged = tf.concat([x_conv, wide], axis=-1)
        return self.combine_fc(merged)

class SensorPredictor(tf.keras.Model):

    def __init__(self, n_devices_out):
        super().__init__()
        self.out_h1 = layers.Dense(int(n_devices_out * 2.25), activation=tf.nn.leaky_relu)
        self.out_h2 = layers.Dense(int(n_devices_out * 1.5), activation=tf.nn.leaky_relu)
        self.out = layers.Dense(n_devices_out)
        self.dropout = layers.Dropout(0.4)

    def call(self, x, training=False):
        x = self.dropout(self.out_h1(x), training=training)
        x = self.dropout(self.out_h2(x), training=training)
        return self.out(x)

class WideDeepNetworkDAICS(tf.keras.Model):
    def __init__(self, window_size_in, window_size_out, n_devices_in, n_devices_out, kernel_size):
        super().__init__()

        self.window_size_in = window_size_in
        self.window_size_out = window_size_out
        self.n_devices_in = n_devices_in
        self.n_devices_out = n_devices_out
        self.kernel_size = kernel_size
        
        self.feature_extractor = FeatureExtractor(window_size_in, window_size_out, n_devices_in, kernel_size)
        self.sensor_predictor = SensorPredictor(n_devices_out)
        self.window_size_out = window_size_out

    def call(self, x, training=False):
        x = self.feature_extractor(x, training=training)
        x = self.sensor_predictor(x, training=training)
        return x[:, -self.window_size_out:, :]

    def get_config(self):
        return {
            "window_size_in": self.window_size_in,
            "window_size_out": self.window_size_out,
            "n_devices_in": self.n_devices_in,
            "n_devices_out": self.n_devices_out,
            "kernel_size": self.kernel_size,
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ThresholdNetworkDAICS(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv1D(2, kernel_size=2, activation="relu")
        self.pool1 = layers.MaxPooling1D(pool_size=2)
        self.conv2 = layers.Conv1D(4, kernel_size=2, activation="relu")
        self.pool2 = layers.MaxPooling1D(pool_size=2)
        self.flatten = layers.Flatten()
        self.out = layers.Dense(1)

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        return self.out(x)

def clone_wide_deep_networks(
    wide_deep_networks: list[WideDeepNetworkDAICS],
    optimizer: tf.keras.optimizers.Optimizer,
    loss: str
) -> list[WideDeepNetworkDAICS]:

    cloned_models = []

    for original_model in wide_deep_networks:
        
        window_size_in = original_model.window_size_in
        window_size_out = original_model.window_size_out
        n_devices_in = original_model.n_devices_in
        n_devices_out = original_model.n_devices_out
        kernel_size = original_model.kernel_size

        if not original_model.built:
            dummy_input = tf.zeros((1, window_size_in, n_devices_in))
            _ = original_model(dummy_input, training=False)

        original_weights = original_model.get_weights()

        cloned_model = WideDeepNetworkDAICS(
            window_size_in=window_size_in,
            window_size_out=window_size_out,
            n_devices_in=n_devices_in,
            n_devices_out=n_devices_out,
            kernel_size=kernel_size
        )

        dummy_input = tf.zeros((1, window_size_in, n_devices_in))
        _ = cloned_model(dummy_input, training=False)

        cloned_model.set_weights(original_weights)
        cloned_model.compile(optimizer=optimizer, loss=loss)

        cloned_models.append(cloned_model)

    return cloned_models

def load_wide_deep_networks() -> list[WideDeepNetworkDAICS]:

    wide_deep_networks = []

    for wide_deep_network in listdir(WIDE_DEEP_NETWORKS):
        model = load_model(join(WIDE_DEEP_NETWORKS, wide_deep_network), custom_objects={"WideDeepNetworkDAICS": WideDeepNetworkDAICS})
        wide_deep_networks.append(model)
    
    return wide_deep_networks

def clone_threshold_networks(
    threshold_networks: list[ThresholdNetworkDAICS],
    optimizer: tf.keras.optimizers.Optimizer,
    loss: str
) -> list[ThresholdNetworkDAICS]:

    cloned_models = []

    for original_model in threshold_networks:

        if not original_model.built:
            dummy_input = tf.zeros((1, WINDOW_PAST, 1))
            _ = original_model(dummy_input)

        original_weights = original_model.get_weights()

        cloned_model = ThresholdNetworkDAICS()

        dummy_input = tf.zeros((1, WINDOW_PAST, 1))
        _ = cloned_model(dummy_input)

        cloned_model.set_weights(original_weights)

        cloned_model.compile(optimizer=optimizer, loss=loss)

        cloned_models.append(cloned_model)

    return cloned_models

def load_threshold_networks() -> list[ThresholdNetworkDAICS]:

    threshold_networks = []

    for threshold_network in listdir(THRESHOLD_NETWORKS):
        model = load_model(join(THRESHOLD_NETWORKS, threshold_network), custom_objects={"ThresholdNetworkDAICS": ThresholdNetworkDAICS})
        threshold_networks.append(model)
    
    return threshold_networks