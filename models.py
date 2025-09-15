from constants import *

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

# ===========================================
# Wide & Deep Network
# ===========================================

class WideDeepNetworkDAICS(keras.Model):
    def __init__(
            self, 
            window_past: int, 
            window_present: int,
            n_inputs: int, 
            n_outputs: int,
            conv_filters: int = 64, 
            conv_kernel: int = KERNEL_SIZE,
            hidden_units: int = 80, 
            dropout: float = 0.4
        ):

        super().__init__()

        self.window_past = window_past
        self.window_present = window_present
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # Client-specific mask (set externally)
        self.mask_out = None

        # Wide branch
        self.fc_wide = layers.Dense(window_present)

        # Deep branch
        self.conv1 = layers.Conv1D(conv_filters, conv_kernel, activation="leaky_relu")
        self.pool1 = layers.MaxPool1D(2)
        self.conv2 = layers.Conv1D(conv_filters * 2, conv_kernel, activation="leaky_relu")
        self.pool2 = layers.MaxPool1D(2)
        self.dropout = layers.Dropout(dropout)

        # Combine
        self.fc_combine = layers.Dense(hidden_units, activation="leaky_relu")

        # Head
        self.fc_out1 = layers.Dense(int(n_outputs * 2.25), activation="leaky_relu")
        self.fc_out2 = layers.Dense(int(n_outputs * 1.5), activation="leaky_relu")
        self.out = layers.Dense(window_present * n_outputs, activation="linear")

    def set_mask(self, mask_out: np.ndarray, window_present: int):
        mask = np.zeros((1, window_present, self.n_outputs), dtype=np.float32)

        for index in mask_out:
            mask[:, :, index] = 1.0

        self.mask_out = tf.constant(mask, dtype=tf.float32)

    def call(self, inputs, training=False):
        x = inputs  # (B, W_in, n_inputs)

        # Wide
        wide = self.fc_wide(tf.transpose(x, perm=[0, 2, 1]))

        # Deep
        d = self.conv1(x)
        d = self.pool1(d)
        d = self.conv2(d)
        d = self.pool2(d)
        d = self.dropout(d, training=training)
        d = tf.reduce_mean(d, axis=1)

        # Combine
        wide_mean = tf.reduce_mean(wide, axis=1)
        fused = tf.concat([wide_mean, d], axis=-1)
        fused = self.fc_combine(fused)

        # Head
        y = self.fc_out1(fused)
        y = self.dropout(y, training=training)
        y = self.fc_out2(y)
        y = self.dropout(y, training=training)
        out = self.out(y)
        out = tf.reshape(out, (-1, self.window_present, self.n_outputs))

        if self.mask_out is not None:
            out = out * self.mask_out

        return out

    def clone(self):

        clone = WideDeepNetworkDAICS(
            self.window_past, self.window_present,
            self.n_inputs, self.n_outputs
        )

        # Build weights
        clone.build(input_shape=(None, self.window_past, self.n_inputs))

        # Copy weights
        clone.set_weights(self.get_weights())

        # Copy mask if present
        if self.mask_out is not None:
            clone.mask_out = tf.identity(self.mask_out)

        # Compile
        clone.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
            loss=LOSS
        )

        return clone
    
    def save(self, *args, **kwargs):

        dummy = tf.zeros((1, self.window_past, self.n_inputs))
        _ = self(dummy) 

        super().save(*args, **kwargs)

    def get_config(self):
        return {
            "window_past": self.window_past,
            "window_present": self.window_present,
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ===========================================
# Threshold Network
# ===========================================

class ThresholdNetworkDAICS(keras.Model):
    def __init__(
            self, 
            window_past: int, 
            window_present: int
        ):

        super().__init__()

        self.window_past = window_past
        self.window_present = window_present
        self.n_inputs = 1

        self.conv1 = layers.Conv1D(2, 2, activation="relu", input_shape=(window_past, 1))
        self.pool1 = layers.MaxPool1D(2)
        self.conv2 = layers.Conv1D(4, 2, activation="relu")
        self.pool2 = layers.MaxPool1D(2)

        self.flatten = layers.Flatten()
        self.fc_out = layers.Dense(1, activation="relu")

        self.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.01),
            loss="mse"
        )

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        out = self.fc_out(x)
        return out

    def clone(self):
        clone = ThresholdNetworkDAICS(self.window_past, self.window_present)

        # Build weights
        clone.build(input_shape=(None, self.window_past, self.n_inputs))

        # Copy weights
        clone.set_weights(self.get_weights())

        # Compile
        clone.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
            loss=LOSS
        )

        return clone
    
    def save(self, *args, **kwargs):
        
        dummy = tf.zeros((1, self.window_past, self.n_inputs))
        _ = self(dummy) 

        super().save(*args, **kwargs)

    def get_config(self):
        return {
            "window_past": self.window_past,
            "window_present": self.window_present,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
