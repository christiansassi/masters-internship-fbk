from constants import *

import numpy as np

from tensorflow.keras.utils import Sequence  # type: ignore

class SlidingWindowGenerator(Sequence):
    def __init__(self, x, y, labels, input_indices, output_indices, batch_size, outputs, shuffle: bool = True):

        self.x = x
        self.y = y
        self.labels = labels

        self.input_indices = input_indices
        self.output_indices = output_indices

        self.batch_size = batch_size

        self.outputs = outputs

        self.indices = np.arange(len(input_indices))

        self.shuffle = shuffle

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx: int):

        batch_ids = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        input_idx = self.input_indices[batch_ids]
        output_idx = self.output_indices[batch_ids]

        x = self.x[input_idx.flatten()].reshape(
            len(batch_ids), input_idx.shape[1], self.x.shape[1]
        )

        y_local = self.y[output_idx.flatten()].reshape(
            len(batch_ids), output_idx.shape[1], self.y.shape[1]
        )

        y_global = np.zeros((len(batch_ids), output_idx.shape[1], len(GLOBAL_OUTPUTS)), dtype=np.float32)

        for local_index, local_output in enumerate(self.outputs):
            global_index = GLOBAL_OUTPUTS.index(local_output)
            y_global[:, :, global_index] = y_local[:, :, local_index]

        return x, y_global
    
    def get_item_with_label(self, idx):

        batch_ids = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        input_idx = self.input_indices[batch_ids]
        output_idx = self.output_indices[batch_ids]

        x = self.x[input_idx.flatten()].reshape(
            len(batch_ids), input_idx.shape[1], self.x.shape[1]
        )

        y_local = self.y[output_idx.flatten()].reshape(
            len(batch_ids), output_idx.shape[1], self.y.shape[1]
        )

        y_global = np.zeros((len(batch_ids), output_idx.shape[1], len(GLOBAL_OUTPUTS)), dtype=np.float32)

        for local_index, local_output in enumerate(self.outputs):
            global_index = GLOBAL_OUTPUTS.index(local_output)
            y_global[:, :, global_index] = y_local[:, :, local_index]

        labels = self.labels[output_idx.flatten()].reshape(
            len(batch_ids), output_idx.shape[1]
        )

        return x, y_global, labels

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indices)
