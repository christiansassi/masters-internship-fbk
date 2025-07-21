from tensorflow.keras.utils import Sequence # type: ignore
import numpy as np

class SlidingWindowGenerator(Sequence):
    def __init__(self, data, input_indices, output_indices, sensor_indices, batch_size):
        self.data = data
        self.input_indices = input_indices
        self.output_indices = output_indices
        self.sensor_indices = sensor_indices
        self.batch_size = batch_size

    def __len__(self):
        return len(self.input_indices) // self.batch_size

    def __getitem__(self, idx):
        i_start = idx * self.batch_size
        i_end = (idx + 1) * self.batch_size

        batch_input_indices = self.input_indices[i_start:i_end]
        batch_output_indices = self.output_indices[i_start:i_end]

        x = self.data[batch_input_indices]
        y = self.data[batch_output_indices][:, :, self.sensor_indices]
        return x, y
