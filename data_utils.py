from constants import *

import numpy as np

import torch
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, df_train, train_input_indices, train_output_indices, input_mask, output_mask):
        self.df_train = df_train
        self.train_input_indices = train_input_indices
        self.train_output_indices = train_output_indices
        self.input_mask = input_mask
        self.output_mask = output_mask

    def __len__(self):
        return len(self.train_input_indices)

    def __getitem__(self, idx):

        # indices for this window
        in_idx = self.train_input_indices[idx].flatten()
        out_idx = self.train_output_indices[idx].flatten()

        # Input window
        df_in = self.df_train[in_idx]
        w_in = np.zeros((WINDOW_PAST, len(GLOBAL_INPUTS)), dtype=np.float32)
        w_in[:, self.input_mask] = df_in

        # Output window
        df_out = self.df_train[out_idx][:, self.output_mask]

        return torch.from_numpy(w_in), torch.from_numpy(df_out.astype(np.float32))
