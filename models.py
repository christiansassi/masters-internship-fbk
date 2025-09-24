import numpy as np

import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def conv1d_output_shape(l_in, kernel_size=1, stride=1, pad=0, dilation=1):
    l_out = np.floor((l_in + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1)
    return int(l_out)

def maxpool1d_output_shape(l_in, kernel_size=1, stride=None, pad=0, dilation=1):
    if stride is None:
        stride = kernel_size
    l_out = np.floor((l_in + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1)
    return int(l_out if l_out != 0 else 1)

class ModelFExtractor(nn.Module):
    def __init__(self, window_size_in, window_size_out, n_devices_in, kernel_size):
        super(ModelFExtractor, self).__init__()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.4)

        # Deep branch
        self.fc13 = nn.Linear(window_size_in, 
                              window_size_in * 3 if window_size_in >= n_devices_in else n_devices_in * 3)
        self.conv = nn.Sequential(
            nn.Conv1d(n_devices_in, 64, kernel_size),
            nn.LeakyReLU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size),
            nn.LeakyReLU(True),
            nn.MaxPool1d(2)
        )
        self.conv_out_channels = 128
        self.maxpool1_out = maxpool1d_output_shape(conv1d_output_shape(
            window_size_in * 3 if window_size_in >= n_devices_in else n_devices_in * 3, kernel_size=kernel_size), kernel_size=2)
        self.maxpool2_out = maxpool1d_output_shape(conv1d_output_shape(
            self.maxpool1_out, kernel_size=kernel_size), kernel_size=2)
        self.out_2 = nn.Linear(self.conv_out_channels, window_size_out)

        # Wide branch
        self.fc20 = nn.Linear(window_size_in, window_size_out)

        # Aggregation
        self.out_h = nn.Linear(self.maxpool2_out + n_devices_in, 80)

    def forward_two(self, x):
        x = self.fc13(x)
        x = self.conv(x)
        x = x.view(x.size(0), self.maxpool2_out, self.conv_out_channels)
        x = self.dropout(x)
        x = self.relu(self.out_2(x))
        return x

    def forward_three(self, x):
        x = self.dropout(self.relu(self.fc20(x)))
        return x

    def forward(self, x_t_1):
        x_t_1 = x_t_1.transpose(2, 1)
        y_t2 = self.forward_two(x_t_1)
        y_t3 = self.forward_three(x_t_1)
        y_t = torch.cat((y_t2, y_t3), dim=1).transpose(2, 1)
        y_t = self.dropout(self.relu(self.out_h(y_t)))
        return y_t

class ModelSensors(nn.Module):
    def __init__(self, n_devices_out):
        super(ModelSensors, self).__init__()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.out_h1 = nn.Linear(80, int(n_devices_out * 2.25))
        self.out_h2 = nn.Linear(int(n_devices_out * 2.25), int(n_devices_out * 1.5))
        self.out = nn.Linear(int(n_devices_out * 1.5), n_devices_out)

    def forward(self, y_t):
        y_t = self.dropout(self.relu(self.out_h1(y_t)))
        y_t = self.dropout(self.relu(self.out_h2(y_t)))
        y_t = self.relu(self.out(y_t))
        return y_t

class PredErrorModel(nn.Module):
    def __init__(self, window_size_in, window_size_out):
        super(PredErrorModel, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 2, 2),
            nn.ReLU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(2, 4, 2),
            nn.ReLU(True),
            nn.MaxPool1d(2),
        )
        self.maxpool1_out = maxpool1d_output_shape(
            conv1d_output_shape(window_size_in, kernel_size=2), kernel_size=2)
        self.maxpool2_out = maxpool1d_output_shape(
            conv1d_output_shape(self.maxpool1_out, kernel_size=2), kernel_size=2)
        self.out_1 = nn.Linear(self.maxpool2_out, 1)

    def forward(self, x_t_1):
        x = x_t_1.transpose(2, 1)
        x = self.conv(x)
        x = self.relu(self.out_1(x))
        return x