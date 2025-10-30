from config import *
from constants import *
from models import ModelFExtractor, ModelSensors, PredErrorModel
import utils

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import scipy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import h5py
import numpy as np

from collections import OrderedDict, deque

from copy import deepcopy

from time import time

import uuid

import matplotlib.pyplot as plt

class Client:
    def __init__(
        self,

        client_id: str,

        df_train: np.ndarray,
        df_val: np.ndarray,
        df_test: np.ndarray,
        df_real: np.ndarray,

        train_input_indices: np.ndarray,
        train_output_indices: np.ndarray,

        val_input_indices: np.ndarray,
        val_output_indices: np.ndarray,

        test_input_indices: np.ndarray,
        test_output_indices: np.ndarray,

        real_input_indices: np.ndarray,
        real_output_indices: np.ndarray,

        inputs: list[str],
        outputs: list[str],

        model_f_extractor: ModelFExtractor = None,
        model_sensor: ModelSensors = None,
        pred_error_model: PredErrorModel = None
    ):
        self.id = client_id

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.df_real = df_real

        self.all_labels = self.df_real[:, -1].astype(int)

        self.num_of_samples = len(self.df_train) + len(self.df_val) + len(self.df_test)

        self.df_train = self.df_train[:, :-1]
        self.df_val = self.df_val[:, :-1]
        self.df_test = self.df_test[:, :-1]
        self.df_real = self.df_real[:, :-1]

        self.train_input_indices = train_input_indices
        self.train_output_indices = train_output_indices

        self.val_input_indices = val_input_indices
        self.val_output_indices = val_output_indices

        self.test_input_indices = test_input_indices
        self.test_output_indices = test_output_indices

        self.real_input_indices = real_input_indices
        self.real_output_indices = real_output_indices

        self.inputs = inputs
        self.outputs = outputs

        self.model_f_extractor = ModelFExtractor(
            window_size_in=WINDOW_PAST, 
            window_size_out=WINDOW_PRESENT, 
            n_devices_in=len(GLOBAL_INPUTS), 
            kernel_size=KERNEL_SIZE
        ) if model_f_extractor is None else deepcopy(model_f_extractor)

        self.model_sensor = ModelSensors(
            n_devices_out=len(self.outputs)
        ) if model_sensor is None else deepcopy(model_sensor)

        self.pred_error_model = PredErrorModel(
            window_size_in=WINDOW_PAST, 
            window_size_out=WINDOW_PRESENT
        ) if pred_error_model is None else deepcopy(pred_error_model)

        self.optimizer = torch.optim.SGD(list(self.model_f_extractor.parameters()) + list(self.model_sensor.parameters()), lr=LEARNING_RATE, momentum=MOMENTUM)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=DAICS_PATIENCE)
        self.criterion = nn.MSELoss()

        self.epochs = 0
        self.steps = 0
        self.score = float("-inf")

        self.input_mask = [list(GLOBAL_INPUTS).index(x) for x in self.inputs]
        self.output_mask = [list(self.inputs).index(item) for item in self.outputs]

        self.val_mask = torch.zeros(BATCH_SIZE, WINDOW_PAST, len(GLOBAL_INPUTS))
        self.val_mask[:, :, self.input_mask] = 1
        self.val_mask = self.val_mask.to(DEVICE)

        self.eval_mask = torch.zeros(BATCH_SIZE, WINDOW_PAST, len(GLOBAL_INPUTS))
        self.eval_mask[:, :, self.input_mask] = 1
        self.eval_mask = self.eval_mask.to(DEVICE)

        self.real_mask = torch.zeros(BATCH_SIZE, WINDOW_PAST, len(GLOBAL_INPUTS))
        self.real_mask[:, :, self.input_mask] = 1
        self.real_mask = self.real_mask.to(DEVICE)

        self.log = lambda msg, end="\n", verbose=False: print(f"{utils.log_timestamp_status()} {msg}", end=end) if verbose else None

    def __str__(self) -> str:
        return self.id

    def train_model_f_extractor_and_sensor(self, model_f_extractor: ModelFExtractor, verbose: bool = False) -> tuple:

        self.model_f_extractor.load_state_dict(model_f_extractor.state_dict())
        
        self.model_f_extractor.to(DEVICE)
        self.model_sensor.to(DEVICE)

        min_train_loss = float("inf")
        min_val_loss = float("inf")

        best_model_f_extractor = None
        best_model_sensor = None

        # # Calculate batch size based on hardware specs
        # batch_size = utils.get_safe_batch_size(model=self.model_f_extractor, sample=np.zeros((1, WINDOW_PAST, len(GLOBAL_INPUTS)), dtype=np.float32))

        # # Calculate data len based on batch_size
        # train_input_indices = self.train_input_indices[:(len(self.train_input_indices) // batch_size) * batch_size]
        # train_output_indices = self.train_output_indices[:(len(self.train_input_indices) // batch_size) * batch_size]

        # # Calculate steps
        # steps = len(train_input_indices) // batch_size

        # # Choose the max steps -> less memory usage
        # self.steps = max(self.steps, steps)

        # # Upper limit for steps
        # self.steps = min(self.steps, MAX_STEPS)

        # Calculate batch_size
        batch_size = len(self.train_input_indices) // self.steps
        batch_size = min(batch_size, WIDE_DEEP_MAX_BATCH_SIZE)

        # Recalculate steps
        self.steps = len(self.train_input_indices) // batch_size
        self.steps = min(self.steps, MAX_STEPS)

        # Recalculate batch_size
        batch_size = len(self.train_input_indices) // self.steps

        # Calculate indices
        train_input_indices = self.train_input_indices[:(len(self.train_input_indices) // batch_size) * batch_size]
        train_output_indices = self.train_output_indices[:(len(self.train_input_indices) // batch_size) * batch_size]

        train_mask = torch.zeros(batch_size, WINDOW_PAST, len(GLOBAL_INPUTS))
        train_mask[:, :, self.input_mask] = 1
        train_mask = train_mask.to(DEVICE)

        for epoch in range(self.epochs):

            # Training
            self.model_f_extractor.train()
            self.model_sensor.train()

            train_loss = 0

            steps = len(train_input_indices) // batch_size

            for step, batch_index in enumerate(np.random.permutation(range(0, steps)), start=1):

                df_in = self.df_train[train_input_indices[batch_index * batch_size: batch_index * batch_size + batch_size].flatten()]
                df_out = self.df_train[train_output_indices[batch_index * batch_size: batch_index * batch_size + batch_size].flatten()][:, self.output_mask]

                # Input
                w_in = np.zeros((len(df_in), len(GLOBAL_INPUTS)), dtype=np.float32)
                w_in[:, self.input_mask] = df_in

                w_in = w_in.reshape(batch_size, WINDOW_PAST, -1)
                w_in = torch.from_numpy(w_in).float().to(DEVICE)

                # Output
                w_out = df_out.reshape(batch_size, WINDOW_PRESENT, -1)
                w_out = torch.from_numpy(w_out).float().to(DEVICE)

                # Reset gradients
                self.optimizer.zero_grad()

                # Forward pass through the feature extractor
                x = self.model_f_extractor(w_in, train_mask)

                # Forward pass through the sensor head
                y = self.model_sensor(x)

                # Compute loss
                loss = self.criterion(y, w_out)

                # One SGD step
                loss.backward()
                self.optimizer.step()

                train_loss = train_loss + loss.item()

                self.log(" " * 100, end="\r", verbose=verbose)
                self.log(f"Epoch: {epoch + 1} / {self.epochs} | Step: {step} / {steps} | Training loss {train_loss / step}", end="\r", verbose=verbose)

            train_loss = train_loss / steps

            min_train_loss = min(min_train_loss, train_loss)

            # Validation
            self.model_f_extractor.eval()
            self.model_sensor.eval()

            val_loss = 0

            steps = len(self.val_input_indices) // BATCH_SIZE

            with torch.no_grad():

                for step, batch_index in enumerate(np.random.permutation(range(0, steps)), start=1):
                    
                    df_in = self.df_val[self.val_input_indices[batch_index * BATCH_SIZE: batch_index * BATCH_SIZE + BATCH_SIZE].flatten()]
                    df_out = self.df_val[self.val_output_indices[batch_index * BATCH_SIZE: batch_index * BATCH_SIZE + BATCH_SIZE].flatten()][:, self.output_mask]

                    # Input
                    w_in = np.zeros((len(df_in), len(GLOBAL_INPUTS)), dtype=np.float32)
                    w_in[:, self.input_mask] = df_in

                    w_in = w_in.reshape(BATCH_SIZE, WINDOW_PAST, -1)
                    w_in = torch.from_numpy(w_in).float().to(DEVICE)

                    # Output
                    w_out = df_out.reshape(BATCH_SIZE, WINDOW_PRESENT, -1)
                    w_out = torch.from_numpy(w_out).float().to(DEVICE)

                    # Forward pass through the feature extractor
                    x = self.model_f_extractor(w_in, self.val_mask)

                    # Forward pass through the sensor head
                    y = self.model_sensor(x)

                    # Compute loss
                    loss = self.criterion(y, w_out)

                    val_loss = val_loss + loss.item()

                    self.log(" " * 100, end="\r", verbose=verbose)
                    self.log(f"Epoch: {epoch + 1} / {self.epochs} | Step: {step} / {steps} | Validation loss: {val_loss / step}", end="\r", verbose=verbose)

            val_loss = val_loss / steps

            # Save best models
            if val_loss < min_val_loss:
                min_val_loss = val_loss

                best_model_f_extractor = self.model_f_extractor.state_dict()
                best_model_sensor = self.model_sensor.state_dict()

            # Decay Learning Rate, pass validation loss for tracking at every epoch
            self.scheduler.step(val_loss)

        self.model_f_extractor.load_state_dict(best_model_f_extractor)
        self.model_sensor.load_state_dict(best_model_sensor)

        self.log(" " * 100, end="\r", verbose=verbose)
        self.log(f"Training loss: {min_train_loss} | Validation loss: {min_val_loss}", verbose=verbose)
        
        return -min_train_loss, -min_val_loss

    def eval_model_f_extractor_and_sensor(self, model_f_extractor: ModelFExtractor, verbose: bool = False) -> float:
        
        self.model_f_extractor.load_state_dict(model_f_extractor.state_dict())

        self.model_f_extractor.to(DEVICE)
        self.model_sensor.to(DEVICE)

        # Evaluation
        self.model_f_extractor.eval()
        self.model_sensor.eval()

        eval_loss = 0

        steps = len(self.test_input_indices) // BATCH_SIZE

        with torch.no_grad():
            
            for step, batch_index in enumerate(np.random.permutation(range(0, steps)), start=1):
                
                df_in = self.df_test[self.test_input_indices[batch_index * BATCH_SIZE: batch_index * BATCH_SIZE + BATCH_SIZE].flatten()]
                df_out = self.df_test[self.test_output_indices[batch_index * BATCH_SIZE: batch_index * BATCH_SIZE + BATCH_SIZE].flatten()][:, self.output_mask]

                # Input
                w_in = np.zeros((len(df_in), len(GLOBAL_INPUTS)), dtype=np.float32)
                w_in[:, self.input_mask] = df_in

                w_in = w_in.reshape(BATCH_SIZE, WINDOW_PAST, -1)
                w_in = torch.from_numpy(w_in).float().to(DEVICE)

                # Output
                w_out = df_out.reshape(BATCH_SIZE, WINDOW_PRESENT, -1)
                w_out = torch.from_numpy(w_out).float().to(DEVICE)

                # Forward pass through the feature extractor
                x = self.model_f_extractor(w_in, self.eval_mask)

                # Forward pass through the sensor head
                y = self.model_sensor(x)

                # Compute loss
                loss = self.criterion(y, w_out)

                eval_loss = eval_loss + loss.item()

                self.log(" " * 100, end="\r", verbose=verbose)
                self.log(f"Step: {step} / {steps} | Evaluation loss: {eval_loss / step}", end="\r", verbose=verbose)
                
        eval_loss = eval_loss / steps

        self.score = -eval_loss

        self.log(" " * 100, end="\r", verbose=verbose)
        self.log(f"Evaluation loss: {eval_loss}", verbose=verbose)

        return -eval_loss

    def train_pred_error_model(self, verbose: bool = False):
        
        self.model_f_extractor.to(DEVICE)
        self.model_sensor.to(DEVICE)

        self.model_f_extractor.eval()
        self.model_sensor.eval()

        criterion = nn.MSELoss(reduction="none")

        def calculate_errors(df, input_indices, output_indices):

            train_input_indices = input_indices[:(len(input_indices) // BATCH_SIZE) * BATCH_SIZE]
            train_output_indices = output_indices[:(len(output_indices) // BATCH_SIZE) * BATCH_SIZE]

            train_mask = torch.zeros(BATCH_SIZE, WINDOW_PAST, len(GLOBAL_INPUTS))
            train_mask[:, :, self.input_mask] = 1
            train_mask = train_mask.to(DEVICE)

            all_predicted = np.zeros(len(df), dtype=np.float32)

            steps = len(train_input_indices) // BATCH_SIZE

            for step in range(0, steps):

                df_in = df[train_input_indices[step * BATCH_SIZE: step * BATCH_SIZE + BATCH_SIZE].flatten()]
                df_out = df[train_output_indices[step * BATCH_SIZE: step * BATCH_SIZE + BATCH_SIZE].flatten()][:, self.output_mask]

                # Input
                w_in = np.zeros((len(df_in), len(GLOBAL_INPUTS)), dtype=np.float32)
                w_in[:, self.input_mask] = df_in

                w_in = w_in.reshape(BATCH_SIZE, WINDOW_PAST, -1)
                w_in = torch.from_numpy(w_in).float().to(DEVICE)

                # Output
                w_out = df_out.reshape(BATCH_SIZE, WINDOW_PRESENT, -1)
                w_out = torch.from_numpy(w_out).float().to(DEVICE)

                # Forward pass through the feature extractor
                x = self.model_f_extractor(w_in, self.eval_mask)

                # Forward pass through the sensor head
                y = self.model_sensor(x)

                # Compute loss
                loss = torch.mean(criterion(y, w_out), dim=2)

                all_predicted[train_input_indices[step * BATCH_SIZE: step * BATCH_SIZE + BATCH_SIZE, 0]] =  loss.detach().cpu().numpy()[:, 0]

                self.log(" " * 100, end="\r", verbose=verbose)
                self.log(f"Calculating errors {step + 1} / {steps}", end="\r", verbose=verbose)
            
            return all_predicted

        all_predicted_val = calculate_errors(df=self.df_val, input_indices=self.val_input_indices, output_indices=self.val_output_indices)
        all_predicted_test = calculate_errors(df=self.df_test, input_indices=self.test_input_indices, output_indices=self.test_output_indices)
        all_predicted = np.concatenate((all_predicted_val, all_predicted_test))

        # Craft input and output for the threshold model
        all_predicted = np.trim_zeros(all_predicted) if not np.all(all_predicted == 0) else all_predicted
        all_predicted = scipy.signal.medfilt(all_predicted, kernel_size=MED_FILTER_LAG)

        input_indices = (np.arange(SAMPLING_START, len(all_predicted), VAL_STEP) - HORIZON - WINDOW_PRESENT)[:, None] - np.arange(1, WINDOW_PAST + 1)
        input_indices = np.sort(input_indices)
        input_indices = input_indices[: (len(input_indices) // BATCH_SIZE) * BATCH_SIZE, :]

        output_indices = np.arange(SAMPLING_START, len(all_predicted), VAL_STEP)[:, None] - np.arange(1, WINDOW_PRESENT + 1)
        output_indices = np.sort(output_indices)
        output_indices = output_indices[: (len(output_indices) // BATCH_SIZE) * BATCH_SIZE, :]

        self.pred_error_model.to(DEVICE)
        
        optimizer = torch.optim.SGD(self.pred_error_model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        min_train_loss = float("inf")

        best_pred_error_model = None

        for epoch in range(THRESHOLD_EPOCHS):

            self.pred_error_model.train()

            train_loss = 0

            steps = len(input_indices) // BATCH_SIZE

            for step, batch_index in enumerate(np.random.permutation(range(0, steps)), start=1):

                df_in = all_predicted[input_indices[batch_index * BATCH_SIZE: batch_index * BATCH_SIZE + BATCH_SIZE]]
                df_out = all_predicted[output_indices[batch_index * BATCH_SIZE: batch_index * BATCH_SIZE + BATCH_SIZE]]

                # Input and output
                w_in = torch.from_numpy(df_in[:, :, None]).float().to(DEVICE)
                w_out = torch.from_numpy(df_out[:, :, None]).float().to(DEVICE)

                optimizer.zero_grad()

                y = self.pred_error_model(w_in.float().to(DEVICE)).abs()

                if torch.all(y == 0).item():
                    optimizer.zero_grad()

                    self.pred_error_model.apply(lambda model: model.reset_parameters() if isinstance(model, nn.Conv1d) or isinstance(model, nn.Linear) else None)

                    y = self.pred_error_model(w_in.float().to(DEVICE)).abs()
                
                # Compute loss
                loss = criterion(y, w_out)

                # One SGD step
                loss.backward(retain_graph=True)
                optimizer.step()

                train_loss = train_loss + loss.detach().cpu().numpy()
                
                self.log(" " * 100, end="\r", verbose=verbose)
                self.log(f"Epoch: {epoch + 1} / {THRESHOLD_EPOCHS} | Step: {step} / {steps} | Training loss: {train_loss / step}", end="\r", verbose=verbose)

            train_loss = train_loss / steps

            # Save best model
            if train_loss < min_train_loss:
                min_train_loss = train_loss

                best_pred_error_model = self.pred_error_model.state_dict()
        
        self.pred_error_model.load_state_dict(best_pred_error_model)

        self.log(" " * 100, end="\r", verbose=verbose)
        self.log(f"Training loss: {min_train_loss}", verbose=verbose)

        return -min_train_loss

    def calculate_threshold_base(self):

        criterion = nn.MSELoss(reduction="none")

        losses = []

        with torch.no_grad():

            steps = len(self.val_input_indices) // BATCH_SIZE

            for step in range(0, steps):

                df_in = self.df_val[self.val_input_indices[step * BATCH_SIZE: step * BATCH_SIZE + BATCH_SIZE].flatten()]
                df_out = self.df_val[self.val_output_indices[step * BATCH_SIZE: step * BATCH_SIZE + BATCH_SIZE].flatten()][:, self.output_mask]

                # Input
                w_in = np.zeros((len(df_in), len(GLOBAL_INPUTS)), dtype=np.float32)
                w_in[:, self.input_mask] = df_in

                w_in = w_in.reshape(BATCH_SIZE, WINDOW_PAST, -1)
                w_in = torch.from_numpy(w_in).float().to(DEVICE)

                # Output
                w_out = df_out.reshape(BATCH_SIZE, WINDOW_PRESENT, -1)
                w_out = torch.from_numpy(w_out).float().to(DEVICE)

                # Forward pass through the feature extractor
                x = self.model_f_extractor(w_in, self.val_mask)

                # Forward pass through the sensor head
                y = self.model_sensor(x)

                # Compute loss
                loss = torch.mean(criterion(y, w_out), dim=2)

                losses.extend(scipy.signal.medfilt(loss.detach().cpu().numpy()[:, 0].flatten(), kernel_size=MED_FILTER_LAG))
            
            steps = len(self.test_input_indices) // BATCH_SIZE

            for step in range(0, steps):

                df_in = self.df_test[self.test_input_indices[step * BATCH_SIZE: step * BATCH_SIZE + BATCH_SIZE].flatten()]
                df_out = self.df_test[self.test_output_indices[step * BATCH_SIZE: step * BATCH_SIZE + BATCH_SIZE].flatten()][:, self.output_mask]

                # Input
                w_in = np.zeros((len(df_in), len(GLOBAL_INPUTS)), dtype=np.float32)
                w_in[:, self.input_mask] = df_in

                w_in = w_in.reshape(BATCH_SIZE, WINDOW_PAST, -1)
                w_in = torch.from_numpy(w_in).float().to(DEVICE)

                # Output
                w_out = df_out.reshape(BATCH_SIZE, WINDOW_PRESENT, -1)
                w_out = torch.from_numpy(w_out).float().to(DEVICE)

                # Forward pass through the feature extractor
                x = self.model_f_extractor(w_in, self.eval_mask)

                # Forward pass through the sensor head
                y = self.model_sensor(x)

                # Compute loss
                loss = torch.mean(criterion(y, w_out), dim=2)

                losses.extend(scipy.signal.medfilt(loss.detach().cpu().numpy()[:, 0].flatten(), kernel_size=MED_FILTER_LAG))

        err_mean = np.mean(losses)
        err_std = np.std(losses)

        self.threshold_base = err_mean + err_std
        
        return losses

    def test(self, verbose: bool = False):
        
        self.threshold_base = 0
        # self.calculate_threshold_base()

        self.model_f_extractor.to(DEVICE)
        self.model_sensor.to(DEVICE)
        self.pred_error_model.to(DEVICE)

        self.model_f_extractor.eval()

        optimizer = torch.optim.SGD(self.pred_error_model.parameters(), lr=LEARNING_RATE)
        criterion1 = nn.MSELoss(reduction="none")
        criterion2 = nn.MSELoss()
        criterion3 = nn.MSELoss()

        # Results storage
        all_predicted_sen = np.zeros(len(self.df_real), dtype=float)
        human_idx_sen = np.zeros(len(self.df_real), dtype=float)
        all_threshold_sen = np.zeros(len(self.df_real), dtype=float)
        thresholds_sen = np.zeros(len(self.df_real), dtype=float)

        for name, param in self.model_sensor.named_children():
            for p in param.parameters():
                p.requires_grad = True

        # Calculate the impact range. The attack goes from start to end. The impact range from end + (start - end)
        attack_indices = np.where(self.all_labels == 1)[0]
        attack_indices = np.split(attack_indices, np.where(np.diff(attack_indices) != 1)[0] + 1)
        attack_indices = [(sub[0], sub[-1]) for sub in attack_indices]

        attack_impact_array = np.array([], dtype=int)
    
        for start, end in attack_indices:
            attack_impact_array = np.concatenate((attack_impact_array, np.arange(end, end + (end - start))))

        attack_impact_array = np.unique(attack_impact_array)

        all_labels_threshold = np.copy(self.all_labels).astype(int)
        all_labels_threshold[attack_impact_array] = 1

        human_inter_counter = 0

        steps = len(self.real_input_indices) // BATCH_SIZE

        before = time()

        threshold_points = deque(maxlen=128)
        error_points = deque(maxlen=128)
        colors = deque(maxlen=128)

        plt.ion()

        fig, ax = plt.subplots()

        threshold_line, = ax.plot([], [], label="Threshold")
        error_scatter = ax.scatter([], [], label="Error", s=20)

        for step in range(0, steps):

            apply_start = self.real_output_indices[step * BATCH_SIZE: step * BATCH_SIZE + BATCH_SIZE][0, 0]
            apply_end = self.real_output_indices[step * BATCH_SIZE: step * BATCH_SIZE + BATCH_SIZE][-1, 0]

            df_in = self.df_real[self.real_input_indices[step * BATCH_SIZE: step * BATCH_SIZE + BATCH_SIZE].flatten()]
            df_out = self.df_real[self.real_output_indices[step * BATCH_SIZE: step * BATCH_SIZE + BATCH_SIZE].flatten()][:, self.output_mask]

            #? Wide Deep

            # Input
            w_in = np.zeros((len(df_in), len(GLOBAL_INPUTS)), dtype=np.float32)
            w_in[:, self.input_mask] = df_in

            w_in = w_in.reshape(BATCH_SIZE, WINDOW_PAST, -1)
            w_in = torch.from_numpy(w_in).float().to(DEVICE)
            
            # Output
            w_out = df_out.reshape(BATCH_SIZE, WINDOW_PRESENT, -1)
            w_out = torch.from_numpy(w_out).float().to(DEVICE)

            # Predict future sensors values
            x = self.model_f_extractor(w_in, self.real_mask)
            y = self.model_sensor(x)

            # Calculate error
            loss = torch.mean(criterion1(y, w_out), dim=2)
            
            # Store the prediction error for further processing
            all_predicted_sen[self.real_output_indices[step * BATCH_SIZE: step * BATCH_SIZE + BATCH_SIZE, 0]] = loss.detach().cpu().numpy()[:, 0]

            # Smooth the prediction error using a median filter
            all_predicted_sen[apply_start - W_ANOMALY * 2: apply_end + 1] = scipy.signal.medfilt(all_predicted_sen[apply_start - W_ANOMALY * 2: apply_end + 1], kernel_size=MED_FILTER_LAG)

            #? Threshold

            # Input
            threshold_input_indices = (np.arange(apply_start, apply_end + 1) - HORIZON)[:, None] - np.arange(1, WINDOW_PAST + 1)

            w_in = torch.from_numpy(all_predicted_sen[threshold_input_indices][:, :, None]).float().to(DEVICE)
        
            # Predict threshold
            self.pred_error_model.zero_grad()

            threshold = self.pred_error_model(w_in).abs()

            # Calculate loss and do one SGD step
            loss = criterion2(torch.squeeze(threshold, dim=2), loss)
            loss.backward(retain_graph=True)

            optimizer.step()

            # Use the maximum predicted threshold for anomaly detection
            threshold = torch.max(threshold, dim=1)[0].detach().cpu().numpy()

            # If the threshold is zero, repeat the process
            if np.all(threshold == 0):

                # Predict threshold
                self.pred_error_model.zero_grad()

                self.pred_error_model.apply(lambda model: model.reset_parameters() if isinstance(model, nn.Conv1d) or isinstance(model, nn.Linear) else None)

                threshold = self.pred_error_model(w_in).abs()

                # Calculate loss and do one SGD step
                loss = criterion2(torch.squeeze(threshold, dim=2), loss)
                loss.backward(retain_graph=True)

                optimizer.step()

                # Use the maximum predicted threshold for anomaly detection
                threshold = torch.max(threshold, dim=1)[0].detach().cpu().numpy()

            # Sum the predicted threshold to the threshold base
            threshold = threshold + self.threshold_base

            # Store the calculated threshold
            thresholds_sen[apply_start:apply_end + 1] = np.squeeze(threshold)

            idx_win_thr = np.arange(apply_start, apply_end + 1)[:, None] - np.arange(W_ANOMALY)

            all_threshold_sen[np.arange(apply_start, apply_end + 1)] = np.all(all_predicted_sen[idx_win_thr] > threshold, 1)
            
            # DAICS avoids flagging anomalies that last for only a few samples (transient deviations) - Section VII-C
            # if np.any(all_threshold_sen[np.arange(apply_start, apply_end + 1)] == 1) and np.count_nonzero(all_threshold_sen[np.arange(apply_start, apply_end + 1)]) <= W_GRACE:
                
            #     # Prepare the optimizer
            #     ftune_optimizer = torch.optim.SGD(self.model_sensor.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, dampening=0.9, weight_decay=0.001)
            #     ftune_scheduler = ReduceLROnPlateau(ftune_optimizer)

            #     # Fine-tune the output section
            #     for epoch in range(T_EPOCHS):

            #         self.model_sensor.zero_grad()

            #         f_extracted = self.model_f_extractor(w_in[:-4], self.real_mask[:-4])
            #         y_t_sen = self.model_sensor(f_extracted)

            #         loss = criterion3(y_t_sen, w_out[:-4])
            #         loss.backward()

            #         ftune_optimizer.step()
            #         ftune_scheduler.step(loss)
                
            #     all_threshold_sen[np.arange(apply_start, apply_end + 1)] = 0
            
            # The condition is met when a false positive occours
            if np.any(all_threshold_sen[np.arange(apply_start, apply_end + 1)] == 1) and np.all(all_labels_threshold[np.arange(apply_start, apply_end + 1)] == 0):

                # Increment the human intervention counter
                human_inter_counter = human_inter_counter + 1

                # Flag the indices of human intervention
                human_idx_sen[np.arange(apply_start, apply_end + 1)] = 1

                # Prepare the optimizer
                ftune_optimizer = torch.optim.SGD(self.model_sensor.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, dampening=0.9, weight_decay=0.001)
                ftune_scheduler = ReduceLROnPlateau(ftune_optimizer)

                # Fine-tune the output section
                for epoch in range(T_EPOCHS):

                    self.model_sensor.zero_grad()

                    f_extracted = self.model_f_extractor(w_in[:-4], self.real_mask[:-4])
                    y_t_sen = self.model_sensor(f_extracted)

                    loss = criterion3(y_t_sen, w_out[:-4])
                    loss.backward()

                    ftune_optimizer.step()
                    ftune_scheduler.step(loss)

            self.log(msg=f"Step {step} / {steps}", end="\r", verbose=verbose)

            threshold_points.extend(thresholds_sen[apply_start:apply_end + 1])
            error_points.extend(np.min(all_predicted_sen[idx_win_thr], axis=1))

            x_vals = np.arange(len(threshold_points))

            # Compute colors based on threshold comparison
            # colors = ["green" if e < t else "red" for e, t in zip(error_points, threshold_points)]
            colors.extend(["green" if not a else "red" for a in np.max(self.all_labels[idx_win_thr] == 1, axis=1)])

            # Update threshold line
            threshold_line.set_data(x_vals, threshold_points)

            # Remove old scatter (if any) and redraw new one
            if "error_scatter" in locals():
                error_scatter.remove()

            error_scatter = ax.scatter(x_vals, error_points, c=colors, s=25)

            ax.set_xlim(0, 128)
            ax.set_ylim(np.min([np.min(threshold_points), np.min(error_points)])/2, np.max([np.max(threshold_points), np.max(error_points)])*2)

            plt.pause(0.01)

        plt.ioff()
        plt.show()

        after = time()

        self.log(msg=f"Time elapsed on the attack dataset: {after - before}", verbose=verbose)
        self.log(msg=f"Number of human interventions: {human_inter_counter}", verbose=verbose)

        all_threshold = all_threshold_sen[: len(self.all_labels)]
        all_labels_threshold = np.copy(self.all_labels)[: len(self.all_labels)]

        # Attack impact is part of the attack
        # Ref: http://dx.doi.org/10.1145/3196494.3196546
        all_labels_threshold[attack_impact_array] = all_threshold[attack_impact_array]

        tn, fp, fn, tp = confusion_matrix(all_labels_threshold, all_threshold).ravel()

        self.log(msg=f"Accuracy: {accuracy_score(all_labels_threshold, all_threshold)}", verbose=verbose)
        self.log(msg=f"Precision: {precision_score(all_labels_threshold, all_threshold)}", verbose=verbose)
        self.log(msg=f"Recall: {recall_score(all_labels_threshold, all_threshold)}", verbose=verbose)
        self.log(msg=f"False positive rate: {fp / (fp + tn)}", verbose=verbose)
        self.log(msg=f"False negative rate: {fn / (tp + fn)}", verbose=verbose)
        self.log(msg=f"F1 oneclass score: {f1_score(all_labels_threshold, all_threshold)}", verbose=verbose)

    def set_model_f_extractor(self, model_f_extractor: ModelFExtractor | OrderedDict):
        self.model_f_extractor.load_state_dict(model_f_extractor.state_dict() if isinstance(model_f_extractor, ModelFExtractor) else model_f_extractor)

    def get_model_f_extractor(self) -> ModelFExtractor:
        return deepcopy(self.model_f_extractor)
    
    def set_model_sensor(self, model_sensor: ModelSensors | OrderedDict):
        self.model_sensor.load_state_dict(model_sensor.state_dict() if isinstance(model_sensor, ModelSensors) else model_sensor)

    def get_model_sensor(self) -> ModelFExtractor:
        return deepcopy(self.model_sensor)

    def set_pred_error_model(self, pred_error_model: PredErrorModel | OrderedDict):
        self.pred_error_model.load_state_dict(pred_error_model.state_dict() if isinstance(pred_error_model, PredErrorModel) else pred_error_model)

    def get_pred_error_model(self) -> PredErrorModel:
        return deepcopy(self.pred_error_model)

def generate_non_iid_clients(model_f_extractor: ModelFExtractor = None, model_sensor: ModelSensors = None) -> list[Client]:

    truncate_windows = lambda x, y: (
        x[: (len(x) // BATCH_SIZE) * BATCH_SIZE],
        y[: (len(y) // BATCH_SIZE) * BATCH_SIZE]
    )

    clients: list[Client] = []

    hf = h5py.File(name=OUTPUT_FILE, mode="r")

    normal = hf["normal"]
    attack = hf["attack"]

    for key in normal.keys():
        normal_data = normal[key]
        attack_data = attack[key]

        inputs = normal_data.attrs["inputs"][:]
        outputs = normal_data.attrs["outputs"][:]

        if not len(outputs):
            continue

        df_train = normal_data["df_normal_train"][:]
        df_val = normal_data["df_normal_val"][:]
        df_test = normal_data["df_normal_test"][:]
        df_real = attack_data["df_attack"][:]

        train_input_indices = normal_data["df_normal_train_input_indices"][:]
        train_output_indices = normal_data["df_normal_train_output_indices"][:]

        val_input_indices = normal_data["df_normal_val_input_indices"][:]
        val_output_indices = normal_data["df_normal_val_output_indices"][:]

        test_input_indices = normal_data["df_normal_test_input_indices"][:]
        test_output_indices = normal_data["df_normal_test_output_indices"][:]

        real_input_indices = attack_data["df_attack_input_indices"][:]
        real_output_indices = attack_data["df_attack_output_indices"][:]

        # train_input_indices, train_output_indices = truncate_windows(train_input_indices, train_output_indices)
        val_input_indices, val_output_indices = truncate_windows(val_input_indices, val_output_indices)
        test_input_indices, test_output_indices = truncate_windows(test_input_indices, test_output_indices)
        real_input_indices, real_output_indices = truncate_windows(real_input_indices, real_output_indices)

        client_id = f"{'-'.join(sorted(normal_data.attrs['inputs'][:]))}"
        client_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, client_id))

        client = Client(
            client_id=client_id,

            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            df_real=df_real,

            train_input_indices=train_input_indices,
            train_output_indices=train_output_indices,

            val_input_indices=val_input_indices,
            val_output_indices=val_output_indices,

            test_input_indices=test_input_indices,
            test_output_indices=test_output_indices,

            real_input_indices=real_input_indices,
            real_output_indices=real_output_indices,

            inputs=inputs,
            outputs=outputs,

            model_f_extractor=model_f_extractor,
            model_sensor=model_sensor
        )

        clients.append(client)

    hf.close()

    return clients