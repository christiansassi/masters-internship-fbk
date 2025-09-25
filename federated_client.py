from config import *
from constants import *
from models import ModelFExtractor, ModelSensors
import utils

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import h5py
import numpy as np

from copy import deepcopy

import uuid

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

        model_f_extractor = None,
        model_sensor = None
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

    def __str__(self) -> str:
        return self.id

    def train_model_f_extractor(self, model_f_extractor: ModelFExtractor, verbose: bool = False) -> tuple:

        log = lambda msg, end="\n": print(f"{utils.log_timestamp_status()} {msg}", end=end) if verbose else None

        self.model_f_extractor = deepcopy(model_f_extractor)
        
        self.model_f_extractor.to(DEVICE)
        self.model_sensor.to(DEVICE)

        self.optimizer = torch.optim.SGD(list(self.model_f_extractor.parameters()) + list(self.model_sensor.parameters()), lr=LEARNING_RATE, momentum=MOMENTUM)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=DAICS_PATIENCE)
        self.criterion = nn.MSELoss()

        min_train_loss = float("inf")
        min_val_loss = float("inf")

        best_model_f_extractor = None
        best_model_sensor = None

        # Calculate batch size
        #TODO adjust batch_size dynamically
        batch_size = BATCH_SIZE # len(self.train_input_indices) // self.steps

        train_mask = torch.zeros(batch_size, WINDOW_PAST, len(GLOBAL_INPUTS))
        train_mask[:, :, self.input_mask] = 1
        train_mask = train_mask.to(DEVICE)

        train_input_indices = self.train_input_indices[:(len(self.train_input_indices) // batch_size) * batch_size]
        train_output_indices = self.train_output_indices[:(len(self.train_input_indices) // batch_size) * batch_size]

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

                log(" " * 100, end="\r")
                log(f"Epoch: {epoch} / {self.epochs} | Step: {step} / {steps} | Training loss: {train_loss / step}", end="\r")

            train_loss = train_loss / steps # self.steps

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

                    log(" " * 100, end="\r")
                    log(f"Epoch: {epoch} / {self.epochs} | Step: {step} / {steps} | Validation loss: {val_loss / step}", end="\r")

            val_loss = val_loss / steps # self.steps

            # Save best models
            if val_loss < min_val_loss:
                min_val_loss = val_loss

                best_model_f_extractor = deepcopy(self.model_f_extractor)
                best_model_sensor = deepcopy(self.model_sensor)

            # Decay Learning Rate, pass validation loss for tracking at every epoch
            self.scheduler.step(val_loss)

        self.model_f_extractor = deepcopy(best_model_f_extractor)
        self.model_sensor = deepcopy(best_model_sensor)

        log(" " * 100, end="\r")
        log(f"Training loss: {min_train_loss} | Validation loss: {min_val_loss}")
        
        return -min_train_loss, -min_val_loss

    def eval_model_f_extractor(self, model_f_extractor: ModelFExtractor, verbose: bool = False) -> float:
        
        log = lambda msg, end="\n": print(f"{utils.log_timestamp_status()} {msg}", end=end) if verbose else None

        self.model_f_extractor = deepcopy(model_f_extractor)

        self.model_f_extractor.to(DEVICE)
        self.model_sensor.to(DEVICE)

        # Training
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

                log(" " * 100, end="\r")
                log(f"Step: {step} / {steps} | Evaluation loss: {eval_loss / step}", end="\r")
                
        eval_loss = eval_loss / steps

        self.score = -eval_loss

        log(" " * 100, end="\r")
        log(f"Evaluation loss: {eval_loss}")

        return -eval_loss

    def set_model_f_extractor(self, model_f_extractor: ModelFExtractor):
        self.model_f_extractor = deepcopy(model_f_extractor)

    def get_model_f_extractor(self) -> ModelFExtractor:
        return deepcopy(self.model_f_extractor)
    
    def set_model_sensor(self, model_sensor: ModelSensors):
        self.model_sensor = deepcopy(model_sensor)

    def get_model_sensor(self) -> ModelFExtractor:
        return deepcopy(self.model_sensor)

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

            inputs=normal_data.attrs["inputs"][:],
            outputs=normal_data.attrs["outputs"][:],

            model_f_extractor=model_f_extractor,
            model_sensor=model_sensor
        )

        clients.append(client)

    hf.close()

    return clients