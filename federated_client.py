import config
import constants

from models import ModelFExtractor, ModelSensors, PredErrorModel

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import scipy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import h5py
import numpy as np

from collections import OrderedDict

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
        outputs: list[str]
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
            window_size_in=constants.daics.WINDOW_PAST, 
            window_size_out=constants.daics.WINDOW_PRESENT, 
            n_devices_in=len(constants.GLOBAL_INPUTS), 
            kernel_size=constants.daics.KERNEL_SIZE
        )

        active_stages = [[] for _ in constants.DEFAULT_STAGES]

        for output in self.outputs:
            for index, stage in enumerate(constants.DEFAULT_STAGES):
                
                if output in stage:
                    active_stages[index].append(output)
                    break
        
        self.model_sensors = []
        model_sensors_params = []

        for active_stage in active_stages:

            if len(active_stage) == 0:
                continue
            
            model_sensor = ModelSensors(
                n_devices_out=len(active_stage)
            )

            self.model_sensors.append(model_sensor)
            model_sensors_params = model_sensors_params + list(model_sensor.parameters())

        self.pred_error_models = [PredErrorModel(
            window_size_in=constants.daics.WINDOW_PAST, 
            window_size_out=constants.daics.WINDOW_PRESENT
        ) for _ in range(len(self.model_sensors))]

        self.optimizer = torch.optim.SGD(list(self.model_f_extractor.parameters()) + model_sensors_params, lr=constants.daics.LEARNING_RATE, momentum=constants.daics.MOMENTUM)
        # self.optimizer = torch.optim.Adam(list(self.model_f_extractor.parameters()) + model_sensors_params, lr=constants.daics.LEARNING_RATE, weight_decay=constants.daics.WEIGHT_DECAY)
        # self.optimizer = torch.optim.AdamW(list(self.model_f_extractor.parameters()) + model_sensors_params, lr=constants.daics.LEARNING_RATE, betas=(0.9, 0.9), weight_decay=constants.daics.WEIGHT_DECAY)

        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=constants.daics.DAICS_PATIENCE)
        self.criterion = nn.MSELoss()

        self.epochs = 0
        self.steps = 0
        self.batch_size = 0
        self.score = float("-inf")

        self.input_mask = [list(constants.GLOBAL_INPUTS).index(x) for x in self.inputs]
        self.output_mask = [[list(self.inputs).index(output) for output in outputs] for outputs in active_stages if len(outputs)]

        self.mask = torch.zeros(constants.daics.BATCH_SIZE, constants.daics.WINDOW_PAST, len(constants.GLOBAL_INPUTS))
        self.mask[:, :, self.input_mask] = 1
        self.mask = self.mask.to(config.DEVICE)

    def __str__(self) -> str:
        return self.id

    def train_model_f_extractor_and_sensors(self, model_f_extractor: ModelFExtractor, ) -> tuple:

        self.model_f_extractor.to(config.DEVICE)

        for model_sensor in self.model_sensors:
            model_sensor.to(config.DEVICE)

        self.model_f_extractor.load_state_dict(deepcopy(model_f_extractor.state_dict()))

        min_train_loss = float("inf")
        min_val_loss = float("inf")

        best_model_f_extractor = None
        best_model_sensor = None

        config.printplus(f"Steps before: {self.steps}", log_only=True)
        config.printplus(f"Batch size before: {self.batch_size}", log_only=True)
        config.printplus(f"Epochs before: {self.epochs}", log_only=True)

        steps = max(constants.flad.MIN_STEPS, min(self.steps, constants.flad.MAX_STEPS))
        batch_size = max(constants.daics.BATCH_SIZE, min(len(self.train_input_indices) // steps, config.WIDE_DEEP_MAX_BATCH_SIZE))
        steps = max(constants.flad.MIN_STEPS, min(len(self.train_input_indices) // batch_size, constants.flad.MAX_STEPS))

        self.steps = steps
        self.batch_size = batch_size

        config.printplus(f"Steps after: {self.steps}", log_only=True)
        config.printplus(f"Batch size after: {self.batch_size}", log_only=True)
        config.printplus(f"Epochs after: {self.epochs}", log_only=True)

        # Calculate indices
        train_input_indices = self.train_input_indices[:self.steps * self.batch_size]
        train_output_indices = self.train_output_indices[:self.steps * self.batch_size]

        train_mask = torch.zeros(self.batch_size, constants.daics.WINDOW_PAST, len(constants.GLOBAL_INPUTS))
        train_mask[:, :, self.input_mask] = 1
        train_mask = train_mask.to(config.DEVICE)

        for epoch in range(self.epochs):

            # Training
            self.model_f_extractor.train()

            for model_sensor in self.model_sensors:
                model_sensor.train()

            train_loss = 0

            for step, batch_index in enumerate(np.random.permutation(range(0, self.steps)), start=1):
                
                # Input
                df_in = self.df_train[train_input_indices[batch_index * self.batch_size: batch_index * self.batch_size + self.batch_size].flatten()]

                w_in = np.zeros((len(df_in), len(constants.GLOBAL_INPUTS)), dtype=np.float32)
                w_in[:, self.input_mask] = df_in

                w_in = w_in.reshape(self.batch_size, constants.daics.WINDOW_PAST, -1)
                w_in = torch.from_numpy(w_in).float().to(config.DEVICE)

                # Reset gradients
                # self.model_f_extractor.zero_grad()
                self.optimizer.zero_grad(set_to_none=True)

                # Forward pass through the feature extractor
                x = self.model_f_extractor(w_in, train_mask)

                # Forward pass through the sensor head
                loss = 0

                for index, (model_sensor, output_mask) in enumerate(zip(self.model_sensors, self.output_mask), start=1):
                    
                    # Output
                    df_out = self.df_train[train_output_indices[batch_index * self.batch_size: batch_index * self.batch_size + self.batch_size].flatten()][:, output_mask]

                    w_out = df_out.reshape(self.batch_size, constants.daics.WINDOW_PRESENT, -1)
                    w_out = torch.from_numpy(w_out).float().to(config.DEVICE)

                    # model_sensor.zero_grad()

                    y = model_sensor(x)

                    # Compute loss
                    tmp = self.criterion(y, w_out)

                    #! CLIP THE LOSS
                    tmp = torch.clamp(tmp, max=2.0)

                    loss = loss + tmp

                    config.printplus(f"Training loss #{index}: {tmp}", log_only=True)

                #! SAFETY CHECK FOR NON-FINITE LOSS
                if not torch.isfinite(loss):
                    config.printplus(f"Skipping batch due to invalid loss ({loss.item()})")
                    continue

                # One step
                loss.backward()

                #! CLIP GRADIENTS
                torch.nn.utils.clip_grad_norm_(
                    parameters=list(self.model_f_extractor.parameters()) + [p for m in self.model_sensors for p in m.parameters()],
                    max_norm=1.0
                )
                
                self.optimizer.step()

                train_loss = train_loss + loss.detach()

                config.printplus(" " * 100, end="\r")
                config.printplus(f"Epoch: {epoch + 1} / {self.epochs} | Step: {step} / {self.steps} | Training loss {train_loss.item() / step}", end="\r")

            train_loss = train_loss.item() / self.steps

            min_train_loss = min(min_train_loss, train_loss)

            # Validation
            self.model_f_extractor.eval()

            for model_sensor in self.model_sensors:
                model_sensor.eval()

            val_loss = 0

            steps = len(self.val_input_indices) // constants.daics.BATCH_SIZE

            with torch.no_grad():

                for step, batch_index in enumerate(np.random.permutation(range(0, steps)), start=1):
                    
                    # Input
                    df_in = self.df_val[self.val_input_indices[batch_index * constants.daics.BATCH_SIZE: batch_index * constants.daics.BATCH_SIZE + constants.daics.BATCH_SIZE].flatten()]

                    w_in = np.zeros((len(df_in), len(constants.GLOBAL_INPUTS)), dtype=np.float32)
                    w_in[:, self.input_mask] = df_in

                    w_in = w_in.reshape(constants.daics.BATCH_SIZE, constants.daics.WINDOW_PAST, -1)
                    w_in = torch.from_numpy(w_in).float().to(config.DEVICE)

                    # Forward pass through the feature extractor
                    x = self.model_f_extractor(w_in, self.mask)

                    # Forward pass through the sensor head
                    loss = 0

                    for index, (model_sensor, output_mask) in enumerate(zip(self.model_sensors, self.output_mask), start=1):

                        # Output
                        df_out = self.df_val[self.val_output_indices[batch_index * constants.daics.BATCH_SIZE: batch_index * constants.daics.BATCH_SIZE + constants.daics.BATCH_SIZE].flatten()][:, output_mask]

                        w_out = df_out.reshape(constants.daics.BATCH_SIZE, constants.daics.WINDOW_PRESENT, -1)
                        w_out = torch.from_numpy(w_out).float().to(config.DEVICE)

                        y = model_sensor(x)

                        # Compute loss
                        tmp = self.criterion(y, w_out)

                        loss = loss + tmp

                        config.printplus(f"Validation loss #{index}: {tmp}", log_only=True)

                    val_loss = val_loss + loss.detach()

                    config.printplus(" " * 100, end="\r")
                    config.printplus(f"Epoch: {epoch + 1} / {self.epochs} | Step: {step} / {steps} | Validation loss: {val_loss.item() / step}", end="\r")

            val_loss = val_loss.item() / steps

            # Save best models
            if val_loss < min_val_loss:
                min_val_loss = val_loss

                best_model_f_extractor = deepcopy(self.model_f_extractor.state_dict())
                best_model_sensors = [deepcopy(model_sensor.state_dict()) for model_sensor in self.model_sensors]

            # Decay Learning Rate, pass validation loss for tracking at every epoch
            self.scheduler.step(val_loss)

        self.model_f_extractor.load_state_dict(deepcopy(best_model_f_extractor))
        
        for model_sensor, best_model_sensor in zip(self.model_sensors, best_model_sensors):
            model_sensor.load_state_dict(deepcopy(best_model_sensor))

        config.printplus(" " * 100, end="\r")
        config.printplus(f"Training loss: {min_train_loss} | Validation loss: {min_val_loss}")
        
        return -min_train_loss, -min_val_loss

    def eval_model_f_extractor_and_sensor(self, model_f_extractor: ModelFExtractor, ) -> float:
        
        self.model_f_extractor.to(config.DEVICE)

        for model_sensor in self.model_sensors:
            model_sensor.to(config.DEVICE)

        self.model_f_extractor.load_state_dict(deepcopy(model_f_extractor.state_dict()))

        # Evaluation
        self.model_f_extractor.eval()
        
        for model_sensor in self.model_sensors:
            model_sensor.eval()

        eval_loss = 0

        steps = len(self.test_input_indices) // constants.daics.BATCH_SIZE

        with torch.no_grad():
            
            for step, batch_index in enumerate(np.random.permutation(range(0, steps)), start=1):
                
                # Input
                df_in = self.df_test[self.test_input_indices[batch_index * constants.daics.BATCH_SIZE: batch_index * constants.daics.BATCH_SIZE + constants.daics.BATCH_SIZE].flatten()]

                w_in = np.zeros((len(df_in), len(constants.GLOBAL_INPUTS)), dtype=np.float32)
                w_in[:, self.input_mask] = df_in

                w_in = w_in.reshape(constants.daics.BATCH_SIZE, constants.daics.WINDOW_PAST, -1)
                w_in = torch.from_numpy(w_in).float().to(config.DEVICE)

                # Forward pass through the feature extractor
                x = self.model_f_extractor(w_in, self.mask)

                # Forward pass through the sensor head
                loss = 0

                for index, (model_sensor, output_mask) in enumerate(zip(self.model_sensors, self.output_mask), start=1):

                    # Output
                    df_out = self.df_test[self.test_output_indices[batch_index * constants.daics.daics.BATCH_SIZE: batch_index * constants.daics.BATCH_SIZE + constants.daics.BATCH_SIZE].flatten()][:, output_mask]

                    # Output
                    w_out = df_out.reshape(constants.daics.BATCH_SIZE, constants.daics.WINDOW_PRESENT, -1)
                    w_out = torch.from_numpy(w_out).float().to(config.DEVICE)

                    y = model_sensor(x)

                    # Compute loss
                    tmp = self.criterion(y, w_out)

                    loss = loss + tmp

                    config.printplus(f"Evaluation loss #{index}: {tmp}", log_only=True)

                eval_loss = eval_loss + loss.detach()

                config.printplus(" " * 100, end="\r")
                config.printplus(f"Step: {step} / {steps} | Evaluation loss: {eval_loss.item() / step}", end="\r")
                
        eval_loss = eval_loss.item() / steps

        self.score = -eval_loss

        config.printplus(" " * 100, end="\r")
        config.printplus(f"Evaluation loss: {eval_loss}")

        return -eval_loss

    def train_pred_error_model(self, ):
        
        self.model_f_extractor.to(config.DEVICE)
        
        for model_sensor in self.model_sensors:
            model_sensor.to(config.DEVICE)

        self.model_f_extractor.eval()
        
        for model_sensor in self.model_sensors:
            model_sensor.eval()

        criterion = nn.MSELoss(reduction="none")

        def calculate_errors(df, input_indices, output_indices):

            train_input_indices = input_indices[:(len(input_indices) // constants.daics.BATCH_SIZE) * constants.daics.BATCH_SIZE]
            train_output_indices = output_indices[:(len(output_indices) // constants.daics.BATCH_SIZE) * constants.daics.BATCH_SIZE]

            train_mask = torch.zeros(constants.daics.BATCH_SIZE, constants.daics.WINDOW_PAST, len(constants.GLOBAL_INPUTS))
            train_mask[:, :, self.input_mask] = 1
            train_mask = train_mask.to(config.DEVICE)

            all_predicted = [np.zeros(len(df), dtype=np.float32) for _ in range(len(self.model_sensors))]

            steps = len(train_input_indices) // constants.daics.BATCH_SIZE

            for step in range(0, steps):
                
                # Input
                df_in = df[train_input_indices[step * constants.daics.BATCH_SIZE: step * constants.daics.BATCH_SIZE + constants.daics.BATCH_SIZE].flatten()]

                w_in = np.zeros((len(df_in), len(constants.GLOBAL_INPUTS)), dtype=np.float32)
                w_in[:, self.input_mask] = df_in

                w_in = w_in.reshape(constants.daics.BATCH_SIZE, constants.daics.WINDOW_PAST, -1)
                w_in = torch.from_numpy(w_in).float().to(config.DEVICE)

                # Forward pass through the feature extractor
                x = self.model_f_extractor(w_in, self.mask)

                # Forward pass through the sensor head
                for index, (model_sensor, output_mask) in enumerate(zip(self.model_sensors, self.output_mask)):
                    
                    # Output
                    df_out = df[train_output_indices[step * constants.daics.BATCH_SIZE: step * constants.daics.BATCH_SIZE + constants.daics.BATCH_SIZE].flatten()][:, output_mask]
                    
                    w_out = df_out.reshape(constants.daics.BATCH_SIZE, constants.daics.WINDOW_PRESENT, -1)
                    w_out = torch.from_numpy(w_out).float().to(config.DEVICE)

                    y = model_sensor(x)

                    # Compute loss
                    loss = torch.mean(criterion(y, w_out), dim=2)

                    all_predicted[index][train_output_indices[step * constants.daics.BATCH_SIZE: step * constants.daics.BATCH_SIZE + constants.daics.BATCH_SIZE, 0]] = loss.detach().cpu().numpy()[:, 0]

                config.printplus(" " * 100, end="\r")
                config.printplus(f"Calculating errors {step + 1} / {steps}", end="\r")

            return all_predicted

        all_predicted_val = calculate_errors(df=self.df_val, input_indices=self.val_input_indices, output_indices=self.val_output_indices)
        all_predicted_test = calculate_errors(df=self.df_test, input_indices=self.test_input_indices, output_indices=self.test_output_indices)

        criterion = nn.MSELoss()

        best_pred_error_models = []

        losses = []

        for index, pred_error_model in enumerate(self.pred_error_models):
            
            all_predicted = np.concatenate((np.trim_zeros(all_predicted_val[index]), np.trim_zeros(all_predicted_test[index])))
            all_predicted = scipy.signal.medfilt(all_predicted, kernel_size=constants.daics.MED_FILTER_LAG)

            input_indices = (np.arange(constants.daics.SAMPLING_START, len(all_predicted), constants.daics.VAL_STEP) - constants.daics.HORIZON - constants.daics.WINDOW_PRESENT)[:, None] - np.arange(1, constants.daics.WINDOW_PAST + 1)
            input_indices = np.sort(input_indices)
            input_indices = input_indices[: (len(input_indices) // constants.daics.BATCH_SIZE) * constants.daics.BATCH_SIZE, :]

            output_indices = np.arange(constants.daics.SAMPLING_START, len(all_predicted), constants.daics.VAL_STEP)[:, None] - np.arange(1, constants.daics.WINDOW_PRESENT + 1)
            output_indices = np.sort(output_indices)
            output_indices = output_indices[: (len(output_indices) // constants.daics.BATCH_SIZE) * constants.daics.BATCH_SIZE, :]

            pred_error_model.to(config.DEVICE)
        
            optimizer = torch.optim.SGD(pred_error_model.parameters(), lr=constants.daics.LEARNING_RATE)
            # optimizer = torch.optim.Adam(pred_error_model.parameters(), lr=constants.daics.LEARNING_RATE, weight_decay=constants.daics.WEIGHT_DECAY)
            # optimizer = torch.optim.AdamW(pred_error_model.parameters(), lr=constants.daics.LEARNING_RATE, betas=(0.9, 0.9), weight_decay=constants.daics.WEIGHT_DECAY)

            min_train_loss = float("inf")

            for epoch in range(constants.daics.THRESHOLD_EPOCHS):
                
                pred_error_model.train()

                train_loss = 0

                steps = len(input_indices) // constants.daics.BATCH_SIZE

                for step, batch_index in enumerate(np.random.permutation(range(0, steps)), start=1):

                    df_in = all_predicted[input_indices[batch_index * constants.daics.BATCH_SIZE: batch_index * constants.daics.BATCH_SIZE + constants.daics.BATCH_SIZE]]
                    df_out = all_predicted[output_indices[batch_index * constants.daics.BATCH_SIZE: batch_index * constants.daics.BATCH_SIZE + constants.daics.BATCH_SIZE]]

                    # Input and output
                    w_in = torch.from_numpy(df_in[:, :, None]).float().to(config.DEVICE)
                    w_out = torch.from_numpy(df_out[:, :, None]).float().to(config.DEVICE)

                    # pred_error_model.zero_grad()
                    optimizer.zero_grad(set_to_none=True)

                    y = pred_error_model(w_in.float().to(config.DEVICE)).abs()

                    if torch.all(y == 0).item():
                        # pred_error_model.zero_grad()

                        pred_error_model.apply(lambda model: model.reset_parameters() if isinstance(model, nn.Conv1d) or isinstance(model, nn.Linear) else None)

                        y = pred_error_model(w_in.float().to(config.DEVICE)).abs()
                    
                    # Compute loss
                    loss = criterion(y, w_out)

                    train_loss = train_loss +  loss.detach()

                    # One step
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    config.printplus(" " * 100, end="\r")
                    config.printplus(f"[{index + 1} / {len(self.model_sensors)}] Epoch: {epoch + 1} / {constants.daics.THRESHOLD_EPOCHS} | Step: {step} / {steps} | Training loss: {train_loss.item() / step}", end="\r")
                
                train_loss = train_loss.item() / steps

                # Save best model
                if train_loss < min_train_loss:
                    min_train_loss = train_loss

                    best_pred_error_models.append(deepcopy(pred_error_model.state_dict()))
            
            losses.append(min_train_loss)
        
        for pred_error_model, best_pred_error_model in zip(self.pred_error_models, best_pred_error_models):
            pred_error_model.load_state_dict(deepcopy(best_pred_error_model))
        
        config.printplus(" " * 100, end="\r")
        config.printplus(f"Training loss: {np.mean(losses)}")

        return losses

    def calculate_threshold_base(self, ):

        self.model_f_extractor.to(config.DEVICE)

        self.model_f_extractor.eval()
        
        for model_sensor in self.model_sensors:
            model_sensor.to(config.DEVICE)
    
        criterion = nn.MSELoss(reduction="none")

        def calculate_errors(df, input_indices, output_indices):

            errors = [[] for _ in range(len(self.model_sensors))]

            steps = len(input_indices) // constants.daics.BATCH_SIZE

            for step in range(0, steps):
                
                # Input
                df_in = df[input_indices[step * constants.daics.BATCH_SIZE: step * constants.daics.BATCH_SIZE + constants.daics.BATCH_SIZE].flatten()]

                w_in = np.zeros((len(df_in), len(constants.GLOBAL_INPUTS)), dtype=np.float32)
                w_in[:, self.input_mask] = df_in

                w_in = w_in.reshape(constants.daics.BATCH_SIZE, constants.daics.WINDOW_PAST, -1)
                w_in = torch.from_numpy(w_in).float().to(config.DEVICE)

                # Forward pass through the feature extractor
                x = self.model_f_extractor(w_in, self.mask)

                # Forward pass through the sensor head
                for index, (model_sensor, output_mask) in enumerate(zip(self.model_sensors, self.output_mask)):
                    
                    # Output
                    df_out = df[output_indices[step * constants.daics.BATCH_SIZE: step * constants.daics.BATCH_SIZE + constants.daics.BATCH_SIZE].flatten()][:, output_mask]

                    w_out = df_out.reshape(constants.daics.BATCH_SIZE, constants.daics.WINDOW_PRESENT, -1)
                    w_out = torch.from_numpy(w_out).float().to(config.DEVICE)

                    y = model_sensor(x)

                    # Compute loss
                    loss = torch.mean(criterion(y, w_out), dim=2)
                
                    errors[index].append(scipy.signal.medfilt(loss.detach().cpu().numpy()[:, 0].flatten(), kernel_size=constants.daics.MED_FILTER_LAG))

                config.printplus(" " * 100, end="\r")
                config.printplus(f"Calculating errors {step + 1} / {steps}", end="\r")
            
            return [item for loss in errors for item in loss]

        all_errors = []
        all_errors.extend(calculate_errors(df=self.df_val, input_indices=self.val_input_indices, output_indices=self.val_output_indices))
        all_errors.extend(calculate_errors(df=self.df_test, input_indices=self.test_input_indices, output_indices=self.test_output_indices))

        err_mean = np.mean(all_errors)
        err_std = np.std(all_errors)

        threshold_base = err_mean + err_std
        
        self.threshold_base = threshold_base

        return threshold_base

    def test(self):
        
        self.calculate_threshold_base()

        self.model_f_extractor.to(config.DEVICE)

        for model_sensor in self.model_sensors:
            model_sensor.to(config.DEVICE)

        for pred_error_model in self.pred_error_models:
            pred_error_model.to(config.DEVICE)

        self.model_f_extractor.eval()

        all_predicted_sen = []
        all_threshold_sen = []
        all_threshold_act = np.zeros(len(self.df_real), dtype=float)
        thresholds_sen = []
        human_idx_sen = []

        pred_error_optimizer = []

        for model_sensor, pred_error_model in zip(self.model_sensors, self.pred_error_models):
            pred_error_optimizer.append(torch.optim.SGD(pred_error_model.parameters(), lr=constants.daics.LEARNING_RATE))

            for name, param in model_sensor.named_children():
                for p in param.parameters():
                    p.requires_grad = True
            
            all_predicted_sen.append(np.zeros(len(self.df_real), dtype=float))
            human_idx_sen.append(np.zeros(len(self.df_real), dtype=float))
            all_threshold_sen.append(np.zeros(len(self.df_real), dtype=float))
            thresholds_sen.append(np.zeros(len(self.df_real), dtype=float))

        actuators = [list(self.inputs).index(item) for item in self.inputs if item in set(constants.ACTUATORS)]

        database_actuators = np.concatenate((
            self.df_val[:, actuators],
            self.df_test[:, actuators]
        ))

        database_actuators, indices, unique_counts = np.unique(database_actuators, axis=0, return_index=True, return_counts=True)
        
        attack_indices = np.where(self.all_labels == 1)[0]
        attack_indices = np.split(attack_indices, np.where(np.diff(attack_indices) != 1)[0] + 1)
        attack_indices = [(sub[0], sub[-1]) for sub in attack_indices]

        attack_impact_array = np.array([], dtype=int)

        all_labels_threshold = np.copy(self.all_labels).astype(int)
        all_labels_threshold[attack_impact_array] = 1

        human_inter_counter = 0
        actuation_alarm = 0

        steps = len(self.real_output_indices) // constants.daics.BATCH_SIZE

        criterion1 = nn.MSELoss(reduction="none") # test_loss_function
        criterion2 = nn.MSELoss() # threshold_loss_function
        criterion3 = nn.MSELoss() # ftune_loss_function

        for step in range(0, steps):

            apply_thr_start = self.real_output_indices[step * constants.daics.BATCH_SIZE: step * constants.daics.BATCH_SIZE + constants.daics.BATCH_SIZE][0, 0]
            apply_thr_end = self.real_output_indices[step * constants.daics.BATCH_SIZE: step * constants.daics.BATCH_SIZE + constants.daics.BATCH_SIZE][-1, 0]
        
            # Input
            df_in = self.df_real[self.real_input_indices[step * constants.daics.BATCH_SIZE: step * constants.daics.BATCH_SIZE + constants.daics.BATCH_SIZE].flatten()]

            w_in = np.zeros((len(df_in), len(constants.GLOBAL_INPUTS)), dtype=np.float32)
            w_in[:, self.input_mask] = df_in

            w_in = w_in.reshape(constants.daics.BATCH_SIZE, constants.daics.WINDOW_PAST, -1)
            w_in = torch.from_numpy(w_in).float().to(config.DEVICE)

            # Forward pass through the feature extractor
            x = self.model_f_extractor(w_in, self.mask)
        
            set_database_actuators = set(map(tuple, database_actuators))

            window_t_actuator = self.df_real[np.unique(self.real_output_indices[step * constants.daics.BATCH_SIZE:  step * constants.daics.BATCH_SIZE + constants.daics.BATCH_SIZE].flatten())][:, actuators][:constants.daics.BATCH_SIZE, :]

            tmp = [(set(map(tuple, np.expand_dims(x_window_t_actuator, axis=0))) & set_database_actuators) == set() for  x_window_t_actuator in window_t_actuator]

            if np.any(tmp):
                actuation_alarm = 1
                all_threshold_act[np.arange(apply_thr_start, apply_thr_end + 1)] = tmp
            
            else:
                actuation_alarm = 0
            
            idx_threshold = (np.arange(apply_thr_start, apply_thr_end + 1) - constants.daics.HORIZON)[:, None] - np.arange(1, constants.daics.WINDOW_PAST + 1)

            for index, (model_sensor, pred_error_model, output_mask) in enumerate(zip(self.model_sensors, self.pred_error_models, self.output_mask)):
                    
                # Output
                df_out = self.df_real[self.real_output_indices[step * constants.daics.BATCH_SIZE: step * constants.daics.BATCH_SIZE + constants.daics.BATCH_SIZE].flatten()][:, output_mask]

                w_out = df_out.reshape(constants.daics.BATCH_SIZE, constants.daics.WINDOW_PRESENT, -1)
                w_out = torch.from_numpy(w_out).float().to(config.DEVICE)

                y = model_sensor(x)

                pred_error_sen = torch.mean(criterion1(y, w_out), dim=2)

                all_predicted_sen[index][self.real_output_indices[step * constants.daics.BATCH_SIZE: step * constants.daics.BATCH_SIZE + constants.daics.BATCH_SIZE, 0]] = pred_error_sen.detach().cpu().numpy()[:, 0]
                all_predicted_sen[index][apply_thr_start - constants.daics.W_ANOMALY * 2: apply_thr_end + 1] = scipy.signal.medfilt(all_predicted_sen[index][apply_thr_start - constants.daics.W_ANOMALY * 2: apply_thr_end + 1], kernel_size=constants.daics.MED_FILTER_LAG)

                threshold_wt_1 = torch.from_numpy(all_predicted_sen[index][idx_threshold][:, :, None]).float().to(config.DEVICE)

                pred_error_model.zero_grad()

                threshold = pred_error_model(threshold_wt_1).abs()

                thresh_loss = criterion2(torch.squeeze(threshold, dim=2), pred_error_sen)
                thresh_loss.backward(retain_graph=True)

                pred_error_optimizer[index].step()

                threshold = torch.max(threshold, dim=1)[0].detach().cpu().numpy()

                if np.all(threshold == 0):
                    pred_error_model.zero_grad()
                    pred_error_model.apply(lambda model: model.reset_parameters() if isinstance(model, nn.Conv1d) or isinstance(model, nn.Linear) else None)

                    threshold = pred_error_model(threshold_wt_1).abs()

                    thresh_loss = criterion2(torch.squeeze(threshold, dim=2), pred_error_sen)
                    thresh_loss.backward(retain_graph=True)
                    
                    pred_error_optimizer[index].step()

                    threshold = torch.max(threshold, dim=1)[0].detach().cpu().numpy()
                
                threshold = threshold + self.threshold_base

                thresholds_sen[index][apply_thr_start: apply_thr_end + 1] = np.squeeze(threshold)

                idx_win_thr = np.arange(apply_thr_start, apply_thr_end + 1)[:, None] - np.arange(constants.daics.W_ANOMALY)

                all_threshold_sen[index][np.arange(apply_thr_start, apply_thr_end + 1)] = np.all(all_predicted_sen[index][idx_win_thr] > threshold, 1)

            if np.any(all_threshold_act[np.arange(apply_thr_start, apply_thr_end + 1)] == 1) and np.count_nonzero(all_threshold_act[np.arange(apply_thr_start, apply_thr_end + 1)]) <= constants.daics.W_GRACE:

                database_actuators = np.concatenate((database_actuators, window_t_actuator), axis=0)
                database_actuators = np.unique(database_actuators, axis=0)

                all_threshold_act[np.arange(apply_thr_start, apply_thr_end + 1)] = 0

            if np.any(all_threshold_act[np.arange(apply_thr_start, apply_thr_end + 1)] == 1) and np.all(all_labels_threshold[np.arange(apply_thr_start, apply_thr_end + 1)] == 0):

                database_actuators = np.concatenate((database_actuators, window_t_actuator), axis=0)
                database_actuators = np.unique(database_actuators, axis=0) 

                human_inter_counter = human_inter_counter + 1 
        
            for index, (model_sensor, pred_error_model, output_mask) in enumerate(zip(self.model_sensors, self.pred_error_models, self.output_mask)):

                if np.any(all_threshold_sen[index][np.arange(apply_thr_start, apply_thr_end + 1)] == 1) and np.count_nonzero(all_threshold_sen[index][np.arange(apply_thr_start, apply_thr_end + 1)]) <= constants.daics.W_GRACE:
                    
                    df_out = self.df_real[self.real_output_indices[step * constants.daics.BATCH_SIZE: step * constants.daics.BATCH_SIZE + constants.daics.BATCH_SIZE].flatten()][:, output_mask]

                    w_out = df_out.reshape(constants.daics.BATCH_SIZE, constants.daics.WINDOW_PRESENT, -1)
                    w_out = torch.from_numpy(w_out).float().to(config.DEVICE)

                    # Prepare the optimizer
                    ftune_optimizer = torch.optim.SGD(model_sensor.parameters(), lr=constants.daics.LEARNING_RATE, momentum=0.9, dampening=0.9, weight_decay=0.001)
                    ftune_scheduler = ReduceLROnPlateau(ftune_optimizer)
                    # Fine-tune the output section
                    for epoch in range(constants.daics.T_EPOCHS):
                        model_sensor.zero_grad()
                        f_extracted = self.model_f_extractor(w_in[:-4].float().to(config.DEVICE))
                        y_t_sen = model_sensor(f_extracted)
                        loss = criterion3(y_t_sen, w_out[:-4])
                        loss.backward()
                        ftune_optimizer.step()
                        ftune_scheduler.step(loss)

                    # The alarm is silenced
                    all_threshold_sen[index][np.arange(apply_thr_start, apply_thr_end + 1)] = 0

                if np.any(all_threshold_sen[index][np.arange(apply_thr_start, apply_thr_end + 1)] == 1) and np.all(all_labels_threshold[np.arange(apply_thr_start, apply_thr_end + 1)] == 0):
                    
                    df_out = self.df_real[self.real_output_indices[step * constants.daics.BATCH_SIZE: step * constants.daics.BATCH_SIZE + constants.daics.BATCH_SIZE].flatten()][:, output_mask]

                    w_out = df_out.reshape(constants.daics.BATCH_SIZE, constants.daics.WINDOW_PRESENT, -1)
                    w_out = torch.from_numpy(w_out).float().to(config.DEVICE)

                    # Increment the human intervention counter
                    human_inter_counter += 1
                    # Flag the indices of human intervention for debugging purposes
                    human_idx_sen[index][np.arange(apply_thr_start, apply_thr_end + 1)] = 1
                    ftune_optimizer = torch.optim.SGD(model_sensor.parameters(), lr=constants.daics.LEARNING_RATE, momentum=constants.daics.MOMENTUM, dampening=0.9, weight_decay=0.001)
                    ftune_scheduler = ReduceLROnPlateau(ftune_optimizer)
                    # Fine-tune the output section
                    for epoch in range(constants.daics.T_EPOCHS):
                        model_sensor.zero_grad()
                        w_t_1 = w_in.detach().clone()
                        f_extracted = self.model_f_extractor(w_t_1[:-4].float().to(config.DEVICE))
                        y_t_sen = model_sensor(f_extracted)
                        loss = criterion3(y_t_sen, w_out[:-4])
                        loss.backward()
                        ftune_optimizer.step()
                        ftune_scheduler.step(loss)

            config.printplus(" " * 100, end="\r")
            config.printplus(f"Step {step + 1} / {steps}", end="\r")

        print("Number of human interventions: ", human_inter_counter)
        # end for enumerate(dl_test):                                            
        # Combining the anomaly detection results of all output sections
        all_threshold = np.zeros(len(self.df_real), dtype=float)
        for all_thr in all_threshold_sen:
            all_threshold = np.logical_or(all_threshold, all_thr)
        # Combine with anomaly detection in actuators
        all_threshold = np.logical_or(all_threshold, all_threshold_act)
        # Make sure that all arrays have the same length
        all_threshold = all_threshold[: len(self.all_labels)]
        all_labels_threshold = np.copy(self.all_labels)[: len(self.all_labels)]
        # Attack impact is part of the attack
        # Ref: http://dx.doi.org/10.1145/3196494.3196546
        all_labels_threshold[attack_impact_array] = all_threshold[attack_impact_array]
        # Calculate the confusion matrix
        tn, fp, fn, tp = confusion_matrix(all_labels_threshold, all_threshold).ravel()
        print("accuracy {}".format(accuracy_score(all_labels_threshold, all_threshold)))
        print("precision {}".format(precision_score(all_labels_threshold, all_threshold)))
        print("recall {}".format(recall_score(all_labels_threshold, all_threshold)))
        print("false positive rate", fp/(fp+tn))
        print("false negative rate", fn/(tp+fn))
        print("\x1b[6;30;42m f1_oneclass_score \x1b[0m {}".format(f1_score(all_labels_threshold, all_threshold)))

    def set_model_f_extractor(self, model_f_extractor: ModelFExtractor | OrderedDict):
        self.model_f_extractor.load_state_dict(deepcopy(model_f_extractor.state_dict()) if isinstance(model_f_extractor, ModelFExtractor) else deepcopy(model_f_extractor))

    def set_model_sensors(self, model_sensors: list[ModelSensors | OrderedDict]):
        for model_sensor, loaded_model_sensor in zip(self.model_sensors, model_sensors):
            model_sensor.load_state_dict(deepcopy(loaded_model_sensor.state_dict()) if isinstance(loaded_model_sensor, ModelSensors) else deepcopy(loaded_model_sensor))

    def set_pred_error_models(self, pred_error_models: PredErrorModel | OrderedDict):
        for pred_error_model, loaded_pred_error_model in zip(self.pred_error_models, pred_error_models):
            pred_error_model.load_state_dict(deepcopy(loaded_pred_error_model.state_dict()) if isinstance(loaded_pred_error_model, PredErrorModel) else deepcopy(loaded_pred_error_model))

def generate_non_iid_clients() -> list[Client]:

    truncate_windows = lambda x, y: (
        x[: (len(x) // constants.daics.BATCH_SIZE) * constants.daics.BATCH_SIZE],
        y[: (len(y) // constants.daics.BATCH_SIZE) * constants.daics.BATCH_SIZE]
    )

    clients: list[Client] = []

    hf = h5py.File(name=constants.OUTPUT_FILE, mode="r")

    normal = hf["normal"]
    attack = hf["attack"]

    for index, key in enumerate(normal.keys(), start = 1):
        config.printplus(f"Creating clients {index} / {len(normal)}", end="\r")

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
            outputs=outputs
        )

        clients.append(client)

    hf.close()

    config.printplus(" "*50, end="\r")
    config.printplus(f"Created {len(clients)} clients")

    return clients