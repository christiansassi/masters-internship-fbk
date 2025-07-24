from constants import (
    OUTPUT_FILE,
    SENSOR_GROUPS_INDICES,
    N_CLIENTS,
    WINDOW_PAST,
    WINDOW_PRESENT,
    FEATURES_IN,
    KERNEL_SIZE,
    LEARNING_RATE,
    MOMENTUM,
    LOSS,
    BATCH_SIZE,
    MAX_EPOCHS,
    W_ANOMALY,
    W_GRACE,
    SENSORS_GROUPS,
    MED_FILTER_LAG,
    CACHE_CLIENTS
)

import config

from models import (
    WideDeepNetworkDAICS, 
    clone_wide_deep_networks,
    ThresholdNetworkDAICS,
    clone_threshold_networks,
    load_wide_deep_networks,
    load_threshold_networks
)

from generators import SlidingWindowGenerator

import h5py
import numpy as np

from uuid import uuid4

import tensorflow as tf

import scipy.signal

from os import makedirs, listdir
from os.path import join

import pickle

class Client:

    def __init__(
        self,

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

        wide_deep_networks: list[WideDeepNetworkDAICS],
        threshold_networks: list[ThresholdNetworkDAICS]
    ):
        
        self.id = str(uuid4())

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.df_real = df_real

        self.all_labels = self.df_real[:, -1].astype(int)

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

        self.wide_deep_networks = []
        self.wide_deep_epochs = 0
        self.wide_deep_steps = 0
        self.wide_deep_score = 0

        self.wide_deep_fit_data = []
        self.wide_deep_real_data = []

        for index in range(len(SENSOR_GROUPS_INDICES)):

            x = SlidingWindowGenerator(
                data=self.df_train,
                input_indices=self.train_input_indices,
                output_indices=self.train_output_indices,
                sensor_indices=SENSOR_GROUPS_INDICES[index],
                batch_size=BATCH_SIZE
            )

            validation_data = SlidingWindowGenerator(
                data=self.df_val,
                input_indices=self.val_input_indices,
                output_indices=self.val_output_indices,
                sensor_indices=SENSOR_GROUPS_INDICES[index],
                batch_size=BATCH_SIZE
            )

            self.wide_deep_fit_data.append((x, validation_data))

            x = SlidingWindowGenerator(
                data=self.df_real,
                input_indices=self.real_input_indices,
                output_indices=self.real_output_indices,
                sensor_indices=SENSOR_GROUPS_INDICES[index],
                batch_size=BATCH_SIZE
            )

            self.wide_deep_real_data.append(x)

        self.wide_deep_eval_data = []

        for index in range(len(SENSOR_GROUPS_INDICES)):

            x = SlidingWindowGenerator(
                data=self.df_test,
                input_indices=self.test_input_indices,
                output_indices=self.test_output_indices,
                sensor_indices=SENSOR_GROUPS_INDICES[index],
                batch_size=BATCH_SIZE
            )

            self.wide_deep_eval_data.append(x)
        
        if len(wide_deep_networks) != len(SENSOR_GROUPS_INDICES):

            for sensors_indices in SENSOR_GROUPS_INDICES:

                model = WideDeepNetworkDAICS(
                    window_size_in=WINDOW_PAST,
                    window_size_out=WINDOW_PRESENT,
                    n_devices_in=FEATURES_IN,
                    n_devices_out=len(sensors_indices),
                    kernel_size=KERNEL_SIZE
                )

                model.compile(
                    optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
                    loss=LOSS
                )

                self.wide_deep_networks.append(model)
        
        else:
            self.wide_deep_networks = clone_wide_deep_networks(
                wide_deep_networks=wide_deep_networks,
                optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
                loss=LOSS
            )

        self.threshold_networks = []
        self.threshold_epochs = 0
        self.threshold_steps = 0
        self.threshold_score = 0
        self.threshold_networks_fit_data = []
        self.threshold_networks_eval_data = []

        if len(threshold_networks) != len(SENSOR_GROUPS_INDICES):

            for _ in SENSOR_GROUPS_INDICES:
                model = ThresholdNetworkDAICS()

                model.compile(
                    optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
                    loss=LOSS
                )

                self.threshold_networks.append(model)

        else:
            self.threshold_networks = clone_threshold_networks(
                threshold_networks=threshold_networks,
                optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
                loss=LOSS
            )

    def __str__(self) -> str:
        return self.id

    def train_wide_deep_network(self, wide_deep_networks: list[WideDeepNetworkDAICS]):
        
        self.wide_deep_networks = clone_wide_deep_networks(
            wide_deep_networks=wide_deep_networks,
            optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
            loss=LOSS
        )

        for index, model in enumerate(self.wide_deep_networks):
            
            x, validation_data = self.wide_deep_fit_data[index]

            model.fit(
                x=x,

                validation_data=validation_data,

                batch_size=BATCH_SIZE,
                
                epochs=self.wide_deep_epochs,
                steps_per_epoch=self.wide_deep_steps,

                verbose=config.VERBOSE
            )
    
    def eval_wide_deep_network(self, wide_deep_networks: list[WideDeepNetworkDAICS]) -> float:

        scores = []

        for index, model in enumerate(wide_deep_networks):

            x = self.wide_deep_eval_data[index]

            score = model.evaluate(
                x=x,

                batch_size=BATCH_SIZE,

                verbose=config.VERBOSE
            )
        
            scores.append(score)
        
        self.wide_deep_score = -np.mean(scores)

        return self.wide_deep_score

    def set_wide_deep_networks(self, wide_deep_networks: list[WideDeepNetworkDAICS]):

        self.wide_deep_networks = clone_wide_deep_networks(
            wide_deep_networks=wide_deep_networks,
            optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
            loss=LOSS
        )

    def get_wide_deep_networks(self) -> list[WideDeepNetworkDAICS]:

        return clone_wide_deep_networks(
            wide_deep_networks=self.wide_deep_networks,
            optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
            loss=LOSS
        )

    def train_threshold_network(self, threshold_networks: list[ThresholdNetworkDAICS], clear_cache: bool = False):

        self.threshold_networks = clone_threshold_networks(
            threshold_networks=threshold_networks,
            optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
            loss=LOSS
        )

        if clear_cache or not len(self.threshold_networks_fit_data):

            self.threshold_networks_fit_data = []

            for index, wide_deep_network in enumerate(self.wide_deep_networks):

                chunk = []

                for label in ["train", "val"]:
                    predicted = wide_deep_network.predict(
                        x=getattr(self, f"df_{label}")[getattr(self, f"{label}_input_indices")],

                        batch_size=BATCH_SIZE,

                        verbose=config.VERBOSE
                    )

                    ground_truth = getattr(self, f"df_{label}")[getattr(self, f"{label}_output_indices")][:, :, SENSOR_GROUPS_INDICES[index]]

                    errors = np.mean((predicted - ground_truth) ** 2, axis=-1)

                    error_series = np.zeros(len(getattr(self, f"df_{label}")))
                    error_sums = np.zeros_like(error_series)
                    error_counts = np.zeros_like(error_series)

                    for indices, error in zip(getattr(self, f"{label}_output_indices"), errors):
                        for idx, e in zip(indices, error):
                            error_sums[idx] += e
                            error_counts[idx] += 1

                    nonzero_mask = error_counts > 0
                    error_series[nonzero_mask] = error_sums[nonzero_mask] / error_counts[nonzero_mask]

                    input_windows = np.flip(np.arange(WINDOW_PAST - 1, len(error_series) - WINDOW_PRESENT + 1)[:, None] - np.arange(WINDOW_PAST), axis=1)
                    output_windows = (np.arange(WINDOW_PAST, len(error_series) - WINDOW_PRESENT + 1)[:, None] + np.arange(WINDOW_PRESENT))

                    input_windows = input_windows[:(len(input_windows) // BATCH_SIZE) * BATCH_SIZE]
                    output_windows = output_windows[:len(input_windows)]

                    x = error_series[input_windows][:, :, None]
                    y = np.max(error_series[output_windows], axis=1, keepdims=True)

                    chunk.append((x, y))

                self.threshold_networks_fit_data.append(chunk)

        for index, threshold_network in enumerate(self.threshold_networks):

            chunk = self.threshold_networks_fit_data[index]
            (x_train, y_train), (x_val, y_val) = chunk

            threshold_network.fit(
                x=x_train,
                y=y_train,

                validation_data=(x_val, y_val),

                batch_size=BATCH_SIZE,
                epochs=self.threshold_epochs,
                steps_per_epoch=self.threshold_steps,

                verbose=config.VERBOSE
            )

    def eval_threshold_network(self, threshold_networks: list[ThresholdNetworkDAICS], clear_cache: bool = False) -> float:

        if clear_cache or not len(self.threshold_networks_eval_data):

            self.threshold_networks_eval_data = []

            for index, wide_deep_network in enumerate(self.wide_deep_networks):

                predicted = wide_deep_network.predict(
                    x=self.df_test[self.test_input_indices],

                    batch_size=BATCH_SIZE,

                    verbose=config.VERBOSE
                )

                ground_truth = self.df_test[self.test_output_indices][:, :, SENSOR_GROUPS_INDICES[index]]
                errors = np.mean((predicted - ground_truth) ** 2, axis=-1)

                error_series = np.zeros(len(self.df_test))
                error_sums = np.zeros_like(error_series)
                error_counts = np.zeros_like(error_series)

                for indices, error in zip(self.test_output_indices, errors):
                    for idx, e in zip(indices, error):
                        error_sums[idx] += e
                        error_counts[idx] += 1

                nonzero_mask = error_counts > 0
                error_series[nonzero_mask] = error_sums[nonzero_mask] / error_counts[nonzero_mask]

                input_windows = np.flip((np.arange(WINDOW_PAST - 1, len(error_series) - WINDOW_PRESENT + 1)[:, None] - np.arange(WINDOW_PAST)), axis=1)
                output_windows = (np.arange(WINDOW_PAST, len(error_series) - WINDOW_PRESENT + 1)[:, None] + np.arange(WINDOW_PRESENT))

                input_windows = input_windows[:(len(input_windows) // BATCH_SIZE) * BATCH_SIZE]
                output_windows = output_windows[:len(input_windows)]

                x = error_series[input_windows][:, :, None]
                y = np.max(error_series[output_windows], axis=1, keepdims=True)

                self.threshold_networks_eval_data.append((x, y))

        scores = []

        for index, model in enumerate(threshold_networks):

            x_test, y_test = self.threshold_networks_eval_data[index]

            score = model.evaluate(
                x=x_test,
                y=y_test,

                batch_size=BATCH_SIZE,

                verbose=config.VERBOSE
            )

            scores.append(score)

        self.threshold_score = -np.mean(scores)

        return self.threshold_score

    def set_threshold_networks(self, threshold_networks: list[ThresholdNetworkDAICS]):

        self.threshold_networks = clone_threshold_networks(
            threshold_networks=threshold_networks,
            optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
            loss=LOSS
        )
    
    def get_threshold_networks(self) -> list[ThresholdNetworkDAICS]:

        return clone_threshold_networks(
            threshold_networks=self.threshold_networks,
            optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
            loss=LOSS
        )
    
    def calcuate_threshold_base(self) -> list[float]:

        all_smoothed_errors = []

        for index, wide_deep_network in enumerate(self.wide_deep_networks):

            predicted = wide_deep_network.predict(
                x=self.df_val[self.val_input_indices],

                batch_size=BATCH_SIZE,

                verbose=config.VERBOSE
            )

            ground_truth = self.df_val[self.val_output_indices][:, :, SENSOR_GROUPS_INDICES[index]]
            errors = np.mean((predicted - ground_truth) ** 2, axis=-1)

            error_series = np.zeros(len(self.df_test))
            error_sums = np.zeros_like(error_series)
            error_counts = np.zeros_like(error_series)

            for indices, error in zip(self.test_output_indices, errors):
                for idx, e in zip(indices, error):
                    error_sums[idx] += e
                    error_counts[idx] += 1

            nonzero_mask = error_counts > 0
            error_series[nonzero_mask] = error_sums[nonzero_mask] / error_counts[nonzero_mask]

            smoothed_series = scipy.signal.medfilt(error_series[nonzero_mask], kernel_size=59)

            all_smoothed_errors.extend(smoothed_series.tolist())

        err_mean = np.mean(all_smoothed_errors)
        err_std = np.std(all_smoothed_errors)
        t_base = err_mean + err_std

        return [t_base for _ in SENSOR_GROUPS_INDICES]

    def simulate(self):

        threshold_base = self.calcuate_threshold_base()

        wide_deep_networks_real_errors = [[] for _ in SENSOR_GROUPS_INDICES]

        for index, wide_deep_network in enumerate(self.wide_deep_networks):

            predicted_sensors = wide_deep_network.predict(
                x=self.wide_deep_real_data[index],

                batch_size=BATCH_SIZE,

                verbose=config.VERBOSE
            )

            real_sensors = self.df_real[self.real_output_indices][:, :, SENSOR_GROUPS_INDICES[index]]

            errors = np.mean((predicted_sensors - real_sensors) ** 2, axis=-1)

            error_series = np.zeros(len(self.df_real))
            error_sums = np.zeros_like(error_series)
            error_counts = np.zeros_like(error_series)

            for indices, error in zip(self.real_output_indices, errors):
                for idx, e in zip(indices, error):
                    error_sums[idx] += e
                    error_counts[idx] += 1

            nonzero_mask = error_counts > 0
            error_series[nonzero_mask] = error_sums[nonzero_mask] / error_counts[nonzero_mask]
            
            wide_deep_networks_real_errors[index] = error_series

        self.wide_deep_networks_real_errors = wide_deep_networks_real_errors

        # for input_window, output_window in zip(self.real_input_indices, self.real_output_indices):

        #     for index, wide_deep_network in enumerate(self.wide_deep_networks):

        #         predicted_sensors = wide_deep_network.predict(
        #             x=np.expand_dims(self.df_real[input_window], axis=0),

        #             batch_size=1,

        #             verbose=config.VERBOSE
        #         )[0]

        #         real_sensors = self.df_real[output_window][:, SENSOR_GROUPS_INDICES[index]]

        #         mse_per_timestep = np.mean((predicted_sensors - real_sensors) ** 2, axis=1)

        #         mse_per_timestep

def generate_iid_clients(wide_deep_networks: list[WideDeepNetworkDAICS] = [], threshold_networks: list[ThresholdNetworkDAICS] = []) -> list[Client]:

    hf = h5py.File(name=OUTPUT_FILE, mode="r")

    df_normal_train = hf["df_normal_train"][:]
    df_normal_val = hf["df_normal_val"][:]
    df_normal_test = hf["df_normal_test"][:]

    df_normal_train_input_indices = np.array_split(hf["df_normal_train_input_indices"][:], N_CLIENTS)
    df_normal_train_output_indices = np.array_split(hf["df_normal_train_output_indices"][:], N_CLIENTS)

    df_normal_val_input_indices = np.array_split(hf["df_normal_val_input_indices"][:], N_CLIENTS)
    df_normal_val_output_indices = np.array_split(hf["df_normal_val_output_indices"][:], N_CLIENTS)

    df_normal_test_input_indices = np.array_split(hf["df_normal_test_input_indices"][:], N_CLIENTS)
    df_normal_test_output_indices = np.array_split(hf["df_normal_test_output_indices"][:], N_CLIENTS)

    df_attack = hf["df_attack"][:]
    df_attack_input_indices = hf["df_attack_input_indices"][:]
    df_attack_output_indices = hf["df_attack_output_indices"][:]

    hf.close()

    truncate_windows = lambda x, y: (x[: (len(x) // BATCH_SIZE) * BATCH_SIZE],
                                     y[: (len(y) // BATCH_SIZE) * BATCH_SIZE])

    clients = []

    for i in range(N_CLIENTS):

        # Split windows
        train_input_indices = df_normal_train_input_indices[i]
        train_output_indices = df_normal_train_output_indices[i]

        val_input_indices = df_normal_val_input_indices[i]
        val_output_indices = df_normal_val_output_indices[i]

        test_input_indices = df_normal_test_input_indices[i]
        test_output_indices = df_normal_test_output_indices[i]

        # Build unique row indices per dataset
        train_indices_used = np.unique(np.concatenate([train_input_indices.flatten(), train_output_indices.flatten()]))
        val_indices_used = np.unique(np.concatenate([val_input_indices.flatten(), val_output_indices.flatten()]))
        test_indices_used = np.unique(np.concatenate([test_input_indices.flatten(), test_output_indices.flatten()]))

        # Slice datasets
        df_train = df_normal_train[train_indices_used]
        df_val = df_normal_val[val_indices_used]
        df_test = df_normal_test[test_indices_used]

        # Map global indices to local indices
        train_map = {idx: j for j, idx in enumerate(train_indices_used)}
        val_map = {idx: j for j, idx in enumerate(val_indices_used)}
        test_map = {idx: j for j, idx in enumerate(test_indices_used)}

        train_in_local = np.vectorize(train_map.get)(train_input_indices)
        train_out_local = np.vectorize(train_map.get)(train_output_indices)

        val_in_local = np.vectorize(val_map.get)(val_input_indices)
        val_out_local = np.vectorize(val_map.get)(val_output_indices)

        test_in_local = np.vectorize(test_map.get)(test_input_indices)
        test_out_local = np.vectorize(test_map.get)(test_output_indices)

        # Truncate to batch-aligned window counts
        train_in_local, train_out_local = truncate_windows(train_in_local, train_out_local)
        val_in_local, val_out_local = truncate_windows(val_in_local, val_out_local)
        test_in_local, test_out_local = truncate_windows(test_in_local, test_out_local)
        df_attack_input_indices, df_attack_output_indices = truncate_windows(df_attack_input_indices, df_attack_output_indices)

        # Create client object
        client = Client(
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            df_real=df_attack,

            train_input_indices=train_in_local,
            train_output_indices=train_out_local,

            val_input_indices=val_in_local,
            val_output_indices=val_out_local,

            test_input_indices=test_in_local,
            test_output_indices=test_out_local,

            real_input_indices=df_attack_input_indices,
            real_output_indices=df_attack_output_indices,

            wide_deep_networks=wide_deep_networks,
            threshold_networks=threshold_networks
        )

        clients.append(client)

    return clients

def save_clients(clients: list[Client]):

    for client in clients:

        client_dir = join(CACHE_CLIENTS, str(client))

        makedirs(client_dir, exist_ok=True)

        wide_deep_networks = client.get_wide_deep_networks()
        threshold_networks = client.get_threshold_networks()

        client.wide_deep_networks = []
        client.threshold_networks = []

        with open(join(client_dir, f"{str(client)}.pkl"), "wb+") as f:
            pickle.dump(client, f)

        client.set_wide_deep_networks(wide_deep_networks=wide_deep_networks)
        client.set_threshold_networks(threshold_networks=threshold_networks)

def load_clients() -> list[Client]:

    clients = []

    wide_deep_networks = load_wide_deep_networks()
    threshold_networks = load_threshold_networks()

    for client_id in listdir(CACHE_CLIENTS):

        client_dir = join(CACHE_CLIENTS, client_id)

        with open(join(client_dir, f"{client_id}.pkl"), "rb") as f:
            client = pickle.load(f)
        
        client.id = client_id

        client.set_wide_deep_networks(wide_deep_networks=wide_deep_networks)
        client.set_threshold_networks(threshold_networks=threshold_networks)

        clients.append(client)
    
    return clients