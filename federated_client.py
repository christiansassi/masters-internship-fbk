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
    SENSORS_GROUPS
)

import config

from models import (
    WideDeepNetworkDAICS, 
    clone_wide_deep_networks,
    ThresholdNetworkDAICS,
    clone_threshold_networks
)

from generators import SlidingWindowGenerator

import h5py
import numpy as np
from collections import deque

from uuid import uuid4

import tensorflow as tf

import matplotlib.pyplot as plt

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

    def set_wide_deep_network(self, wide_deep_networks: list[WideDeepNetworkDAICS]):

        self.threshold_networks = clone_wide_deep_networks(
            wide_deep_networks=wide_deep_networks,
            optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
            loss=LOSS
        )

    def get_wide_deep_network(self) -> list[WideDeepNetworkDAICS]:

        return clone_wide_deep_networks(
            wide_deep_networks=self.threshold_networks,
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
    
    def simulate(self):

        wide_deep_input_windows = self.real_input_indices
        wide_deep_output_windows = self.real_output_indices

        start = 0

        errors = {str(i): {"0": 0., "1": 0., "2": 0., "3": 0., "4": 0., "5": 0., "6": 0.,} for i in range(wide_deep_output_windows[start][0])}
        thresholds = {}

        fig, axes = plt.subplots(3, 2, figsize=(10, 8))
        axes = axes.flatten()

        lines = []
        points = []

        for ax in axes:
            line, = ax.plot([], [], label="Threshold")
            point, = ax.plot([], [], "o", markersize=3, label="Error")

            lines.append({
                "obj": line,
                "x": deque(maxlen=100),
                "y": deque(maxlen=100)
            })

            points.append({
                "obj": point,
                "x": deque(maxlen=100),
                "y": deque(maxlen=100)
            })

        plt.tight_layout()
        plt.ion()
        plt.show()

        # Iterate over the input windows
        for index1 in range(start, len(wide_deep_input_windows)):
            
            # Take the last timestep of the current window.
            # This is the last time we see this timestep so we can calculate the avg of all the cumulated MSEs
            t = str(wide_deep_output_windows[index1][0])

            print(f"{t}", end="\r")

            # Iterate over the wide deep networks
            for index2, wide_deep_network in enumerate(self.wide_deep_networks):

                predicted_sensors = wide_deep_network.predict(
                    x=np.expand_dims(self.df_real[wide_deep_input_windows[index1]], axis=0),

                    batch_size=1,

                    verbose=config.VERBOSE
                )[0]

                real_sensors = self.df_real[wide_deep_output_windows[index1]][:, SENSOR_GROUPS_INDICES[index2]]

                # The mse is calculated between each timestep. So 110 (real) with 110 (predicted)
                # In the end, we obtain a single value (mse) for each timestep 
                mse_per_timestep = np.mean((predicted_sensors - real_sensors) ** 2, axis=1)

                # Since some input windows are overlapping (e.g. 113 is present in multiple windows), 
                # we keep track of each error for each timestep
                for index3, index4 in enumerate(wide_deep_output_windows[index1]):
                    errors.setdefault(str(index4), {}).setdefault(str(index2), []).append(mse_per_timestep[index3])

            for key in errors[t].keys():
                errors[t][key] = np.mean(errors[t][key])
            
            # Do the same for the threshold
            threshold_inputs = [[] for _ in range(len(SENSOR_GROUPS_INDICES))]

            # The input of the threshold networks are generated by taking the last 60 errors from t (excluded)
            for key in range(int(t) - 60, int(t)):
                for index2 in range(len(SENSOR_GROUPS_INDICES)):
                    threshold_inputs[index2].append(errors[str(key)][str(index2)])
            
            # Predict the thresholds for each input
            for index2, (threshold_network, threshold_input) in enumerate(zip(self.threshold_networks, threshold_inputs)):

                predicted_error = threshold_network.predict(
                    x=np.expand_dims(threshold_input, axis=(0, 2)),

                    batch_size=1,

                    verbose=config.VERBOSE
                ).item()

                for index3 in wide_deep_output_windows[index1]:
                    thresholds.setdefault(str(index3), {}).setdefault(str(index2), []).append(predicted_error)
            
            # Similarly to what happened for the errors, also here we calculate the avg of all the cumulated thresholds for the last timestep
            for key in thresholds[t].keys():
                thresholds[t][key] = np.mean(thresholds[t][key])
            
            current_errors = list(errors[t].values())
            current_thresholds = list(thresholds[t].values())

            anomalies = [int(ce > ct) for ce, ct in zip(current_errors, current_thresholds)]
            alarm = sum(anomalies) >= len(anomalies) // 2

            # if not alarm:

            #     for i, (threshold_network, threshold_input) in enumerate(zip(self.threshold_networks, threshold_inputs)):

            #         target = np.array(errors[t][str(i)])
            #         target = np.expand_dims(np.expand_dims(target, axis=0), axis=-1)

            #         threshold_network.fit(
            #             x=np.expand_dims(threshold_input, axis=(0, 2)),
            #             y=target,

            #             batch_size=1,

            #             verbose=config.VERBOSE
            #         )

            for i in range(len(SENSOR_GROUPS_INDICES)):

                lines[i]["x"].append(int(t))
                lines[i]["y"].append(current_thresholds[i])
                lines[i]["obj"].set_data(lines[i]["x"], lines[i]["y"])

                points[i]["x"].append(int(t))
                points[i]["y"].append(current_errors[i])
                points[i]["obj"].set_data(points[i]["x"], points[i]["y"])

                axes[i].set_xlim(
                    min(min(lines[i]["x"]), min(points[i]["x"])),
                    max(max(lines[i]["x"]), max(points[i]["x"]))
                )

                axes[i].set_ylim(
                    min(min(lines[i]["y"]), min(points[i]["y"])),
                    max(max(lines[i]["y"]), max(points[i]["y"]))
                )

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.05)

            # if int(alarm) != self.all_labels[int(t)]:
            #     print("WRONG")

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

        # Truncate to batch-aligned window counts
        train_input_indices, train_output_indices = truncate_windows(train_input_indices, train_output_indices)
        val_input_indices, val_output_indices = truncate_windows(val_input_indices, val_output_indices)
        test_input_indices, test_output_indices = truncate_windows(test_input_indices, test_output_indices)
        df_attack_input_indices, df_attack_output_indices = truncate_windows(df_attack_input_indices, df_attack_output_indices)

        # Create client object
        client = Client(
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            df_real=df_attack,

            train_input_indices=train_input_indices,
            train_output_indices=train_output_indices,

            val_input_indices=val_input_indices,
            val_output_indices=val_output_indices,

            test_input_indices=test_input_indices,
            test_output_indices=test_output_indices,

            real_input_indices=df_attack_input_indices,
            real_output_indices=df_attack_output_indices,

            wide_deep_networks=wide_deep_networks,
            threshold_networks=threshold_networks
        )

        clients.append(client)

    return clients
