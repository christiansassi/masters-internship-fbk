import config

from constants import *
from generators import SlidingWindowGenerator
from models import WideDeepNetworkDAICS, ThresholdNetworkDAICS

from os import makedirs
from os.path import isfile

import h5py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import pickle

import uuid

import scipy

import tensorflow as tf

def pad_client_df_to_global(df_client: np.ndarray, client_inputs: list[str]) -> np.ndarray:

    df = np.zeros((df_client.shape[0], len(GLOBAL_INPUTS)), dtype=np.float32)

    for local_index, local_inpiut in enumerate(client_inputs):
        global_index = GLOBAL_INPUTS.index(local_inpiut)
        df[:, global_index] = df_client[:, local_index]

    return df

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

        wide_deep_network: WideDeepNetworkDAICS = None,
        threshold_network: ThresholdNetworkDAICS = None
    ):
        self.id = client_id

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

        self.inputs = inputs
        self.outputs = outputs

        self.mask_indices = [GLOBAL_OUTPUTS.index(tag) for tag in self.outputs]
        
        # Wide Deep Network
        self.wide_deep_network = WideDeepNetworkDAICS(
            window_past=WINDOW_PAST,
            window_present=WINDOW_PRESENT,
            n_inputs=len(GLOBAL_INPUTS),
            n_outputs=len(GLOBAL_OUTPUTS)
        )

        self.wide_deep_network.build(
            input_shape=(None, WINDOW_PAST, len(GLOBAL_INPUTS))
        )

        if wide_deep_network is not None:
            self.wide_deep_network.load_weights(wide_deep_network)

        self.wide_deep_network.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
            loss=LOSS
        )

        self.wide_deep_network.set_mask(self.mask_indices, WINDOW_PRESENT)

        self.wide_deep_epochs = 0
        self.wide_deep_steps = 0
        self.wide_deep_score = 0

        self.wide_deep_train = pad_client_df_to_global(self.df_train, self.inputs)
        self.wide_deep_train = SlidingWindowGenerator(
            x=self.wide_deep_train,
            y=self.wide_deep_train[:, [GLOBAL_OUTPUTS.index(tag) for tag in self.outputs]],
            labels=np.zeros(len(self.wide_deep_train), dtype=np.int32),
            input_indices=self.train_input_indices,
            output_indices=self.train_output_indices,
            batch_size=BATCH_SIZE,
            outputs=self.outputs
        )

        self.wide_deep_val = pad_client_df_to_global(self.df_val, self.inputs)
        self.wide_deep_val = SlidingWindowGenerator(
            x=self.wide_deep_val,
            y=self.wide_deep_val[:, [GLOBAL_OUTPUTS.index(tag) for tag in self.outputs]],
            labels=np.zeros(len(self.wide_deep_val), dtype=np.int32),
            input_indices=self.val_input_indices,
            output_indices=self.val_output_indices,
            batch_size=BATCH_SIZE,
            outputs=self.outputs
        )

        self.wide_deep_test = pad_client_df_to_global(self.df_test, self.inputs)
        self.wide_deep_test = SlidingWindowGenerator(
            x=self.wide_deep_test,
            y=self.wide_deep_test[:, [GLOBAL_OUTPUTS.index(tag) for tag in self.outputs]],
            labels=np.zeros(len(self.wide_deep_test), dtype=np.int32),
            input_indices=self.test_input_indices,
            output_indices=self.test_output_indices,
            batch_size=BATCH_SIZE,
            outputs=self.outputs
        )

        self.wide_deep_real = pad_client_df_to_global(self.df_real, self.inputs)
        self.wide_deep_real = SlidingWindowGenerator(
            x=self.wide_deep_real,
            y=self.wide_deep_real[:, [GLOBAL_OUTPUTS.index(tag) for tag in self.outputs]],
            labels=self.all_labels,
            input_indices=self.real_input_indices,
            output_indices=self.real_output_indices,
            batch_size=BATCH_SIZE,
            outputs=self.outputs,
            shuffle=False
        )

        # Threshold Network
        self.threshold_network = ThresholdNetworkDAICS(
            window_past=WINDOW_PAST,
            window_present=WINDOW_PRESENT
        )

        self.threshold_network.build(
            input_shape=(None, WINDOW_PAST, 1)
        )

        if threshold_network is not None:
            self.threshold_network.load_weights(threshold_network)

        self.threshold_network.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
            loss=LOSS
        )

        self.threshold_epochs = 0
        self.threshold_steps = 0
        self.threshold_score = 0

    def __str__(self) -> str:
        return self.id

    def train_wide_deep_network(self, wide_deep_network: WideDeepNetworkDAICS) -> tuple:

        self.wide_deep_network = wide_deep_network.clone()
        self.wide_deep_network.set_mask(self.mask_indices, WINDOW_PRESENT)

        history = self.wide_deep_network.fit(
            self.wide_deep_train,
            validation_data=self.wide_deep_val,

            epochs=self.wide_deep_epochs,
            steps_per_epoch=self.wide_deep_steps,

            verbose=config.TRAIN_VERBOSE
        )

        return -history.history["loss"][-1], -history.history["val_loss"][-1]
    
    def eval_wide_deep_network(self, wide_deep_network: WideDeepNetworkDAICS) -> float:
        
        self.wide_deep_network = wide_deep_network.clone()
        self.wide_deep_network.set_mask(self.mask_indices, WINDOW_PRESENT)

        score = self.wide_deep_network.evaluate(
            self.wide_deep_test,

            verbose=config.EVAL_VERBOSE
        )
        
        self.wide_deep_score = -score

        return self.wide_deep_score

    def set_wide_deep_network(self, wide_deep_network: WideDeepNetworkDAICS):

        self.wide_deep_network = wide_deep_network.clone()
        self.wide_deep_network.set_mask(self.mask_indices, WINDOW_PRESENT)

    def get_wide_deep_network(self) -> WideDeepNetworkDAICS:
        return self.wide_deep_network.clone()
    
    def load_wide_deep_network(self, weights_file: str):

        self.wide_deep_network = WideDeepNetworkDAICS(
            window_past=WINDOW_PAST,
            window_present=WINDOW_PRESENT,
            n_inputs=len(GLOBAL_INPUTS),
            n_outputs=len(GLOBAL_OUTPUTS)
        )

        self.wide_deep_network.build(
            input_shape=(None, WINDOW_PAST, len(GLOBAL_INPUTS))
        )

        self.wide_deep_network.load_weights(weights_file)
        
        self.wide_deep_network.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
            loss=LOSS
        )
        
        self.wide_deep_network.set_mask(self.mask_indices, WINDOW_PRESENT)

    def _craft_x_and_y(self, df):

        y_true = []
        labels = []

        for index in range(len(df)):
            _, y, label = df.get_item_with_label(index)

            y_true.append(y)
            labels.append(label)

        y_true = np.concatenate(y_true, axis=0)[:, :, [GLOBAL_OUTPUTS.index(tag) for tag in self.outputs]]
        labels = np.concatenate(labels, axis=0)[:, 0]

        y_pred = self.wide_deep_network.predict(
            df, 

            verbose=config.PREDICT_VERBOSE
        )[:, :, [GLOBAL_OUTPUTS.index(tag) for tag in self.outputs]]

        y_true = y_true[:, 0, :]
        y_pred = y_pred[:, 0, :]

        errors = np.mean((y_pred - y_true) ** 2, axis=1)
        errors = scipy.signal.medfilt(errors, kernel_size=MED_FILTER_LAG)
        
        y = errors[WINDOW_PAST + HORIZON - 1:]
        labels = labels[WINDOW_PAST + HORIZON - 1:]

        x = sliding_window_view(errors, window_shape=WINDOW_PAST)[:len(y)][..., None]

        return x, y, labels
    
    def train_threshold_network(self):
        
        x_train, y_train, _ = self._craft_x_and_y(df=self.wide_deep_train)
        x_val, y_val, _ = self._craft_x_and_y(df=self.wide_deep_val)

        history = self.threshold_network.fit(
            x_train, y_train,

            validation_data=(x_val, y_val),

            batch_size=BATCH_SIZE,
            epochs=THRESHOLD_EPOCHS,

            shuffle=True,

            verbose=config.TRAIN_VERBOSE
        )

        return -history.history["loss"][-1], -history.history["val_loss"][-1]
    
    def eval_threshold_network(self):
        
        x_test, y_test, _ = self._craft_x_and_y(df=self.wide_deep_test)

        score = self.threshold_network.evaluate(
            x_test, y_test,

            batch_size=BATCH_SIZE,

            verbose=config.EVAL_VERBOSE
        )
        
        self.threshold_score = -score

        return self.threshold_score

    def set_threshold_network(self, threshold_network: ThresholdNetworkDAICS):
        self.threshold_network = threshold_network.clone()

    def get_threshold_network(self) -> ThresholdNetworkDAICS:
        return self.threshold_network.clone()
    
    def load_threshold_network(self, weights_file: str):

        self.threshold_network = ThresholdNetworkDAICS(
            window_past=WINDOW_PAST,
            window_present=WINDOW_PRESENT
        )

        self.threshold_network.build(
            input_shape=(None, WINDOW_PAST, 1)
        )

        self.wide_deep_network.load_weights(weights_file)

        self.threshold_network.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
            loss=LOSS
        )
    
    def calculate_threshold_base(self):

        all_errors = []

        for df in [self.wide_deep_train, self.wide_deep_val, self.wide_deep_test]:

            y_true = np.concatenate([y for _, y in df], axis=0)[:, :, [GLOBAL_OUTPUTS.index(tag) for tag in self.outputs]]
            y_pred = self.wide_deep_network.predict(
                df, 

                verbose=config.PREDICT_VERBOSE
            )[:, :, [GLOBAL_OUTPUTS.index(tag) for tag in self.outputs]]

            y_true = y_true[:, 0, :]
            y_pred = y_pred[:, 0, :]

            errors = np.mean((y_pred - y_true) ** 2, axis=1)

            all_errors.extend(errors)

        all_errors = scipy.signal.medfilt(all_errors, kernel_size=MED_FILTER_LAG)
        
        return np.mean(all_errors) + np.std(all_errors)

    def run_simulation_v1(self):
        
        #TODO

        cache_file = join(CACHE, f"{self.id}.pkl")

        if not isfile(cache_file):

            x, y, labels = self._craft_x_and_y(df=self.wide_deep_real)
            t_base = self.calculate_threshold_base()

            makedirs(CACHE, exist_ok=True)

            with open(cache_file, "wb+") as f:
                pickle.dump((x, y, labels, t_base), f)
        
        else:

            with open(cache_file, "rb") as f:
                x, y, labels, t_base = pickle.load(f)

        start = 0
        # end = 2574

        x = x[start:]
        y = y[start:]
        labels = labels[start:]

        attacks = np.where(labels == 1)[0]
        attacks = np.split(attacks, np.where(np.diff(attacks) != 1)[0] + 1)
        attacks_index = 0

        warnings = 0

        stats = {
            "true_positives": 0,
            "true_negatives": 0,
            "false_positives": 0,
            "false_negatives": 0
        }

        flag = False

        import matplotlib.pyplot as plt

        folder = join("clients", self.id)

        makedirs(folder, exist_ok=True)

        for index, attack in enumerate(attacks, start=1):
            offset = 100
            start_idx = attack[0] - offset
            end_idx = attack[-1] + offset

            points = y[start_idx:end_idx + 1]
            x = np.arange(len(points))

            # Plot black line
            plt.plot(x, points, color="black")

            # Shade background
            plt.axvspan(offset, offset + len(attack) - 1, color="red", alpha=0.2, label="Attack region")
            plt.axvspan(0, offset, color="green", alpha=0.1, label="Normal region")
            plt.axvspan(offset + len(attack), len(points) - 1, color="green", alpha=0.1)

            # Markers
            attack_x = np.arange(offset, offset + len(attack))
            attack_y = points[attack_x]
            plt.scatter(attack_x, attack_y, color="red")

            normal_x = np.concatenate((x[:offset], x[offset + len(attack):]))
            normal_y = np.concatenate((points[:offset], points[offset + len(attack):]))
            plt.scatter(normal_x, normal_y, color="green")

            plt.xlabel("Time")
            plt.ylabel("Error")
            plt.title(f"Attack #{index}")
            plt.grid(True)
            plt.legend()
            plt.savefig(join(folder, f"attack_{index}.png"), dpi=300, bbox_inches="tight")
            plt.close()

        return

        plt.ion()
        _, ax = plt.subplots()

        window_size = 50
        y_true_history = []
        threshold_history = []
        flags_history = []

        index = -1
        
        negatives_x = []
        negatives_y = []

        while index < len(x):
            
            index = index + 1

            errors_window = x[index][None, ...]
            y_true_val = y[index].item()
            y_pred_val = self.threshold_network.predict(errors_window, verbose=0).item()
            label = labels[index]

            threshold_val = y_pred_val + t_base

            if y_true_val > threshold_val:
                warnings = warnings + 1

                if warnings >= W_ANOMALY:
                    warnings = 0

                    if label == 1 and index >= attacks[attacks_index][0]:
                        
                        #? TRUE POSITIVE
                        stats["true_positives"] = stats["true_positives"] + 1
                        flag = False

                        # We are in an attack chunk
                        # Skip to a normal one
                        index = attacks[attacks_index][-1]
                        attacks_index = attacks_index + 1

                        flags_history = []
                        y_true_history = []
                        threshold_history = []

                    elif index < attacks[attacks_index][0] and not flag:
                        
                        #? FALSE POSITIVE
                        stats["false_positives"] = stats["false_positives"] + 1
                        flag = True

                        # We are in a normal chunk
                        # Skip to an attack one
                        # index = attacks[attacks_index][0]

                        # flags_history = []
                        # y_true_history = []
                        # threshold_history = []

            else:
                warnings = max(0, warnings - 1)

                if label == 1 and index >= attacks[attacks_index][-1]:

                    warnings = 0

                    #? FALSE NEGATIVE
                    stats["false_negatives"] = stats["false_negatives"] + 1
                    flag = False

                    # We are in an attack chunk
                    # Skip to a normal one
                    index = attacks[attacks_index][-1]
                    attacks_index = attacks_index + 1

                    flags_history = []
                    y_true_history = []
                    threshold_history = []
                
                elif index == attacks[attacks_index][0] - 1 and not flag:

                    #? TRUE NEGATIVE
                    stats["true_negatives"] = stats["true_negatives"] + 1
                    flag = True

                    # We are in a normal chunk
                    # Skip to an attack one
                    # index = attacks[attacks_index][0]

                    # flags_history = []
                    # y_true_history = []
                    # threshold_history = []

                # fit
                negatives_x.append(x[index])
                negatives_y.append(y[index])

                if len(negatives_x) == BATCH_SIZE:
                    batch_x = np.stack(negatives_x, axis=0)
                    batch_y = np.stack(negatives_y, axis=0)

                    #self.threshold_network.train_on_batch(batch_x, batch_y)

                    negatives_x.clear()
                    negatives_y.clear()
            
            flags_history.append(bool(label))
            y_true_history.append(y_true_val)
            threshold_history.append(threshold_val)

            if len(y_true_history) > window_size:
                flags_history = flags_history[-window_size:]
                y_true_history = y_true_history[-window_size:]
                threshold_history = threshold_history[-window_size:]

            ax.clear()

            x_vals = list(range(index - len(y_true_history) + 1, index + 1))

            ax.plot(x_vals, y_true_history, "b-", label="y_true")

            for point_x, point_y, f in zip(x_vals, y_true_history, flags_history):
                color = "green" if not f else "red"
                ax.scatter(x=point_x, y=point_y, color=color, s=50, zorder=5)

            ax.plot(range(index - len(threshold_history) + 1, index + 1),
                    threshold_history, "r--", label="threshold")

            ax.set_xlabel("Index")
            ax.set_ylabel("Error")
            ax.set_title(f"Step {index}, label={label}")
            ax.legend()

            plt.pause(0.01)

        plt.ioff()
        plt.show()

def generate_non_iid_clients(wide_deep_network: str = None) -> list[Client]:

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

        train_input_indices, train_output_indices = truncate_windows(train_input_indices, train_output_indices)
        val_input_indices, val_output_indices = truncate_windows(val_input_indices, val_output_indices)
        test_input_indices, test_output_indices = truncate_windows(test_input_indices, test_output_indices)
        real_input_indices, real_output_indices = truncate_windows(real_input_indices, real_output_indices)

        client_id = f"{'-'.join(sorted(normal_data.attrs['inputs'][:]))}"
        client_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, client_id))

        threshold_network = join(THRESHOLD_NETWORK, f"{THRESHOLD_NETWORK_BASENAME}-{client_id}.h5")

        if not isfile(threshold_network):
            threshold_network = None

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

            wide_deep_network=wide_deep_network,
            threshold_network=threshold_network,
        )

        clients.append(client)

    hf.close()

    return clients