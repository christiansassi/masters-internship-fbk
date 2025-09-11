import config

from constants import *
from generators import SlidingWindowGenerator
from models import WideDeepNetworkDAICS, ThresholdNetworkDAICS

import h5py
import numpy as np

from uuid import uuid4

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

        normal_inputs: list[str],
        normal_outputs: list[str],

        attack_inputs: list[str],
        attack_outputs: list[str],

        wide_deep_network: WideDeepNetworkDAICS = None,
        threshold_network: ThresholdNetworkDAICS = None
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

        self.normal_inputs = normal_inputs
        self.normal_outputs = normal_outputs

        self.attack_inputs = attack_inputs
        self.attack_outputs = attack_outputs

        self.mask_indices = [GLOBAL_OUTPUTS.index(tag) for tag in self.normal_outputs]
        
        if wide_deep_network is None:
            self.wide_deep_network = WideDeepNetworkDAICS(
                window_past=WINDOW_PAST,
                window_present=WINDOW_PRESENT,
                n_inputs=len(GLOBAL_INPUTS),
                n_outputs=len(GLOBAL_OUTPUTS)
            )

            self.wide_deep_network.build(
                input_shape=(None, WINDOW_PAST, len(GLOBAL_INPUTS))
            )

            self.wide_deep_network.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
                loss=LOSS
            )

            self.wide_deep_network.set_mask(self.mask_indices, WINDOW_PRESENT)
        else:
            self.wide_deep_network = wide_deep_network.clone()

        self.wide_deep_epochs = 0
        self.wide_deep_steps = 0
        self.wide_deep_score = 0

        self.wide_deep_train = pad_client_df_to_global(self.df_train, self.normal_inputs)
        self.wide_deep_train = SlidingWindowGenerator(
            x= self.wide_deep_train,
            y= self.wide_deep_train[:, [GLOBAL_OUTPUTS.index(tag) for tag in self.normal_outputs]],
            input_indices=self.train_input_indices,
            output_indices=self.train_output_indices,
            batch_size=BATCH_SIZE,
            outputs=self.normal_outputs
        )

        self.wide_deep_val = pad_client_df_to_global(self.df_val, self.normal_inputs)
        self.wide_deep_val = SlidingWindowGenerator(
            x=self.wide_deep_val,
            y=self.wide_deep_val[:, [GLOBAL_OUTPUTS.index(tag) for tag in self.normal_outputs]],
            input_indices=self.val_input_indices,
            output_indices=self.val_output_indices,
            batch_size=BATCH_SIZE,
            outputs=self.normal_outputs
        )

        self.wide_deep_test = pad_client_df_to_global(self.df_test, self.normal_inputs)
        self.wide_deep_test = SlidingWindowGenerator(
            x=self.wide_deep_test,
            y=self.wide_deep_test[:, [GLOBAL_OUTPUTS.index(tag) for tag in self.normal_outputs]],
            input_indices=self.test_input_indices,
            output_indices=self.test_output_indices,
            batch_size=BATCH_SIZE,
            outputs=self.normal_outputs
        )

        if threshold_network is None:
            self.threshold_network = ThresholdNetworkDAICS(
                window_past=WINDOW_PAST,
                window_present=WINDOW_PRESENT
            )

            self.threshold_network.build(
                input_shape=(None, WINDOW_PAST, len(self.normal_inputs))
            )

            self.threshold_network.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
                loss=LOSS
            )

        else:
            self.threshold_network = threshold_network.clone()

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

    def set_wide_deep_network(self, wide_deep_network: WideDeepNetworkDAICS) -> float:

        self.wide_deep_network = wide_deep_network.clone()
        self.wide_deep_network.set_mask(self.mask_indices, WINDOW_PRESENT)

def generate_non_iid_clients(wide_deep_network = None,threshold_network = None) -> list[Client]:

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

        client = Client(
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

            normal_inputs=normal_data.attrs["inputs"][:],
            normal_outputs=normal_data.attrs["outputs"][:],

            attack_inputs=attack_data.attrs["inputs"][:],
            attack_outputs=attack_data.attrs["outputs"][:],

            wide_deep_network=wide_deep_network,
            threshold_network=threshold_network,
        )

        clients.append(client)

    hf.close()

    return clients