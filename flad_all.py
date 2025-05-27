import os
from os.path import join, exists
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_CONSOLE"] = "off"

from types import SimpleNamespace
from uuid import uuid4
import shutil

import dotenv
dotenv.load_dotenv() # load env

import h5py
import numpy as np

# Scikit-learn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Keras
from keras import Input
from keras.models import Model, load_model, clone_model  # type: ignore
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Reshape, Dense  # type: ignore
from keras.optimizers import Adam  # type: ignore
from keras.losses import BinaryCrossentropy  # type: ignore

# TensorFlow Keras
from tensorflow.keras.optimizers import Adam  # type: ignore

from rich.progress import Progress
import wandb

# FLAD hyperparameters (Section 5, Table IV)
MIN_EPOCHS: int = 1
MAX_EPOCHS: int = 5

MIN_STEPS: int = 10
MAX_STEPS: int = 1000

PATIENCE: int = 25

# Test hyperparameters
N_CLIENTS: int = 30
SEED: int = 42

DROPOUT_RATE: float = 0
DENSE_LAYERS: int = 2
HIDDEN_UNITS: int = 10
ACTIVATION: str = "relu"
LEARNING_RATE: float = 0.0001

# Wandb object
run = SimpleNamespace()
run.log = lambda *args: None
run.finish = lambda *args: None

wandb_init = lambda name: wandb.init(
    entity=os.getenv("ENTITY"),
    project=os.getenv("PROJECT"),
    name=name
)

# Files
ROOT: str = "datasets"

SWAT_2015: str = join(ROOT, "SWaT2015")

INPUT_NORMAL: str = join(SWAT_2015, "original", "SWaT_Dataset_Normal.csv") # Only normal records
INPUT_ATTACK: str = join(SWAT_2015, "original", "SWaT_Dataset_Attack.csv") # Normal and attack records

OUTPUT_NORMAL: str = join(SWAT_2015, "processed", "SWaT_Dataset_Normal.hdf5") # Only normal records
OUTPUT_ATTACK: str = join(SWAT_2015, "processed", "SWaT_Dataset_Attack.hdf5") # Normal and attack records

AUTOENCODER_MODEL: str = join(SWAT_2015, "models", "autoencoder.keras") # Autoencoder model
THRESHOLD_MODEL: str = join(SWAT_2015, "models", "threshold.keras") # Threshold model

# Lambdas
clear_console = lambda: os.system("cls" if os.name == "nt" else "clear")
clear_wandb = lambda: shutil.rmtree("wandb") if exists("wandb") else None

def clone(src, weights = None):

    # Clone model
    model = clone_model(src)

    # Build model (if not done yet)
    if not model.built:
        model.build(src.input_shape)
    
    # Set weights from src (input model) or given weights
    model.set_weights(src.get_weights() if weights is None else weights)

    return model

class Client:

    def __init__(self, autoencoder_data: dict, threshold_data: dict):
        
        # Generate an id for the current client (useful while debugging)
        self._id = str(uuid4())

        # Autoencoder model
        self._autoencoder = None

        # Autoencoder data (x_train, x_val, x_test)
        self._autoencoder_data = autoencoder_data

        # Autoencoder info (used by FLAD)
        self._autoencoder_info = {
            "accuracy_score": 0,
            "epochs": 0,
            "steps": 0
        }

        # Threshold model
        self._threshold = None

        # Threshold initial data (it will be modified after the autoencoder training)
        self._threshold_data = threshold_data

        # Threshold info (used by FLAD)
        self._threshold_info = {
            "accuracy_score": 0,
            "epochs": 0,
            "steps": 0
        }

    def __str__(self) -> str:
        return self._id
    
    def autoencoder_train(self, model):
        
        # Clone the input model
        self._autoencoder = clone(src=model)

        # Extract epochs and steps
        epochs = self._autoencoder_info["epochs"]
        steps = self._autoencoder_info["steps"]

        if epochs <= 0 or steps <= 0:
            return

        # Calculate batch size
        batch_size = int(max(len(self._autoencoder_data["x_train"]) // steps, 1))

        # Train the autoencoder
        self._autoencoder.fit(
            self._autoencoder_data["x_train"],
            self._autoencoder_data["x_train"],

            validation_data=(
                self._autoencoder_data["x_val"], 
                self._autoencoder_data["x_val"]
            ),

            epochs=epochs,
            batch_size=batch_size,

            verbose=0
        )

    def autoencoder_evaluate(self) -> float:
        
        # Evaluate autoencoder's performances
        y_pred = self._autoencoder.predict(
            self._autoencoder_data["x_test"],
            verbose=0
        )

        y_true = self._autoencoder_data["x_test"].reshape(y_pred.shape)

        # Calculate the reconstruction error
        # Note: The reconstruction error is negated because this implementation follows FLAD,
        # where the objective is to maximize accuracy. In contrast, autoencoders aim to minimize
        # reconstruction error. By converting the error to a negative value, we align with FLAD's
        # maximization logic while still using the reconstruction loss
        self._autoencoder_info["accuracy_score"] = -np.mean(np.square(y_true - y_pred))

        return self._autoencoder_info["accuracy_score"]

    def threshold_train(self, model) -> None:
        
        def split(_x) -> tuple:
            
            if not len(_x):
                _empty_shape = (0,) + _x.shape[1:]
                return np.empty(_empty_shape), np.empty(_empty_shape), np.empty(_empty_shape)

            # Train 60%
            # Val 20%
            # Test 20%

            _train, _test = train_test_split(
                _x,
                test_size=0.2,
                random_state=SEED
            )

            _train, _val = train_test_split(
                _train,
                test_size=0.2,
                random_state=SEED
            )

            return _train, _val, _test

        def calculate_reconstruction_errors(_x) -> np.ndarray:

            if not len(_x):
                return np.array([]) # Returns (0,) shape
            
            # Calculate the reconstruction error based on the trained autoencoder
            _reconstructions = self._autoencoder.predict(
                _x,
                verbose=0
            )

            _errors = np.mean(np.square(_x - _reconstructions), axis=(1, 2))

            # Ensure errors is always at least 1D, even if it comes out as a scalar
            if _errors.ndim == 0:
                _errors = _errors.reshape(1)

            return _errors
            
        def prepare_x_y(_x_benign, _x_malicious) -> tuple:
            
            # Use the benign samples to extract all the associated reconstruction errors
            _benign_errors = calculate_reconstruction_errors(_x=_x_benign)
            _benign_labels = np.zeros(len(_benign_errors))

            # Use the malicious samples to extract all the associated reconstruction errors
            _malicious_errors = calculate_reconstruction_errors(_x=_x_malicious)
            _malicious_labels = np.ones(len(_malicious_errors))

            # The input of the threshold model will be the combination of the reconstruction errors
            _x = np.concatenate((_benign_errors, _malicious_errors))
            _x = _x.reshape(-1, 1)

            # The output of the threshold model will be 0 or 1, 
            # depending if a reconstruction error refers to benign or malicious sample
            _y = np.concatenate((_benign_labels, _malicious_labels)).astype(int)

            return _x, _y

        # Clone the input model
        self._threshold = clone(src=model)

        # Extract epochs and steps
        epochs = self._threshold_info["epochs"]
        steps = self._threshold_info["steps"]

        if epochs <= 0 or steps <= 0:
            return
        
        # If it is the first time the model is trained
        if len(self._threshold_data) != 6:
            
            x = self._threshold_data["x"]
            y = self._threshold_data["y"]

            # Split benign and malicious samples
            x_benign = x[y == 0]
            x_malicious = x[y == 1]
            
            # Split in train, val and test
            x_train_benign, x_val_benign, x_test_benign = split(_x=x_benign)
            x_train_malicious, x_val_malicious, x_test_malicious = split(_x=x_malicious)

            x_train, y_train = prepare_x_y(_x_benign=x_train_benign, _x_malicious=x_train_malicious)
            x_val, y_val = prepare_x_y(_x_benign=x_val_benign, _x_malicious=x_val_malicious)
            x_test, y_test = prepare_x_y(_x_benign=x_test_benign, _x_malicious=x_test_malicious)

            # Use the same structure as the one used for the autoencoder
            self._threshold_data = {
                "x_train": x_train,
                "y_train": y_train,

                "x_val": x_val,
                "y_val": y_val,

                "x_test": x_test,
                "y_test": y_test
            }
        
        # Calculate batch size
        batch_size = int(max(len(self._threshold_data["x_train"]) // steps, 1))

        # Train the threshold model
        self._threshold.fit(
            self._threshold_data["x_train"],
            self._threshold_data["y_train"],

            epochs=epochs,
            batch_size=batch_size,

            validation_data=(
                self._threshold_data["x_val"], 
                self._threshold_data["y_val"]
            ),

            verbose=0
        )

    def threshold_evaluate(self) -> float:
        
        # Evaluate threshold's performances
        y_pred_proba = self._threshold.predict(
            self._threshold_data["x_test"],
            verbose=0
        )

        # Convert each probability into one of the two classes:
        # - Benign 0
        # - Malicious 1
        #TODO change 0.5. More tests needed
        y_pred_class = (y_pred_proba > 0.5).astype(int)

        # Use the accuracy score to evaluate the performances
        #TODO use F1 score instead?
        score = accuracy_score(self._threshold_data["y_test"], y_pred_class)
        
        self._threshold_info["accuracy_score"] = score

        return self._threshold_info["accuracy_score"]

class Server:

    def __init__(self, autoencoder, threshold, clients: list[Client], min_epochs: int = MIN_EPOCHS, max_epochs: int = MAX_EPOCHS, min_steps: int = MIN_STEPS, max_steps: int = MAX_STEPS, patience: int = PATIENCE):

        self._global_autoencoder = clone(autoencoder)
        self._global_threshold = clone(threshold)

        self._clients = clients

        # Give to each client a copy of the autoencoder and threshold models
        # Normally, this is not required. However, if the entire federated learning process
        # is split into two separate sessions—one for training the autoencoder and another for training the threshold—
        # this prevents errors by ensuring the server has at least one of the two models from the start.
        # This approach makes sense if the autoencoder has already been trained.
        # It does not make sense for the threshold, as it depends on the autoencoder.
        for client in self._clients:
            client._autoencoder = clone(autoencoder)
            client._threshold = clone(threshold)

        # Set FLAD hyperparameters
        #TODO: use different parameters for autoencoder and threshold
        self._min_epochs = min_epochs
        self._max_epochs = max_epochs

        self._min_steps = min_steps
        self._max_steps = max_steps

        self._patience = patience

        self._autoencoder_average_accuracy_score = 0
        self._threshold_average_accuracy_score = 0
    
    def _aggregate_models(self, model_label: str):

        # Get weights from all the clients
        all_weights = [getattr(client, f"_{model_label}").get_weights() for client in self._clients]

        # Calculate the average for each layer
        average_weights = [np.mean(layer, axis=0) for layer in zip(*all_weights)]

        # Create the aggregated model
        global_model = getattr(self, f"_global_{model_label}")
        aggregated_model = clone(src=global_model, weights=average_weights)

        # Update global model
        setattr(self, f"_global_{model_label}", aggregated_model)
    
    def _select_clients(self, model_label: str) -> list:
        
        selected_clients = []

        min_accuracy_score = float("inf")
        max_accuracy_score = float("-inf")

        for client in self._clients:

            # Select clients that are performing poorly (below the mean)
            if getattr(client, f"_{model_label}_info")["accuracy_score"] > getattr(self, f"_{model_label}_average_accuracy_score"):
                continue

            min_accuracy_score = min(min_accuracy_score, getattr(client, f"_{model_label}_info")["accuracy_score"])
            max_accuracy_score = max(max_accuracy_score, getattr(client, f"_{model_label}_info")["accuracy_score"])

            selected_clients.append(client)
        
        for index, client in enumerate(selected_clients):
            
            # Calculate the scaling factor
            if max_accuracy_score != min_accuracy_score:
                scaling_factor = (max_accuracy_score - getattr(client, f"_{model_label}_info")["accuracy_score"]) / (max_accuracy_score - min_accuracy_score)
            else:
                scaling_factor = 0

            # Calculate epochs and steps
            info = getattr(client, f"_{model_label}_info")
            info["epochs"] = int(self._min_epochs + (self._max_epochs - self._min_epochs) * scaling_factor)
            info["steps"] = int(self._min_steps + (self._max_steps - self._min_steps) * scaling_factor)

            selected_clients[index] = client
        
        return selected_clients
    
    def get_global_autoencoder_model(self):
        return clone(src=self._global_autoencoder) if self._global_autoencoder is not None else None
    
    def get_global_threshold_model(self):
        return clone(src=self._global_threshold) if self._global_threshold is not None else None
    
    def federated_learning(self, model_label: str):
        
        clear_console()

        # All the clients partecipate in the first round
        selected_clients = self._select_clients(model_label=model_label)

        best_model = getattr(self, f"_global_{model_label}")
        max_accuracy = float("-inf")
        stopping_counter = 0
        round_num = 0

        while True:

            # Keep track of each round
            round_num = round_num + 1

            #? Logging code
            print(f"[{model_label.upper()}] Round #{round_num} | {stopping_counter} / {self._patience}\n")
            progress_bar = Progress()
            progress_bar.start()
            task_id = progress_bar.add_task(total=len(selected_clients) + len(self._clients), description=f"Running round #{round_num}")
            #? ============

            # Client updates
            # The global model is given to each selected client and trained on their data
            for index, client in enumerate(selected_clients):

                progress_bar.update(task_id=task_id, description=f"Training {client} ({index + 1} / {len(selected_clients)})")

                train_method = getattr(client, f"{model_label}_train")
                train_method(model=getattr(self, f"_global_{model_label}"))

                progress_bar.update(task_id=task_id, advance=1)

            # Evaluate on all clients' test sets
            setattr(self, f"_{model_label}_average_accuracy_score", 0)

            for index, client in enumerate(self._clients):

                progress_bar.update(task_id=task_id, description=f"Evaluating {client} ({index + 1} / {len(self._clients)})")

                evaluate_method = getattr(client, f"{model_label}_evaluate")
                accuracy = evaluate_method()

                setattr(self, f"_{model_label}_average_accuracy_score", getattr(self, f"_{model_label}_average_accuracy_score") + accuracy)

                progress_bar.update(task_id=task_id, advance=1)

            setattr(self, f"_{model_label}_average_accuracy_score", getattr(self, f"_{model_label}_average_accuracy_score") / len(self._clients))

            progress_bar.stop()

            # Check for improvements
            if getattr(self, f"_{model_label}_average_accuracy_score") > max_accuracy:
                max_accuracy = getattr(self, f"_{model_label}_average_accuracy_score")
                best_model = clone(getattr(self, f"_global_{model_label}"))
                stopping_counter = 0
            
            else:
                stopping_counter = stopping_counter + 1

            #? Logging code
            clear_console()

            print(f"[{model_label.upper()}] Round #{round_num} | {stopping_counter} / {self._patience}\n")

            for client in self._clients:

                symbol = "🟢" if client in selected_clients else "🔴"

                print(f"{symbol} {client}: {round(abs(getattr(client, f'_{model_label}_info')['accuracy_score']),4)}")
            
            print(f"\nScore: {round(abs(getattr(self, f'_{model_label}_average_accuracy_score')), 4)}")
            print(f"Best Score: {round(abs(max_accuracy), 4)}\n\n\n")

            run.log({
                "best": round(abs(max_accuracy), 4),
                "round": round_num, 
                "score": abs(getattr(self, f"_{model_label}_average_accuracy_score")), 
                "clients": len(selected_clients)
            })
            #? ============

            # Check stopping condition
            if stopping_counter >= self._patience:
                break
            
            # Select clients for the next round
            selected_clients = self._select_clients(model_label=model_label)

            # Model aggregation
            self._aggregate_models(model_label=model_label)

        # Update global model
        setattr(self, f"_global_{model_label}", clone(src=best_model))

        # Assign the best model to each client
        for client in self._clients:
            setattr(client, f"_{model_label}", clone(src=best_model))

        return best_model

def random_iid_clients(x_autoencoder: np.ndarray, x_threshold: np.ndarray, y_threshold: np.ndarray) -> list:

    #TODO add shuffle

    autoencoder_samples_per_client = len(x_autoencoder) // N_CLIENTS
    threshold_samples_per_client = len(x_threshold) // N_CLIENTS

    clients = []

    # Each client will receive a specific amount of data
    for i in range(N_CLIENTS):

        autoencoder_start = i * autoencoder_samples_per_client
        autoencoder_end = (i+ 1) * autoencoder_samples_per_client if i < N_CLIENTS - 1 else len(x_autoencoder)

        threshold_start = i * threshold_samples_per_client
        threshold_end = (i+ 1) * threshold_samples_per_client if i < N_CLIENTS - 1 else len(x_threshold)

        chunk_autoencoder = x_autoencoder[autoencoder_start:autoencoder_end]

        chunk_threshold_x = x_threshold[threshold_start:threshold_end]
        chunk_threshold_y = y_threshold[threshold_start:threshold_end]

        x_train, x_test = train_test_split(
            chunk_autoencoder,
            test_size=0.2,
            random_state=SEED
        )

        x_train, x_val = train_test_split(
            x_train,
            test_size=0.2,
            random_state=SEED
        )

        assert all(len(obj) > 0 for obj in [x_train, x_val, x_test, chunk_threshold_x, chunk_threshold_y])

        clients.append(Client(

            autoencoder_data={
                "x_train": x_train,
                "x_val": x_val,
                "x_test": x_test
            },

            threshold_data={
                "x": chunk_threshold_x,
                "y": chunk_threshold_y
            }
        ))

    return clients

def random_autoencoder_model(x_shape: tuple):

    _, input_features = x_shape
    input_layer = Input(shape=x_shape) # (timesteps, features)

    # Encoder
    # Layer 1
    x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(input_layer)
    x = MaxPooling1D(pool_size=2, padding="same")(x)

    # Bottleneck Layer (Encoded representation)
    encoded = Conv1D(filters=HIDDEN_UNITS, kernel_size=3, activation="relu", padding="same")(x)

    # Decoder
    # Layer 1
    x = Conv1D(filters=HIDDEN_UNITS, kernel_size=3, activation="relu", padding="same")(encoded)
    x = UpSampling1D(size=2)(x) # (timesteps, hidden_units) e.g., (10, 32)

    x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(x)

    # Output layer: filters should match the number of input features
    # Activation "linear" for reconstructing real-valued data
    output_layer = Conv1D(filters=input_features, kernel_size=3, activation="linear", padding="same")(x)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")

    return autoencoder

def random_threshold_model():

    input_layer = Input(shape=(1,)) 
    x = Dense(32, activation="relu")(input_layer)
    x = Dense(16, activation="relu")(x)
    output_layer = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=BinaryCrossentropy(), metrics=["accuracy"])

    return model

if __name__ == "__main__":

    # TODO Implement operator feedback (by DAICS)
    # TODO Generate an alarm when an anomaly persists for more than N seconds (by DAICS)

    # TODO Implement logic for generating clients:
    #      - Based on attack type (36 different attacks). How should we handle surrounding samples (e.g., normal ones)? (by FLAD)

    # Load dataset
    hf = h5py.File(name=OUTPUT_NORMAL)
    x_autoencoder = np.array(hf["x"])

    hf = h5py.File(name=OUTPUT_ATTACK)
    x_threshold = np.array(hf["x"])
    y_threshold = np.array(hf["y"])

    # Create I.I.D. clients
    clients = random_iid_clients(
        x_autoencoder=x_autoencoder,
        x_threshold=x_threshold,
        y_threshold=y_threshold
    )

    # Create a random autoencoder model
    autoencoder = random_autoencoder_model(x_shape=x_autoencoder.shape[1:])
    # autoencoder = load_model(AUTOENCODER_MODEL)

    # Create a random threshold model
    threshold = random_threshold_model()

    # Create the server
    server = Server(autoencoder=autoencoder, threshold=threshold, clients=clients)

    # Start wandb
    run = wandb_init(name="Autoencoder model")

    # # Federated Learning for the autoencoder
    autoencoder = server.federated_learning(model_label="autoencoder")

    # Save best autoencoder model
    autoencoder.save(AUTOENCODER_MODEL)

    run.finish()

    # Start wandb
    run = wandb_init(name="Threshold model")

    # Federated Learning for the threshold
    threshold = server.federated_learning(model_label="threshold")

    # Save best autoencoder model
    threshold.save(THRESHOLD_MODEL)

    # Stop wandb
    run.finish()

