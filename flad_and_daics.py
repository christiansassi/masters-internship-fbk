# Configure logging
import logging
logging.basicConfig(
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)

logging.info("Importing types")
from types import SimpleNamespace

logging.info("Importing uuid")
from uuid import uuid4

logging.info("Importing threading")
from threading import Thread, Lock

logging.info("Importing shutil")
import shutil

logging.info("Importing rich")
from rich.progress import Progress

logging.info("Importing wandb")
import wandb

logging.info("Importing h5py")
import h5py

logging.info("Importing numpy")
import numpy as np

# Configuring env
logging.info("Setting up the environment")

import os
from os.path import join, exists
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_CONSOLE"] = "off"

# Loading env
logging.info("Loading the environment")

import dotenv
dotenv.load_dotenv()

# Scikit-learn
logging.info("Importing sklearn")

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Keras
logging.info("Importing keras")

from keras import Input
from keras.models import Model, load_model, clone_model  # type: ignore
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Dense  # type: ignore
from keras.optimizers import Adam  # type: ignore

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

TRAIN_RATIO: float = 0.60 # 60% for training
VAL_EACH_RATIO: float = 0.10 # 10% for each of the two validation sets (20% total)
TEST_EACH_RATIO: float = 0.10 # 10% for each of the two test sets (20% total)

# Lambdas
clear_console = lambda: os.system("cls" if os.name == "nt" else "clear")
clear_wandb = lambda: shutil.rmtree("wandb") if exists("wandb") else None

def get_best_round_digit(n1, n2):

    n1 = str(abs(n1))
    n2 = str(abs(n2))

    if "." in n1 and "." not in n2:
        return 4
    
    if "." not in n1 and "." in n2:
        return 4

    n1 = n1.split(".")[-1]
    n2 = n2.split(".")[-1]

    result = []

    for a, b in zip(n1, n2):
        if a == b:
            result.append(a)
        else:
            break

    return len(''.join(result)) + 1

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
        self._threshold_data.update({"dummy": None})

        # Threshold info (used by FLAD)
        self._threshold_info = {
            "accuracy_score": 0,
            "epochs": 0,
            "steps": 0
        }

        # Calculate total number of samples
        self._samples = sum(len(self._autoencoder_data[item]) * factor for item, factor in [("x_train", 1), ("x_val", 2), ("x_test", 2)])

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

    def autoencoder_evaluate(self, model) -> float:
        
        # Clone the input model
        self._autoencoder = clone(src=model)

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

        # Clone the input model
        self._threshold = clone(src=model)

        # Extract epochs and steps
        epochs = self._threshold_info["epochs"]
        steps = self._threshold_info["steps"]

        if epochs <= 0 or steps <= 0:
            return
        
        # Calculate batch size
        batch_size = int(max(len(self._threshold_data["x_train"]) // steps, 1))

        if "dummy" in self._threshold_data:
            
            # Extract reconstruction errors from train set
            self._threshold_data["x_train"] = self._autoencoder.predict(
                self._threshold_data["x_train"],
                verbose=0
            )

            # Extract reconstruction errors from validation set
            self._threshold_data["x_val"] = self._autoencoder.predict(
                self._threshold_data["x_val"],
                verbose=0
            )

            # Extract reconstruction errors from test set
            self._threshold_data["x_test"] = self._autoencoder.predict(
                self._threshold_data["x_test"],
                verbose=0
            )

            del self._threshold_data["dummy"]

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

    def threshold_evaluate(self, model) -> float:
        
        # Clone the input model
        self._threshold = clone(src=model)

        # Evaluate threshold's performances
        y_pred = self._threshold.predict(
            self._threshold_data["x_test"],
            verbose=0
        )

        score = r2_score(self._threshold_data["x_test"], y_pred)
        
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
        #TODO use different parameters for autoencoder and threshold
        self._min_epochs = min_epochs
        self._max_epochs = max_epochs

        self._min_steps = min_steps
        self._max_steps = max_steps

        self._patience = patience

        self._autoencoder_average_accuracy_score = 0
        self._threshold_average_accuracy_score = 0
    
    def _aggregate_models(self, model_label: str, clients: list, weighted: bool = False):

        if not len(clients):
            return

        all_weights = [getattr(client, f"_{model_label}").get_weights() for client in clients]

        num_layers = len(all_weights[0])
        aggregated_weights = [np.zeros_like(all_weights[0][i]) for i in range(num_layers)]
        total_weight = 0

        if weighted:
    
            # Calculate weighted sum
            for client in clients:
                
                # Access samples directly from the client object
                avg_weight = client._samples
                total_weight = total_weight + avg_weight
                client_weights = getattr(client, f"_{model_label}").get_weights()

                for i in range(num_layers):
                    aggregated_weights[i] += client_weights[i] * avg_weight
        else:

            # Calculate simple average (as in your original `_aggregate_models` but applied to specific clients)
            for client in clients:

                total_weight = total_weight + 1 # Each client contributes 1 to the total for unweighted average
                client_weights = getattr(client, f"_{model_label}").get_weights()

                for i in range(num_layers):
                    aggregated_weights[i] += client_weights[i]

        # Perform the final division to get the average
        if total_weight <= 0:
            return
        
        averaged_weights = [w / total_weight for w in aggregated_weights]

        # Create the aggregated model
        global_model = getattr(self, f"_global_{model_label}")

        aggregated_model = clone(src=global_model, weights=averaged_weights)

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
        
        progress_bar = None
        task_id = None

        _evaluate_clients_lock = Lock()
        _progress_bar_lock = Lock()

        def _partition_clients(_clients: list[Client]) -> list[list]:

            _n = os.cpu_count() - 1
            _k = len(_clients)

            _result = []

            if len(_clients) >= _n:

                _avg = len(_clients) // _n
                _remainder = len(_clients) % _n
                _start = 0

                for i in range(_n):
                    _end = _start + _avg + (1 if i < _remainder else 0)
                    _result.append(_clients[_start:_end])
                    _start = _end

            else:

                _result = [[] for _ in range(_k)]

                for i, item in enumerate(_clients):
                    _result[i % _k].append(item)

            return _result

        def _update_clients(_clients: list[Client]):

            for index, client in enumerate(_clients):

                train_method = getattr(client, f"{model_label}_train")
                train_method(model=getattr(self, f"_global_{model_label}"))

                with _progress_bar_lock:
                    progress_bar.update(task_id=task_id, advance=1)

        def _evaluate_clients(_clients: list[Client]):

            for index, client in enumerate(_clients):

                evaluate_method = getattr(client, f"{model_label}_evaluate")
                accuracy = evaluate_method(model=getattr(self, f"_global_{model_label}"))

                with _evaluate_clients_lock:
                    setattr(self, f"_{model_label}_average_accuracy_score", getattr(self, f"_{model_label}_average_accuracy_score") + accuracy)

                with _progress_bar_lock:
                    progress_bar.update(task_id=task_id, advance=1)

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
            print(f"[{model_label.upper()}] Round #{round_num} | {len(selected_clients)} / {len(self._clients)} | {stopping_counter} / {self._patience}\n")
            progress_bar = Progress()
            progress_bar.start()
            task_id = progress_bar.add_task(total=len(selected_clients) + len(self._clients), description=f"Running round #{round_num}")
            #? ============

            #! Client updates
            # The global model is given to each selected client and trained on their data

            # Single thread
            # for index, client in enumerate(selected_clients):

            #     progress_bar.update(task_id=task_id, description=f"Training {client} ({index + 1} / {len(selected_clients)})")

            #     train_method = getattr(client, f"{model_label}_train")
            #     train_method(model=getattr(self, f"_global_{model_label}"))

            #     progress_bar.update(task_id=task_id, advance=1)

            # Multi-thread
            progress_bar.update(task_id=task_id, description=f"Training {len(selected_clients)} client{'s' if len(selected_clients) > 1 else ''}")

            threads = []

            for chunk in _partition_clients(_clients=selected_clients):
                thread = Thread(target=_update_clients, args=(chunk, ))
                thread.start()
                threads.append(thread)
            
            for thread in threads:
                thread.join()

            # Model aggregation
            self._aggregate_models(model_label=model_label, clients=selected_clients)

            #! Evaluate on all clients' test sets
            setattr(self, f"_{model_label}_average_accuracy_score", 0)

            # Single thread
            # for index, client in enumerate(self._clients):

            #     progress_bar.update(task_id=task_id, description=f"Evaluating {client} ({index + 1} / {len(self._clients)})")

            #     evaluate_method = getattr(client, f"{model_label}_evaluate")
            #     accuracy = evaluate_method(model=getattr(self, f"_global_{model_label}"))

            #     setattr(self, f"_{model_label}_average_accuracy_score", getattr(self, f"_{model_label}_average_accuracy_score") + accuracy)

            #     progress_bar.update(task_id=task_id, advance=1)

            # Multi-thread
            progress_bar.update(task_id=task_id, description=f"Evaluating {len(self._clients)} client{'s' if len(self._clients) > 1 else ''}")

            threads = []

            for chunk in _partition_clients(_clients=self._clients):
                thread = Thread(target=_evaluate_clients, args=(chunk, ))
                thread.start()
                threads.append(thread)
            
            for thread in threads:
                thread.join()

            setattr(self, f"_{model_label}_average_accuracy_score", getattr(self, f"_{model_label}_average_accuracy_score") / len(self._clients))

            #? Logging code
            display_digits = get_best_round_digit(getattr(self, f"_{model_label}_average_accuracy_score"), max_accuracy)
            #? ============

            # Check for improvements
            if getattr(self, f"_{model_label}_average_accuracy_score") > max_accuracy:
                max_accuracy = getattr(self, f"_{model_label}_average_accuracy_score")
                best_model = clone(getattr(self, f"_global_{model_label}"))
                stopping_counter = 0

                # Save best autoencoder model
                progress_bar.update(task_id=task_id, description=f"Saving to {eval(f'{model_label.upper()}_MODEL')}")
                best_model.save(eval(f'{model_label.upper()}_MODEL'))
            
            else:
                stopping_counter = stopping_counter + 1

            progress_bar.update(task_id=task_id, advance=1)
            progress_bar.stop()

            #? Logging code
            clear_console()

            print(f"[{model_label.upper()}] Round #{round_num} | {len(selected_clients)} / {len(self._clients)} | {stopping_counter} / {self._patience}\n")

            for client in self._clients:

                symbol = "🟢" if client in selected_clients else "🔴"

                print(f"{symbol} {client}: {round(abs(getattr(client, f'_{model_label}_info')['accuracy_score']),display_digits)}")
            
            print(f"\nScore: {round(abs(getattr(self, f'_{model_label}_average_accuracy_score')), display_digits)}")
            print(f"Best Score: {round(abs(max_accuracy), display_digits)}\n\n\n")

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

        # Update global model
        setattr(self, f"_global_{model_label}", clone(src=best_model))

        # Assign the best model to each client
        for client in self._clients:
            setattr(client, f"_{model_label}", clone(src=best_model))

        return best_model

def random_iid_clients(x: np.ndarray) -> list:

    # Shuffle data
    rng = np.random.default_rng(seed=SEED)
    rng.shuffle(x)

    samples_per_client = len(x) // N_CLIENTS

    clients = []

    # Each client will receive a specific amount of data
    for i in range(N_CLIENTS):

        autoencoder_start = i * samples_per_client
        autoencoder_end = (i+ 1) * samples_per_client if i < N_CLIENTS - 1 else len(x)

        chunk_autoencoder = x[autoencoder_start:autoencoder_end]

        # Step 1: Split into the main training set and a remainder for all validation/test sets
        # test_size here is the proportion of the original data that will be in the remainder
        x_train, remainder_for_val_test = train_test_split(
            chunk_autoencoder, test_size=(1 - TRAIN_RATIO), random_state=SEED
        )

        # Step 2: Split the 'remainder_for_val_test' into two equal pools.
        # This ensures that x_val1 and x_val2 will be equal, and x_test1 and x_test2 will be equal.
        # test_size=0.5 means it splits the current array into two halves.
        val_test_pool_1, val_test_pool_2 = train_test_split(
            remainder_for_val_test, test_size=0.5, random_state=SEED
        )

        # Step 3: From the first pool (val_test_pool_1), get x_val1 and x_test1
        # The test_size here is the proportion of 'test_each_ratio' relative to the size of this current pool.
        # Each pool (val_test_pool_1 and val_test_pool_2) contains (val_each_ratio + test_each_ratio)
        # of the original data. So, the relative test_size is test_each_ratio / (val_each_ratio + test_each_ratio).
        test_proportion_in_pool = TEST_EACH_RATIO / (VAL_EACH_RATIO + TEST_EACH_RATIO)
        x_val1, x_test1 = train_test_split(
            val_test_pool_1, test_size=test_proportion_in_pool, random_state=SEED
        )

        # Step 4: From the second pool (val_test_pool_2), get x_val2 and x_test2
        # Use the exact same proportion as in Step 3 to ensure equal sizes for val and test sets.
        x_val2, x_test2 = train_test_split(
            val_test_pool_2, test_size=test_proportion_in_pool, random_state=SEED
        )

        clients.append(Client(

            autoencoder_data={
                "x_train": x_train,
                "x_val": x_val1,
                "x_test": x_test1
            },

            threshold_data={
                "x_train": x_train,
                "x_val": x_val2,
                "x_test": x_test2
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
    
    # Output layer predicts a continuous value (the "normal" threshold)
    # No activation or "linear" activation for regression
    output_layer = Dense(1, activation="linear")(x) 

    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Use Mean Squared Error (MSE) for regression
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse", metrics=["mae"])
    
    return model

if __name__ == "__main__":

    # TODO Implement operator feedback (by DAICS)
    # TODO Generate an alarm when an anomaly persists for more than N seconds (by DAICS)

    # TODO Implement logic for generating clients:
    #      - Based on attack type (36 different attacks). How should we handle surrounding samples (e.g., normal ones)? (by FLAD)

    # Load dataset
    logging.info(f"Loading {OUTPUT_NORMAL}")

    hf = h5py.File(name=OUTPUT_NORMAL)
    x = np.array(hf["x"])

    # Create a random autoencoder model
    logging.info("Creating autoencoder model")

    autoencoder = random_autoencoder_model(x_shape=x.shape[1:])
    # autoencoder = load_model(AUTOENCODER_MODEL)

    # Create a random threshold model
    logging.info("Creating threshold model")

    threshold = random_threshold_model()
    
    # Create I.I.D. clients
    logging.info(f"Initializing {N_CLIENTS} client{'s' if N_CLIENTS > 1 else ''}")

    clients = random_iid_clients(x=x)

    # Create the server
    logging.info("Initializing server")

    server = Server(autoencoder=autoencoder, threshold=threshold, clients=clients)

    # Start wandb
    run = wandb_init(name="Autoencoder model")

    # Federated Learning for the autoencoder
    autoencoder = server.federated_learning(model_label="autoencoder")

    # Save best autoencoder model
    logging.info(f"Saving {AUTOENCODER_MODEL}")

    autoencoder.save(AUTOENCODER_MODEL)

    run.finish()

    # Start wandb
    run = wandb_init(name="Threshold model")

    # Federated Learning for the threshold
    threshold = server.federated_learning(model_label="threshold")

    # Save best autoencoder model
    logging.info(f"Saving {THRESHOLD_MODEL}")

    threshold.save(THRESHOLD_MODEL)

    # Stop wandb
    run.finish()

