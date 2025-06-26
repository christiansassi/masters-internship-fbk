# Local imports
import config
import utils

# External imports
from os import getenv, listdir, remove
from os.path import join

import h5py
import numpy as np

from keras import Input
from keras.models import Model, load_model, clone_model  # type: ignore
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Dense  # type: ignore
from keras.optimizers import Adam  # type: ignore

import tensorflow as tf

from sklearn.model_selection import train_test_split

from uuid import uuid4

import logging
utils.configure_log()

def clone(src: Model, weights: list | None = None):
    """
    Creates a clone of a Keras model, optionally setting its weights.

    :param src: The source Keras model to clone.
    :type src: `tf.keras.Model`

    :param weights: Optional weights to set for the cloned model. If `None`, the weights from the source model (`src`) are used. Defaults to `None`.
    :type weights: `list | None`

    :returns: A new Keras model instance, cloned from `src` and compiled with the same configuration, with its weights initialized.
    :rtype: `tf.keras.Model`
    """

    # Clone model
    model = clone_model(src)

    # Build model (if not done yet)
    if not model.built:
        model.build(src.input_shape)
    
    # Set weights from src (input model) or given weights
    model.set_weights(src.get_weights() if weights is None else weights)

    if src.optimizer is not None and src.loss is not None:
        model.compile(
            optimizer=src.optimizer.__class__.from_config(src.optimizer.get_config()),
            loss=src.loss,
            metrics=src.metrics
        )
        
    return model

class Client:
    def __init__(self, autoencoder_data: dict):

        # Generate an id for the current client (useful while debugging)
        self._id: str = str(uuid4())

        # Autoencoder model
        self._autoencoder: Model = None

        # Autoencoder data (x_train, x_val, x_test)
        self._autoencoder_data: dict = autoencoder_data

        # Autoencoder info (used by FLAD)
        self._autoencoder_info: dict = {
            "accuracy_score": 0,
            "epochs": 0,
            "steps": 0
        }

        # Threshold model
        self._threshold: Model = None

        # Threshold data (x_train, x_val, x_test) initially set to None
        self._threshold_data: dict = {}

        # Threshold info
        self._threshold_info: dict = {
            "accuracy_score": 0,
            "epochs": config.FLADHyperparameters.MAX_EPOCHS,
            "steps": config.FLADHyperparameters.MAX_STEPS
        }

        # Calculate total samples
        self._samples: int = sum([len(self._autoencoder_data[key]) for key in self._autoencoder_data.keys()])

    def __str__(self) -> str:
        return self._id
    
    def _autoencoder_train(self, model: Model):
        """
        Trains the local autoencoder model using the provided Keras autoencoder model as a baseline.

        :param model: The Keras autoencoder model to use as the baseline for training.
        :type model: `tf.keras.Model`
        """
        
        # Clone the input model
        self._autoencoder = clone(src=model)

        # Extract epochs and steps
        epochs = self._autoencoder_info["epochs"]
        steps = self._autoencoder_info["steps"]

        if epochs <= 0 or steps <= 0:
            return
        
        # Calculate train and validation batch size
        train_batch_size = int(max(len(self._autoencoder_data["x_train"]) // steps, 1))
        val_batch_size = int(max(len(self._autoencoder_data["x_val"]) // steps, 1))

        # Calculate steps
        steps_per_epoch = len(self._autoencoder_data["x_train"]) // train_batch_size
        validation_steps = len(self._autoencoder_data["x_val"]) // val_batch_size

        x_train = tf.data.Dataset.from_tensor_slices(
            (self._autoencoder_data["x_train"], self._autoencoder_data["x_train"])
        ).batch(batch_size=train_batch_size, drop_remainder=True).cache().repeat().prefetch(buffer_size=tf.data.AUTOTUNE) # Use train_batch_size

        x_val = tf.data.Dataset.from_tensor_slices(
            (self._autoencoder_data["x_val"], self._autoencoder_data["x_val"])
        ).batch(batch_size=val_batch_size, drop_remainder=True).cache().repeat().prefetch(buffer_size=tf.data.AUTOTUNE) # Use val_batch_size

        # Train the autoencoder
        self._autoencoder.fit(
            x_train,

            validation_data=x_val,

            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,

            verbose=config.VERBOSE
        )
    
    def _autoencoder_evaluate(self, model: Model) -> float:
        """
        Evaluates the given autoencoder model using the current client's data.
        The provided autoencoder model should ideally be the result of an aggregation of models.

        :param model: The Keras autoencoder model to be evaluated.
        :type model: `tf.keras.Model`

        :return: The accuracy score.
        :rtype: `float`
        """
        
        # Clone the input model to ensure evaluation is on a fresh copy or specific aggregated model
        self._autoencoder = clone(src=model)

        # Extract steps
        steps = self._autoencoder_info["steps"]

        if steps <= 0:
            return

        # Calculate batch size
        batch_size = int(max(len(self._autoencoder_data["x_test"]) // steps, 1))

        # Create a TensorFlow Dataset for the test data.
        # `drop_remainder=True` is used to ensure all batches passed to `predict` have
        # the same shape, which helps prevent `tf.function` retracing warnings.
        x_test_dataset = tf.data.Dataset.from_tensor_slices(
            tensors=self._autoencoder_data["x_test"]
        ).batch(batch_size=batch_size, drop_remainder=True)

        # Perform prediction using the batched dataset.
        y_pred = self._autoencoder.predict(
            x_test_dataset,
            verbose=config.VERBOSE
        )

        # Reconstruct the ground truth (y_true) from the same batched dataset.
        # This is crucial because `drop_remainder=True` means `y_pred` will not
        # include predictions for any partial last batch, so `y_true` must match
        # the exact data that `y_pred` was generated from.
        y_true_list = []

        for batch in x_test_dataset:
            y_true_list.append(batch.numpy()) # Convert TensorFlow tensor batch back to NumPy array

        y_true = np.concatenate(y_true_list, axis=0) # Combine all batches into a single NumPy array

        # Calculate the reconstruction error.
        # As per the project's FLAD specific implementation, the error is negated
        # to align with an objective of maximizing accuracy, even though autoencoders
        # traditionally minimize reconstruction error.
        self._autoencoder_info["accuracy_score"] = -np.mean(np.square(y_true - y_pred))

        # Return the calculated "accuracy score" (negated reconstruction error).
        return self._autoencoder_info["accuracy_score"]

    def _threshold_train(self, model: Model):
        """
        Trains the local threshold model using the provided Keras threshold model as a baseline.

        :param model: The Keras threshold model to use as the baseline for training.
        :type model: `tf.keras.Model`
        """

        # Clone the input model
        self._threshold = clone(src=model)

        # Extract steps
        steps = self._threshold_info["steps"]
        epochs = self._threshold_info["epochs"]

        # Derive the threshold data from the autoencoder data
        for key in ["x_train", "x_val", "x_test"]:

            data = self._autoencoder_data[key]

            # Calculate batch size for prediction.
            # This ensures consistent batch sizes if `drop_remainder=True` is used.
            # The batch size is derived from the training data parameters for consistency.
            batch_size = int(max(len(data) // steps, 1))

            # Create a TensorFlow Dataset for the input data.
            dataset = tf.data.Dataset.from_tensor_slices(
                tensors=data
            ).batch(batch_size=batch_size, drop_remainder=False)

            # Perform prediction using the batched dataset.
            y_pred = self._autoencoder.predict(
                dataset,
                verbose=config.VERBOSE
            )

            # Reconstruct y_true from the same batched dataset to ensure matching shapes
            # even if drop_remainder was used (though we set it to False here for full data).
            y_true_list = []

            for batch in dataset:
                y_true_list.append(batch.numpy())

            y_true = np.concatenate(y_true_list, axis=0)

            # Calculate the squared reconstruction error for each sample.
            self._threshold_data[key] = np.mean(np.square(y_true - y_pred), axis=(1, 2))
        
        # Calculate train and validation batch size
        train_batch_size = int(max(len(self._threshold_data["x_train"]) // steps, 1))
        val_batch_size = int(max(len(self._threshold_data["x_val"]) // steps, 1))

        # Calculate steps
        steps_per_epoch = len(self._threshold_data["x_train"]) // train_batch_size
        validation_steps = len(self._threshold_data["x_val"]) // val_batch_size

        x_train = tf.data.Dataset.from_tensor_slices(
            (self._threshold_data["x_train"], self._threshold_data["x_train"])
        ).batch(batch_size=train_batch_size, drop_remainder=True).cache().repeat().prefetch(buffer_size=tf.data.AUTOTUNE)

        x_val = tf.data.Dataset.from_tensor_slices(
            (self._threshold_data["x_val"], self._threshold_data["x_val"])
        ).batch(batch_size=val_batch_size, drop_remainder=True).cache().repeat().prefetch(buffer_size=tf.data.AUTOTUNE)

        # Train the threshold
        self._threshold.fit(
            x_train,

            validation_data=x_val,

            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,

            verbose=config.VERBOSE
        )

    def _threshold_evaluate(self) -> float:
        """
        Evaluates the given threshold model using the current client's data.

        :param model: The Keras threshold model to be evaluated.
        :type model: `tf.keras.Model`

        :return: The accuracy score.
        :rtype: `float`
        """

        # Extract steps
        steps = self._threshold_info["steps"]

        # Calculate batch size
        batch_size = int(max(len(self._threshold_data["x_test"]) // steps, 1))

        # Create a TensorFlow Dataset for the test data.
        # `drop_remainder=True` is used to ensure all batches passed to `predict` have
        # the same shape, which helps prevent `tf.function` retracing warnings.
        x_test_dataset = tf.data.Dataset.from_tensor_slices(
            tensors=self._threshold_data["x_test"]
        ).batch(batch_size=batch_size, drop_remainder=True)

        # Perform prediction using the batched dataset.
        y_pred = self._threshold.predict(
            x_test_dataset,
            verbose=config.VERBOSE
        )

        # Reconstruct the ground truth (y_true) from the same batched dataset.
        # This is crucial because `drop_remainder=True` means `y_pred` will not
        # include predictions for any partial last batch, so `y_true` must match
        # the exact data that `y_pred` was generated from.
        y_true_list = []

        for batch in x_test_dataset:
            y_true_list.append(batch.numpy()) # Convert TensorFlow tensor batch back to NumPy array

        y_true = np.concatenate(y_true_list, axis=0) # Combine all batches into a single NumPy array

        # Calculate the reconstruction error.
        # As per the project's FLAD specific implementation, the error is negated
        # to align with an objective of maximizing accuracy, even though autoencoders
        # traditionally minimize reconstruction error.
        self._threshold_info["accuracy_score"] = np.mean(np.square(y_true - y_pred))

        return self._threshold_info["accuracy_score"]

    def get_autoencoder_model(self) -> Model:
        """
        Returns the autoencoder model of the current client

        :return: The autoencoder model.
        :rtype: `tf.keras.Model`
        """
        return clone(src=self._threshold)

    def get_threshold_model(self) -> Model:
        """
        Returns the threshold model of the current client

        :return: The threshold model.
        :rtype: `tf.keras.Model`
        """
        return clone(src=self._threshold)

class Server:
    def __init__(
            self, 
            autoencoder: Model, 
            threshold: Model, 
            clients: list[Client], 
            min_epochs: int = config.FLADHyperparameters.MIN_EPOCHS,
            max_epochs: int = config.FLADHyperparameters.MAX_EPOCHS,
            min_steps: int = config.FLADHyperparameters.MIN_STEPS,
            max_steps: int = config.FLADHyperparameters.MAX_STEPS,
            patience: int = config.FLADHyperparameters.PATIENCE
        ):

        # Set the initial global autoencoder and threshold models
        self._global_autoencoder: Model = clone(src=autoencoder)
        self._global_threshold: Model = clone(src=threshold)

        # Set the clients
        self._clients: list[Client] = clients
        
        #! This is not needed during the deployment
        for client in self._clients:
            client._autoencoder = clone(src=self._global_autoencoder)
            client._threshold = clone(src=self._global_threshold)

        # Set FLAD hyperparameters
        self._min_epochs: int = min_epochs
        self._max_epochs: int = max_epochs
        self._min_steps: int = min_steps
        self._max_steps: int = max_steps
        self._patience: int = patience

        self._autoencoder_average_accuracy_score: float = 0
    
    def _aggregate_models(self, clients: list[Client], weighted: bool = False):
        """
        Aggregates the autoencoder models from the given clients to update the global autoencoder model.

        :param clients: A list of clients whose models will contribute to the aggregation.
        :type clients: `list[Client]`

        :param weighted: If `True`, the aggregation will be weighted based on the number of samples from each client. Defaults to `False`.
        :type weighted: `bool`
        """
        
        if not len(clients):
            return

        # Retrieve weights from the first client. 
        # All clients are expected to have identical weight structures.
        weights = [clients[0]._autoencoder.get_weights()]

        num_layers = len(weights[0])
        aggregated_weights = [np.zeros_like(weights[0][i]) for i in range(num_layers)]
        total_weight = 0

        if weighted:
    
            # Calculate weighted sum
            for client in clients:
                
                # Access samples directly from the client object
                avg_weight = client._samples
                total_weight = total_weight + avg_weight
                client_weights = client._autoencoder.get_weights()

                for i in range(num_layers):
                    aggregated_weights[i] = aggregated_weights[i] + client_weights[i] * avg_weight
        else:

            # Calculate simple average
            for client in clients:

                total_weight = total_weight + 1 # Each client contributes 1 to the total for unweighted average
                client_weights = client._autoencoder.get_weights()

                for i in range(num_layers):
                    aggregated_weights[i] = aggregated_weights[i] + client_weights[i]

        # Perform the final division to get the average
        averaged_weights = [w / total_weight for w in aggregated_weights]

        # Update the global model with the aggregated weights
        self._global_autoencoder = clone(src=self._global_autoencoder, weights=averaged_weights)

    def _select_clients(self) -> list[Client]:
        """
        Selects clients based on the FLAD logic.

        :return: A list of the selected clients.
        :rtype: `list[Client]`
        """

        selected_clients = []

        min_accuracy_score = float("inf")
        max_accuracy_score = float("-inf")

        for client in self._clients:

            # Select clients that are performing poorly (below the mean)
            if client._autoencoder_info["accuracy_score"] > self._autoencoder_average_accuracy_score:
                continue

            min_accuracy_score = min(min_accuracy_score, client._autoencoder_info["accuracy_score"])
            max_accuracy_score = max(max_accuracy_score, client._autoencoder_info["accuracy_score"])

            selected_clients.append(client)

        for index, client in enumerate(selected_clients):

            # Calculate the scaling factor
            if max_accuracy_score != min_accuracy_score:
                scaling_factor = (max_accuracy_score - client._autoencoder_info["accuracy_score"]) / (max_accuracy_score - min_accuracy_score)
            else:
                scaling_factor = 0

            # Calculate epochs and steps
            client._autoencoder_info["epochs"] = int(self._min_epochs + (self._max_epochs - self._min_epochs) * scaling_factor)
            client._autoencoder_info["steps"] = int(self._min_steps + (self._max_steps - self._min_steps) * scaling_factor)

            selected_clients[index] = client

        return selected_clients
    
    def federated_learning(self):

        # All the clients partecipate in the first round
        selected_clients = self._select_clients()

        best_model = self._global_autoencoder
        max_accuracy_score = float("-inf")
        round_num = 0
        stop_counter = 0

        while True:

            print("")
            logging.info(f"---------- Round {round_num + 1} ----------")

            # Keep track of each round
            round_num = round_num + 1

            # Update clients
            print(f"{utils.log_timestamp_status()} Updating {len(selected_clients)} client(s): 0 / {len(selected_clients)}", end="\r" if len(selected_clients) > 1 else "\n")

            for index, client in enumerate(selected_clients):
                client._autoencoder_train(model=self._global_autoencoder)

                print(f"{utils.log_timestamp_status()} Updating {len(selected_clients)} client(s): {index + 1} / {len(selected_clients)}", end="\r" if index != len(selected_clients) - 1 else "\n")
            
            # Model aggregation
            logging.info(f"Aggregating models of {len(selected_clients)} client(s)")

            self._aggregate_models(clients=self._clients)

            # Evaluate clients
            print(f"{utils.log_timestamp_status()} Evaluating {len(self._clients)} client(s): 0 / {len(self._clients)}", end="\r" if len(self._clients) > 1 else "\n")

            self._autoencoder_average_accuracy_score = 0

            for index, client in enumerate(self._clients):

                accuracy = client._autoencoder_evaluate(model=self._global_autoencoder)

                self._autoencoder_average_accuracy_score = self._autoencoder_average_accuracy_score + accuracy

                print(f"{utils.log_timestamp_status()} Evaluating {len(self._clients)} client(s): {index + 1} / {len(self._clients)}", end="\r" if index != len(self._clients) - 1 else "\n")
            
            self._autoencoder_average_accuracy_score = self._autoencoder_average_accuracy_score / len(self._clients)

            logging.info(f"Current accuracy score: {utils.dynamic_round(value=self._autoencoder_average_accuracy_score, reference_value=max_accuracy_score)}")
            logging.info(f"Max accuracy score: {utils.dynamic_round(value=max_accuracy_score, reference_value=self._autoencoder_average_accuracy_score)}")

            #? Wandb log
            run.log({
                "round": round_num, 
                "clients": len(selected_clients),
                "score": self._autoencoder_average_accuracy_score, 
                "best": max_accuracy_score,
                "stop_counter": stop_counter
            })
            #? ---

            # Check for improvements
            if self._autoencoder_average_accuracy_score > max_accuracy_score:
                max_accuracy_score = self._autoencoder_average_accuracy_score
                best_model = clone(src=self._global_autoencoder)
                stop_counter = 0

                # Remove any previous saved model
                for f in listdir(config.ModelConfig.AUTOENCODER_MODEL_ROOT):
                    if f.endswith(config.ModelConfig.MODEL_EXTENSION):
                        remove(join(config.ModelConfig.AUTOENCODER_MODEL_ROOT, f))

                best_model.save(filepath=config.ModelConfig.autoencoder_model(accuracy=utils.dynamic_round(max_accuracy_score, self._autoencoder_average_accuracy_score)), overwrite=True)

            else:
                stop_counter = stop_counter + 1

            logging.info(f"Stop counter: {stop_counter} / {self._patience}")

            # Check stop conditions
            if stop_counter >= self._patience:
                break

            # Select clients for the next round
            selected_clients = self._select_clients()
        
        # Update global model
        self._global_autoencoder = clone(src=best_model)

        # Assign the best model to each client
        for client in self._clients:
            client._autoencoder = clone(src=best_model)

        # Save best model
        best_model.save(filepath=config.ModelConfig.autoencoder_model(), overwrite=True)

        return best_model

def random_autoencoder_model(x_shape: tuple, hidden_units: int = 10, learning_rate: float = 0.0001) -> Model:
    """
    Constructs and compiles a 1D Convolutional Autoencoder model.

    :param x_shape: The shape of the input data (timesteps, features).
    :type x_shape: `tuple`

    :param hidden_units: The number of filters in the bottleneck layer.
    :type hidden_units: `int`

    :param learning_rate: The learning rate for the Adam optimizer.
    :type learning_rate: `float`

    :returns: The compiled Keras Autoencoder model.
    :rtype: `tf.keras.Model`
    """

    _, input_features = x_shape
    input_layer = Input(shape=x_shape)

    # Encoder
    x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(input_layer)
    x = MaxPooling1D(pool_size=2, padding="same")(x)
    encoded = Conv1D(filters=hidden_units, kernel_size=3, activation="relu", padding="same")(x)

    # Decoder
    x = Conv1D(filters=hidden_units, kernel_size=3, activation="relu", padding="same")(encoded)
    x = UpSampling1D(size=2)(x)
    x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(x)

    # Output layer
    output_layer = Conv1D(filters=input_features, kernel_size=3, activation="linear", padding="same")(x)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

    return autoencoder

def random_threshold_model(learning_rate: float = 0.0001) -> Model:
    """
    Constructs and compiles a simple Dense Neural Network for threshold prediction.

    :param learning_rate: The learning rate for the Adam optimizer.
    :type learning_rate: float

    :returns: The compiled Keras Threshold model.
    :rtype: `tf.keras.Model`
    """

    input_layer = Input(shape=(1,))
    x = Dense(32, activation="relu")(input_layer)
    x = Dense(16, activation="relu")(x)

    # Output layer predicts a continuous value (the "normal" threshold)
    output_layer = Dense(1, activation="linear")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])

    return model

def random_iid_clients(x: np.ndarray) -> list:
    """
    Shuffles data and distributes it IID across a specified number of clients.

    Each client receives a portion of the data, further split into
    training, validation, and test sets.

    :param x: The input dataset to be distributed.
    :type x: `np.ndarray`

    :returns: A list of Client objects, each containing their respective data splits.
    :rtype: `list`
    """

    # Shuffle data
    np.random.default_rng(seed=config.DatasetConfig.SEED).shuffle(x)

    samples_per_client = len(x) // config.FLADHyperparameters.N_CLIENTS

    clients = []

    # Each client will receive a specific amount of data (train, validation and test)
    for i in range(config.FLADHyperparameters.N_CLIENTS):

        autoencoder_start = i * samples_per_client
        autoencoder_end = (i+ 1) * samples_per_client if i < config.FLADHyperparameters.N_CLIENTS - 1 else len(x)

        autoencoder_data = x[autoencoder_start:autoencoder_end]

        x_train, x_test = train_test_split(
            autoencoder_data, test_size=config.DatasetConfig.TEST_SIZE, random_state=config.DatasetConfig.SEED
        )

        x_test, x_val = train_test_split(
            x_test, test_size=config.DatasetConfig.VAL_SIZE, random_state=config.DatasetConfig.SEED
        )

        clients.append(Client(
            autoencoder_data={
                "x_train": x_train,
                "x_val": x_val,
                "x_test": x_test
            }
        ))

    return clients

if __name__ == "__main__":

    utils.clear_console()

    # Load dataset
    logging.info(f"Loading dataset")
    hf = h5py.File(name=config.DatasetConfig.OUTPUT_NORMAL)
    x = np.array(hf["x"]).astype(np.float32)

    if config.RUN_TYPE in [config.RUN_TYPE.ALL, config.RUN_TYPE.AUTOENCODER]:

        # Create a random autoencoder model
        logging.info(f"Creating autoencoder model")
        autoencoder = random_autoencoder_model(x_shape=x.shape[1:])
    
    else:

        # Loading autoencoder model
        logging.info(f"Loading autoencoder model")
        autoencoder = load_model(getenv("AUTOENCODER_MODEL"))

    # Create a random threshold model
    logging.info(f"Creating threshold model")
    threshold = random_threshold_model()

    # Create I.I.D. clients
    logging.info(f"Creating {config.FLADHyperparameters.N_CLIENTS} client(s)")
    clients = random_iid_clients(x=x)

    # Create the server
    logging.info(f"Initializing server")
    server = Server(autoencoder=autoencoder, threshold=threshold, clients=clients)

    if config.RUN_TYPE in [config.RUN_TYPE.ALL, config.RUN_TYPE.AUTOENCODER]:

        # Start federated learning
        run = config.WandbConfig.init_run(name="Autoencoder model")

        logging.info(f"Starting federated learning")
        autoencoder = server.federated_learning()

        run.finish()

    if config.RUN_TYPE in [config.RUN_TYPE.ALL, config.RUN_TYPE.THRESHOLD]:

        # Train the threshold model of each client
        for index, client in enumerate(clients):
            
            # Train model
            print(f"{utils.log_timestamp_status()}[{index + 1} / {len(clients)}] Training {str(client)}", end="\r")
            client._threshold_train(model=threshold)

            # Evaluate model
            print(f"{utils.log_timestamp_status()}[{index + 1} / {len(clients)}] Evaluating {str(client)}", end="\r")
            accuracy = client._threshold_evaluate()

            # Save model
            client.get_threshold_model().save(filepath=config.ModelConfig.threshold_model(client_id=str(client), accuracy=accuracy), overwrite=True)
        
        print(" "*100, end="\r")
        print(f"Trained and Evaluated {len(clients)} client(s)")