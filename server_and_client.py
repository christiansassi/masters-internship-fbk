# Local imports
import config
import utils

# External imports
from os import listdir, remove, mkdir
from os.path import join, exists

import shutil

import numpy as np

from keras.models import Model, clone_model  # type: ignore

import tensorflow as tf

import uuid
from uuid import uuid4

from time import time

import pickle

import logging

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

    model = clone_model(src)

    if not model.built:
        model.build(src.input_shape)

    model.set_weights(src.get_weights() if weights is None else weights)

    if src.optimizer is not None and src.loss is not None:

        cleaned_metrics = []


        if hasattr(src, "metrics") and src.metrics is not None:

            for m in src.metrics:

                metric_name = m if isinstance(m, str) else getattr(m, "name", None)

                if metric_name and metric_name != "loss" and metric_name != src.loss:
                    cleaned_metrics.append(m)

        model.compile(
            optimizer=src.optimizer.__class__.from_config(src.optimizer.get_config()),
            loss=src.loss,
            metrics=cleaned_metrics
        )
    
    return model


class Client:
    def __init__(self, autoencoder_data: dict, client_id: str = None):

        # Generate an id for the current client (useful while debugging) or take the one given in input
        try:
            if str(uuid.UUID(client_id, version=4)).lower() != client_id.lower():
                client_id = str(uuid4())
        except:
            client_id = str(uuid4())

        self._id: str = client_id

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
            "epochs": config.FLADAndDAICSHyperparameters.MAX_EPOCHS,
            "steps": config.FLADAndDAICSHyperparameters.MAX_STEPS
        }

        # Calculate total samples
        self._samples: int = sum([len(self._autoencoder_data[key]) for key in self._autoencoder_data.keys()])

        # Set t_base to None
        self._t_base = None

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

    def _calculate_t_base(self):

        # Create a TensorFlow Dataset for the test data.
        x = tf.data.Dataset.from_tensor_slices(
            tensors=self._autoencoder_data["x_test"]
        ).batch(batch_size=32, drop_remainder=False)

        # Perform prediction using the batched dataset.
        y_pred = self._autoencoder.predict(
            x,
            verbose=config.VERBOSE
        )

        # Reconstruct the ground truth (y_true) from the same batched dataset.
        y_true_list = []

        for batch in x:
            y_true_list.append(batch.numpy())

        y_true = np.concatenate(y_true_list, axis=0)

        reconstruction_errors = np.mean(np.square(y_true - y_pred), axis=(1, 2))

        self._t_base = reconstruction_errors.mean() # + reconstruction_errors.std()

    def set_autoencoder_model(self, model: Model):
        """
        Set the autoencoder model of the current client

        :param model: The Keras autoencoder model to be set.
        :type model: `tf.keras.Model`
        """
        self._autoencoder = clone(src=model)

    def get_autoencoder_model(self) -> Model:
        """
        Returns the autoencoder model of the current client.

        :return: The autoencoder model.
        :rtype: `tf.keras.Model`
        """
        return clone(src=self._autoencoder)

    def set_threshold_model(self, model: Model):
        """
        Set the threshold model of the current client

        :param model: The Keras threshold model to be set.
        :type model: `tf.keras.Model`
        """
        self._threshold = clone(src=model)

    def get_threshold_model(self) -> Model:
        """
        Returns the threshold model of the current client.

        :return: The threshold model.
        :rtype: `tf.keras.Model`
        """
        return clone(src=self._threshold)

    def export(self) -> tuple:
        """
        Saves the current class instance as a pickle file, excluding the TensorFlow models (which are set to `None` for pickling). 
        The TensorFlow models are saved separately using their dedicated functions.
        When loading this pickled instance, remember to load the associated models from their respective Keras files.

        :return: The client's folder path and the file paths for its pickle file, autoencoder Keras model, and threshold Keras model.
        :rtype: `tuple`
        """

        # Get export paths
        folder, client, autoencoder, threshold = config.ServerAndClientConfig.export_client(client=self)

        if exists(folder):
            shutil.rmtree(folder)

        mkdir(folder)

        original_autoencoder = clone(self._autoencoder)
        original_threshold = clone(self._threshold)

        # Save class obj without tensorflow models
        self._autoencoder = None
        self._threshold = None

        pickle.dump(self, open(client, "wb+"))

        # Save tensorflow models separatedly
        original_autoencoder.save(autoencoder)
        original_threshold.save(threshold)

        # Restore tensorflow models
        self._autoencoder = original_autoencoder
        self._threshold = original_threshold

        return folder, client, autoencoder, threshold

class Server:
    def __init__(
            self, 
            autoencoder: Model, 
            clients: list[Client], 
            min_epochs: int = config.FLADAndDAICSHyperparameters.MIN_EPOCHS,
            max_epochs: int = config.FLADAndDAICSHyperparameters.MAX_EPOCHS,
            min_steps: int = config.FLADAndDAICSHyperparameters.MIN_STEPS,
            max_steps: int = config.FLADAndDAICSHyperparameters.MAX_STEPS,
            patience: int = config.FLADAndDAICSHyperparameters.PATIENCE
        ):

        # Set the initial global autoencoder model
        self._global_autoencoder: Model = clone(src=autoencoder)

        # Set the clients
        self._clients: list[Client] = clients
        
        #! This is not needed during the deployment
        for client in self._clients:

            if client._autoencoder is None:
                client._autoencoder = clone(src=self._global_autoencoder)

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
        """
        Orchestrates the federated learning process for the autoencoder model.
        This function manages rounds of client selection, model training on clients,
        global model aggregation, and evaluation, continuing until a stop condition
        (patience limit) is met. It tracks the best performing global model and
        saves it.
        """

        # All the clients partecipate in the first round
        selected_clients = self._select_clients()

        best_model = self._global_autoencoder
        max_accuracy_score = float("-inf")
        round_num = 0
        stop_counter = 0

        while True:

            start = time()

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

            time_per_round = time() - start

            #? Wandb log
            run.log({ # type: ignore
                "round": round_num, 
                "clients": len(selected_clients),
                "score": self._autoencoder_average_accuracy_score, 
                "best": max_accuracy_score,
                "stop_counter": stop_counter,
                "time_per_round": time_per_round
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
            logging.info(f"Time per round: {time_per_round}")

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
    
    def get_autoencoder_model(self) -> Model:
        """
        Returns the global autoencoder model.

        :return: The global autoencoder model.
        :rtype: `tf.keras.Model`
        """
        return clone(self._global_autoencoder)