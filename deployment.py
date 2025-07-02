# Local imports
import config
import utils
from server_and_client import Server, Client

# External imports
from os import getenv, listdir
from os.path import join

import numpy as np
import h5py

from keras.models import Model, load_model   # type: ignore

import pickle

from rich.progress import Progress

import logging

from concurrent.futures import ThreadPoolExecutor

def load_client(client_id: str) -> Client:

    client = pickle.load(open(join(config.ServerAndClientConfig.CLIENT_ROOT, client_id, f"client{config.ServerAndClientConfig.CLIENT_EXTENSION}"), "rb"))
    client.set_autoencoder_model(load_model(join(config.ServerAndClientConfig.CLIENT_ROOT, client_id, f"{config.ModelConfig.AUTOENCODER_MODEL_BASENAME}{config.ModelConfig.MODEL_EXTENSION}")))
    client.set_threshold_model(load_model(join(config.ServerAndClientConfig.CLIENT_ROOT, client_id, f"{config.ModelConfig.THRESHOLD_MODEL_BASENAME}{config.ModelConfig.MODEL_EXTENSION}")))

    return client

def simulate(client: Client, x: np.array, y: np.array, progress, task_id) -> dict:

    anomalies = 0
    false_positives = 0

    benign_true = 0
    benign_false = 0

    anomaly_true = 0
    anomaly_false = 0

    # It is necessary to process each sample one by one like a real scenario
    for index, (sample, output) in enumerate(zip(x, y)):
        
        y_pred = client._autoencoder.predict(
            np.expand_dims(sample, axis=0),
            verbose=config.VERBOSE
        )[0]

        reconstruction_error = np.mean(np.square(sample - y_pred))

        y_pred = client._threshold.predict(
            np.expand_dims(reconstruction_error, axis=0),
            verbose=config.VERBOSE
        )[0]

        # Predicted class
        #TODO better check how daics manages this part
        predicted_class = int(reconstruction_error > max(y_pred.item(), client._t_base.item() if isinstance(client._t_base, np.ndarray) else client._t_base))

        # If it is benign, update the threshold model
        if predicted_class == config.DatasetConfig.NORMAL_LABEL:

            # Save the sample (in case of a complete retraining)
            client._threshold_data["x_test"] = np.append(client._threshold_data["x_test"], reconstruction_error)

            # Update the threshold model
            client._threshold.train_on_batch(
                np.expand_dims(reconstruction_error, axis=0),
                np.expand_dims(reconstruction_error, axis=0)
            )

            # Reset the alerts
            anomalies = 0

        else:

            anomalies = anomalies + 1

            if anomalies != config.FLADAndDAICSHyperparameters.W_ANOMALY:
                continue

            false_positive = output == config.DatasetConfig.NORMAL_LABEL

            if not false_positive:
                continue

            false_positives = false_positives + 1

            #TODO add retraining logic
        
        if output == config.DatasetConfig.NORMAL_LABEL:

            if output == predicted_class:
                benign_true = benign_true + 1
            else:
                benign_false = benign_false + 1
            
        else:

            if output == predicted_class:
                anomaly_true = anomaly_true + 1
            else:
                anomaly_false = anomaly_false + 1

        progress.update(task_id=task_id, description=f"[{index+1} / {len(x)}] Benign: {benign_true} | {benign_false}  -  Anomaly: {anomaly_true} | {anomaly_false}")

    return {
        "id": str(client),
        "benign": {
            "true": benign_true,
            "false": benign_false
        },
        "anomaly": {
            "true": anomaly_true,
            "false": anomaly_false
        }
    }

if __name__ == "__main__":

    utils.clear_console()

    # Load dataset
    logging.info(f"Loading dataset")
    hf = h5py.File(name=config.DatasetConfig.OUTPUT_ATTACK)
    x = np.array(hf["x"])
    y = np.array(hf["y"])

    # Loading clients
    logging.info(f"Loading client(s)")

    clients = []

    with ThreadPoolExecutor() as executor:

        futures = []

        for client_id in listdir(config.ServerAndClientConfig.CLIENT_ROOT):
            futures.append(executor.submit(load_client, client_id))
        
        for future in futures:
            clients.append(future.result())

    logging.info(f"Created {len(clients)} client(s)")

    # Calculate t_base and max(threshold_outputs)
    logging.info(f"Calculating t_base for {len(clients)} clients")

    for client in clients:
        client._calculate_t_base()
    
    # with ThreadPoolExecutor() as executor:

    #     futures = []

    #     for client in clients:
    #         futures.append(executor.submit(client._calculate_t_base))
        
    #     for future in futures:
    #         future.result()

    logging.info(f"Calculated t_base for {len(clients)} client(s)")

    results = []

    with Progress(transient=True) as progress:
    
        with ThreadPoolExecutor() as executor:
            futures = []
            
            for client in clients:
                task_id = progress.add_task(f"Simulating {str(client)}", total=len(x))
                futures.append(executor.submit(simulate, client, x, y, progress, task_id))
            
            for future in futures:
                results.append(future.result())
    
    results