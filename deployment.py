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

import logging

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

    for client_dir in listdir(config.ServerAndClientConfig.CLIENT_ROOT):

        client = pickle.load(open(join(config.ServerAndClientConfig.CLIENT_ROOT, client_dir, f"client{config.ServerAndClientConfig.CLIENT_EXTENSION}"), "rb"))
        client.set_autoencoder_model(load_model(join(config.ServerAndClientConfig.CLIENT_ROOT, client_dir, f"{config.ModelConfig.AUTOENCODER_MODEL_BASENAME}{config.ModelConfig.MODEL_EXTENSION}")))
        client.set_threshold_model(load_model(join(config.ServerAndClientConfig.CLIENT_ROOT, client_dir, f"{config.ModelConfig.THRESHOLD_MODEL_BASENAME}{config.ModelConfig.MODEL_EXTENSION}")))

        clients.append(client)

    logging.info(f"Created {len(clients)} client(s)")

    # Calculate t_base and max(threshold_outputs)
    for index, client in enumerate(clients):

        print(f"{utils.log_timestamp_status()}[{index + 1} / {len(clients)}] Calculating t_base for {str(client)}", end="\r")
        client._calculate_t_base()
    
    print(" "*100, end="\r")
    print(f"{utils.log_timestamp_status()} Calculated t_base for {len(clients)} client(s)")

    # Simulate real data
    for client in clients:
        
        # print(" "*100, end="\r")

        # correct = 0
        # wrong = 0

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
            predicted_class = int(reconstruction_error > max(y_pred, client._t_base))

            # If it is benign, update the threshold model
            if predicted_class == config.DatasetConfig.NORMAL_LABEL:

                # Save the sample (in case of a complete retraining)
                client._threshold_data["x_test"] = np.append(client._threshold_data["x_test"], reconstruction_error)

                # Update the threshold model
                client._threshold.train_on_batch(
                    np.expand_dims(reconstruction_error, axis=0),
                    np.expand_dims(reconstruction_error, axis=0)
                )

            #TODO add alert window
            #TODO add operator feedback
            #TODO add retraining logic

            # if output == predicted_class:
            #     correct = correct + 1
            # else:
            #     wrong = wrong + 1

            # print(f"[{index + 1} / {len(x)}] Correct: {correct} | Wrong: {wrong}", end="\r")

    #TODO