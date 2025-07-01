# Local imports
import config
import utils
from server_and_client import Server, Client

# External imports
from os import getenv, listdir
from os.path import join

from keras.models import Model, load_model   # type: ignore

import pickle

import logging

if __name__ == "__main__":

    utils.clear_console()

    # Load dataset
    # logging.info(f"Loading dataset")
    # hf = h5py.File(name=config.DatasetConfig.OUTPUT_ATTACK)
    # x = hf["x"]
    # y = hf["y"]

    # Loading clients
    logging.info(f"Loading client(s)")
    clients = []

    for client_dir in listdir(config.ServerAndClientConfig.CLIENT_ROOT):

        client = pickle.load(open(join(config.ServerAndClientConfig.CLIENT_ROOT, client_dir, f"client{config.ServerAndClientConfig.CLIENT_EXTENSION}"), "rb"))
        client.set_autoencoder_model(load_model(join(config.ServerAndClientConfig.CLIENT_ROOT, client_dir, f"{config.ModelConfig.AUTOENCODER_MODEL_BASENAME}{config.ModelConfig.MODEL_EXTENSION}")))
        client.set_threshold_model(load_model(join(config.ServerAndClientConfig.CLIENT_ROOT, client_dir, f"{config.ModelConfig.THRESHOLD_MODEL_BASENAME}{config.ModelConfig.MODEL_EXTENSION}")))

        clients.append(client)

    logging.info(f"Created {len(clients)} client(s)")

    # Simulate real data
    #TODO