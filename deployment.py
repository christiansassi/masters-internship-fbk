# Local imports
import config
import utils
from server_and_client import Server, Client

# External imports
from os import getenv, listdir
from os.path import join

from keras.models import Model, load_model   # type: ignore

import logging

if __name__ == "__main__":

    utils.clear_console()

    # Load dataset
    logging.info(f"Loading dataset")
    hf = h5py.File(name=config.DatasetConfig.OUTPUT_ATTACK)
    x = hf["x"]
    y = hf["y"]

    # Load autoencoder model
    logging.info(f"Loading autoencoder model")
    autoencoder = load_model(config.ModelConfig.FINAL_AUTOENCODER_MODEL)

    # Loading threshold models while creating new clients
    logging.info(f"Loading threshold model(s)")
    clients = []

    for item in listdir(config.ModelConfig.FINAL_THRESHOLD_MODEL_ROOT):

        if not item.endswith(config.ModelConfig.MODEL_EXTENSION):
            continue

        client = Client(
            autoencoder_data={},
            client_id=item.strip(f"{config.ModelConfig.THRESHOLD_MODEL_BASENAME}-").strip(config.ModelConfig.FINAL_THRESHOLD_MODEL_ROOT)
        )

        client.set_threshold_model(
            model=load_model(join(config.ModelConfig.FINAL_THRESHOLD_MODEL_ROOT, item))
        )

        clients.append(client)

    logging.info(f"Created {len(clients)} client(s)")

    # Simulate real data
    #TODO