# Local imports
import config
import utils
from server_and_client import Server, Client

# External imports
from os import getenv, listdir
from os.path import join

from keras.models import Model  # type: ignore
from keras.saving import load_model #type: ignore

import logging

if __name__ == "__main__":

    # Load autoencoder model
    logging.info(f"Loading autoencoder model")
    autoencoder = load_model(r"C:\Users\sassi\Desktop\masters-internship-fbk\models\autoencoder\autoencoder.keras")

    # Loading threshold models
    for item in listdir(config.ModelConfig.FINAL_THRESHOLD_MODEL_ROOT):

        if not item.endswith(config.ModelConfig.MODEL_EXTENSION):
            continue

        threshold = load_model(join(config.ModelConfig.FINAL_THRESHOLD_MODEL_ROOT, item))

        client = Client(
            autoencoder_data={},
            client_id=item.strip(f"{config.ModelConfig.THRESHOLD_MODEL_BASENAME}-").strip(config.ModelConfig.FINAL_THRESHOLD_MODEL_ROOT)
        )