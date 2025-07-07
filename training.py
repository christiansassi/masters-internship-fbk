# Local imports
import config
import utils
from server_and_client import Server, Client
from autoencoder_and_threshold import autoencoder_model, threshold_model

# External imports
import h5py
import numpy as np

from keras.models import load_model  # type: ignore

from sklearn.model_selection import train_test_split

import logging

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

    samples_per_client = len(x) // config.FLADAndDAICSHyperparameters.N_CLIENTS

    clients = []

    # Each client will receive a specific amount of data (train, validation and test)
    for i in range(config.FLADAndDAICSHyperparameters.N_CLIENTS):

        autoencoder_start = i * samples_per_client
        autoencoder_end = (i+ 1) * samples_per_client if i < config.FLADAndDAICSHyperparameters.N_CLIENTS - 1 else len(x)

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
    x = np.array(hf["x"])

    if config.RUN_TYPE in [config.RunType.ALL, config.RunType.AUTOENCODER]:

        # Create a random autoencoder model
        logging.info(f"Creating autoencoder model")
        autoencoder = autoencoder_model(x_shape=x.shape[1:])
    
    else:

        # load autoencoder model
        logging.info(f"Loading autoencoder model")
        autoencoder = load_model(r"models\autoencoder\autoencoder-lstm.keras")

    # Create a random threshold model
    logging.info(f"Creating threshold model")
    threshold = threshold_model()

    # Create I.I.D. clients
    logging.info(f"Creating {config.FLADAndDAICSHyperparameters.N_CLIENTS} client(s)")
    clients = random_iid_clients(x=x)

    # Create the server
    logging.info(f"Initializing server")
    server = Server(autoencoder=autoencoder, clients=clients)

    if config.RUN_TYPE in [config.RunType.ALL, config.RunType.AUTOENCODER]:

        # Start federated learning

        logging.info(f"Starting federated learning")
        server.federated_learning()

    if config.RUN_TYPE in [config.RunType.ALL, config.RunType.THRESHOLD]:

        # Train the threshold model of each client
        for index, client in enumerate(clients):
            
            # Train model
            print(f"{utils.log_timestamp_status()}[{index + 1} / {len(clients)}] Training {str(client)}", end="\r")
            client._threshold_train(model=threshold)

            # Evaluate model
            # print(f"{utils.log_timestamp_status()}[{index + 1} / {len(clients)}] Evaluating {str(client)}", end="\r")
            # accuracy = client._threshold_evaluate()

            # Save model
            client.get_threshold_model().save(filepath=config.ModelConfig.threshold_model(client_id=str(client)), overwrite=True)
        
        print(" "*100, end="\r")
        print(f"{utils.log_timestamp_status()} Trained and Evaluated {len(clients)} client(s)")
    
    if config.RUN_TYPE != config.RunType.NONE:

        # Save the clients
        for index, client in enumerate(clients):

            print(f"{utils.log_timestamp_status()}[{index + 1} / {len(clients)}] Saving {str(client)}", end="\r")
            client.export()

        print(" "*100, end="\r")
        print(f"{utils.log_timestamp_status()} Saved {len(clients)} client(s)")