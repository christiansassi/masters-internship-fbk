# Local imports
import config
import utils
from server_and_client import Server, Client

# External imports
from os import getenv

import h5py
import numpy as np

from keras import Input
from keras.models import Model, load_model  # type: ignore
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Dense  # type: ignore
from keras.optimizers import Adam  # type: ignore

from sklearn.model_selection import train_test_split

import logging

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
    x = hf["x"]

    if config.RUN_TYPE in [config.RUN_TYPE.ALL, config.RUN_TYPE.AUTOENCODER]:

        # Create a random autoencoder model
        logging.info(f"Creating autoencoder model")
        autoencoder = random_autoencoder_model(x_shape=x.shape[1:])
    
    else:

        # load autoencoder model
        logging.info(f"Loading autoencoder model")
        autoencoder = load_model(config.ModelConfig.FINAL_AUTOENCODER_MODEL_ROOT)

    # Create a random threshold model
    logging.info(f"Creating threshold model")
    threshold = random_threshold_model()

    # Create I.I.D. clients
    logging.info(f"Creating {config.FLADHyperparameters.N_CLIENTS} client(s)")
    clients = random_iid_clients(x=x)

    # Create the server
    logging.info(f"Initializing server")
    server = Server(autoencoder=autoencoder, clients=clients)

    if config.RUN_TYPE in [config.RUN_TYPE.ALL, config.RUN_TYPE.AUTOENCODER]:

        # Start federated learning
        run = config.WandbConfig.init_run(name="Autoencoder model")

        logging.info(f"Starting federated learning")
        server.federated_learning()

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
            client.get_threshold_model().save(filepath=config.ModelConfig.threshold_model(client_id=str(client)), overwrite=True)
        
        print(" "*100, end="\r")
        print(f"Trained and Evaluated {len(clients)} client(s)")