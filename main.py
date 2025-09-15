import config
from constants import *
import utils

from federated_client import generate_non_iid_clients
from federated_server import Server

from os import makedirs

if __name__ == "__main__":

    utils.clear_console()

    # Instantiate clients
    clients = generate_non_iid_clients()

    # Instantiate server
    server = Server(clients=clients, wide_deep_network=f"{join(WIDE_DEEP_NETWORK, WIDE_DEEP_NETWORK_BASENAME)}.h5")

    # Wide Deep Network training (federated)
    # server.federated_learning()

    # Threshold Network training (local)
    server.train_threshold_networks()