import config
from constants import *
import utils

from federated_client import generate_non_iid_clients
from federated_server import Server

if __name__ == "__main__":

    utils.clear_console()

    # # Instantiate clients
    clients = generate_non_iid_clients()

    # Instantiate server
    server = Server(clients=clients)

    # Wide Deep Network training (federated)
    if config.WIDE_DEEP_NETWORK:
        server.federated_learning()

    # # Threshold Network training (local)
    # if config.THRESHOLD_NETWORK:
    #     server.train_threshold_networks()
    
    # # Simulation
    # if config.SIMULATION:

    #     for client in clients:
    #         client.run_simulation_v1()