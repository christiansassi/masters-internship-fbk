import config
from constants import *
import utils

from federated_client import generate_non_iid_clients, generate_daics_client
from federated_server import Server

from os import listdir

if __name__ == "__main__":

    utils.clear_console()

    # # Instantiate clients
    clients = generate_non_iid_clients()

    for client in clients:
        client.epochs = WIDE_DEEP_EPOCHS
        client.train_model_f_extractor(model_f_extractor=client.model_f_extractor)

    # # Instantiate server
    # server = Server(clients=clients)

    # # Wide Deep Network training (federated)
    # if config.WIDE_DEEP_NETWORK:
    #     server.federated_learning()

    # # Threshold Network training (local)
    # if config.THRESHOLD_NETWORK:
    #     server.train_threshold_networks()
    
    # # Simulation
    # if config.SIMULATION:

    #     for client in clients:
    #         client.run_simulation_v1()