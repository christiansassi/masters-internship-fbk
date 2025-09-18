import config
from constants import *
import utils

from federated_client import generate_non_iid_clients
from federated_server import Server

from os import listdir

if __name__ == "__main__":

    utils.clear_console()

    # Instantiate clients
    clients = generate_non_iid_clients(wide_deep_network=f"{join(WIDE_DEEP_NETWORK, WIDE_DEEP_NETWORK_BASENAME)}.h5")

    # Instantiate server
    server = Server(clients=clients, wide_deep_network=f"{join(WIDE_DEEP_NETWORK, WIDE_DEEP_NETWORK_BASENAME)}.h5")

    # Wide Deep Network training (federated)
    if config.WIDE_DEEP_NETWORK:
        server.federated_learning()

    # Threshold Network training (local)
    if config.THRESHOLD_NETWORK:
        server.train_threshold_networks()
    
    # Simulation
    if config.SIMULATION:

        clients[-1].run_simulation_v1()

        # for client in clients:
        #     client.run_simulation_v1()