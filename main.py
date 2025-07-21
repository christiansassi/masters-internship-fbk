from constants import (
    WIDE_DEEP_NETWORKS_LABEL,
    THRESHOLD_NETWORKS_LABEL
)

import config
import utils

from models import load_wide_deep_networks, load_threshold_networks

from federated_server import Server
from federated_client import generate_iid_clients

if __name__ == "__main__":
    
    utils.clear_console() 

    # Load wide deep networks
    wide_deep_networks = [] # load_wide_deep_networks()

    # Load threshold networks
    threshold_networks = [] # load_threshold_networks()
    
    # Generate clients
    clients = generate_iid_clients(
        wide_deep_networks=wide_deep_networks,
        threshold_networks=threshold_networks
    )

    # # Create server
    server = Server(
        clients=clients,
        wide_deep_networks=wide_deep_networks,
        threshold_networks=threshold_networks
    )

    if not len(wide_deep_networks):
        # Federated learning (wide deep networks)
        server.federated_learning(label=WIDE_DEEP_NETWORKS_LABEL)

    # if not len(threshold_networks):
    #     # Federated learning (threshold networks)
    #     server.federated_learning(label=THRESHOLD_NETWORKS_LABEL)
    
    # Simulate deploy
    # for client in clients:
    #     client.simulate()