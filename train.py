from constants import (
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
    wide_deep_networks = load_wide_deep_networks()

    # Generate clients
    clients = generate_iid_clients()

    # Create server
    server = Server(
        clients=clients,
        wide_deep_networks=wide_deep_networks
    )

    # Federated learning (threshold networks)
    server.federated_learning(label=THRESHOLD_NETWORKS_LABEL)