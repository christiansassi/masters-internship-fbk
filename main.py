import config
import utils

from federated_client import generate_non_iid_clients
from federated_server import Server

if __name__ == "__main__":

    utils.clear_console()

    clients = generate_non_iid_clients()

    server = Server(clients=clients)
    server.federated_learning()
