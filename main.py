import config
from constants import *
import utils

from federated_client import generate_non_iid_clients
from federated_server import Server

from os import makedirs

from datetime import datetime

import torch

if __name__ == "__main__":

    utils.clear_console()

    # Instantiate clients
    clients = generate_non_iid_clients()

    # Instantiate server
    server = Server(clients=clients)

    # Wide Deep Network training (federated)
    if config.WIDE_DEEP_NETWORK:
        server.federated_learning()

    # Threshold Network training (local)
    if config.THRESHOLD_NETWORK:
        
        makedirs(name=THRESHOLD_NETWORK_CHECKPOINT, exist_ok=True)

        session_id = str(int(datetime.now().timestamp()))

        model_path = join(THRESHOLD_NETWORK_CHECKPOINT, f"{THRESHOLD_NETWORK_BASENAME}-{session_id}.pt")
        model_dict = {}

        wide_deep_network = torch.load(join(WIDE_DEEP_NETWORK, f"{WIDE_DEEP_NETWORK_BASENAME}.pt"), map_location=config.DEVICE)

        for index, client in enumerate(clients, start=1):

            print(f"{utils.log_timestamp_status()} Client {index} / {len(clients)}")

            client.set_model_f_extractor(wide_deep_network["model_f_extractor"])
            client.set_model_sensor(wide_deep_network["model_sensors"][str(client)])

            train_loss = client.train_pred_error_model(verbose=config.VERBOSE)

            model_dict[str(client)] = client.pred_error_model.state_dict()
        
        torch.save(model_dict, model_path)

    # # Simulation
    # if config.SIMULATION:

    #     for client in clients:
    #         client.run_simulation_v1()