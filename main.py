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
        
        makedirs(name=THRESHOLD_NETWORK, exist_ok=True)

        session_id = str(int(datetime.now().timestamp()))

        model_path = join(THRESHOLD_NETWORK, f"{THRESHOLD_NETWORK_BASENAME}-{session_id}.pt")
        model_dict = {}

        run = config.WandbConfig.init_run(name=f"[{'GPU' if config.GPU else 'CPU'}] Threshold Network", tags=["pytorch", "threshold", "all_sensors"])

        wide_deep_network = torch.load(join(WIDE_DEEP_NETWORK, f"{WIDE_DEEP_NETWORK_BASENAME}.pt"), map_location=config.DEVICE)

        data = []

        for index, client in enumerate(clients, start=1):

            print(f"{utils.log_timestamp_status()} Client {index} / {len(clients)}")

            client.set_model_f_extractor(wide_deep_network["model_f_extractor"])
            client.set_model_sensor(wide_deep_network["model_sensors"][str(client)])

            train_loss = client.train_pred_error_model(verbose=config.VERBOSE)

            model_dict[str(client)] = client.pred_error_model.state_dict()

            data.append([f"{'-'.join(sorted(client.inputs))}", train_loss])

        torch.save(model_dict, model_path)

        table = config.WandbConfig.table(data=data, columns=["client", "loss"])
        bar_plot = config.WandbConfig.plot_bar(
            table=table,
            label="client",
            value="loss",
            title=f"Training Loss"
        )

        run.log({"threshold_network": bar_plot})
        run.finish()

    # Simulation
    if config.SIMULATION:
        
        wide_deep_network = torch.load(join(WIDE_DEEP_NETWORK, f"{WIDE_DEEP_NETWORK_BASENAME}.pt"), map_location=config.DEVICE)
        threshold_network = torch.load(join(THRESHOLD_NETWORK, f"{THRESHOLD_NETWORK_BASENAME}.pt"), map_location=config.DEVICE)
        
        for client in clients:
            client.set_model_f_extractor(wide_deep_network["model_f_extractor"])
            client.set_model_sensor(wide_deep_network["model_sensors"][str(client)])
            client.set_pred_error_model(threshold_network[str(client)])
            client.test(verbose=True)