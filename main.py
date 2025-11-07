from config import *
from constants import *

from federated_client import generate_non_iid_clients
from federated_server import Server

from os import makedirs

from datetime import datetime

import math
import numpy as np
from copy import deepcopy

import torch

if __name__ == "__main__":

    # Instantiate clients
    clients = generate_non_iid_clients()

    # Adjust MAX_EPOCHS and MAX_STEPS to match DAICS min BATCH_SIZE and hardware max BATCH_SIZE
    min_batch_size = daics.BATCH_SIZE
    max_batch_size = WIDE_DEEP_MAX_BATCH_SIZE

    max_size = len(max(clients, key=lambda x: len(x.df_train)).df_train)
    min_size = len(min(clients, key=lambda x: len(x.df_train)).df_train)

    flad.MAX_STEPS = math.ceil(max_size / min_batch_size) 
    flad.MIN_STEPS = max(1, math.floor(min_size / max_batch_size))
    
    printplus("")
    printplus(f"EPOCHS {flad.MIN_EPOCHS} - {flad.MAX_EPOCHS}")
    printplus(f"STEPS {flad.MIN_STEPS} - {flad.MAX_STEPS}")
    printplus(f"BATCH_SIZE {min_batch_size} - {max_batch_size}")

    # Instantiate server
    server = Server(clients=clients)

    # Wide Deep Network training (federated)
    if WIDE_DEEP_NETWORK:
        server.federated_learning()

    # Threshold Network training (local)
    if THRESHOLD_NETWORK:
        
        makedirs(name=THRESHOLD_NETWORK, exist_ok=True)

        session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        session_path = join(THRESHOLD_NETWORK, session_id)
        makedirs(name=session_path, exist_ok=True)

        session_id = str(int(datetime.now().timestamp()))

        model_path = join(session_path, f"{THRESHOLD_NETWORK_BASENAME}.pt")
        model_dict = {}

        run = WandbConfig.init_run(name=f"[{'GPU' if GPU else 'CPU'}] Threshold Network", tags=["pytorch", "threshold", "sensors_and_actuators"])

        wide_deep_network = torch.load(join(WIDE_DEEP_NETWORK, f"{WIDE_DEEP_NETWORK_BASENAME}.pt"), map_location=DEVICE)

        data = []

        for index, client in enumerate(clients, start=1):

            printplus(f"Client {index} / {len(clients)}")

            client.set_model_f_extractor(wide_deep_network["model_f_extractor"])
            client.set_model_sensors(wide_deep_network["model_sensors"][str(client)])

            train_loss = client.train_pred_error_model()

            model_dict[str(client)] = [deepcopy(pred_error_model.state_dict()) for pred_error_model in client.pred_error_models]

            data.append([f"{'-'.join(sorted(client.inputs))}", np.mean(train_loss)])

        torch.save(model_dict, model_path)

        table = WandbConfig.table(data=data, columns=["client", "loss"])
        bar_plot = WandbConfig.plot_bar(
            table=table,
            label="client",
            value="loss",
            title=f"Training Loss"
        )

        run.printplus({"threshold_network": bar_plot})
        run.finish()

    # Simulation
    if SIMULATION:
        
        wide_deep_network = torch.load(join(WIDE_DEEP_NETWORK, f"{WIDE_DEEP_NETWORK_BASENAME}.pt"), map_location=DEVICE)
        threshold_network = torch.load(join(THRESHOLD_NETWORK, f"{THRESHOLD_NETWORK_BASENAME}.pt"), map_location=DEVICE)
        
        client = max(clients, key=lambda x: len(x.inputs))
        client.set_model_f_extractor(wide_deep_network["model_f_extractor"])
        client.set_model_sensors(wide_deep_network["model_sensors"][str(client)])
        client.set_pred_error_models(threshold_network[str(client)])
        client.test()
    
    