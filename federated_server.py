from config import *
from constants import *
from federated_client import *
from models import *
import utils

from os import makedirs

from time import time
from datetime import datetime

import logging

import torch

class Server:

    def __init__(self, clients: list[Client]):
        
        self.clients = clients

        self.model_f_extractor = ModelFExtractor(
            window_size_in=WINDOW_PAST, 
            window_size_out=WINDOW_PRESENT, 
            n_devices_in=len(GLOBAL_INPUTS), 
            kernel_size=KERNEL_SIZE
        )

        self.score = float("inf")

        for client in self.clients:
            client.set_model_f_extractor(model_f_extractor=self.model_f_extractor)

    def select_clients(self) -> list[Client]:

        selected_clients = []

        min_score = float("inf")
        max_score = float("-inf")

        for client in self.clients:

            if client.score > self.score:
                continue

            min_score = min(min_score, client.score)
            max_score = max(max_score, client.score)

            selected_clients.append(client)
        
        for index, client in enumerate(selected_clients):

            if max_score != min_score:
                scaling_factor = (max_score - client.score) / (max_score - min_score)
            else:
                scaling_factor = 0

            client.epochs = int(MIN_EPOCHS + (MAX_EPOCHS - MIN_EPOCHS) * scaling_factor)
            client.steps = int(MIN_STEPS + (MAX_STEPS - MIN_STEPS) * scaling_factor)

            selected_clients[index] = client
        
        return selected_clients

    def aggregate_networks(self, clients: list[Client], weighted: bool = False) -> ModelFExtractor:
        
        # Deepcopy global model structure from the first client
        global_model = deepcopy(self.model_f_extractor)
        global_model = global_model.to(DEVICE)

        global_state = global_model.state_dict()

        # Prepare accumulators
        for key in global_state.keys():
            global_state[key] = torch.zeros_like(global_state[key])

        # Compute total weight (for weighted averaging)
        total_weight = sum(client.num_of_samples for client in clients) if weighted else len(clients)

        # Aggregate parameters
        for client in clients:

            client_state = client.model_f_extractor.state_dict()
            weight = client.num_of_samples if weighted else 1

            for key in global_state.keys():
                global_state[key] += (client_state[key] * (weight / total_weight))

        # Load averaged weights into global model
        global_model.load_state_dict(global_state)

        return global_model
    
    def federated_learning(self):

        session_id = str(int(datetime.now().timestamp()))

        run = WandbConfig.init_run(f"[{'GPU' if GPU else 'CPU'}] Wide Deep Network")

        map_clients_ids = {str(client): f"{'-'.join(sorted(client.inputs))}" for client in self.clients}

        losses = {client: {
            "train_loss": float("-inf"),
            "val_loss": float("-inf"),
            "eval_loss": float("-inf")
        } for client in map_clients_ids.values()}

        round_num = 0
        stop_counter = 0

        best_score = float("-inf")
        best_model_f_extractor = deepcopy(self.model_f_extractor)

        while True:

            start = time()

            round_num = round_num + 1

            print("")
            logging.info(f"---------- Round {round_num} ----------")

            # Select clients
            selected_clients = self.select_clients()

            # Updat eclients
            for index, client in enumerate(selected_clients):
                print(f"{utils.log_timestamp_status()} Training {index + 1} / {len(selected_clients)}", end="\r")
                train_loss, val_loss = client.train_model_f_extractor(model_f_extractor=self.model_f_extractor)

                client_id = map_clients_ids[str(client)]

                losses[client_id]["train_loss"] = train_loss
                losses[client_id]["val_loss"] = val_loss
            
            logging.info(f"Trained {len(selected_clients)} clients")

            # Model aggregations
            logging.info(f"Aggregating models")
            self.model_f_extractor = self.aggregate_networks(clients=self.clients)

            # Evaluate clients
            self.score = 0

            for index, client in enumerate(self.clients):
                print(f"{utils.log_timestamp_status()} Evaluating {index + 1} / {len(self.clients)}", end="\r")

                eval_loss = client.eval_model_f_extractor(model_f_extractor=self.model_f_extractor)

                client_id = map_clients_ids[str(client)]

                losses[client_id]["eval_loss"] = eval_loss

                self.score = self.score + eval_loss

            self.score = self.score / len(self.clients)

            logging.info(f"Evaluated {len(self.clients)} clients")

            # Check for improvements
            logging.info(f"Current score: {self.score}")
            logging.info(f"Best score: {best_score}")

            if self.score > best_score:
                stop_counter = 0

                best_score = self.score
                best_model_f_extractor = deepcopy(self.model_f_extractor)

                makedirs(name=WIDE_DEEP_NETWORK_CHECKPOINT, exist_ok=True)

                model_path = join(WIDE_DEEP_NETWORK_CHECKPOINT, f"{WIDE_DEEP_NETWORK_BASENAME}-{session_id}.pt")
                model_dict = {
                    "model_f_extractor": best_model_f_extractor.state_dict(),
                    "model_sensors": {
                        str(client): client.model_sensor.state_dict()
                    for client in self.clients}
                }

                torch.save(model_dict, model_path)

            else:
                stop_counter = stop_counter + 1
            
            # Check stop conditions
            logging.info(f"Patience {stop_counter} / {PATIENCE}")

            log = {
                "round": round_num,
                "selected_clients": len(selected_clients),
                "score": self.score,
                "best": best_score,
                "stop_counter": stop_counter,
                "time_per_round": time() - start,
            }

            log.update(losses)

            run.log(log)

            if stop_counter >= PATIENCE:
                break
        
        self.model_f_extractor = deepcopy(best_model_f_extractor)

        for client in self.clients:
            client.set_model_f_extractor(model_f_extractor=self.model_f_extractor)

        run.finish()