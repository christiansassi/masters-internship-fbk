from config import *
from constants import *
from federated_client import *
from models import *

from os import makedirs

from time import time
from datetime import datetime

import logging

import torch

class Server:

    def __init__(self, clients: list[Client]):
        
        self.clients = clients

        self.model_f_extractor = ModelFExtractor(
            window_size_in=daics.WINDOW_PAST, 
            window_size_out=daics.WINDOW_PRESENT, 
            n_devices_in=len(GLOBAL_INPUTS), 
            kernel_size=daics.KERNEL_SIZE
        )

        self.score = float("inf")
    
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
        
        for client in selected_clients:

            if max_score != min_score:
                scaling_factor = (client.score - max_score) / (min_score - max_score)
            else:
                scaling_factor = 1

            client.epochs = max(flad.MIN_EPOCHS, int(flad.MIN_EPOCHS + (flad.MAX_EPOCHS - flad.MIN_EPOCHS) * scaling_factor))
            client.steps  = max(flad.MIN_STEPS, int(flad.MIN_STEPS  + (flad.MAX_STEPS  - flad.MIN_STEPS ) * scaling_factor))
            
        return selected_clients


    def aggregate_networks(self, clients: list[Client], weighted: bool = False) -> ModelFExtractor:
        
        # Deepcopy global model structure from the first client
        global_model = deepcopy(self.model_f_extractor)
        global_model.to(DEVICE)

        global_state = deepcopy(global_model.state_dict())

        # Prepare accumulators
        for key in global_state.keys():
            global_state[key] = torch.zeros_like(global_state[key])

        # Compute total weight (for weighted averaging)
        total_weight = sum(client.num_of_samples for client in clients) if weighted else len(clients)

        # Aggregate parameters
        for client in clients:

            client_state = deepcopy(client.model_f_extractor.state_dict())
            weight = client.num_of_samples if weighted else 1

            for key in global_state.keys():
                global_state[key] += (client_state[key] * (weight / total_weight))

        # Load averaged weights into global model
        global_model.load_state_dict(deepcopy(global_state))

        return global_model
    
    def federated_learning(self):
        
        makedirs(name=WIDE_DEEP_NETWORK, exist_ok=True)

        session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        session_path = join(WIDE_DEEP_NETWORK, session_id)
        makedirs(name=session_path, exist_ok=True)

        checkpoint_path = join(session_path, CHECKPOINTS)
        makedirs(name=checkpoint_path)

        run = WandbConfig.init_run(name=f"[{'GPU' if GPU else 'CPU'}] Wide Deep Network", tags=["pytorch", "wide-deep", "sensors_and_actuators"])

        map_clients_ids = {str(client): f"{'-'.join(sorted(client.inputs))}" for client in self.clients}

        stats = {client: {
            "train_loss": float("-inf"),
            "val_loss": float("-inf"),
            "eval_loss": float("-inf"),
            "epochs": None,
            "steps": None,
            "batch_size": None,
            "selected": None
        } for client in map_clients_ids.values()}

        round_num = 0
        stop_counter = 0

        best_score = float("-inf")
        best_model_f_extractor = deepcopy(self.model_f_extractor)

        while True:

            round_num = round_num + 1

            round_path = join(checkpoint_path, f"round_{round_num}")
            makedirs(name=round_path, exist_ok=True)

            printplus("")
            printplus(f"---------- Round {round_num} ----------")
            
            for client in self.clients:
                client_id = map_clients_ids[str(client)]
                stats[client_id]["selected"] = 0

            #? Select clients
            start = time()
            selected_clients = self.select_clients()
            elapsed = max(0, time() - start)

            #? Update eclients
            start = time()

            for index, client in enumerate(selected_clients, start=1):
                printplus(f"Training {index} / {len(selected_clients)}")
                train_loss, val_loss = client.train_model_f_extractor_and_sensors(model_f_extractor=self.model_f_extractor)

                client_id = map_clients_ids[str(client)]

                stats[client_id]["train_loss"] = train_loss
                stats[client_id]["val_loss"] = val_loss
                stats[client_id]["selected"] = 1

            elapsed = elapsed + max(0, time() - start)

            #! CHECKPOINT #######
            for client in self.clients:
                torch.save({
                    "model_f_extractor": deepcopy(client.model_f_extractor.state_dict()),
                    "model_sensors": [deepcopy(model_sensor.state_dict()) for model_sensor in client.model_sensors]
                }, join(round_path, f"{str(client.id)}.pt"))
            #!###################

            logging.info(f"Trained {len(selected_clients)} clients")

            #? Model aggregations
            logging.info(f"Aggregating models")
            start = time()
            self.model_f_extractor = self.aggregate_networks(clients=self.clients)
            elapsed = elapsed + max(0, time() - start)

            #? Evaluate clients
            start = time()

            self.score = 0

            for index, client in enumerate(self.clients, start=1):
                printplus(f"Evaluating {index} / {len(self.clients)}")

                eval_loss = client.eval_model_f_extractor_and_sensor(model_f_extractor=self.model_f_extractor)

                client_id = map_clients_ids[str(client)]

                stats[client_id]["eval_loss"] = eval_loss
                stats[client_id]["epochs"] = client.epochs
                stats[client_id]["steps"] = client.steps
                stats[client_id]["batch_size"] = client.batch_size

                self.score = self.score + eval_loss

            self.score = self.score / len(self.clients)

            elapsed = elapsed + max(0, time() - start)

            logging.info(f"Evaluated {len(self.clients)} clients")

            #? Check for improvements
            logging.info(f"Current score: {self.score}")
            logging.info(f"Best score: {best_score}")

            if self.score > best_score:
                stop_counter = 0

                best_score = self.score
                best_model_f_extractor = deepcopy(self.model_f_extractor)

            else:
                stop_counter = stop_counter + 1
            
            #! CHECKPOINT #######
            torch.save({
                "model_f_extractor": deepcopy(best_model_f_extractor.state_dict()),
                "model_sensors": {
                    str(client): [deepcopy(model_sensor.state_dict()) for model_sensor in client.model_sensors]
                for client in self.clients}
            }, join(round_path, f"{WIDE_DEEP_NETWORK_BASENAME}.pt"))
            #!###################

            logging.info(f"Patience {stop_counter} / {flad.FLAD_PATIENCE}")

            log = {
                "round": round_num,
                "selected_clients": len(selected_clients),
                "score": self.score,
                "best": best_score,
                "stop_counter": stop_counter,
                "time_per_round": elapsed,
            }

            log.update(stats)

            run.printplus(log)

            #? Check stop conditions
            if stop_counter >= flad.FLAD_PATIENCE:
                break
        
        self.model_f_extractor = deepcopy(best_model_f_extractor)

        for client in self.clients:
            client.set_model_f_extractor(model_f_extractor=self.model_f_extractor)

        torch.save({
            "model_f_extractor": deepcopy(best_model_f_extractor.state_dict()),
            "model_sensors": {
                str(client): [deepcopy(model_sensor.state_dict()) for model_sensor in client.model_sensors]
            for client in self.clients}
        }, join(session_path, f"{WIDE_DEEP_NETWORK_BASENAME}.pt"))

        run.finish()