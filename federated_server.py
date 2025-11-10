import config
import constants

from federated_client import Client

from models import ModelFExtractor

from os import makedirs
from os.path import join

from time import time
from datetime import datetime

from copy import deepcopy

import torch

class Server:

    def __init__(self, clients: list[Client]):
        
        self.clients = clients

        self.model_f_extractor = ModelFExtractor(
            window_size_in=constants.daics.WINDOW_PAST, 
            window_size_out=constants.daics.WINDOW_PRESENT, 
            n_devices_in=len(constants.GLOBAL_INPUTS), 
            kernel_size=constants.daics.KERNEL_SIZE
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

            client.epochs = max(constants.flad.MIN_EPOCHS, int(constants.flad.MIN_EPOCHS + (constants.flad.MAX_EPOCHS - constants.flad.MIN_EPOCHS) * scaling_factor))
            client.steps  = max(constants.flad.MIN_STEPS, int(constants.flad.MIN_STEPS  + (constants.flad.MAX_STEPS  - constants.flad.MIN_STEPS ) * scaling_factor))
            
        return selected_clients


    def aggregate_networks(self, clients: list[Client], weighted: bool = False) -> ModelFExtractor:
        
        # Deepcopy global model structure from the first client
        global_model = deepcopy(self.model_f_extractor)
        global_model.to(config.DEVICE)

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
        
        makedirs(name=constants.WIDE_DEEP_NETWORK, exist_ok=True)

        session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        session_path = join(constants.WIDE_DEEP_NETWORK, session_id)
        makedirs(name=session_path, exist_ok=True)

        checkpoint_path = join(session_path, constants.CHECKPOINTS)
        makedirs(name=checkpoint_path)

        run = config.WandbConfig.init_run(name=f"[{'GPU' if config.GPU else 'CPU'}] Wide Deep Network", tags=["pytorch", "wide-deep", "sensors_and_actuators"])

        map_clients_ids = {str(client): f"{'-'.join(sorted(client.inputs))}" for client in self.clients}

        stats = {client: {
            "train_loss": float("-inf"),
            "val_loss": float("-inf"),
            "eval_loss": float("-inf"),
            "epochs": None,
            "steps": None,
            "batch_size": None,
            "selected": None,
            "time": None
        } for client in map_clients_ids.values()}

        round_num = 0
        stop_counter = 0

        best_score = float("-inf")
        best_model_f_extractor = deepcopy(self.model_f_extractor)

        select_clients_time = 0
        update_clients_time = 0
        aggregate_network_time = 0
        evaluate_clients_time = 0

        while True:

            round_num = round_num + 1

            round_path = join(checkpoint_path, f"round_{round_num}")
            makedirs(name=round_path, exist_ok=True)

            config.printplus("")
            config.printplus(f"---------- Round {round_num} ----------")
            
            for client in self.clients:
                client_id = map_clients_ids[str(client)]
                stats[client_id]["selected"] = 0
                stats[client_id]["time"] = 0

            #? Select clients
            start = time()
            selected_clients = self.select_clients()
            select_clients_time = max(0, time() - start)
            elapsed = select_clients_time

            #? Update eclients
            start = time()

            for index, client in enumerate(selected_clients, start=1):
                config.printplus(f"Training {index} / {len(selected_clients)}")

                client_start = time()
                train_loss, val_loss = client.train_model_f_extractor_and_sensors(model_f_extractor=self.model_f_extractor)
                client_end = time()

                client_id = map_clients_ids[str(client)]

                stats[client_id]["train_loss"] = train_loss
                stats[client_id]["val_loss"] = val_loss
                stats[client_id]["selected"] = 1
                stats[client_id]["time"] = client_end - client_start

            update_clients_time = max(0, time() - start)
            elapsed = elapsed + update_clients_time

            #! CHECKPOINT #######
            for client in self.clients:
                torch.save({
                    "model_f_extractor": deepcopy(client.model_f_extractor.state_dict()),
                    "model_sensors": [deepcopy(model_sensor.state_dict()) for model_sensor in client.model_sensors]
                }, join(round_path, f"{str(client.id)}.pt"))
            #!###################

            config.printplus(f"Trained {len(selected_clients)} clients")

            #? Model aggregations
            config.printplus(f"Aggregating models")
            start = time()

            self.model_f_extractor = self.aggregate_networks(clients=self.clients)

            aggregate_network_time = max(0, time() - start)
            elapsed = elapsed + aggregate_network_time

            #? Evaluate clients
            start = time()

            self.score = 0

            for index, client in enumerate(self.clients, start=1):
                config.printplus(f"Evaluating {index} / {len(self.clients)}")

                client_start = time()
                eval_loss = client.eval_model_f_extractor_and_sensor(model_f_extractor=self.model_f_extractor)
                client_end = time()

                client_id = map_clients_ids[str(client)]

                stats[client_id]["eval_loss"] = eval_loss
                stats[client_id]["epochs"] = client.epochs
                stats[client_id]["steps"] = client.steps
                stats[client_id]["batch_size"] = client.batch_size
                stats[client_id]["time"] = client_end - client_start

                self.score = self.score + eval_loss

            self.score = self.score / len(self.clients)
            
            evaluate_clients_time = max(0, time() - start)
            elapsed = elapsed + evaluate_clients_time

            config.printplus(f"Evaluated {len(self.clients)} clients")

            #? Check for improvements
            config.printplus(f"Current score: {self.score}")
            config.printplus(f"Best score: {best_score}")

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
            }, join(round_path, f"{constants.WIDE_DEEP_NETWORK_BASENAME}.pt"))
            #!###################

            config.printplus(f"Patience {stop_counter} / {constants.flad.FLAD_PATIENCE}")

            log = {
                "round": round_num,
                "selected_clients": len(selected_clients),
                "score": self.score,
                "best": best_score,
                "stop_counter": stop_counter,
                "time_per_round": elapsed,
                "select_clients_time": select_clients_time,
                "update_clients_time": update_clients_time,
                "aggregate_network_time": aggregate_network_time,
                "evaluate_clients_time": evaluate_clients_time
            }

            log.update(stats)

            run.log(log)

            #? Check stop conditions
            if stop_counter >= constants.flad.FLAD_PATIENCE:
                break
        
        self.model_f_extractor = deepcopy(best_model_f_extractor)

        for client in self.clients:
            client.set_model_f_extractor(model_f_extractor=self.model_f_extractor)

        torch.save({
            "model_f_extractor": deepcopy(best_model_f_extractor.state_dict()),
            "model_sensors": {
                str(client): [deepcopy(model_sensor.state_dict()) for model_sensor in client.model_sensors]
            for client in self.clients}
        }, join(session_path, f"{constants.WIDE_DEEP_NETWORK_BASENAME}.pt"))

        run.finish()