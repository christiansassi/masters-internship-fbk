import config
import utils
from constants import *
from models import WideDeepNetworkDAICS, ThresholdNetworkDAICS
from federated_client import *

from os.path import join
from os import makedirs

import logging

from time import time
from datetime import datetime

import tensorflow as tf

class Server:

    def __init__(self, clients: list[Client], wide_deep_network: WideDeepNetworkDAICS = None):
        
        self.clients = clients

        if wide_deep_network is None:
            self.wide_deep_network = WideDeepNetworkDAICS(
                window_past=WINDOW_PAST,
                window_present=WINDOW_PRESENT,
                n_inputs=len(GLOBAL_INPUTS),
                n_outputs=len(GLOBAL_OUTPUTS)
            )

            self.wide_deep_network.build(
                input_shape=(None, WINDOW_PAST, len(GLOBAL_INPUTS))
            )

            self.wide_deep_network.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
                loss=LOSS
            )
        
        else:
            self.wide_deep_network = wide_deep_network.clone()

        self.wide_deep_score = 0

        for client in self.clients:
            client.set_wide_deep_network(wide_deep_network=self.wide_deep_network)

    def select_clients(self) -> list[Client]:

        selected_clients = []

        min_score = float("inf")
        max_score = float("-inf")

        for client in self.clients:

            if client.wide_deep_score > self.wide_deep_score:
                continue

            min_score = min(min_score, client.wide_deep_score)
            max_score = max(max_score, client.wide_deep_score)

            selected_clients.append(client)
        
        for index, client in enumerate(selected_clients):

            if max_score != min_score:
                scaling_factor = (max_score - client.wide_deep_score) / (max_score - min_score)
            else:
                scaling_factor = 0

            client.wide_deep_epochs = int(MIN_EPOCHS + (MAX_EPOCHS - MIN_EPOCHS) * scaling_factor)
            client.wide_deep_steps = int(MIN_STEPS + (MAX_STEPS - MIN_STEPS) * scaling_factor)

            selected_clients[index] = client
        
        return selected_clients

    def aggregate_networks(self, clients: list[Client], weighted: bool = False) -> WideDeepNetworkDAICS:
        
        old_weights = [client.wide_deep_network.get_weights() for client in clients]
        new_weights = []

        if not weighted:

            for weights in zip(*old_weights):
                new_weights.append(np.mean(np.stack(weights, axis=0), axis=0))
            
        else:

            sample_counts = [len(client.df_train) + len(client.df_val) + len(client.df_test) for client in clients]
            total = np.sum(sample_counts)

            for weights in zip(*old_weights):
                weighted_sum = np.sum([w * (n/total) for w, n in zip(weights, sample_counts)], axis=0)
                new_weights.append(weighted_sum)
            
        wide_deep_network = self.wide_deep_network.clone()
        wide_deep_network.set_weights(new_weights)

        return wide_deep_network

    def federated_learning(self):

        session_id = str(int(datetime.now().timestamp()))

        run = config.WandbConfig.init_run(f"Wide Deep Network")

        map_clients_ids = {str(client): f"{'-'.join(client.normal_inputs)}" for index, client in enumerate(self.clients, start=1)}

        losses = {client: {
            "loss": float("-inf"),
            "val_loss": float("-inf"),
            "eval_loss": float("-inf")
        } for client in map_clients_ids.values()}

        round_num = 0
        stop_counter = 0

        best_score = float("-inf")
        best_wide_deep_network = self.wide_deep_network.clone()

        while True:

            start = time()

            round_num = round_num + 1

            print("")
            logging.info(f"---------- Round {round_num} ----------")

            # Select clients
            selected_clients = self.select_clients()

            # Update clients
            for index, client in enumerate(selected_clients):
                print(f"{utils.log_timestamp_status()} Training {index + 1} / {len(selected_clients)}", end="\r")
                loss, val_loss = client.train_wide_deep_network(wide_deep_network=self.wide_deep_network)

                client_id = map_clients_ids[str(client)]

                losses[client_id]["loss"] = loss
                losses[client_id]["val_loss"] = val_loss

            logging.info(f"Trained {len(selected_clients)} clients")

            # Model aggregation
            logging.info(f"Aggregating models")
            self.wide_deep_network = self.aggregate_networks(clients=self.clients)

            # Evaluate clients
            self.wide_deep_score = 0

            for index, client in enumerate(self.clients):
                print(f"{utils.log_timestamp_status()} Evaluating {index + 1} / {len(self.clients)}", end="\r")

                score = client.eval_wide_deep_network(wide_deep_network=self.wide_deep_network)

                client_id = map_clients_ids[str(client)]

                losses[client_id]["eval_loss"] = score
    
                self.wide_deep_score = self.wide_deep_score + score
            
            self.wide_deep_score = self.wide_deep_score / len(self.clients)

            logging.info(f"Evaluated {len(self.clients)} clients")

            # Check for improvements
            logging.info(f"Current score: {self.wide_deep_score}")
            logging.info(f"Best score: {best_score}")

            if self.wide_deep_score > best_score:
                stop_counter = 0

                best_score = self.wide_deep_score
                best_wide_deep_network = self.wide_deep_network.clone()

                # Save the checkpoint
                makedirs(name=WIDE_DEEP_NETWORK_CHECKPOINT, exist_ok=True)

                best_wide_deep_network.save_weights(filepath=f"{join(WIDE_DEEP_NETWORK_CHECKPOINT, WIDE_DEEP_NETWORK_BASENAME)}-{session_id}.h5")

            else:
                stop_counter = stop_counter + 1
            
            # Check stop conditions
            logging.info(f"Patience {stop_counter} / {PATIENCE}")

            log = {
                "round": round_num,
                "selected_clients": len(selected_clients),
                "score": self.wide_deep_score,
                "best": best_score,
                "stop_counter": stop_counter,
                "time_per_round": time() - start,
            }

            log.update(losses)

            run.log(log)

            if stop_counter >= PATIENCE:
                break
        
        self.wide_deep_network = best_wide_deep_network.clone()

        for client in self.clients:
            client.set_wide_deep_network(wide_deep_network=self.wide_deep_network)
        
        makedirs(name=WIDE_DEEP_NETWORK, exist_ok=True)

        self.wide_deep_network.save_weights(filepath=f"{join(WIDE_DEEP_NETWORK, WIDE_DEEP_NETWORK_BASENAME)}-{session_id}.h5")

        run.finish()