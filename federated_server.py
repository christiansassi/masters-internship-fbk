from constants import (
    MIN_EPOCHS,
    MAX_EPOCHS,
    MIN_STEPS,
    MAX_STEPS,
    WINDOW_PAST,
    WINDOW_PRESENT,
    FEATURES_IN,
    SENSOR_GROUPS_INDICES,
    KERNEL_SIZE,
    LEARNING_RATE,
    MOMENTUM,
    LOSS,
    PATIENCE,
    WIDE_DEEP_NETWORKS_TMP,
    WIDE_DEEP_NETWORKS_BASENAME,
    THRESHOLD_NETWORKS_TMP,
    THRESHOLD_NETWORKS_BASENAME
)

import config

from models import clone_wide_deep_networks, clone_threshold_networks
from federated_client import *

import tensorflow as tf

from os.path import join
from os import makedirs

from shutil import rmtree

import logging

from time import time

class Server:

    def __init__(self, clients: list[Client], wide_deep_networks: list[WideDeepNetworkDAICS] = [], threshold_networks: list[ThresholdNetworkDAICS] = []):
        
        self.clients = clients

        # Generate wide deep networks if not provided
        self.wide_deep_networks = []
        self.wide_deep_score = 0

        if len(wide_deep_networks) != len(SENSOR_GROUPS_INDICES):

            for sensors_indices in SENSOR_GROUPS_INDICES:

                model = WideDeepNetworkDAICS(
                    window_size_in=WINDOW_PAST,
                    window_size_out=WINDOW_PRESENT,
                    n_devices_in=FEATURES_IN,
                    n_devices_out=len(sensors_indices),
                    kernel_size=KERNEL_SIZE,
                    sensor_groups=SENSOR_GROUPS_INDICES
                )

                model.compile(
                    optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
                    loss=LOSS
                )

                self.wide_deep_networks.append(model)

        else:
            self.wide_deep_networks = clone_wide_deep_networks(
                wide_deep_networks=wide_deep_networks,
                optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
                loss=LOSS
            )

        # Generate threshold networks if not provided
        self.threshold_networks = []
        self.threshold_score = 0

        if len(threshold_networks) != len(SENSOR_GROUPS_INDICES):

            for _ in SENSOR_GROUPS_INDICES:
                model = ThresholdNetworkDAICS()

                model.compile(
                    optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
                    loss=LOSS
                )

                self.threshold_networks.append(model)

        else:
            self.threshold_networks = clone_threshold_networks(
                threshold_networks=threshold_networks,
                optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
                loss=LOSS
            )

        # Align the client's models with the server's
        for client in self.clients:
            client.set_wide_deep_network(wide_deep_networks=self.wide_deep_networks)
            client.set_threshold_networks(threshold_networks=self.threshold_networks)

    def select_clients(self, label: str) -> list[Client]:

        selected_clients = []

        min_score = float("inf")
        max_score = float("-inf")

        for client in self.clients:

            if getattr(client, f"{label}_score") > getattr(self, f"{label}_score"):
                continue

            min_score = min(min_score, getattr(client, f"{label}_score"))
            max_score = max(max_score, getattr(client, f"{label}_score"))

            selected_clients.append(client)
        
        for index, client in enumerate(selected_clients):

            if max_score != min_score:
                scaling_factor = (max_score - getattr(client, f"{label}_score")) / (max_score - min_score)
            else:
                scaling_factor = 0

            setattr(client, f"{label}_epochs", int(MIN_EPOCHS + (MAX_EPOCHS - MIN_EPOCHS) * scaling_factor))
            setattr(client, f"{label}_steps", int(MIN_STEPS + (MAX_STEPS - MIN_STEPS) * scaling_factor))

            selected_clients[index] = client
        
        return selected_clients

    def aggregate_networks(self, label: str, clients: list[Client], weighted: bool = False) -> list[WideDeepNetworkDAICS] | list[ThresholdNetworkDAICS]:

        aggregated_models = []

        for index in range(len(getattr(self, f"{label}_networks"))):

            group_models = [getattr(client, f"{label}_networks")[index] for client in clients]

            if weighted:
                total_samples = sum(len(client.train_input_indices) for client in clients)
                weights = [len(client.train_input_indices) / total_samples for client in clients]
            
            else:
                weights = [1.0 / len(clients)] * len(clients)
            
            averaged_weights = []

            for layer_index in range(len(group_models[0].get_weights())):
                layer_weights = [model.get_weights()[layer_index] for model in group_models]
                weighted_sum = sum(weight * layer_weight for weight, layer_weight in zip(weights, layer_weights))
                averaged_weights.append(weighted_sum)

            new_model = eval(f"clone_{label}_networks")(
                [group_models[0]],
                optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
                loss=LOSS
            )[0]

            new_model.set_weights(averaged_weights)
            aggregated_models.append(new_model)

        return aggregated_models

    def federated_learning(self, label: str):

        run = config.WandbConfig.init_run(f"{label.replace('_',' ').title()} Networks")

        selected_clients = self.select_clients(label=label)

        networks = getattr(self, f"{label}_networks")

        round_num = 0
        max_score = float("-inf")
        stop_counter = 0

        while True:

            start = time()

            round_num = round_num + 1

            print("")
            logging.info(f"---------- Round {round_num} ----------")

            # Update clients
            for index, client in enumerate(selected_clients):
                logging.info(f"Training {index + 1} / {len(selected_clients)}")
                getattr(client, f"train_{label}_network")(getattr(self, f"{label}_networks"))
            
            logging.info(f"Trained {len(selected_clients)} clients")

            # Model aggregation
            logging.info(f"Aggregating models")
            setattr(
                self, 
                f"{label}_networks", 
                self.aggregate_networks(
                    label=label,
                    clients=self.clients
                )
            )

            # Evaluate clients
            setattr(
                self,
                f"{label}_score",
                0
            )

            for index, client in enumerate(self.clients):
                logging.info(f"Evaluating {index + 1} / {len(self.clients)}")
                score = getattr(client, f"eval_{label}_network")(getattr(self, f"{label}_networks"))
            
                setattr(
                self,
                f"{label}_score",
                    getattr(self, f"{label}_score") + score
                )
            
            setattr(
                self,
                f"{label}_score",
                    getattr(self, f"{label}_score") / len(self.clients)
                )
            
            logging.info(f"Evaluated {len(self.clients)} clients")

            # Check for improvements
            logging.info(f"Current score: {getattr(self, f'{label}_score')}")
            logging.info(f"Best score: {max_score}")

            if getattr(self, f"{label}_score") > max_score:
                max_score = getattr(self, f"{label}_score")

                networks = eval(f"clone_{label}_networks")(
                    getattr(self, f"{label}_networks"),
                    optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
                    loss=LOSS
                )

                stop_counter = 0

                root = eval(f"{label.upper()}_NETWORKS_TMP")
                basename = eval(f"{label.upper()}_NETWORKS_BASENAME")

                makedirs(root, exist_ok=True)

                # Delete everything
                for index, network in enumerate(networks):
                    filepath = join(root, f"{basename}_{str(index + 1)}")
                    rmtree(filepath, ignore_errors=True)

                # Save
                for index, network in enumerate(networks):
                    network.save(join(root, f"{basename}_{str(index + 1)}"), save_format="tf")
            
            else:
                stop_counter = stop_counter + 1
            
            # Check stop conditions
            logging.info(f"Patience {stop_counter} / {PATIENCE}")

            run.log({ # type: ignore
                "round": round_num, 
                "clients": len(selected_clients),
                "score": getattr(self, f"{label}_score"), 
                "best": max_score,
                "stop_counter": stop_counter,
                "time_per_round": time() - start
            })

            if stop_counter >= PATIENCE:
                break
            
            # Select new clients
            selected_clients = self.select_clients(label=label)

        # Update global models
        setattr(
            self, 
            f"{label}_networks",
            eval(f"clone_{label}_networks")(
                networks,
                optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
                loss=LOSS
            )
        )

        # Update clients
        for client in self.clients:
            setattr(
                client, 
                f"{label}_networks",
                eval(f"clone_{label}_networks")(
                    networks,
                    optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
                    loss=LOSS
                )
            )
        
        run.finish()