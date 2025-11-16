import pandas as pd
import matplotlib.pyplot as plt

from os.path import join
from os import makedirs

import numpy as np

from uuid import uuid4

DATA: str = "data"
FIGURES: str = "figures"
CLIENTS: str = "clients"

def generate_time_per_round(root: str, x: str):

    makedirs(join(root, FIGURES), exist_ok=True)

    df = pd.read_csv(join(root, DATA, "time_per_round.csv"))

    columns = [column for column in df.columns if not any(key in column for key in [x, "__MIN", "__MAX"])]

    x = df[x] + 1
    y = df[columns]

    _, ax = plt.subplots(figsize=(8, 4))

    indices = [i - 1 if i != 0 else i for i in range(0, len(x), 10)]
    indices.append(len(x)- 1)

    ax.plot(
        x,
        y,
        color="blue",

        # marker="s",
        # markersize=6,
        # markevery=indices,
        
        linewidth=2,
        label="Time (s)"
    )

    ax.set_title("Time Per Round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Time (s)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)

    ax.legend(frameon=True, edgecolor="black")

    ax.set_xticks(x.iloc[indices])

    plt.tight_layout()
    plt.savefig(join(root, FIGURES, "time_per_round.png"), dpi=600, bbox_inches="tight")

def generate_best_loss(root: str, x: str):

    makedirs(join(root, FIGURES), exist_ok=True)

    df = pd.read_csv(join(root, DATA, "best_loss.csv"))

    columns = [column for column in df.columns if not any(key in column for key in [x, "__MIN", "__MAX"])]

    x = df[x] + 1
    y = df[columns]

    indices = [i - 1 if i != 0 else i for i in range(0, len(x), 10)]
    indices.append(len(x)- 1)

    _, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        x,
        y,
        color="blue",

        # marker="s",
        # markersize=6,
        # markevery=indices,

        linewidth=2,
        label="Loss"
    )

    ax.set_title("Best Loss")
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)

    ax.legend(frameon=True, edgecolor="black")

    ax.set_xticks(x.iloc[indices])

    plt.tight_layout()
    plt.savefig(join(root, FIGURES, "best_loss.png"), dpi=600, bbox_inches="tight")

def generate_training_loss(root: str, x: str):

    makedirs(join(root, FIGURES), exist_ok=True)

    df = pd.read_csv(join(root, DATA, "training_loss.csv"))

    columns = [column for column in df.columns if not any(key in column for key in [x, "__MIN", "__MAX"])]

    x = df[x] + 1
    y = df[columns]

    _, ax = plt.subplots(figsize=(8, 4))

    indices = [i - 1 if i != 0 else i for i in range(0, len(x), 10)]
    indices.append(len(x)- 1)

    for index, column in enumerate(columns, start=1):
        ax.plot(
            x,
            y[column],

            linewidth=2,
            label=f"Model Sensor #{index}"
        )

    ax.set_title("Training Loss")
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)

    ax.legend(frameon=True, edgecolor="black")

    ax.set_xticks(x.iloc[indices])

    plt.tight_layout()
    plt.savefig(join(root, FIGURES, "training_loss.png"), dpi=600, bbox_inches="tight")

def generate_validation_loss(root: str, x: str):

    makedirs(join(root, FIGURES), exist_ok=True)

    df = pd.read_csv(join(root, DATA, "validation_loss.csv"))

    columns = [column for column in df.columns if not any(key in column for key in [x, "__MIN", "__MAX"])]

    x = df[x] + 1
    y = df[columns]

    _, ax = plt.subplots(figsize=(8, 4))

    indices = [i - 1 if i != 0 else i for i in range(0, len(x), 10)]
    indices.append(len(x)- 1)

    for index, column in enumerate(columns, start=1):
        ax.plot(
            x,
            y[column],

            linewidth=2,
            label=f"Model Sensor #{index}"
        )

    ax.set_title("Validation Loss")
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)

    ax.legend(frameon=True, edgecolor="black")

    ax.set_xticks(x.iloc[indices])

    plt.tight_layout()
    plt.savefig(join(root, FIGURES, "validation_loss.png"), dpi=600, bbox_inches="tight")

def generate_evaluation_loss(root: str, x: str):

    makedirs(join(root, FIGURES), exist_ok=True)

    df = pd.read_csv(join(root, DATA, "evaluation_loss.csv"))

    columns = [column for column in df.columns if not any(key in column for key in [x, "__MIN", "__MAX"])]

    x = df[x] + 1
    y = df[columns]

    _, ax = plt.subplots(figsize=(8, 4))

    indices = [i - 1 if i != 0 else i for i in range(0, len(x), 10)]
    indices.append(len(x)- 1)

    for index, column in enumerate(columns, start=1):
        ax.plot(
            x,
            y[column],

            linewidth=2,
            label=f"Model Sensor #{index}"
        )

    ax.set_title("Evaluation Loss")
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)

    ax.legend(frameon=True, edgecolor="black")

    ax.set_xticks(x.iloc[indices])

    plt.tight_layout()
    plt.savefig(join(root, FIGURES, "evaluation_loss.png"), dpi=600, bbox_inches="tight")

def generate_threshold_loss(root: str, multiple_clients: bool):

    makedirs(join(root, FIGURES), exist_ok=True)
    makedirs(join(root, FIGURES, CLIENTS), exist_ok=True)

    df = pd.read_csv(join(root, DATA, "threshold_loss.csv"))

    if not multiple_clients:
        clients = {"client": {}}

        for loss in df["loss"].tolist():
            clients["client"][str(uuid4())] = abs(loss)

    else:
        clients = {entry.split("_")[0]: {} for entry in df["client"]}
        
        for key in clients.keys():
            for entry, loss in df[["client", "loss"]].itertuples(index=False):
                if key in entry:
                    clients[key][entry.replace(f"{key}_", "")] = loss

    for index, (_, stages) in enumerate(clients.items(), start=1):

        _, ax = plt.subplots(figsize=(8, 4))

        ax.bar(
            np.arange(len(stages)),
            list(stages.values()),
            color="blue",
        )

        labels = [f"Threshold Model #{i + 1}" for i in range(len(stages))]

        ax.set_xticks(np.arange(len(stages)))
        ax.set_xticklabels(labels, rotation=45, ha="right")

        ax.set_title("Training Loss")
        ax.set_xlabel("")
        ax.set_ylabel("Loss")

        ax.set_axisbelow(True)
        ax.grid(True, which="both", linestyle=":", linewidth=0.8)

        plt.tight_layout()
        plt.savefig(join(root, FIGURES, CLIENTS, f"client-{index}_threshold_loss.png"), dpi=600, bbox_inches="tight")

# Time Per Round
generate_time_per_round(root=r"graphs\daics", x="Step")
generate_time_per_round(root=r"graphs\project", x="Step")

# Best Loss
generate_best_loss(root=r"graphs\daics", x="Step")
generate_best_loss(root=r"graphs\project", x="Step")

# Training Loss
generate_training_loss(root=r"graphs\daics", x="Step")
generate_training_loss(root=r"graphs\project", x="Step")

# Validation Loss
generate_validation_loss(root=r"graphs\daics", x="Step")
generate_validation_loss(root=r"graphs\project", x="Step")

# Evaluation Loss
generate_evaluation_loss(root=r"graphs\project", x="Step")

# Threshold Loss
generate_threshold_loss(root=r"graphs\daics", multiple_clients=False)
generate_threshold_loss(root=r"graphs\project",multiple_clients=True)