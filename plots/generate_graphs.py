import pandas as pd
import matplotlib.pyplot as plt

from os.path import join
from os import makedirs, getenv

import numpy as np

from uuid import uuid4

import wandb
import pandas as pd

import dotenv
dotenv.load_dotenv()

ROOT: str = "plots"
DATA: str = "data"
FIGURES: str = "figures"
CLIENTS: str = "clients"

def _get_time_per_round_adj():

    api = wandb.Api()
    run = api.run(getenv("WIDE_DEEP_NETWORK_RUN"))

    history = run.history()

    columns = list(history.columns)

    select_clients_time = history["select_clients_time"].tolist()
    update_clients_time = history["update_clients_time"].tolist()
    aggregate_network_time = history["aggregate_network_time"].tolist()
    evaluate_clients_time = history["evaluate_clients_time"].tolist()

    clients = {}

    for column in columns:

        if column in ["stop_counter", "select_clients_time", "score", "update_clients_time", "_runtime", "time_per_round", "best", "round", "aggregate_network_time", "_step", "_timestamp", "evaluate_clients_time", "selected_clients"]:
            continue

        (name, metric) = column.split(".")

        clients.setdefault(name, {})[metric] = history[column].tolist()

    times = {
        "select_clients": [],
        "update_clients": [],
        "aggregate_clients": [],
        "evaluate_clients": [],
        "total": []
    }

    for round_number in range(len(history)):

        selected_clients = {}

        for client, metrics in clients.items():

            if not metrics["selected"][round_number]:
                continue

            selected_clients[client] = {metric: values[round_number] for metric, values in metrics.items()}
        
        times["select_clients"].append(select_clients_time[round_number])

        total_iterations = 0
        max_iterations = 0

        for client, metrics in selected_clients.items():
            iterations = metrics["epochs"] * (metrics["steps"] + 1543)

            total_iterations = total_iterations + iterations
            max_iterations = max(max_iterations, iterations)
        
        times["update_clients"].append(max_iterations * update_clients_time[round_number] / total_iterations)

        times["aggregate_clients"].append(aggregate_network_time[round_number])
        times["evaluate_clients"].append(evaluate_clients_time[round_number] / len(clients))

        times["total"].append(times["select_clients"][-1] + times["update_clients"][-1] + times["aggregate_clients"][-1] + times["evaluate_clients"][-1])

    return times

def generate_time_per_round(root: str, x: str, title: str):

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

        linewidth=2,
        label="Time (s)"
    )

    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Time (s)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)

    ax.legend(frameon=True, edgecolor="black")

    ax.set_xticks(x.iloc[indices])

    plt.tight_layout()
    plt.savefig(join(root, FIGURES, "time_per_round.png"), dpi=600, bbox_inches="tight")

def generate_best_loss(root: str, x: str, title: str):

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
        abs(y),
        color="blue",

        linewidth=2,
        label="Loss"
    )

    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)

    ax.legend(frameon=True, edgecolor="black")

    ax.set_xticks(x.iloc[indices])

    plt.tight_layout()
    plt.savefig(join(root, FIGURES, "best_loss.png"), dpi=600, bbox_inches="tight")

def generate_training_loss(root: str, x: str, prefix: str, title: str):

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
            abs(y[column]),

            linewidth=2,
            label=f"{prefix} #{index}"
        )

    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)

    ax.legend(frameon=True, edgecolor="black")

    ax.set_xticks(x.iloc[indices])

    plt.tight_layout()
    plt.savefig(join(root, FIGURES, "training_loss.png"), dpi=600, bbox_inches="tight")

def generate_validation_loss(root: str, x: str, prefix: str, title: str):

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
            abs(y[column]),

            linewidth=2,
            label=f"{prefix} #{index}"
        )

    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)

    ax.legend(frameon=True, edgecolor="black")

    ax.set_xticks(x.iloc[indices])

    plt.tight_layout()
    plt.savefig(join(root, FIGURES, "validation_loss.png"), dpi=600, bbox_inches="tight")

def generate_evaluation_loss(root: str, x: str, prefix: str, title: str):

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
            abs(y[column]),

            linewidth=2,
            label=f"{prefix} #{index}"
        )

    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)

    ax.legend(frameon=True, edgecolor="black")

    ax.set_xticks(x.iloc[indices])

    plt.tight_layout()
    plt.savefig(join(root, FIGURES, "evaluation_loss.png"), dpi=600, bbox_inches="tight")

def generate_threshold_loss(root: str, multiple_clients: bool, title: str):

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

        bars = ax.bar(
            np.arange(len(stages)),
            list(stages.values()),
            color="blue",
        )

        for bar in bars:
            height = bar.get_height()
    
            if abs(height) < 1e-3:
                label = f"{height:.2e}"
            else:
                label = f"{height:.3f}"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                abs(height),
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )

        labels = [f"Threshold Model #{i + 1}" for i in range(len(stages))]

        ax.set_xticks(np.arange(len(stages)))
        ax.set_xticklabels(labels, rotation=45, ha="right")

        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Loss")

        ax.set_axisbelow(True)
        ax.grid(True, which="both", linestyle=":", linewidth=0.8)

        ymax = max(stages.values())
        plt.ylim(top=ymax * 1.15)

        plt.tight_layout()
        plt.savefig(join(root, FIGURES, CLIENTS, f"client-{index}_threshold_loss.png"), dpi=600, bbox_inches="tight")


def generate_stop_counter(root: str, x: str, title: str):

    makedirs(join(root, FIGURES), exist_ok=True)

    df = pd.read_csv(join(root, DATA, "stop_counter.csv"))

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

        linewidth=2,
        label="Patience"
    )

    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Patience")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)

    ax.legend(frameon=True, edgecolor="black")

    ax.set_xticks(x.iloc[indices])

    plt.tight_layout()
    plt.savefig(join(root, FIGURES, "stop_counter.png"), dpi=600, bbox_inches="tight")

def generate_evaluation_loss_comparison(project: str, daics: str, x: str, prefix: str, title: str):

    # PROJECT
    df = pd.read_csv(join(project, DATA, "evaluation_loss.csv"))

    columns = [column for column in df.columns if not any(key in column for key in [x, "__MIN", "__MAX"])]

    x_plot = df[x] + 1
    y_plot = df[columns]

    _, ax = plt.subplots(figsize=(8, 4))

    indices = [i - 1 if i != 0 else i for i in range(0, len(x_plot), 10)]
    indices.append(len(x_plot)- 1)

    for index, column in enumerate(columns, start=1):
        ax.plot(
            x_plot,
            abs(y_plot[column]),

            linewidth=2,
            label=f"{prefix} #{index}",
            zorder=3
        )

    # DAICS
    df = pd.read_csv(join(daics, DATA, "best_loss.csv"))

    columns = [column for column in df.columns if not any(key in column for key in [x, "__MIN", "__MAX"])]

    y_plot = df[columns]
    y_plot = abs(y_plot.iloc[:, 0]).tolist() + [abs(y_plot.iloc[:, 0].tolist()[-1])] * (len(x_plot) - len(y_plot.iloc[:, 0]))

    ax.plot(
        x_plot,
        y_plot,
        color="black",
        # linestyle="dotted",
        linewidth=3.5,
        zorder=2
    )

    ax.plot(
        x_plot,
        y_plot,
        linewidth=2,
        color = "yellow",
        label="DAICS",
        zorder=2
    )
    
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)

    ax.legend(frameon=True, edgecolor="black")

    ax.set_xticks(x_plot.iloc[indices])

    plt.tight_layout()
    plt.savefig(join(ROOT, "evaluation_loss_comparison.png"), dpi=600, bbox_inches="tight")

def generate_stop_counter_with_best_loss(root: str, x: str, title: str):

    makedirs(join(root, FIGURES), exist_ok=True)

    # Time Per Round
    df = pd.read_csv(join(root, DATA, "stop_counter.csv"))

    columns = [column for column in df.columns if not any(key in column for key in [x, "__MIN", "__MAX"])]

    x_time = df[x] + 1
    y_time = df[columns]

    _, ax = plt.subplots(figsize=(8, 4))

    indices = [i - 1 if i != 0 else i for i in range(0, len(x_time), 10)]
    indices.append(len(x_time) - 1)

    # Plot Time
    ax.plot(
        x_time,
        y_time,
        color="blue",
        linewidth=2,
        label="Patience"
    )

    ax.set_ylabel("Patience")
    ax.set_xlabel("Round")
    ax.set_title(title)

    ax.grid(True, which="both", linestyle=":", linewidth=0.8)
    ax.set_xticks(x_time.iloc[indices])

    # Best Loss
    df = pd.read_csv(join(root, DATA, "best_loss.csv"))
    columns = [column for column in df.columns if not any(key in column for key in [x, "__MIN", "__MAX"])]

    x_loss = df[x] + 1
    y_loss = df[columns]

    # Create secondary y-axis on the right
    ax2 = ax.twinx()

    ax2.plot(
        x_loss,
        abs(y_loss),
        color="red",
        linewidth=2,
        label="Loss"
    )

    ax2.set_ylabel("Loss")

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=True, edgecolor="black", bbox_to_anchor=(0.59, 1.0))

    plt.tight_layout()
    plt.savefig(join(ROOT, "stop_counter_with_best_loss.png"), dpi=600, bbox_inches="tight")

def generate_time_per_round_comparison(daics: str, x: str, title: str):

    # PROJECT
    times = _get_time_per_round_adj()

    x_plot = list(range(1,len(times["total"])+1))
    y = times["total"]

    _, ax = plt.subplots(figsize=(8, 4))

    indices = [i - 1 if i != 0 else i for i in range(0, len(x_plot), 10)]
    indices.append(len(x_plot)- 1)

    ax.plot(
        x_plot,
        y,
        linewidth=2,
        label="Time (s)"
    )

    # DAICS
    df = pd.read_csv(join(daics, DATA, "time_per_round.csv"))

    columns = [column for column in df.columns if not any(key in column for key in [x, "__MIN", "__MAX"])]

    y_plot = df[columns]
    y_plot = abs(y_plot.iloc[:, 0]).tolist() + [abs(y_plot.iloc[:, 0].tolist()[-1])] * (len(x_plot) - len(y_plot.iloc[:, 0]))

    ax.plot(
        x_plot,
        y_plot,
        linewidth=2,
        label="DAICS",
    )

    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Time (s)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)

    ax.legend(frameon=True, edgecolor="black")

    ax.set_xticks([x_plot[index] for index in indices])

    plt.tight_layout()
    plt.savefig(join(ROOT, "time_per_round_comparison.png"), dpi=600, bbox_inches="tight")

def generate_time_per_round_adj(root: str, title: str):

    # PROJECT
    times = _get_time_per_round_adj()

    x = list(range(1,len(times["total"])+1))
    y = times["total"]

    _, ax = plt.subplots(figsize=(8, 4))

    indices = [i - 1 if i != 0 else i for i in range(0, len(x), 10)]
    indices.append(len(x)- 1)

    ax.plot(
        x,
        y,
        color="blue",
        linewidth=2,
        label="Time (s)"
    )

    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Time (s)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)

    ax.legend(frameon=True, edgecolor="black")

    ax.set_xticks([x[index] for index in indices])

    plt.tight_layout()
    plt.savefig(join(root, FIGURES, "time_per_round_adj.png"), dpi=600, bbox_inches="tight")

# Time Per Round
generate_time_per_round(root=r"plots\daics", x="Step", title="Wide Deep Neural Network - Time Per Round")
generate_time_per_round(root=r"plots\project", x="Step", title="Wide Deep Neural Network - Time Per Round")
generate_time_per_round_adj(root=r"plots\project", title="Wide Deep Neural Network - Time Per Round")

# Best Loss
generate_best_loss(root=r"plots\daics", x="Step", title="Wide Deep Neural Network - Best Loss")
generate_best_loss(root=r"plots\project", x="Step", title="Wide Deep Neural Network - Best Loss")

# Training Loss
generate_training_loss(root=r"plots\daics", x="Step", prefix="Model Sensor", title="Wide Deep Neural Network - Losses")
generate_training_loss(root=r"plots\project", x="Step", prefix="Client", title="Wide Deep Neural Network - Losses")

# Validation Loss
generate_validation_loss(root=r"plots\daics", x="Step", prefix="Model Sensor", title="Wide Deep Neural Network - Losses")
generate_validation_loss(root=r"plots\project", x="Step", prefix="Client", title="Wide Deep Neural Network - Losses")

# Evaluation Loss
generate_evaluation_loss(root=r"plots\project", x="Step", prefix="Client", title="Wide Deep Neural Network - Losses")

# Threshold Loss
generate_threshold_loss(root=r"plots\daics", multiple_clients=False, title="Threshold Neural Network - Losses")
generate_threshold_loss(root=r"plots\project", multiple_clients=True, title="Threshold Neural Network - Losses")

# Stop Counter
generate_stop_counter(root=r"plots\project", x="Step", title="Wide Deep Neural Network - Stop Counter")

# Evaluation Loss comparison
generate_evaluation_loss_comparison(project=r"plots\project", daics=r"plots\daics", x="Step", prefix="Client", title="Wide Deep Neural Networks - Losses (Centralized vs. Decentralized)")

# Time Per Round comparison
generate_time_per_round_comparison(daics=r"plots\daics", x="Step", title="Wide Deep Neural Network - Time Per Round (Centralized vs. Decentralized)")

# Stop Counter with Loss
generate_stop_counter_with_best_loss(root=r"plots\project", x="Step", title="Wide Deep Neural Network - Stop Counter and Best Loss")