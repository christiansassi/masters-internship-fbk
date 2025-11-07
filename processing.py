import config
from constants import *

from os import makedirs

import numpy as np

import h5py
import pandas as pd

from itertools import groupby

def clean_dataset(src: str) -> pd.DataFrame:

    # Load dataset
    df = pd.read_csv(filepath_or_buffer=src)

    # Drop NaNs
    df = df.dropna()

    # Clear column names
    df.columns = df.columns.str.strip()

    # Fix typos
    df["Normal/Attack"] = df["Normal/Attack"].replace({
        "A ttack": "Attack"
    })

    # Keep only sensors and actuators
    df[GLOBAL_INPUTS] = df[GLOBAL_INPUTS].astype(float)

    df["Normal/Attack"] = df["Normal/Attack"].map({"Normal": 0, "Attack": 1})
    df["Normal/Attack"] = df["Normal/Attack"].astype(int)

    df = df[GLOBAL_INPUTS + ["Normal/Attack"]]

    return df

def normalize_datasets(*datasets: tuple[pd.DataFrame]) -> tuple[pd.DataFrame]:

    # Stack data across all datasets to compute global min/max
    full_data = np.vstack([dataset[GLOBAL_INPUTS].to_numpy() for dataset in datasets])

    # Compute min/max
    min_v = full_data.min(axis=0)
    max_v = full_data.max(axis=0)

    # Match SWaT handling:
    # if min == max, force min = 0
    # if max == 0, force max = 1
    min_v = np.where(min_v == max_v, np.zeros_like(min_v), min_v)
    max_v = np.where(max_v == 0., np.ones_like(max_v), max_v)

    # Lambda normalization (same as SWaT script)
    normalize = lambda arr: np.clip((arr - min_v) / (max_v - min_v), 0, 1)

    results = []
    for dataset in datasets:
        scaled = normalize(dataset[GLOBAL_INPUTS].to_numpy())
        out = pd.DataFrame(
            data=np.hstack([scaled, dataset[["Normal/Attack"]].to_numpy()]),
            columns=GLOBAL_INPUTS + ["Normal/Attack"]
        )
        results.append(out)

    return tuple(results)

def split_clients(df: pd.DataFrame) -> list[pd.DataFrame]:

    clients = [df[stage + ["Normal/Attack"]].copy() for stage in STAGES]

    attack_indices = df.index[df["Normal/Attack"] == 1].tolist()

    attack_chunks = []

    for _, group in groupby(enumerate(attack_indices), key=lambda t: t[1] - t[0]):
        attack_chunks.append([v for _, v in group])

    if not len(attack_chunks):
        return clients

    for client in clients:
        client["Normal/Attack"] = 0

        for attack_index, attack_labels in enumerate(ATTACKS):

            # if not set(attack_labels) & set(client.columns):
            #     continue
            
            if not (set(attack_labels) & set(client.columns)) & set(GLOBAL_OUTPUTS):
                continue

            client.loc[attack_chunks[attack_index], "Normal/Attack"] = 1

    return clients

def split_train_val_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    total = len(df)

    df_train = df[:int(TRAIN * total)]
    df_val = df[len(df_train):len(df_train) + int(VAL * total)]
    df_test = df[len(df_train) + len(df_val):]

    return df_train, df_val, df_test

def prepare_sliding_windows(df_train: pd.DataFrame = None, df_val: pd.DataFrame = None, df_test: pd.DataFrame = None) -> tuple:

    windows = []

    if df_train is not None:
        windows.append((df_train, daics.TRAIN_STEP))
    
    if df_val is not None:
        windows.append((df_val, daics.VAL_STEP))
    
    if df_test is not None:
        windows.append((df_test, daics.TEST_STEP))

    results = []

    for df, step in windows:

        # Input

        # Generates a vector of indices from SAMPLING_START to len(df_*) with a step size of *_STEP,
        # shifted backward by HORIZON + WINDOW_PRESENT.
        # These indices are expanded into windows of size equal to len(np.arange(1, WINDOW_PAST + 1)).
        # Each window is built by subtracting values from np.arange(1, WINDOW_PAST + 1) from each index,
        # resulting in past time steps for prediction.
        # Finally, the windows are trimmed to be divisible by BATCH_SIZE.

        input_indices = (np.arange(daics.SAMPLING_START, len(df), step) - daics.HORIZON - daics.WINDOW_PRESENT)[:, None] - np.arange(1, daics.WINDOW_PAST + 1)
        input_indices = np.sort(input_indices)
        input_indices = input_indices[: (len(input_indices) // daics.BATCH_SIZE) * daics.BATCH_SIZE, :]

        results.append(input_indices)

        # Output

        # Generates a vector of indices from SAMPLING_START to len(df_*) with a step size of *_STEP.
        # These indices are expanded into windows of size equal to len(np.arange(1, WINDOW_PRESENT + 1)).
        # Each window is built by subtracting values from np.arange(1, WINDOW_PRESENT + 1) from each index,
        # resulting in the target indices to be predicted.
        # Finally, the windows are trimmed to be divisible by BATCH_SIZE.

        output_indices = np.arange(daics.SAMPLING_START, len(df), step)[:, None] - np.arange(1, daics.WINDOW_PRESENT + 1)
        output_indices = np.sort(output_indices)
        output_indices = output_indices[: (len(output_indices) // daics.BATCH_SIZE) * daics.BATCH_SIZE, :]

        results.append(output_indices)

    return tuple(results)

if __name__ == "__main__":

    # Create output dir if it doesn't exist
    makedirs(name=OUTPUT_DIR, exist_ok=True)

    # Prepare dataset
    df_normal = clean_dataset(src=INPUT_NORMAL_FILE)
    df_attack = clean_dataset(src=INPUT_ATTACK_FILE)

    df_normal, df_attack = normalize_datasets(df_normal, df_attack)

    # Federated clients
    clients_normal = split_clients(df=df_normal)
    clients_attack = split_clients(df=df_attack)

    hf = h5py.File(name=OUTPUT_FILE, mode="w")
    group_normal = hf.create_group(f"normal")
    group_attack = hf.create_group(f"attack")

    for index, client in enumerate(clients_normal, start=1):

        df_normal_train, df_normal_val, df_normal_test = split_train_val_test(df=client)

        (
            df_normal_train_input_indices, 
            df_normal_train_output_indices, 
            
            df_normal_val_input_indices, 
            df_normal_val_output_indices, 
            
            df_normal_test_input_indices, 
            df_normal_test_output_indices
        ) = prepare_sliding_windows(df_train=df_normal_train, df_val=df_normal_val, df_test=df_normal_test)

        group = group_normal.create_group(f"client-{index}")
        group.attrs["columns"] = list(client.columns)
        group.attrs["inputs"] = [column for column in list(client.columns) if column != "Normal/Attack"]
        group.attrs["outputs"] = [column for column in list(client.columns) if column in GLOBAL_OUTPUTS]

        group.create_dataset("df_normal_train", data=df_normal_train.values)
        group.create_dataset("df_normal_val", data=df_normal_val.values)
        group.create_dataset("df_normal_test", data=df_normal_test.values)

        group.create_dataset("df_normal_train_input_indices", data=df_normal_train_input_indices)
        group.create_dataset("df_normal_train_output_indices", data=df_normal_train_output_indices)

        group.create_dataset("df_normal_val_input_indices", data=df_normal_val_input_indices)
        group.create_dataset("df_normal_val_output_indices", data=df_normal_val_output_indices)

        group.create_dataset("df_normal_test_input_indices", data=df_normal_test_input_indices)
        group.create_dataset("df_normal_test_output_indices", data=df_normal_test_output_indices)
    
    for index, client in enumerate(clients_attack, start=1):

        (
            df_attack_input_indices,
            df_attack_output_indices
        ) = prepare_sliding_windows(df_test=client)

        group = group_attack.create_group(f"client-{index}")
        group.attrs["columns"] = list(client.columns)
        group.attrs["inputs"] = [column for column in list(client.columns) if column != "Normal/Attack"]
        group.attrs["outputs"] = [column for column in list(client.columns) if column in GLOBAL_OUTPUTS]

        group.create_dataset("df_attack", data=client.values)

        group.create_dataset("df_attack_input_indices", data=df_attack_input_indices)
        group.create_dataset("df_attack_output_indices", data=df_attack_output_indices)
    
    hf.close()