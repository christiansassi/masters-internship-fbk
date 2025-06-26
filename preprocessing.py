# Local imports
import config
import utils

# External imports
import pandas as pd
import h5py
import numpy as np

from rich.progress import track

import logging

def load_dataset(input_src: str) -> pd.DataFrame:
    """
    Loads the dataset and performs initial preprocessing, including:
    - Stripping column names
    - Converting data to the correct types
    - Keeping only relevant columns
    - Fixing typos
    - Dropping NaN values

    :param input_src: The path to the dataset file to be loaded.
    :type input_src: `str`

    :return: The preprocessed dataframe.
    :rtype: `pd.DataFrame`
    """

    logging.info(f"Loading {input_src}")

    # Read dataset
    df = pd.read_csv(filepath_or_buffer=input_src, sep=",")

    # Strip columns names
    df.columns = df.columns.str.strip()

    # Convert data to the correct type
    df[config.DatasetConfig.SENSORS] = df[config.DatasetConfig.SENSORS].astype(float)

    # Keep relevant columns
    df = df[config.DatasetConfig.SENSORS + ["Normal/Attack"]]

    # Fix "A ttack"
    df["Normal/Attack"] = df["Normal/Attack"].str.replace(" ", "", regex=False)

    # Drop NaNs
    df.dropna(inplace=True)

    return df

def normalize_datasets(*datasets) -> list[pd.DataFrame]:
    """
    Normalizes one or more given datasets.

    :param datasets: One or more datasets to be normalized.
    :type datasets: `Any`

    :return: A list containing the normalized dataset(s).
    :rtype: `list[pd.DataFrame]`
    """

    logging.info(f"Normalizing {len(datasets)} dataset{'s' if len(datasets) > 1 else ''}")

    # Extract only sensor columns from all datasets
    sensor_data = np.concatenate([df[config.DatasetConfig.SENSORS].values for df in datasets], axis=0)

    # Calculate min and max for sensors
    global_min = sensor_data.min(axis=0)
    global_max = sensor_data.max(axis=0)

    # Set min to 0 if it is equal to max
    global_min = np.where(global_min == global_max, np.zeros_like(global_min), global_min)

    # Set max to 1 if it is equal to 0
    global_max = np.where(global_max == 0., np.ones_like(global_max), global_max)

    # Normalize only the sensor columns
    normalized_datasets = []

    for dataset in datasets:

        dataset_copy = dataset.copy()

        dataset_copy[config.DatasetConfig.SENSORS] = (dataset_copy[config.DatasetConfig.SENSORS] - global_min) / (global_max - global_min)
        dataset_copy[config.DatasetConfig.SENSORS] = dataset_copy[config.DatasetConfig.SENSORS].clip(lower=0, upper=1)

        normalized_datasets.append(dataset_copy)

    return normalized_datasets

def process_dataset(df: pd.DataFrame, output_src: str):
    """
    Processes a dataset and saves it. This includes:
    - Extracting labels and converting them into integers (one-hot encoding).
    - Organizing rows into windows according to `config.DatasetConfig.ROWS_PER_SAMPLE`.

    :param df: The dataset to be processed.
    :type df: `pd.DataFrame`

    :param output_src: The path to the output file where the processed dataset will be saved.
    :type output_src: `str`
    """

    # Extract labels and convert them into int
    labels = df["Normal/Attack"].map({"Normal": 0, "Attack": 1}).astype(np.int32)
    
    # Organize rows in windows
    num_samples = df.shape[0] - (config.DatasetConfig.ROWS_PER_SAMPLE + 1)
    sensor_data = df[config.DatasetConfig.SENSORS].to_numpy()
    labels_array = labels.to_numpy()

    x = np.empty((num_samples, config.DatasetConfig.ROWS_PER_SAMPLE, len(config.DatasetConfig.SENSORS)), dtype=np.float32)
    y = np.empty(num_samples, dtype=labels_array.dtype)

    for i in track(range(num_samples), description=f"{utils.log_timestamp_status()} Preparing {output_src}"):

        x[i] = sensor_data[i : i + config.DatasetConfig.ROWS_PER_SAMPLE]
        y[i] = labels_array[i + config.DatasetConfig.ROWS_PER_SAMPLE]

    # Save outputs
    hf = h5py.File(name=output_src, mode="w")
    hf.create_dataset(name="x", data=x)
    hf.create_dataset(name="y", data=y)
    hf.close()

if __name__ == "__main__":

    utils.clear_console()

    # Load Normal and Attack datasets
    df_normal = load_dataset(input_src=config.DatasetConfig.INPUT_NORMAL)
    df_attack = load_dataset(input_src=config.DatasetConfig.INPUT_ATTACK)

    # Normalize Normal and Attack datasets
    df_normal, df_attack = normalize_datasets(df_normal, df_attack)

    # Process Normal and Attack datasets
    process_dataset(df=df_normal, output_src=config.DatasetConfig.OUTPUT_NORMAL)
    process_dataset(df=df_attack, output_src=config.DatasetConfig.OUTPUT_ATTACK)