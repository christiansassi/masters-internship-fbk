# Imports
import pandas as pd
import h5py
import numpy as np

from rich.progress import track

import os
from os.path import join

from datetime import datetime

# Configure warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
import logging
logging.basicConfig(
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)

# Constants
ROOT: str = "datasets"

SWAT_2015: str = join(ROOT, "SWaT2015")

INPUT_NORMAL: str = join(SWAT_2015, "original", "SWaT_Dataset_Normal.csv") # Only normal records
INPUT_ATTACK: str = join(SWAT_2015, "original", "SWaT_Dataset_Attack.csv") # Normal and attack records

OUTPUT_NORMAL: str = join(SWAT_2015, "processed", "SWaT_Dataset_Normal.hdf5") # Only normal records
OUTPUT_ATTACK: str = join(SWAT_2015, "processed", "SWaT_Dataset_Attack.hdf5") # Normal and attack records

NORMALIZE: bool = False

ROWS_PER_SAMPLE: int = 10
SENSORS: list[str] = [
    "FIT101", "LIT101", "MV101", "P101", "P102", "AIT201", "AIT202",
    "AIT203", "FIT201", "MV201", "P201", "P202", "P203", "P204", "P205",
    "P206", "DPIT301", "FIT301", "LIT301", "MV301", "MV302", "MV303",
    "MV304", "P301", "P302", "AIT401", "AIT402", "FIT401", "LIT401",
    "P401", "P402", "P403", "P404", "UV401", "AIT501", "AIT502", "AIT503",
    "AIT504", "FIT501", "FIT502", "FIT503", "FIT504", "P501", "P502",
    "PIT501", "PIT502", "PIT503", "FIT601", "P601", "P602", "P603"
]

# Lambdas
clear_console = lambda: os.system("cls" if os.name == "nt" else "clear")
log_status = lambda: f"[{datetime.now().strftime('%H:%M:%S')}][{logging.getLevelName(logging.getLogger().getEffectiveLevel())}]"

def load_dataset(input_src: str):

    logging.info(f"Loading {input_src}")

    # Read dataset
    df = pd.read_csv(filepath_or_buffer=input_src, sep=",")

    # Strip columns names
    df.columns = df.columns.str.strip()

    # Convert data to the correct type
    df[SENSORS] = df[SENSORS].astype(float)

    # Keep relevant columns
    df = df[SENSORS + ["Normal/Attack"]]

    # Fix "A ttack"
    df["Normal/Attack"] = df["Normal/Attack"].str.replace(" ", "", regex=False)

    # Drop NaNs
    df.dropna(inplace=True)

    return df

def normalize_datasets(*datasets):

    logging.info(f"Normalizing {len(datasets)} dataset{'s' if len(datasets) > 1 else ''}")

    # Extract only sensor columns from all datasets
    sensor_data = np.concatenate([df[SENSORS].values for df in datasets], axis=0)

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

        dataset_copy[SENSORS] = (dataset_copy[SENSORS] - global_min) / (global_max - global_min)
        dataset_copy[SENSORS] = dataset_copy[SENSORS].clip(lower=0, upper=1)

        normalized_datasets.append(dataset_copy)

    return normalized_datasets


def process_dataset(df: pd.DataFrame, output_src: str):

    # Extract labels and convert them in int
    labels = df["Normal/Attack"].map({"Normal": 0, "Attack": 1}).astype(np.int32)
    
    # Organize rows in windows
    num_samples = df.shape[0] - (ROWS_PER_SAMPLE + 1)
    sensor_data = df[SENSORS].to_numpy()
    labels_array = labels.to_numpy()

    x = np.empty((num_samples, ROWS_PER_SAMPLE, len(SENSORS)), dtype=sensor_data.dtype)
    y = np.empty(num_samples, dtype=labels_array.dtype)

    for i in track(range(num_samples), description=f"{log_status()} Preparing {output_src}"):

        x[i] = sensor_data[i : i + ROWS_PER_SAMPLE]
        y[i] = labels_array[i + ROWS_PER_SAMPLE]

    # Save outputs
    hf = h5py.File(name=output_src, mode="w")
    hf.create_dataset(name="x", data=x)
    hf.create_dataset(name="y", data=y)
    hf.close()

if __name__ == "__main__":

    clear_console()

    df_normal = load_dataset(input_src=INPUT_NORMAL)
    df_attack = load_dataset(input_src=INPUT_ATTACK)

    df_normal, df_attack = normalize_datasets(df_normal, df_attack)

    process_dataset(df=df_normal, output_src=OUTPUT_NORMAL)
    process_dataset(df=df_attack, output_src=OUTPUT_ATTACK)