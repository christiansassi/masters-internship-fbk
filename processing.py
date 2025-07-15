from constants import (
    ACTUATORS_SENSORS,
    WINDOW_PAST,
    HORIZON,
    WINDOW_PRESENT,
    SAMPLING_START,
    TRAIN_STEP,
    VAL_STEP,
    TEST_STEP,
    BATCH_SIZE,
    INPUT_NORMAL_FILE,
    INPUT_ATTACK_FILE,
    OUTPUT_DIR,
    OUTPUT_FILE,
    TRAIN,
    VAL
)

from os import makedirs

import numpy as np

import h5py
import pandas as pd

def clean_dataset(src: str) -> pd.DataFrame:

    # Load dataset
    df = pd.read_csv(filepath_or_buffer=src)

    # Drop NaNs
    df = df.dropna()

    # Clear column names
    df.columns = df.columns.str.strip()

    # Keep only sensors and actuators
    df = df[ACTUATORS_SENSORS].astype(float)

    return df

def normalize_datasets(*datasets: tuple[pd.DataFrame]) -> tuple[pd.DataFrame]:

    # Vertically stack all the data
    full_data = np.vstack([dataset.values for dataset in datasets])

    # Get min and max value
    min_v = np.minimum(full_data.min(axis=0), 0)
    max_v = np.maximum(full_data.max(axis=0), 1)

    # Normalize each dataset
    normalize = lambda x: np.clip((x - min_v) / (max_v - min_v), 0, 1)

    return (
        pd.DataFrame(data=normalize(dataset.values), columns=ACTUATORS_SENSORS)
        for dataset in datasets
    )

def split_train_val_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    total = len(df)

    df_train = df[:int(TRAIN * total)]
    df_val = df[len(df_train):len(df_train) + int(VAL * total)]
    df_test = df[len(df_train) + len(df_val):]

    return df_train, df_val, df_test

def prepare_sliding_windows(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> tuple:

    results = []

    for df, step in [(df_train, TRAIN_STEP), (df_val, VAL_STEP), (df_test, TEST_STEP)]:

        # Input

        # Generates a vector of indices from SAMPLING_START to len(df_*) with a step size of *_STEP,
        # shifted backward by HORIZON + WINDOW_PRESENT.
        # These indices are expanded into windows of size equal to len(np.arange(1, WINDOW_PAST + 1)).
        # Each window is built by subtracting values from np.arange(1, WINDOW_PAST + 1) from each index,
        # resulting in past time steps for prediction.
        # Finally, the windows are trimmed to be divisible by BATCH_SIZE.

        input_indices = (np.arange(SAMPLING_START, len(df), step) - HORIZON - WINDOW_PRESENT)[:, None] - np.arange(1, WINDOW_PAST + 1)
        input_indices = np.sort(input_indices)
        input_indices = input_indices[: (len(input_indices) // BATCH_SIZE) * BATCH_SIZE, :]

        results.append(input_indices)

        # Output

        # Generates a vector of indices from SAMPLING_START to len(df_*) with a step size of *_STEP.
        # These indices are expanded into windows of size equal to len(np.arange(1, WINDOW_PRESENT + 1)).
        # Each window is built by subtracting values from np.arange(1, WINDOW_PRESENT + 1) from each index,
        # resulting in the target indices to be predicted.
        # Finally, the windows are trimmed to be divisible by BATCH_SIZE.

        output_indices = np.arange(SAMPLING_START, len(df), step)[:, None] - np.arange(1, WINDOW_PRESENT + 1)
        output_indices = np.sort(output_indices)
        output_indices = output_indices[: (len(output_indices) // BATCH_SIZE) * BATCH_SIZE, :]

        results.append(output_indices)

    return tuple(results)

if __name__ == "__main__":

    # Create output dir if it doesn't exist
    makedirs(name=OUTPUT_DIR, exist_ok=True)

    df_normal = clean_dataset(src=INPUT_NORMAL_FILE)
    df_attack = clean_dataset(src=INPUT_ATTACK_FILE)

    df_normal, df_attack = normalize_datasets(df_normal, df_attack)

    df_normal_train, df_normal_val, df_normal_test = split_train_val_test(df=df_normal)

    (
        df_normal_train_input_indices, 
        df_normal_train_output_indices, 
        
        df_normal_val_input_indices, 
        df_normal_val_output_indices, 
        
        df_normal_test_input_indices, 
        df_normal_test_output_indices
    ) = prepare_sliding_windows(df_train=df_normal_train, df_val=df_normal_val, df_test=df_normal_test)

    # Save everything
    hf = h5py.File(name=OUTPUT_FILE, mode="w")

    hf.create_dataset("df_normal_train", data=df_normal_train.values)
    hf.create_dataset("df_normal_val", data=df_normal_val.values)
    hf.create_dataset("df_normal_test", data=df_normal_test.values)

    hf.create_dataset("df_attack", data=df_attack.values)

    hf.create_dataset("df_normal_train_input_indices", data=df_normal_train_input_indices)
    hf.create_dataset("df_normal_train_output_indices", data=df_normal_train_output_indices)

    hf.create_dataset("df_normal_val_input_indices", data=df_normal_val_input_indices)
    hf.create_dataset("df_normal_val_output_indices", data=df_normal_val_output_indices)

    hf.create_dataset("df_normal_test_input_indices", data=df_normal_test_input_indices)
    hf.create_dataset("df_normal_test_output_indices", data=df_normal_test_output_indices)

    hf.close()
