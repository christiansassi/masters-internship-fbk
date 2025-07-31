from os.path import join

from time import time

#? === Constants from DAICS ===
ACTUATORS_SENSORS: list[str] = [
    "FIT101", "LIT101", "MV101", "P101", "P102", "AIT201", "AIT202", "AIT203",
    "FIT201", "MV201", "P201", "P202", "P203", "P204", "P205", "P206",
    "DPIT301", "FIT301", "LIT301", "MV301", "MV302", "MV303", "MV304",
    "P301", "P302", "AIT401", "AIT402", "FIT401", "LIT401", "P401", "P402",
    "P403", "P404", "UV401", "AIT501", "AIT502", "AIT503", "AIT504",
    "FIT501", "FIT502", "FIT503", "FIT504", "P501", "P502", "PIT501",
    "PIT502", "PIT503", "FIT601", "P601", "P602", "P603"
]

FEATURES_IN: int = len(ACTUATORS_SENSORS)

SENSORS_GROUPS: list[list] = [
    ["FIT101", "LIT101"],
    ["AIT201", "AIT202", "AIT203", "FIT201"],
    ["DPIT301", "FIT301", "LIT301"],
    ["AIT401", "AIT402", "FIT401", "LIT401"],
    ["AIT501", "AIT502", "AIT503", "AIT504", "FIT501", "FIT502", "FIT503", "FIT504", "PIT501", "PIT502", "PIT503"],
    ["FIT601"]
]

SENSOR_GROUPS_INDICES: list[list] = [
    [0, 1], 
    [5, 6, 7, 8], 
    [16, 17, 18], 
    [25, 26, 27, 28], 
    [34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46], 
    [47]
]

WINDOW_PAST: int = 60
HORIZON: int = 50
WINDOW_PRESENT: int = 4
SAMPLING_START: int = WINDOW_PAST + HORIZON + WINDOW_PRESENT

TRAIN_STEP: int = 1
VAL_STEP: int = 1
TEST_STEP: int = 1
BATCH_SIZE: int = 32
T_EPOCHS: int = 10

KERNEL_SIZE = 1
LEARNING_RATE: float = 0.01
MOMENTUM: float = 0.9
LOSS: str = "mse"

W_ANOMALY: int = 15

# According to : https://www.researchgate.net/publication/305809559
# Some of the attacks have a stronger effect on the dynamics of system and causing more time
# for the system to stabilize (after the attack). Simpler attacks, such as those that effect flow rates,
# require less time to stabilize. Also, some attacks do not take effect immediately (attack impact is seen after the attack's end).
# Based on that, attack impact is considered as part of the attack, and we avoid human intervention on the period just after the attack
W_GRACE: int = 60

MED_FILTER_LAG: int = 59

#? === Constants from FLAD ===

MIN_EPOCHS: int = 1
MAX_EPOCHS: int = 5
MIN_STEPS: int = 10
MAX_STEPS: int = 1000
PATIENCE: int = 25

#? === Train, Val, Test ===
TRAIN: float = 0.8
VAL: float = 0.1
TEST: float = 0.1

#? === Paths ===
ROOT_DIR: str = join("datasets", "SWaT2015")

INPUT_DIR: str = join(ROOT_DIR, "original")
INPUT_NORMAL_FILE: str = join(INPUT_DIR, "SWaT_Dataset_Normal.csv")
INPUT_ATTACK_FILE: str = join(INPUT_DIR, "SWaT_Dataset_Attack.csv")

OUTPUT_DIR: str = join(ROOT_DIR, "processed")
OUTPUT_FILE: str = join(OUTPUT_DIR, "SWaT_Dataset.h5")

MODELS: str = "models"
MODELS_TMP: str = join(MODELS, "tmp", str(int(time())))

WIDE_DEEP_NETWORKS_BASENAME: str = "wide_deep_networks"
WIDE_DEEP_NETWORK_BASENAME: str = "wide_deep_network"
WIDE_DEEP_NETWORKS_LABEL: str = "wide_deep"
WIDE_DEEP_NETWORKS: str = join(MODELS, WIDE_DEEP_NETWORKS_BASENAME)
WIDE_DEEP_NETWORKS_TMP: str = join(MODELS_TMP, WIDE_DEEP_NETWORKS_BASENAME)

THRESHOLD_NETWORKS_BASENAME: str = "threshold_networks"
THRESHOLD_NETWORK_BASENAME: str = "threshold_network"
THRESHOLD_NETWORKS_LABEL: str = "threshold"
THRESHOLD_NETWORKS: str = join(MODELS, THRESHOLD_NETWORKS_BASENAME)
THRESHOLD_NETWORKS_TMP: str = join(MODELS_TMP, THRESHOLD_NETWORKS_BASENAME)

CACHE: str = "cache"
CACHE_CLIENTS: str = join(CACHE, "clients")

#? === Federated Learning

N_CLIENTS: int = 15