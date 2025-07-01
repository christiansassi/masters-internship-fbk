import os

#? --- Environment Setup ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_CONSOLE"] = "off"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from os.path import join, exists
from os import makedirs, getcwd

import dotenv

import tensorflow as tf

from datetime import datetime

from types import SimpleNamespace
from enum import Enum

# Load Environment Variables
dotenv.load_dotenv()

#? --- User Settings ---
USE_GPU: bool = True
WANDB: bool = True
VERBOSE: int = 0

if not USE_GPU and len(tf.config.list_physical_devices("GPU")) > 0:
    tf.config.set_visible_devices([], "GPU")

#? --- FLAD Configuration ---
class FLADHyperparameters:
    """
    Hyperparameters for FLAD
    """

    MIN_EPOCHS: int = 1
    MAX_EPOCHS: int = 5
    MIN_STEPS: int = 10
    MAX_STEPS: int = 1000
    PATIENCE: int = 25

    N_CLIENTS: int = os.cpu_count() - 1 # 13 50 90

class ServerAndClientConfig:
    """
    Configuration for server and client obj
    """

    CLIENT_ROOT: str = join(getcwd(), "clients")
    CLIENT_EXTENSION: str = ".pkl"

    @classmethod
    def export_client(cls, client) -> tuple:
        return (
            join(cls.CLIENT_ROOT, str(client)),
            join(cls.CLIENT_ROOT, str(client), f"client{cls.CLIENT_EXTENSION}"),
            join(cls.CLIENT_ROOT, str(client), f"{ModelConfig.AUTOENCODER_MODEL_BASENAME}{ModelConfig.MODEL_EXTENSION}"),
            join(cls.CLIENT_ROOT, str(client), f"{ModelConfig.THRESHOLD_MODEL_BASENAME}{ModelConfig.MODEL_EXTENSION}")
        )

#? --- Dataset Configuration ---
class DatasetConfig:
    """
    Configuration for dataset paths, processing, and structure.
    """

    ROOT: str = join(getcwd(), "datasets")

    DATASET_NAME: str = "SWaT2015"
    DATASET_PATH: str = join(ROOT, DATASET_NAME)

    INPUT_NORMAL: str = join(DATASET_PATH, "original", "SWaT_Dataset_Normal.csv")
    INPUT_ATTACK: str = join(DATASET_PATH, "original", "SWaT_Dataset_Attack.csv")
    OUTPUT_NORMAL: str = join(DATASET_PATH, "processed", "SWaT_Dataset_Normal.hdf5")
    OUTPUT_ATTACK: str = join(DATASET_PATH, "processed", "SWaT_Dataset_Attack.hdf5")

    NORMALIZE: bool = False
    ROWS_PER_SAMPLE: int = 10
    SEED: int = 4

    TRAIN_SIZE: float = 0.7
    VAL_SIZE: float = (1.0 - TRAIN_SIZE) / 2.0
    TEST_SIZE: float = (1.0 - TRAIN_SIZE) / 2.0

    SENSORS: list[str] = [
        "FIT101", "LIT101", "MV101", "P101", "P102", "AIT201", "AIT202",
        "AIT203", "FIT201", "MV201", "P201", "P202", "P203", "P204", "P205",
        "P206", "DPIT301", "FIT301", "LIT301", "MV301", "MV302", "MV303",
        "MV304", "P301", "P302", "AIT401", "AIT402", "FIT401", "LIT401",
        "P401", "P402", "P403", "P404", "UV401", "AIT501", "AIT502", "AIT503",
        "AIT504", "FIT501", "FIT502", "FIT503", "FIT504", "P501", "P502",
        "PIT501", "PIT502", "PIT503", "FIT601", "P601", "P602", "P603"
    ]

    NORMAL_LABEL: int = 0
    ATTACK_LABEL: int = 1

#? --- Model Configuration ---
class ModelConfig:
    """
    Paths to saved models
    """
    
    MODEL_ROOT: str = join(DatasetConfig.DATASET_PATH, "models")
    MODEL_RUNTIME: str = join(MODEL_ROOT, f"{int(datetime.now().timestamp())}")
    MODEL_EXTENSION: str = ".keras"

    FINAL_MODEL_ROOT: str = join(getcwd(), "models")

    AUTOENCODER_MODEL_BASENAME: str = "autoencoder"
    AUTOENCODER_MODEL_ROOT: str = join(MODEL_RUNTIME, AUTOENCODER_MODEL_BASENAME)
    AUTOENCODER_MODEL: str = join(AUTOENCODER_MODEL_ROOT, f"{AUTOENCODER_MODEL_BASENAME}{MODEL_EXTENSION}")
    FINAL_AUTOENCODER_MODEL_ROOT: str = join(FINAL_MODEL_ROOT, AUTOENCODER_MODEL_BASENAME)
    FINAL_AUTOENCODER_MODEL: str = join(FINAL_AUTOENCODER_MODEL_ROOT, f"{AUTOENCODER_MODEL_BASENAME}{MODEL_EXTENSION}")

    THRESHOLD_MODEL_BASENAME: str = "threshold"
    THRESHOLD_MODEL_ROOT: str = join(MODEL_RUNTIME, THRESHOLD_MODEL_BASENAME)
    THRESHOLD_MODEL: str = join(THRESHOLD_MODEL_ROOT, f"{THRESHOLD_MODEL_BASENAME}{MODEL_EXTENSION}")
    FINAL_THRESHOLD_MODEL_ROOT: str = join(FINAL_MODEL_ROOT, THRESHOLD_MODEL_BASENAME)

    @classmethod
    def autoencoder_model(cls, accuracy: float = None) -> str:
        return join(cls.AUTOENCODER_MODEL_ROOT, f"{cls.AUTOENCODER_MODEL_BASENAME}{'_' + str(accuracy) if accuracy is not None else ''}{cls.MODEL_EXTENSION}")

    @classmethod
    def threshold_model(cls, client_id: str, accuracy: float = None) -> str:
        return join(cls.THRESHOLD_MODEL_ROOT, f"{cls.THRESHOLD_MODEL_BASENAME}-{client_id}{'_' + str(accuracy) if accuracy is not None else ''}{cls.MODEL_EXTENSION}")

#? --- Wandb Configuration ---
class WandbConfig:
    """
    Configuration for Weights & Biases logging.
    """

    ENTITY: str = os.getenv("ENTITY")
    PROJECT: str = os.getenv("PROJECT")
    
    @classmethod
    def init_run(cls, name: str):
        """
        Initializes a Weights & Biases run with predefined settings.
        """

        if WANDB:
            
            # Dynamic import
            import wandb

            # Init Wandb obj
            return wandb.init(
                entity=cls.ENTITY,
                project=cls.PROJECT,
                name=name
            )
        
        else:

            # Init dummy Wandb obj
            run = SimpleNamespace()
            run.log = lambda *args: None
            run.finish = lambda *args: None

            return run

#? --- Script Configuration ---
class RunType(Enum):

    NONE: int = -1
    ALL: int = 0
    AUTOENCODER: int = 1
    THRESHOLD: int = 2

RUN_TYPE = RunType.THRESHOLD

# Create folders
if RUN_TYPE in [RunType.ALL, RunType.AUTOENCODER, RunType.THRESHOLD]:

    if not exists(ModelConfig.MODEL_RUNTIME):
        makedirs(name=ModelConfig.MODEL_RUNTIME, exist_ok=True)

    if not exists(ModelConfig.AUTOENCODER_MODEL_ROOT):
        makedirs(name=ModelConfig.AUTOENCODER_MODEL_ROOT, exist_ok=True)

    if not exists(ModelConfig.THRESHOLD_MODEL_ROOT):
        makedirs(name=ModelConfig.THRESHOLD_MODEL_ROOT, exist_ok=True)