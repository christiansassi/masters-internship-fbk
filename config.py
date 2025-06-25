import os

#? --- Environment Setup ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_CONSOLE"] = "off"

from os.path import join, exists
from os import makedirs

import dotenv

import tensorflow as tf

from datetime import datetime

# Disable FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load Environment Variables
dotenv.load_dotenv()

#? --- Global Script Settings ---
USE_GPU: bool = True
WANDB: bool = True
VERBOSE: int = 0
N_CLIENTS: int = os.cpu_count() - 1

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


#? --- Dataset Configuration ---
class DatasetConfig:
    """
    Configuration for dataset paths, processing, and structure.
    """

    ROOT: str = "datasets"
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

#? --- Model Configuration ---
class ModelConfig:
    """
    Paths to saved models
    """
    
    _MODEL_ROOT: str = join(DatasetConfig.DATASET_PATH, "models")
    _MODEL_RUNTIME: str = join(_MODEL_ROOT, f"{int(datetime.now().timestamp())}")
    _MODEL_EXTENSION: str = ".keras"

    _AUTOENCODER_MODEL_ROOT: str = join(_MODEL_RUNTIME, "autoencoder")
    _AUTOENCODER_MODEL_BASENAME: str = "autoencoder"
    _AUTOENCODER_MODEL: str = join(_AUTOENCODER_MODEL_ROOT, f"{_AUTOENCODER_MODEL_BASENAME}{_MODEL_EXTENSION}") # Autoencoder model

    _THRESHOLD_MODEL_ROOT: str = join(_MODEL_RUNTIME, "threshold")
    _THRESHOLD_MODEL_BASENAME: str = "threshold"
    THRESHOLD_MODEL: str = join(_THRESHOLD_MODEL_ROOT, f"{_THRESHOLD_MODEL_BASENAME}{_MODEL_EXTENSION}") # Threshold model

    @classmethod
    def autoencoder_model(cls, accuracy: float = None) -> str:
        return join(cls._AUTOENCODER_MODEL_ROOT, f"{cls._AUTOENCODER_MODEL_BASENAME}-{str(accuracy)}{cls._MODEL_EXTENSION}")

    @classmethod
    def threshold_model(cls, accuracy: float = None) -> str:
        return join(cls._THRESHOLD_MODEL_ROOT, f"{cls._THRESHOLD_MODEL_BASENAME}-{str(accuracy)}{cls._MODEL_EXTENSION}")

# Create folders
if not exists(ModelConfig._MODEL_RUNTIME):
    makedirs(name=ModelConfig._MODEL_RUNTIME, exist_ok=True)

if not exists(ModelConfig._AUTOENCODER_MODEL_ROOT):
    makedirs(name=ModelConfig._AUTOENCODER_MODEL_ROOT, exist_ok=True)

if not exists(ModelConfig._THRESHOLD_MODEL_ROOT):
    makedirs(name=ModelConfig._THRESHOLD_MODEL_ROOT, exist_ok=True)

#? --- Wandb Configuration ---
if WANDB:
    import wandb

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

            return wandb.init(
                entity=cls.ENTITY,
                project=cls.PROJECT,
                name=name
            )