import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_CONSOLE"] = "off"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf

import dotenv
dotenv.load_dotenv()

from types import SimpleNamespace

VERBOSE: int = 1
GPU: bool = True
WANDB: bool = True

if GPU:
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    tf.config.set_visible_devices([], "GPU")

class WandbConfig:

    ENTITY: str = os.getenv("ENTITY")
    PROJECT: str = os.getenv("PROJECT")
    
    @classmethod
    def init_run(cls, name: str):

        if WANDB:
            
            import wandb

            return wandb.init(
                entity=cls.ENTITY,
                project=cls.PROJECT,
                name=name
            )
        
        else:

            run = SimpleNamespace()
            run.log = lambda *args: None
            run.finish = lambda *args: None

            return run