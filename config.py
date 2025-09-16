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

WIDE_DEEP_NETWORK: bool = False
THRESHOLD_NETWORK: bool = False

TRAIN_VERBOSE: int = 1
EVAL_VERBOSE: int = 1
PREDICT_VERBOSE: int = 1

GPU: bool = False
WANDB: bool = True

if GPU:

    GPU = False

    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
        GPU = True

else:
    tf.config.set_visible_devices([], "GPU")

if WANDB:        
    import wandb

class WandbConfig:

    ENTITY: str = os.getenv("ENTITY")
    PROJECT: str = os.getenv("PROJECT")
    
    @classmethod
    def init_run(cls, name: str):

        if WANDB:
            
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
    
    @classmethod
    def table(cls, *args, **kwargs):

        if WANDB:
            return wandb.Table(*args, **kwargs)

        else:
            return None
    
    @classmethod
    def plot_bar(cls, *args, **kwargs):

        if WANDB:
            return wandb.plot.bar(*args, **kwargs)

        else:
            return None
    
    @staticmethod
    def safe_log(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                pass

        return wrapper

    @staticmethod
    def safe_finish(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                pass

        return wrapper