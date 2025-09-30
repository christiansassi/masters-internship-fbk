import os

os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_CONSOLE"] = "off"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import dotenv
dotenv.load_dotenv()

import torch
import platform
import psutil
import uuid

from types import SimpleNamespace

WIDE_DEEP_NETWORK: bool = False
THRESHOLD_NETWORK: bool = True
SIMULATION: bool = False

GPU: bool = True
WANDB: bool = True

VERBOSE: bool = True

import torch
torch.set_default_tensor_type("torch.FloatTensor")

if GPU:

    GPU = False

    if torch.cuda.is_available():
        GPU = True
        DEVICE = torch.device("cuda:0")
        hardware = f"{torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory}"

    else:
        DEVICE = torch.device("cpu")
        hardware = f"{platform.processor()} {psutil.virtual_memory()}"
else:
    DEVICE = torch.device("cpu")
    hardware = f"{platform.processor()} {psutil.virtual_memory()}"

hardware = str(uuid.uuid5(uuid.NAMESPACE_DNS, hardware))

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

#TODO add other hardware specs
BENCHMARKS = {
    "be0582c6-0ec0-504a-842e-b1ff65901906": 19794, # NVIDIA GeForce GTX 1660 Ti 6441992192
}

WIDE_DEEP_MAX_BATCH_SIZE = BENCHMARKS.get(hardware, float("inf"))