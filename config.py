import os
from os import makedirs
from os.path import join, basename

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

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

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
    def init_run(cls, name: str, tags: list[str] = []):

        if WANDB:
            
            return wandb.init(
                entity=cls.ENTITY,
                project=cls.PROJECT,
                name=name,
                tags=tags
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
    "be0582c6-0ec0-504a-842e-b1ff65901906": 19776, # NVIDIA GeForce GTX 1660 Ti 6441992192
}

WIDE_DEEP_MAX_BATCH_SIZE = BENCHMARKS.get(hardware, float("inf"))

import random
import numpy as np

# Fix seeds
# source: https://github.com/pytorch/pytorch/issues/7068
def seed_torch(seed=1000):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

import sys
import traceback
from datetime import datetime

ROOT_LOGS = "logs"
SESSION_LOGS = join(ROOT_LOGS, f"{'.py'.join(basename(sys.modules['__main__'].__file__).split('.py')[:-1])}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

makedirs(ROOT_LOGS, exist_ok=True)

_console = sys.__stdout__
_log = open(SESSION_LOGS, "w+")

sys.stdout = _log

def printplus(msg: str, end: str = "\n", log_only: bool = False):

    if VERBOSE != 1:
        return

    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end)

    if log_only == False:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end, file=_console)

def exception_hook(exc_type, exc_value, exc_traceback):

    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))

    _log.write(tb)
    _log.flush()

    _console.write(tb)
    _console.flush()

    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = exception_hook

os.system("cls" if os.name == "nt" else "clear")