import os

os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_CONSOLE"] = "off"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import dotenv
dotenv.load_dotenv()

from types import SimpleNamespace

WIDE_DEEP_NETWORK: bool = True
THRESHOLD_NETWORK: bool = False
SIMULATION: bool = False

GPU: bool = True
WANDB: bool = False

import torch
torch.set_default_tensor_type("torch.FloatTensor")

if GPU:

    GPU = False

    if torch.cuda.is_available():
        GPU = True
        DEVICE = torch.device("cuda:0")

    else:
        DEVICE = torch.device("cpu")

else:
    DEVICE = torch.device("cpu")

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