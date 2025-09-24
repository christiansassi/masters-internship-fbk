import config

import os
from os.path import exists
import shutil

import logging

from datetime import datetime

def clear_console():
    os.system("cls" if os.name == "nt" else "clear")

def clear_wandb_cache():
    if exists("wandb"):
        shutil.rmtree("wandb")

def configure_log(level: int = logging.INFO):

    logging.basicConfig(
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=level
    )
    
def log_timestamp_status() -> str:
    return f"[{datetime.now().strftime('%H:%M:%S')}][{logging.getLevelName(logging.getLogger().getEffectiveLevel())}]"

configure_log()