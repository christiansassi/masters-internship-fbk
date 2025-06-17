import os
from os.path import exists
import shutil

import logging

from datetime import datetime

def clear_console():
    """
    Clears the console screen.
    """

    os.system("cls" if os.name == "nt" else "clear")

def clear_wandb_cache():
    """
    Removes the local Weights & Biases cache directory.
    """

    if exists("wandb"):
        shutil.rmtree("wandb")

def configure_log(log_level: int = logging.INFO):
    """
    Configures the basic logging settings.
    
    Sets the log message format, time format, and minimum logging level.

    :param log_level: The logging level. Defaults to `logging.INFO`.
    :type log_level: Literal
    """

    _log_levels = [0, 10, 20, 30, 40, 50]

    assert log_level in _log_levels, f"Invalid log level. Accepted values are: {_log_levels}."

    logging.basicConfig(
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        log_level=log_level
    )
    
def log_timestamp_status():
    """
    Returns a formatted string with current timestamp and log level.
    """

    return f"[{datetime.now().strftime('%H:%M:%S')}][{logging.getLevelName(logging.getLogger().getEffectiveLevel())}]"