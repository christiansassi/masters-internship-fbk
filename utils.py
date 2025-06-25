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
    :type log_level: `Literal`
    """

    _log_levels = [0, 10, 20, 30, 40, 50]

    assert log_level in _log_levels, f"Invalid log level. Accepted values are: {_log_levels}."

    logging.basicConfig(
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        log_level=log_level
    )
    
def log_timestamp_status() -> str:
    """
    Returns a formatted string with current timestamp and log level.
    
    :return: The formatted string.
    :rtype: `str`
    """

    return f"[{datetime.now().strftime('%H:%M:%S')}][{logging.getLevelName(logging.getLogger().getEffectiveLevel())}]"

def dynamic_round(value: int | float, reference_value: int | float) -> float:
    """
    Dynamically rounds a numerical `value` based on its relationship and
    the decimal precision of a `reference_value`.

    - If `value` and `reference_value` have different signs, `value` is returned as is.
    - If `value` is a float and `reference_value` is an integer, `value` is truncated
      to 4 decimal places (plus the first digit after the decimal if it's not part of the common prefix).
    - If `value` is an integer and `reference_value` is a float, `value` is returned as is.
    - If both `value` and `reference_value` have decimal parts, and their string
      representations (excluding signs) are identical, `value` is truncated to
      its decimal part, keeping leading zeros plus an additional 4 digits.
    - Otherwise (both are floats or both are integers but not identical in string representation),
      `value` is truncated to a length determined by the common prefix of their
      decimal parts plus two additional digits.

    :param value: The number to be dynamically rounded.
    :type value: `int | float`

    :param reference_value: The reference number used to determine the rounding behavior.
    :type reference_value: `int | float`

    :returns: The dynamically rounded or truncated float value.
    :rtype: `float`
    """

    if (value < 0 and reference_value > 0) or (value > 0 and reference_value < 0):
        return value

    n1 = str(abs(value))
    n2 = str(abs(reference_value))

    if "." in n1 and "." not in n2:
        value = str(value)
        return float(value[:value.find(".") + 4 + 1])
    
    elif "." not in n1 and "." in n2:
        return value

    elif n1 == n2:
        
        digits = 0

        n = n1.split(".")[-1]

        for c in n:
            
            if c != "0":
                break

            digits = digits + 1
        
        digits = digits + 4

        value = str(value)
        return float(value[:value.find(".") + digits + 1])

    else:
        n1 = n1.split(".")[-1]
        n2 = n2.split(".")[-1]

        result = []

        for a, b in zip(n1, n2):
            if a == b:
                result.append(a)
            else:
                break


        value = str(value)
        return float(value[:value.find(".") + len(''.join(result)) + 2])