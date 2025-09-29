import config

import os
from os.path import exists
import shutil

import psutil

import logging

from datetime import datetime

import torch

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

def get_safe_batch_size(model, sample, target_frac=0.75, n_runs=50, grad_factor=2.0):

    if config.GPU:
        device = "cuda:0"

        model = model.to(device)
        sample = torch.from_numpy(sample).float().to(device)

        # Get total GPU memory
        free, total = torch.cuda.mem_get_info(device)
        budget = int(total * target_frac)

        # Measure per-sample memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        with torch.no_grad():
            _ = model(sample)

        mem_per_sample = torch.cuda.max_memory_allocated(device)

    else:
        device = "cpu"
        
        model = model.to(device)
        sample = torch.from_numpy(sample).float().to(device)

        proc = psutil.Process(os.getpid())
        total = psutil.virtual_memory().total
        budget = int(total * target_frac)

        deltas = []

        for _ in range(n_runs):
            before = proc.memory_info().rss

            with torch.no_grad():
                _ = model(sample)

            after = proc.memory_info().rss

            delta = after - before

            if delta > 0:
                deltas.append(delta)

        mem_per_sample = max(deltas)

    # Account for backward pass (gradients roughly double usage)
    mem_per_sample = int(mem_per_sample * grad_factor)

    # Convert to MB for reporting
    mem_per_sample_mb = mem_per_sample / 1024**2
    budget_mb = budget / 1024**2

    # Safe batch size
    safe_batch = max(1, budget // mem_per_sample)

    return safe_batch


configure_log()