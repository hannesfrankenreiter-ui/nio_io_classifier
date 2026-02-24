"""
src/utils/reproducibility.py
Setzt alle relevanten Seeds für reproduzierbare Experimente.
"""
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Setzt Python-, NumPy- und PyTorch-Seeds + deterministische Flags."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Für CPU-Training hat cudnn keinen Effekt, schadet aber nicht:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Reproducibility] Seed gesetzt: {seed}")