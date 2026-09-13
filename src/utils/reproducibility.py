"""
src/utils/reproducibility.py
Setzt alle relevanten Seeds für reproduzierbare Experimente.
"""
import functools
import os
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Setzt Python-, NumPy- und PyTorch-Seeds + cuDNN-Flags.

    deterministic=True  -> weitgehend reproduzierbar, aber langsamer (kein cuDNN-Autotuning,
                           kein TF32). NICHT bit-exakt: dafür fehlen bewusst
                           torch.use_deterministic_algorithms(True) und
                           CUBLAS_WORKSPACE_CONFIG — beide würden bei den hier
                           genutzten Backward-Kernels Laufzeitfehler auslösen.
                           Nicht-deterministische Kernel bleiben also aktiv.
    deterministic=False -> cudnn.benchmark + TF32 an: spürbar schneller bei fester
                           Bildgröße; Seeds wirken weiter, Läufe sind aber nicht mehr
                           bit-identisch (nur numerisches Rauschen, statistisch äquivalent).
    """
    # Python-Hash-Seed (Wörterbuch-Reihenfolge, set-Iteration)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA-Seeds (falls GPU verfügbar)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if not deterministic:
        # TF32 für schnellere fp32-Matmuls auf Ampere+/Blackwell (schadet mit AMP nicht)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"[Reproducibility] Seed gesetzt: {seed} (deterministic={deterministic})")


def _worker_init(worker_id: int, base_seed: int) -> None:
    """Seeded einen einzelnen DataLoader-Worker (Modulebene -> picklebar)."""
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_worker_init_fn(seed: int):
    """
    Gibt eine worker_init_fn zurück, die DataLoader-Worker seeded.

    Wichtig: Muss picklebar sein, weil Windows (spawn) / Python 3.14 (forkserver)
    die Funktion an die Worker-Prozesse serialisieren. Eine lokale Closure wäre das
    nicht -> daher functools.partial einer Funktion auf Modulebene.
    """
    return functools.partial(_worker_init, base_seed=seed)
