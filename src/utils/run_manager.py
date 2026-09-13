"""
src/utils/run_manager.py
Erstellt automatisch einen datierten Experiment-Ordner unter outputs/runs/.
"""
import os
import shutil
from datetime import datetime

import yaml


def create_run_dir(config: dict, run_name: str = None) -> str:
    """
    Erstellt die vollständige Experiment-Ordnerstruktur und gibt den Pfad zurück.

    Auto-Format: <timestamp>_<backbone>_lr<lr>_bs<bs>
    Überschreibbar per --run_name Argument.
    """
    if run_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        backbone  = config.get("backbone", "model")
        lr        = config.get("learning_rate", 0)
        bs        = config.get("batch_size", 0)
        run_name  = f"{timestamp}_{backbone}_lr{lr}_bs{bs}"

    run_dir = os.path.join("outputs", "runs", run_name)

    subdirs = [
        "checkpoints",
        "metrics",
        "preds",
        os.path.join("xai", "integrated_gradients"),
        os.path.join("xai", "rueckprojektion"),
    ]
    for sub in subdirs:
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    return run_dir


def save_config(config: dict, run_dir: str, config_path: str = None) -> None:
    """Kopiert die originale config.yaml in den Run-Ordner (oder schreibt dict)."""
    dst = os.path.join(run_dir, "config.yaml")
    if config_path and os.path.exists(config_path):
        shutil.copy(config_path, dst)
    else:
        with open(dst, "w") as f:
            yaml.dump(config, f, default_flow_style=False)