"""
src/training/trainer.py
Vollständige Trainingsschleife:
  - CrossEntropyLoss mit optionalen Klassengewichten
  - TensorBoard + CSV Logging
  - Checkpointing (best.pt / last.pt)
  - Early Stopping
  - Konfigurierbarer Optimizer + LR-Scheduler
"""
import csv
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.early_stopping import EarlyStopping


# ================================================================
#  Optimizer-Builder
# ================================================================
def build_optimizer(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    name = config.get("optimizer", "adam").lower()
    lr   = float(config.get("learning_rate", 1e-4))
    wd   = float(config.get("weight_decay",  1e-4))

    trainable = [p for p in model.parameters() if p.requires_grad]

    if name == "adam":
        return torch.optim.Adam(trainable, lr=lr, weight_decay=wd)
    elif name == "adamw":
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
    elif name == "sgd":
        mom = float(config.get("momentum", 0.9))
        return torch.optim.SGD(trainable, lr=lr, momentum=mom, weight_decay=wd)
    else:
        raise ValueError(f"Unbekannter Optimizer: '{name}'. Waehle adam | adamw | sgd.")


# ================================================================
#  Scheduler-Builder
# ================================================================
def build_scheduler(config: dict, optimizer: torch.optim.Optimizer):
    name   = config.get("scheduler", "cosine").lower()
    epochs = config.get("epochs", 30)

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("scheduler_step_size", 10),
            gamma=config.get("scheduler_gamma", 0.1),
        )
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=config.get("scheduler_patience", 5),
        )
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unbekannter Scheduler: '{name}'.")


# ================================================================
#  Trainer
# ================================================================
class Trainer:
    """
    Kapselt die gesamte Trainingslogik.

    Args:
        model         : Das zu trainierende nn.Module
        train_loader  : DataLoader für Trainingsdaten
        val_loader    : DataLoader für Validierungsdaten
        config        : Config-Dict aus YAML
        run_dir       : Pfad zum aktuellen Experiment-Ordner
        class_weights : Optionaler Tensor [w_io, w_nio] für CrossEntropyLoss
        logger        : Python-Logger-Instanz
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        run_dir: str,
        class_weights: Optional[torch.Tensor] = None,
        logger=None,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.run_dir      = run_dir
        self.logger       = logger

        self.device = torch.device("cpu")
        self.model.to(self.device)

        # ---- Loss ------------------------------------------------
        if config.get("use_class_weights", True) and class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # ---- Optimizer + Scheduler -------------------------------
        self.optimizer = build_optimizer(config, model)
        self.scheduler = build_scheduler(config, self.optimizer)

        # ---- Early Stopping --------------------------------------
        self.early_stopping: Optional[EarlyStopping] = None
        self.early_stopping_monitor = str(
            config.get("early_stopping_monitor", "val_loss")
        ).lower()
        if self.early_stopping_monitor not in {"val_loss", "val_acc"}:
            raise ValueError(
                "early_stopping_monitor muss 'val_loss' oder 'val_acc' sein."
            )
        if config.get("early_stopping", True):
            es_mode = "min" if self.early_stopping_monitor == "val_loss" else "max"
            self.early_stopping = EarlyStopping(
                patience=config.get("early_stopping_patience", 10),
                delta=config.get("early_stopping_delta",    0.001),
                mode=es_mode,
            )

        # ---- TensorBoard -----------------------------------------
        self.tb_writer = None
        if config.get("log_tensorboard", True):
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(run_dir, "tensorboard")
            self.tb_writer = SummaryWriter(log_dir=tb_dir)

        # ---- CSV-Logging -----------------------------------------
        self.train_csv = os.path.join(run_dir, "metrics", "train_metrics.csv")
        self.val_csv   = os.path.join(run_dir, "metrics", "val_metrics.csv")
        self._init_csv_files()

        # ---- State -----------------------------------------------
        self.best_val_acc = 0.0
        self.current_epoch = 0

    # ----------------------------------------------------------
    def _init_csv_files(self):
        for path in [self.train_csv, self.val_csv]:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "loss", "accuracy"])

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    # ----------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, train: bool):
        """Führt eine Epoche durch. Gibt (avg_loss, accuracy) zurück."""
        self.model.train(train)
        total_loss = 0.0
        correct    = 0
        total      = 0

        with torch.set_grad_enabled(train):
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss   = self.criterion(logits, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds       = logits.argmax(dim=1)
                correct    += (preds == labels).sum().item()
                total      += images.size(0)

        return total_loss / total, correct / total

    # ----------------------------------------------------------
    def _append_csv(self, path: str, epoch: int, loss: float, acc: float):
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{loss:.6f}", f"{acc:.6f}"])

    def _save_checkpoint(self, filename: str):
        path = os.path.join(self.run_dir, "checkpoints", filename)
        torch.save(
            {
                "epoch":              self.current_epoch,
                "model_state_dict":   self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_acc":       self.best_val_acc,
                "config":             self.config,
            },
            path,
        )

    # ----------------------------------------------------------
    def train(self):
        """Haupttrainingsschleife."""
        epochs = self.config.get("epochs", 30)

        self._log("=" * 60)
        self._log(f"Starte Training: {epochs} Epochen | Device: {self.device}")
        if self.early_stopping:
            self._log(
                "Early Stopping aktiv: "
                f"monitor={self.early_stopping_monitor}, "
                f"patience={self.early_stopping.patience}, "
                f"delta={self.early_stopping.delta}"
            )
        self._log("=" * 60)

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch

            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss,   val_acc   = self._run_epoch(self.val_loader,   train=False)

            # Scheduler-Step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            self._log(
                f"Epoch {epoch:03d}/{epochs}  |  "
                f"Train  loss={train_loss:.4f}  acc={train_acc:.4f}  |  "
                f"Val    loss={val_loss:.4f}  acc={val_acc:.4f}  |  "
                f"LR={current_lr:.2e}"
            )

            # CSV
            self._append_csv(self.train_csv, epoch, train_loss, train_acc)
            self._append_csv(self.val_csv,   epoch, val_loss,   val_acc)

            # TensorBoard
            if self.tb_writer:
                self.tb_writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
                self.tb_writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)
                self.tb_writer.add_scalar("LR", current_lr, epoch)

            # Checkpointing
            self._save_checkpoint("last.pt")
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint("best.pt")
                self._log(f"  ★  Neues Best-Modell gespeichert (val_acc={val_acc:.4f})")

            # Early Stopping
            if self.early_stopping:
                monitor_value = (
                    val_loss if self.early_stopping_monitor == "val_loss" else val_acc
                )
                if self.early_stopping.step(monitor_value):
                    self._log(
                        f"Early Stopping ausgeloest nach Epoche {epoch} "
                        f"({self.early_stopping_monitor}={monitor_value:.4f})."
                    )
                    break

        if self.tb_writer:
            self.tb_writer.close()

        self._log("=" * 60)
        self._log(f"Training beendet. Beste Val-Accuracy: {self.best_val_acc:.4f}")
        self._log("=" * 60)



