"""
src/training/trainer.py
Vollständige Trainingsschleife:
  - CrossEntropyLoss mit optionalen Klassengewichten + Label Smoothing
  - Mixed Precision Training (AMP, nur CUDA)
  - Gradient Clipping
  - TensorBoard + CSV Logging
  - Checkpointing (best.pt / last.pt) nach konfigurierbarer Metrik
  - Early Stopping (gleiche Metrik wie Checkpointing)
  - Konfigurierbarer Optimizer + LR-Scheduler
"""
import csv
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from sklearn.metrics import f1_score as sklearn_f1
from torch.utils.data import DataLoader

from src.models.resnet import head_param_prefixes
from src.training.early_stopping import EarlyStopping


# ================================================================
#  Optimizer-Builder
# ================================================================
def _ohne_weight_decay(pname: str, param) -> bool:
    """Parameter, die uebliche Rezepte von der Weight Decay ausnehmen.

    Betrifft alle eindimensionalen Parameter (Biases sowie die Gains von
    BatchNorm/LayerNorm) und die LayerScale-Faktoren von ConvNeXt. Sie zu
    schrumpfen hat keine regularisierende Wirkung, verschiebt aber die
    Normierung bzw. daempft bei ConvNeXt die Residualzweige – bei wd = 0,05,
    wie ConvNeXt es braucht, ist der Effekt erheblich. Bei wd = 1e-4 (ResNet-
    Rezept) faellt es kaum ins Gewicht, deshalb ist die Ausnahme abschaltbar.
    """
    return param.ndim <= 1 or "layer_scale" in pname


def build_optimizer(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    """Baut den Optimizer mit getrennten Parametergruppen fuer Kopf und Backbone.

    backbone_learning_rate: null -> beide Gruppen nutzen learning_rate (Verhalten
    wie vor der Umstellung). Ein kleinerer Wert (z.B. 1e-5 gegen 1e-4 am Kopf)
    stabilisiert das Fine-Tuning des vortrainierten Backbones deutlich.

    no_weight_decay_on_norm_bias: true -> Norm-Gains, Biases und LayerScale
    bekommen weight_decay = 0. Standard ist false (bisheriges Verhalten).
    """
    name = config.get("optimizer", "adam").lower()
    lr   = float(config.get("learning_rate", 1e-4))
    wd   = float(config.get("weight_decay",  1e-4))
    wd_ausnahme = bool(config.get("no_weight_decay_on_norm_bias", False))

    backbone_lr_cfg = config.get("backbone_learning_rate", None)
    backbone_lr     = lr if backbone_lr_cfg is None else float(backbone_lr_cfg)

    head_prefixes = head_param_prefixes(config.get("backbone", "resnet18"))
    # Vier Toepfe: Kopf/Backbone x mit/ohne Weight Decay. Ohne die Ausnahme
    # bleiben die "_nodecay"-Toepfe leer und es entstehen wie bisher zwei Gruppen.
    toepfe = {"head": [], "head_nodecay": [], "backbone": [], "backbone_nodecay": []}
    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        basis = "head" if any(pname.startswith(p) for p in head_prefixes) else "backbone"
        if wd_ausnahme and _ohne_weight_decay(pname, param):
            basis += "_nodecay"
        toepfe[basis].append(param)

    groups = []
    for schluessel, params in toepfe.items():
        if not params:
            continue
        groups.append({
            "params": params,
            "lr": lr if schluessel.startswith("head") else backbone_lr,
            "weight_decay": 0.0 if schluessel.endswith("_nodecay") else wd,
            "name": schluessel,
        })

    if not groups:
        raise ValueError(
            "Keine trainierbaren Parameter gefunden – pruefe 'freeze_backbone'."
        )

    if name == "adam":
        return torch.optim.Adam(groups, lr=lr, weight_decay=wd)
    elif name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
    elif name == "sgd":
        mom = float(config.get("momentum", 0.9))
        return torch.optim.SGD(groups, lr=lr, momentum=mom, weight_decay=wd)
    else:
        raise ValueError(f"Unbekannter Optimizer: '{name}'. Waehle adam | adamw | sgd.")


# ================================================================
#  Scheduler-Builder
# ================================================================
def build_scheduler(config: dict, optimizer: torch.optim.Optimizer, monitor_mode: str = "max"):
    """Baut den LR-Scheduler.

    monitor_mode ("min"/"max") stammt aus early_stopping_monitor und steuert die
    Richtung von ReduceLROnPlateau – frueher war der Modus hart auf "max"
    verdrahtet und wurde immer mit der Accuracy gesteppt, was bei
    early_stopping_monitor: val_loss genau falsch herum war.
    """
    name    = config.get("scheduler", "cosine").lower()
    epochs  = config.get("epochs", 30)
    warmup  = int(config.get("warmup_epochs", 0))

    if name == "plateau":
        if warmup > 0:
            raise ValueError(
                "warmup_epochs laesst sich nicht mit scheduler: plateau kombinieren – "
                "ReduceLROnPlateau wird auf eine Metrik gesteppt, nicht auf die Epoche."
            )
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=monitor_mode,
            patience=config.get("scheduler_patience", 5),
        )

    # Nach dem Warmup bleiben epochs - warmup Epochen fuer den Hauptverlauf.
    rest = max(1, epochs - warmup)

    if name == "cosine":
        # T_max = volle geplante Trainingsdauer. Frueher min(epochs, patience*3):
        # bei epochs=50/patience=10 war T_max=30, wodurch die Lernrate ab Epoche 31
        # wieder anstieg statt auszulaufen.
        haupt = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=rest)
    elif name == "step":
        haupt = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("scheduler_step_size", 10),
            gamma=config.get("scheduler_gamma", 0.1),
        )
    elif name == "none":
        haupt = None
    else:
        raise ValueError(f"Unbekannter Scheduler: '{name}'.")

    if warmup <= 0:
        return haupt

    # Lineares Warmup ab 1 % der Ziel-Lernrate. Ohne Warmup macht AdamW auf
    # vortrainierten Gewichten gleich in der ersten Epoche grosse Schritte,
    # bevor die Momentschaetzungen eingeschwungen sind – bei ConvNeXt (LayerNorm,
    # LayerScale) zerstoert das die vortrainierten Merkmale sichtbar.
    aufwaermen = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup
    )
    if haupt is None:
        return aufwaermen
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[aufwaermen, haupt], milestones=[warmup]
    )


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

        # ---- Device (GPU wenn verfügbar) -------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # ---- Mixed Precision (AMP) – nur CUDA --------------------
        self.use_amp = (self.device.type == "cuda") and config.get("use_amp", True)
        self.scaler  = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # ---- Gradient Clipping -----------------------------------
        self.grad_clip = float(config.get("grad_clip_max_norm", 1.0))

        # ---- Loss ------------------------------------------------
        label_smoothing = float(config.get("label_smoothing", 0.0))
        if config.get("use_class_weights", True) and class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(self.device),
                label_smoothing=label_smoothing,
            )
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # ---- Checkpoint-Metrik (Early Stopping + best.pt) --------
        # Muss VOR dem Scheduler stehen: ReduceLROnPlateau braucht die Richtung.
        # Die Validierung bleibt bewusst vor _init_csv_files() – sonst wuerde ein
        # ungueltiger Monitor erst nach dem Anlegen der CSV-Dateien auffallen.
        self.checkpoint_monitor = str(
            config.get("early_stopping_monitor", "val_f1")
        ).lower()
        valid_monitors = {"val_loss", "val_acc", "val_f1"}
        if self.checkpoint_monitor not in valid_monitors:
            raise ValueError(
                f"early_stopping_monitor muss einer von {valid_monitors} sein."
            )
        self.checkpoint_mode = "min" if self.checkpoint_monitor == "val_loss" else "max"
        self.best_val_metric = float("inf") if self.checkpoint_mode == "min" else 0.0
        self.best_epoch: Optional[int] = None

        # best.pt nutzt dieselbe Delta-Schwelle wie Early Stopping. Vorher zaehlte
        # jede noch so kleine Verbesserung als "neues Bestmodell", waehrend Early
        # Stopping sie schon als Stillstand wertete – so konservierte best.pt
        # Ausreisser einer verrauschten Val-Kurve.
        self.checkpoint_delta = float(config.get("early_stopping_delta", 0.001))

        # ---- BatchNorm-Statistiken einfrieren --------------------
        self.freeze_bn_stats = bool(config.get("freeze_bn_stats", True))

        # ---- Optimizer + Scheduler -------------------------------
        self.optimizer = build_optimizer(config, model)
        self.scheduler = build_scheduler(config, self.optimizer, self.checkpoint_mode)

        # Clipping ueber genau die Parameter, die der Optimizer auch aktualisiert
        self.clip_params = [
            p for group in self.optimizer.param_groups for p in group["params"]
        ]

        # ---- Early Stopping --------------------------------------
        self.early_stopping: Optional[EarlyStopping] = None
        if config.get("early_stopping", True):
            self.early_stopping = EarlyStopping(
                patience=config.get("early_stopping_patience", 10),
                delta=self.checkpoint_delta,
                mode=self.checkpoint_mode,
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
        self.current_epoch = 0

    # ----------------------------------------------------------
    def _init_csv_files(self):
        for path in [self.train_csv, self.val_csv]:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "loss", "accuracy", "f1"])

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    # ----------------------------------------------------------
    def _set_frozen_bn_to_eval(self) -> int:
        """Eingefrorene BatchNorm-Schichten in den eval-Modus zwingen.

        requires_grad=False friert nur weight/bias ein. running_mean und
        running_var sind dagegen Buffer und liefen im train-Modus weiter mit –
        die ImageNet-Statistiken des eingefrorenen Backbones drifteten also bei
        jeder Epoche, obwohl die Schichten als "eingefroren" galten. Die
        Validierung laeuft in eval() mit genau diesen driftenden Statistiken,
        was starke Spruenge in der Val-Kurve erzeugt.

        Eine BN-Schicht gilt als eingefroren, wenn sie eigene Parameter hat und
        keiner davon trainiert wird. Der BatchNorm1d im Klassifikationskopf
        (src/models/resnet.py, _build_head) ist trainierbar und bleibt deshalb
        bewusst im train-Modus.

        Gibt die Anzahl der umgeschalteten Schichten zurueck.
        """
        n_frozen = 0
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                params = list(module.parameters(recurse=False))
                if params and not any(p.requires_grad for p in params):
                    module.eval()
                    n_frozen += 1
        return n_frozen

    # ----------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, train: bool):
        """
        Führt eine Epoche durch.
        Gibt (avg_loss, accuracy, f1_nio) zurück.
        f1_nio ist F1-Score der NIO-Klasse (pos_label=1).
        """
        self.model.train(train)
        # Muss NACH train(train) stehen, sonst wird das eval() wieder ueberschrieben.
        if train and self.freeze_bn_stats:
            self._set_frozen_bn_to_eval()
        total_loss = 0.0
        all_preds:  list = []
        all_labels: list = []

        with torch.set_grad_enabled(train):
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    logits = self.model(images)
                    loss   = self.criterion(logits, labels)

                if train:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    # Gradient Clipping (falls aktiviert)
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.clip_params, self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item() * images.size(0)
                preds       = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        n        = len(all_labels)
        avg_loss = total_loss / n
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / n
        f1       = float(sklearn_f1(all_labels, all_preds, pos_label=1, zero_division=0))

        return avg_loss, accuracy, f1

    # ----------------------------------------------------------
    def _append_csv(self, path: str, epoch: int, loss: float, acc: float, f1: float):
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{loss:.6f}", f"{acc:.6f}", f"{f1:.6f}"])

    def _save_checkpoint(self, filename: str, versuche: int = 5, wartezeit: float = 3.0):
        """Checkpoint schreiben, mit Wiederholung bei gesperrter Datei.

        Das Projekt liegt in einem OneDrive-Ordner. OneDrive oeffnet eine
        gerade geschriebene Datei zum Hochladen und haelt sie dabei kurz
        exklusiv; faellt das mit dem naechsten torch.save zusammen, wirft
        Windows Fehlercode 32 (Sharing Violation). Ohne Wiederholung reisst
        das ein laufendes Training nach Stunden ab - beobachtet am 19.08.2026
        in Epoche 9 des Laufs S2_01, bei last.pt (137 MB).

        Fuenf Versuche mit 3 s Abstand ueberbruecken den Upload; klappt es
        danach immer noch nicht, fliegt der Fehler weiter - ein still
        uebersprungener Checkpoint waere schlimmer als ein Abbruch.
        """
        path = os.path.join(self.run_dir, "checkpoints", filename)
        zustand = {
            "epoch":                self.current_epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_metric":      self.best_val_metric,
            "checkpoint_monitor":   self.checkpoint_monitor,
            "config":               self.config,
        }
        for versuch in range(1, versuche + 1):
            try:
                torch.save(zustand, path)
                return
            except (OSError, RuntimeError) as fehler:
                if versuch == versuche:
                    raise
                self._log(f"  [!] {filename} nicht schreibbar (Versuch "
                          f"{versuch}/{versuche}): {fehler}. Neuer Versuch in "
                          f"{wartezeit:.0f} s - vermutlich haelt OneDrive die Datei.")
                time.sleep(wartezeit)

    # ----------------------------------------------------------
    def train(self):
        """Haupttrainingsschleife."""
        epochs = self.config.get("epochs", 30)

        self._log("=" * 60)
        self._log(
            f"Starte Training: {epochs} Epochen | Device: {self.device} | "
            f"AMP: {self.use_amp} | Grad-Clip: {self.grad_clip}"
        )
        if self.early_stopping:
            self._log(
                "Early Stopping aktiv: "
                f"monitor={self.checkpoint_monitor}, "
                f"patience={self.early_stopping.patience}, "
                f"delta={self.early_stopping.delta}"
            )

        # Lernraten pro Parametergruppe belegen
        group_info = "  ".join(
            f"{g.get('name', f'group{i}')}={g['lr']:.2e} "
            f"({sum(p.numel() for p in g['params']):,} Params)"
            for i, g in enumerate(self.optimizer.param_groups)
        )
        self._log(f"Optimizer-Gruppen: {group_info}")

        # Nachweis, dass der BatchNorm-Freeze greift
        if self.freeze_bn_stats:
            n_bn_total = sum(
                1 for m in self.model.modules()
                if isinstance(m, nn.modules.batchnorm._BatchNorm)
            )
            n_bn_frozen = self._set_frozen_bn_to_eval()
            self._log(
                f"BatchNorm-Freeze aktiv: {n_bn_frozen} von {n_bn_total} BN-Schichten "
                f"bleiben auf eval (running_mean/var eingefroren)"
            )
        else:
            self._log(
                "BatchNorm-Freeze DEAKTIVIERT (freeze_bn_stats: false) – "
                "running_mean/var des Backbones laufen im Training weiter mit."
            )
        self._log("=" * 60)

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch

            train_loss, train_acc, train_f1 = self._run_epoch(self.train_loader, train=True)
            val_loss,   val_acc,   val_f1   = self._run_epoch(self.val_loader,   train=False)

            # Checkpoint-Metrik auswählen (wird auch vom Plateau-Scheduler gebraucht)
            monitor_values = {
                "val_loss": val_loss,
                "val_acc":  val_acc,
                "val_f1":   val_f1,
            }
            monitor_value = monitor_values[self.checkpoint_monitor]

            # Scheduler-Step – Plateau reagiert auf DIE ueberwachte Metrik,
            # nicht mehr fest auf die Accuracy.
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(monitor_value)
                else:
                    self.scheduler.step()

            lrs = {
                group.get("name", f"group{i}"): group["lr"]
                for i, group in enumerate(self.optimizer.param_groups)
            }
            lr_str = "  ".join(f"{name}={lr:.2e}" for name, lr in lrs.items())

            self._log(
                f"Epoch {epoch:03d}/{epochs}  |  "
                f"Train  loss={train_loss:.4f}  acc={train_acc:.4f}  f1={train_f1:.4f}  |  "
                f"Val    loss={val_loss:.4f}  acc={val_acc:.4f}  f1={val_f1:.4f}  |  "
                f"LR {lr_str}"
            )

            # CSV
            self._append_csv(self.train_csv, epoch, train_loss, train_acc, train_f1)
            self._append_csv(self.val_csv,   epoch, val_loss,   val_acc,   val_f1)

            # TensorBoard
            if self.tb_writer:
                self.tb_writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
                self.tb_writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)
                self.tb_writer.add_scalars("F1_NIO",   {"train": train_f1,   "val": val_f1},   epoch)
                self.tb_writer.add_scalars("LR", lrs, epoch)

            # Neues Bestmodell nur bei einer Verbesserung um mehr als delta –
            # dieselbe Schwelle, die auch Early Stopping anlegt.
            if self.best_epoch is None:
                is_best = True
            elif self.checkpoint_mode == "max":
                is_best = monitor_value > self.best_val_metric + self.checkpoint_delta
            else:
                is_best = monitor_value < self.best_val_metric - self.checkpoint_delta

            self._save_checkpoint("last.pt")
            if is_best:
                self.best_val_metric = monitor_value
                self.best_epoch      = epoch
                self._save_checkpoint("best.pt")
                self._log(
                    f"  ★  Neues Best-Modell ({self.checkpoint_monitor}={monitor_value:.4f})"
                )

            # Early Stopping
            if self.early_stopping:
                if self.early_stopping.step(monitor_value):
                    self._log(
                        f"Early Stopping ausgeloest nach Epoche {epoch} "
                        f"({self.checkpoint_monitor}={monitor_value:.4f})."
                    )
                    break

        if self.tb_writer:
            self.tb_writer.close()

        self._log("=" * 60)
        self._log(
            f"Training beendet. Bestes {self.checkpoint_monitor}: {self.best_val_metric:.4f}"
        )
        self._log("=" * 60)
