"""
src/evaluation/evaluator.py
Berechnet und speichert Metriken für einen Split (test oder val):
  - Accuracy, Precision, Recall, F1 (binär, pos_label=1 → "nio")
  - ROC-AUC + Average Precision (threshold-unabhängig)
  - Rechenzeit je Bild + trainierbare Parameter (fuer den Backbone-Vergleich)
  - Confusion Matrix als PNG
  - ROC-Kurve als PNG
  - Predictions als CSV

Die Klassenentscheidung nutzt einen konfigurierbaren Schwellenwert
(prob_nio >= threshold → nio) statt argmax. Beides ist bei threshold=0.5
identisch, aber nur so laesst sich ein auf val bestimmter Betriebspunkt
unveraendert auf test anwenden. Die Semantik entspricht exakt
src/inference/predict.py, damit Evaluation und Inferenz nicht auseinanderlaufen.

HINWEIS: pos_label=1 entspricht der Klasse "nio" (Index 1).
         Das ist die industriell relevante Fehlerklasse.
"""
import csv
import json
import os
import time
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.models.resnet import count_trainable_params
from src.utils.plot_style import BLUE, GRID, MUT, curve_style, style_axes

matplotlib.use("Agg")  # Kein Display nötig

CLASSES = ["io", "nio"]

# Beschriftung der Plots je Split
_SPLIT_LABEL = {"test": "Testset", "val": "Validierungsset"}


def find_best_threshold(
    labels, nio_probs, default: float = 0.5
) -> Tuple[float, float]:
    """Sucht den F1-optimalen Schwellenwert für die NIO-Klasse.

    Wird auf dem Validierungsset aufgerufen; der gefundene Wert wird danach
    unveraendert auf das Testset angewendet und dort NICHT nachoptimiert.

    Gibt (threshold, erreichter_f1) zurueck. Faellt auf `default` zurueck, wenn
    keine sinnvolle Suche moeglich ist (nur eine Klasse vorhanden oder leeres
    Threshold-Array).
    """
    labels    = np.asarray(labels)
    nio_probs = np.asarray(nio_probs)

    if len(np.unique(labels)) < 2:
        return float(default), float("nan")

    precision, recall, thresholds = precision_recall_curve(
        labels, nio_probs, pos_label=1
    )
    if thresholds.size == 0:
        return float(default), float("nan")

    # precision/recall haben genau ein Element mehr als thresholds
    prec  = precision[:-1]
    rec   = recall[:-1]
    denom = prec + rec
    f1    = np.where(denom > 0, 2 * prec * rec / np.maximum(denom, 1e-12), 0.0)

    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


class Evaluator:
    """
    Evaluiert ein Modell auf einem DataLoader und speichert alle Ergebnisse.

    Args:
        model      : Trainiertes nn.Module (bereits auf device geladen)
        loader     : DataLoader des zu evaluierenden Splits
        run_dir    : Pfad zum Experiment-Ordner
        device     : torch.device
        logger     : optionaler Python-Logger
        threshold  : Entscheidungsschwelle (prob_nio >= threshold → nio)
        split_name : "test" oder "val". Steuert Dateinamen und Plot-Titel.
                     Bei "test" bleiben die historischen Dateinamen erhalten,
                     val-Artefakte bekommen ein "_val"-Suffix.
    """

    def __init__(
        self,
        model: nn.Module,
        loader: DataLoader,
        run_dir: str,
        device: torch.device,
        logger=None,
        threshold: float = 0.5,
        split_name: str = "test",
    ):
        self.model      = model
        self.loader     = loader
        self.run_dir    = run_dir
        self.device     = device
        self.logger     = logger
        self.threshold  = float(threshold)
        self.split_name = split_name
        # Forward-Pass-Ergebnisse, damit Schwellensuche und finale Auswertung
        # nicht zweimal ueber den Datensatz laufen muessen
        self._cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
        # Mittlere Vorwaertsdurchlaufzeit je Bild (ms), in collect() gemessen
        self._inference_ms: float = float("nan")

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    @property
    def _suffix(self) -> str:
        """Dateinamens-Suffix: test behaelt die historischen Namen."""
        return "" if self.split_name == "test" else f"_{self.split_name}"

    @property
    def _label(self) -> str:
        return _SPLIT_LABEL.get(self.split_name, self.split_name)

    # ----------------------------------------------------------
    def collect(self) -> Tuple[np.ndarray, np.ndarray]:
        """Forward-Pass über den Loader. Gibt (labels, probs) zurück.

        Wird gecacht: ein zweiter Aufruf (z.B. nach dem Setzen eines neuen
        Schwellenwerts) laeuft nicht erneut ueber den Datensatz.

        Nebenbei wird die reine Modellrechenzeit je Bild gemessen (ohne Laden
        und ohne Geraetetransfer, damit der Backbone-Vergleich nur die
        Architektur misst). Der erste Batch dient als Warmlauf.
        """
        if self._cache is not None:
            return self._cache

        self.model.eval()
        all_labels, all_probs = [], []
        on_cuda   = self.device.type == "cuda"
        fwd_s, fwd_n     = 0.0, 0   # ohne Warmlauf
        warm_s, warm_n   = 0.0, 0   # Rueckfall bei nur einem Batch

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.loader):
                images = images.to(self.device)

                if on_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                logits = self.model(images)
                if on_cuda:
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0

                if batch_idx == 0:
                    warm_s, warm_n = dt, images.size(0)
                else:
                    fwd_s += dt
                    fwd_n += images.size(0)

                probs  = torch.softmax(logits, dim=1)

                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        if fwd_n == 0:          # Datensatz passte in einen einzigen Batch
            fwd_s, fwd_n = warm_s, warm_n
        if fwd_n > 0:
            self._inference_ms = 1000.0 * fwd_s / fwd_n

        self._cache = (np.array(all_labels), np.array(all_probs))
        return self._cache

    # ----------------------------------------------------------
    def evaluate(self) -> dict:
        """Führt Evaluation durch. Gibt Metriken-Dict zurück."""
        all_labels, all_probs = self.collect()
        nio_probs  = all_probs[:, 1]       # Konfidenz für NIO-Klasse
        # Schwellenwert statt argmax – bei 0.5 identisch, aber uebertragbar
        all_preds  = (nio_probs >= self.threshold).astype(int)

        # ---- Klassische Metriken ---------------------------------
        metrics = {
            "accuracy":  float(accuracy_score(all_labels, all_preds)),
            "precision": float(precision_score(all_labels, all_preds, pos_label=1, zero_division=0)),
            "recall":    float(recall_score(all_labels,    all_preds, pos_label=1, zero_division=0)),
            "f1":        float(f1_score(all_labels,        all_preds, pos_label=1, zero_division=0)),
            "n_samples": int(len(all_labels)),
            "n_io":      int((all_labels == 0).sum()),
            "n_nio":     int((all_labels == 1).sum()),
            "note":      "pos_label=1 entspricht Klasse 'nio' (Fehlerklasse)",
            "threshold": self.threshold,
            "split":     self.split_name,
            # Nebenkriterien des Backbone-Vergleichs (vgl. Kapitel 4.5.2)
            "inference_ms_per_image": float(self._inference_ms),
            "n_params_trainable":     int(count_trainable_params(self.model)),
        }

        # ---- Threshold-unabhängige Metriken (ROC-AUC, AP) -------
        n_unique = len(np.unique(all_labels))
        if n_unique >= 2:
            metrics["roc_auc"]           = float(roc_auc_score(all_labels, nio_probs))
            metrics["average_precision"] = float(average_precision_score(all_labels, nio_probs))
        else:
            metrics["roc_auc"]           = float("nan")
            metrics["average_precision"] = float("nan")
            self._log(f"[WARNUNG] Nur eine Klasse im {self._label} – ROC-AUC nicht berechenbar.")

        # ---- Vergleichswerte beim Standard-Schwellenwert 0.5 -----
        # Macht in der Thesis belegbar, was die Schwellenwahl gebracht hat.
        if abs(self.threshold - 0.5) > 1e-9:
            preds_05 = (nio_probs >= 0.5).astype(int)
            metrics["metrics_at_0.5"] = {
                "accuracy":  float(accuracy_score(all_labels, preds_05)),
                "precision": float(precision_score(all_labels, preds_05, pos_label=1, zero_division=0)),
                "recall":    float(recall_score(all_labels,    preds_05, pos_label=1, zero_division=0)),
                "f1":        float(f1_score(all_labels,        preds_05, pos_label=1, zero_division=0)),
            }

        # Logging
        self._log(f"\n=== Metriken ({self._label}, threshold={self.threshold:.4f}) ===")
        for k, v in metrics.items():
            if isinstance(v, float):
                self._log(f"  {k:<20}: {v:.4f}")
            elif isinstance(v, dict):
                inner = "  ".join(f"{ik}={iv:.4f}" for ik, iv in v.items())
                self._log(f"  {k:<20}: {inner}")
            else:
                self._log(f"  {k:<20}: {v}")

        # ---- Speichern -------------------------------------------
        metrics_path = os.path.join(
            self.run_dir, "metrics", f"{self.split_name}_metrics.json"
        )
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        self._log(f"Metriken gespeichert → {metrics_path}")

        # ---- Confusion Matrix (labels=[0,1] sichert immer 2×2-Matrix) ----
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        self._save_confusion_matrix(cm)

        # ---- ROC-Kurve -------------------------------------------
        if n_unique >= 2:
            self._save_roc_curve(all_labels, nio_probs, metrics["roc_auc"])

        # ---- Prediction CSV --------------------------------------
        self._save_predictions(all_preds, all_labels, all_probs)

        return metrics

    # ----------------------------------------------------------
    def _save_confusion_matrix(self, cm: np.ndarray):
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(CLASSES, fontsize=12)
        ax.set_yticklabels(CLASSES, fontsize=12)
        ax.set_xlabel("Vorhergesagt", fontsize=12)
        ax.set_ylabel("Tatsächlich",  fontsize=12)
        ax.set_title(f"Confusion Matrix ({self._label})", fontsize=13)

        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = 100 * cm[i, j] / total if total > 0 else 0
                ax.text(
                    j, i,
                    f"{cm[i, j]}\n({pct:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=11,
                )

        plt.tight_layout()
        path = os.path.join(
            self.run_dir, "metrics", f"confusion_matrix{self._suffix}.png"
        )
        plt.savefig(path, dpi=120)
        plt.close()
        self._log(f"Confusion Matrix gespeichert → {path}")

    # ----------------------------------------------------------
    def _save_roc_curve(self, labels: np.ndarray, nio_probs: np.ndarray, auc: float):
        fpr, tpr, _ = roc_curve(labels, nio_probs, pos_label=1)

        with curve_style():
            fig, ax = plt.subplots(figsize=(5.2, 5.2))
            # Diagonale (Zufallsklassifikator) hinter der Kurve
            ax.plot([0, 1], [0, 1], color=MUT, lw=1.4, linestyle="--", zorder=2)
            ax.plot(fpr, tpr, color=BLUE, lw=2.2, zorder=4)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xlabel("Falsch-Positiv-Rate")
            ax.set_ylabel("Richtig-Positiv-Rate")
            ax.set_title(f"ROC-Kurve (NIO, {self._label})")
            ax.set_aspect("equal")
            style_axes(ax, grid_axis="both")
            ax.text(
                0.58, 0.16, f"AUC = {auc:.3f}".replace(".", ","),
                fontsize=13, fontweight="bold", color=BLUE,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=GRID),
            )

            path = os.path.join(
                self.run_dir, "metrics", f"roc_curve{self._suffix}.png"
            )
            fig.tight_layout()
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
        self._log(f"ROC-Kurve gespeichert → {path}")

    # ----------------------------------------------------------
    def _save_predictions(self, preds, labels, probs):
        path = os.path.join(
            self.run_dir, "preds", f"example_predictions{self._suffix}.csv"
        )
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "true_label", "predicted_label",
                              "prob_io", "prob_nio", "correct"])
            for i, (pred, label, prob) in enumerate(zip(preds, labels, probs)):
                writer.writerow([
                    i,
                    CLASSES[label],
                    CLASSES[pred],
                    f"{prob[0]:.4f}",
                    f"{prob[1]:.4f}",
                    int(pred == label),
                ])
        self._log(f"Predictions gespeichert → {path}")
