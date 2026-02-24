"""
src/evaluation/evaluator.py
Berechnet und speichert Test-Metriken:
  - Accuracy, Precision, Recall, F1 (binär, pos_label=1 → "nio")
  - Confusion Matrix als PNG
  - Predictions als CSV

HINWEIS: pos_label=1 entspricht der Klasse "nio" (Index 1).
         Das ist die industriell relevante Fehlerklasse.
"""
import csv
import json
import os
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

matplotlib.use("Agg")  # Kein Display nötig

CLASSES = ["io", "nio"]


class Evaluator:
    """
    Evaluiert ein Modell auf einem DataLoader und speichert alle Ergebnisse.

    Args:
        model     : Trainiertes nn.Module (bereits auf device geladen)
        loader    : Test-DataLoader
        run_dir   : Pfad zum Experiment-Ordner
        device    : torch.device
        logger    : optionaler Python-Logger
    """

    def __init__(
        self,
        model: nn.Module,
        loader: DataLoader,
        run_dir: str,
        device: torch.device,
        logger=None,
    ):
        self.model   = model
        self.loader  = loader
        self.run_dir = run_dir
        self.device  = device
        self.logger  = logger

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    # ----------------------------------------------------------
    def evaluate(self) -> dict:
        """Führt Evaluation durch. Gibt Metriken-Dict zurück."""
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for images, labels in self.loader:
                images  = images.to(self.device)
                logits  = self.model(images)
                probs   = torch.softmax(logits, dim=1)
                preds   = probs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs  = np.array(all_probs)

        # ---- Metriken ----------------------------------------
        metrics = {
            "accuracy":  float(accuracy_score(all_labels, all_preds)),
            "precision": float(precision_score(all_labels, all_preds, pos_label=1, zero_division=0)),
            "recall":    float(recall_score(all_labels,    all_preds, pos_label=1, zero_division=0)),
            "f1":        float(f1_score(all_labels,        all_preds, pos_label=1, zero_division=0)),
            "n_samples": int(len(all_labels)),
            "n_io":      int((all_labels == 0).sum()),
            "n_nio":     int((all_labels == 1).sum()),
            "note": "pos_label=1 entspricht Klasse 'nio' (Fehlerklasse)",
        }

        self._log("\n=== Test-Metriken ===")
        for k, v in metrics.items():
            if isinstance(v, float):
                self._log(f"  {k:<12}: {v:.4f}")
            else:
                self._log(f"  {k:<12}: {v}")

        # ---- Speichern ---------------------------------------
        metrics_path = os.path.join(self.run_dir, "metrics", "test_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        self._log(f"Metriken gespeichert → {metrics_path}")

        # ---- Confusion Matrix --------------------------------
        cm = confusion_matrix(all_labels, all_preds)
        self._save_confusion_matrix(cm)

        # ---- Prediction CSV ----------------------------------
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
        ax.set_title("Confusion Matrix (Testset)", fontsize=13)

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
        path = os.path.join(self.run_dir, "metrics", "confusion_matrix.png")
        plt.savefig(path, dpi=120)
        plt.close()
        self._log(f"Confusion Matrix gespeichert → {path}")

    # ----------------------------------------------------------
    def _save_predictions(self, preds, labels, probs):
        path = os.path.join(self.run_dir, "preds", "example_predictions.csv")
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