import csv
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.utils.plot_style import BLUE, RED, curve_style, style_axes


def _read_metrics_csv(path: str):
    """Liest epoch, loss, accuracy und (falls vorhanden) f1 aus einer Metrik-CSV."""
    epochs, losses, accs, f1s = [], [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))
            accs.append(float(row["accuracy"]))
            f1s.append(float(row["f1"]) if row.get("f1") not in (None, "") else float("nan"))
    return epochs, losses, accs, f1s


def _xticks(epochs):
    """Wenige, ganzzahlige x-Ticks (Anfang/Mitte/Ende) wie in der Präsentation."""
    if len(epochs) <= 5:
        return epochs
    return sorted({epochs[0], epochs[len(epochs) // 2], epochs[-1]})


def _zoom_ylim(*series):
    """Untere Grenze knapp unter das Minimum (für Accuracy/F1, damit man Details sieht)."""
    lo = min(min(s) for s in series)
    return (max(0.0, lo - 0.03), 1.01)


def plot_training_curves(run_dir: str, logger=None) -> str | None:
    """
    Zeichnet Trainings-/Validierungskurven (Loss, Accuracy, F1) und einen
    separaten Validierungsplot – einheitlich im Softmax-Stil (Pfeil-Achsen,
    gedeckte Palette, feines Raster). Rückgabe: Pfad des Hauptplots oder None.
    """
    metrics_dir = os.path.join(run_dir, "metrics")
    train_csv = os.path.join(metrics_dir, "train_metrics.csv")
    val_csv = os.path.join(metrics_dir, "val_metrics.csv")

    if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
        return None

    tr_ep, tr_loss, tr_acc, tr_f1 = _read_metrics_csv(train_csv)
    va_ep, va_loss, va_acc, va_f1 = _read_metrics_csv(val_csv)
    if not tr_ep or not va_ep:
        return None

    xt = _xticks(tr_ep)

    with curve_style():
        # ---- Kombinierte Kurven: Loss / Accuracy / F1 (Training vs. Validierung) ----
        panels = [
            ("Loss", tr_loss, va_loss, None),
            ("Accuracy", tr_acc, va_acc, _zoom_ylim(tr_acc, va_acc)),
            ("F1-Score", tr_f1, va_f1, _zoom_ylim(tr_f1, va_f1)),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
        for ax, (title, tr_y, va_y, ylim) in zip(axes, panels):
            ax.plot(tr_ep, tr_y, color=BLUE, linewidth=1.8, label="Training")
            ax.plot(va_ep, va_y, color=RED, linewidth=1.8, linestyle="--", label="Validierung")
            ax.set_title(title)
            ax.set_xlabel("Epoche")
            ax.set_xticks(xt)
            if ylim is not None:
                ax.set_ylim(*ylim)
            else:
                ax.set_ylim(bottom=0)
            style_axes(ax, grid_axis="y")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, 1.0), columnspacing=2.6, handlelength=2.4)

        out_path = os.path.join(metrics_dir, "training_curves.png")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

        # ---- Separater Validierungsplot (Loss / Accuracy / F1) ----
        vpanels = [
            ("Validation Loss", va_loss, None),
            ("Validation Accuracy", va_acc, _zoom_ylim(va_acc)),
            ("Validation F1-Score", va_f1, _zoom_ylim(va_f1)),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
        for ax, (title, y, ylim) in zip(axes, vpanels):
            ax.plot(va_ep, y, color=BLUE, linewidth=1.8)
            ax.set_title(title)
            ax.set_xlabel("Epoche")
            ax.set_xticks(xt)
            if ylim is not None:
                ax.set_ylim(*ylim)
            else:
                ax.set_ylim(bottom=0)
            style_axes(ax, grid_axis="y")

        val_out_path = os.path.join(metrics_dir, "val_curves.png")
        fig.tight_layout()
        fig.savefig(val_out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    if logger:
        logger.info(f"Metrik-Plot gespeichert: {out_path}")
        logger.info(f"Val-Metrik-Plot gespeichert: {val_out_path}")
    return out_path
