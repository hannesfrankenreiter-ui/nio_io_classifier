import csv
import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def _read_metrics_csv(path: str):
    epochs, losses, accs = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))
            accs.append(float(row["accuracy"]))
    return epochs, losses, accs


def plot_training_curves(run_dir: str, logger=None) -> str | None:
    """
    Plot train/val loss and accuracy curves from CSV logs.
    Returns output path if plot was created, otherwise None.
    """
    metrics_dir = os.path.join(run_dir, "metrics")
    train_csv = os.path.join(metrics_dir, "train_metrics.csv")
    val_csv = os.path.join(metrics_dir, "val_metrics.csv")

    if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
        return None

    tr_ep, tr_loss, tr_acc = _read_metrics_csv(train_csv)
    va_ep, va_loss, va_acc = _read_metrics_csv(val_csv)
    if not tr_ep or not va_ep:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(tr_ep, tr_loss, label="train", linewidth=2)
    axes[0].plot(va_ep, va_loss, label="val", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(tr_ep, tr_acc, label="train", linewidth=2)
    axes[1].plot(va_ep, va_acc, label="val", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    out_path = os.path.join(metrics_dir, "training_curves.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()

    # Separate validation-only plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(va_ep, va_loss, label="val", linewidth=2, color="tab:orange")
    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(va_ep, va_acc, label="val", linewidth=2, color="tab:green")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    val_out_path = os.path.join(metrics_dir, "val_curves.png")
    plt.tight_layout()
    plt.savefig(val_out_path, dpi=140, bbox_inches="tight")
    plt.close()

    if logger:
        logger.info(f"Metrik-Plot gespeichert: {out_path}")
        logger.info(f"Val-Metrik-Plot gespeichert: {val_out_path}")
    return out_path
