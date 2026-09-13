"""
src/inference/predict.py
Inference-Modul: Einzelbild oder Ordner klassifizieren.

Neue Features:
  - DataLoader-basierte Batch-Inferenz (statt Python-Schleife)
  - Konfigurierbarer Entscheidungsschwellenwert (threshold)
  - Test-Time Augmentation (TTA)

CLI-Nutzung:
    python predict.py --checkpoint outputs/runs/<run>/checkpoints/best.pt \
                      --config     outputs/runs/<run>/config.yaml \
                      --input      path/to/image_or_folder \
                      [--output    results.csv]
"""
import argparse
import csv
import os
import sys

import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.dataset import bildgroesse  # noqa: E402  (braucht den Pfad oben)

CLASSES = ["io", "nio"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ================================================================
#  Hilfsfunktionen
# ================================================================
def load_model(checkpoint_path: str, config: dict):
    """Lädt Modell aus Checkpoint."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.models.resnet import build_model

    model = build_model(config)
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def get_transform(config: dict) -> T.Compose:
    """Erstellt Inferenz-Transform (kein Augmentation)."""
    norm  = config.get("augmentation", {}).get("normalize", {})
    mean  = norm.get("mean", [0.485, 0.456, 0.406])
    std   = norm.get("std",  [0.229, 0.224, 0.225])
    return T.Compose([
        T.Resize(bildgroesse(config)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def _get_tta_transform(config: dict) -> T.Compose:
    """Augmented transform für Test-Time Augmentation."""
    norm  = config.get("augmentation", {}).get("normalize", {})
    mean  = norm.get("mean", [0.485, 0.456, 0.406])
    std   = norm.get("std",  [0.229, 0.224, 0.225])
    return T.Compose([
        T.Resize(bildgroesse(config)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def predict_single(
    model,
    transform,
    image_path: str,
    config: dict = None,
) -> dict:
    """
    Klassifiziert ein einzelnes Bild.

    Args:
        model       : Geladenes nn.Module (eval-Modus)
        transform   : Standard-Inferenz-Transform
        image_path  : Pfad zum Eingabebild
        config      : Optionales Config-Dict (für threshold und TTA)

    Returns:
        Dict mit predicted_class, confidence, prob_io, prob_nio
    """
    config = config or {}
    threshold   = float(config.get("threshold", 0.5))
    tta_enabled = config.get("tta_enabled", False)
    tta_n       = int(config.get("tta_n", 5))

    image = Image.open(image_path).convert("RGB")
    model.eval()  # BatchNorm1d erfordert eval-Modus bei batch_size=1

    if tta_enabled and tta_n > 1:
        # Test-Time Augmentation: 1× Original + (tta_n-1)× augmentiert
        tta_transform = _get_tta_transform(config)
        all_probs = []
        with torch.no_grad():
            # Original
            tensor = transform(image).unsqueeze(0)
            all_probs.append(torch.softmax(model(tensor), dim=1).squeeze(0))
            # Augmentierte Versionen
            for _ in range(tta_n - 1):
                aug_tensor = tta_transform(image).unsqueeze(0)
                all_probs.append(torch.softmax(model(aug_tensor), dim=1).squeeze(0))
        probs = torch.stack(all_probs).mean(0)
    else:
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).squeeze(0)

    # Konfigurierbarer Schwellenwert für NIO-Klasse (pos_label=1)
    pred_idx = 1 if float(probs[1]) >= threshold else 0

    return {
        "path":            image_path,
        "predicted_class": CLASSES[pred_idx],
        "confidence":      float(probs[pred_idx]),
        "prob_io":         float(probs[0]),
        "prob_nio":        float(probs[1]),
    }


# ================================================================
#  Batch-Inferenz Dataset
# ================================================================
class _InferenceDataset(Dataset):
    """Leichtgewichtiges Dataset für Batch-Inferenz aus Bildpfaden."""

    def __init__(self, paths: list, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path  = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), path


def predict_batch(
    model,
    transform,
    folder: str,
    config: dict = None,
) -> list:
    """
    Klassifiziert alle Bilder in einem Ordner via DataLoader (batch-parallel).

    Args:
        model     : Geladenes nn.Module (eval-Modus)
        transform : Standard-Inferenz-Transform
        folder    : Ordner mit Eingangsbildern
        config    : Optionales Config-Dict (batch_size, threshold)

    Returns:
        Liste von Ergebnis-Dicts (path, predicted_class, confidence, prob_io, prob_nio)
    """
    config      = config or {}
    threshold   = float(config.get("threshold", 0.5))
    batch_size  = int(config.get("batch_size", 16))
    num_workers = int(config.get("num_workers", 0))

    paths = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
    )
    if not paths:
        print(f"[WARNUNG] Keine Bilder gefunden in: {folder}")
        return []

    dataset = _InferenceDataset(paths, transform)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    results = []
    model.eval()
    with torch.no_grad():
        for tensors, batch_paths in loader:
            logits = model(tensors)
            probs  = torch.softmax(logits, dim=1)
            for i, path in enumerate(batch_paths):
                p = probs[i]
                pred_idx = 1 if float(p[1]) >= threshold else 0
                results.append({
                    "path":            path,
                    "predicted_class": CLASSES[pred_idx],
                    "confidence":      float(p[pred_idx]),
                    "prob_io":         float(p[0]),
                    "prob_nio":        float(p[1]),
                })

    return results


# ================================================================
#  CLI
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="IO/NIO Inference CLI")
    parser.add_argument("--checkpoint", required=True,
                        help="Pfad zu best.pt (oder last.pt)")
    parser.add_argument("--config",     required=True,
                        help="Pfad zur config.yaml des Runs")
    parser.add_argument("--input",      required=True,
                        help="Pfad zu einem Bild oder einem Ordner")
    parser.add_argument("--output",     default=None,
                        help="Optionaler Pfad zum Speichern der Ergebnisse (CSV)")
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Ueberschreibt den Schwellenwert aus der Config. Den auf VAL "
                             "bestimmten Wert findest du in metrics/threshold.json des Runs.")
    args = parser.parse_args()

    # Config laden
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.threshold is not None:
        config["threshold"] = args.threshold
        print(f"[INFO] Schwellenwert per CLI gesetzt: {args.threshold}")

    model     = load_model(args.checkpoint, config)
    transform = get_transform(config)

    # Eingabe auflösen
    if os.path.isdir(args.input):
        results = predict_batch(model, transform, args.input, config=config)
    elif os.path.isfile(args.input):
        results = [predict_single(model, transform, args.input, config=config)]
    else:
        print(f"[FEHLER] '{args.input}' ist weder Datei noch Ordner.")
        sys.exit(1)

    # Ausgabe
    print(f"\n{'Datei':<50} {'Klasse':<8} {'Konfidenz':>10}  prob_io  prob_nio")
    print("-" * 85)
    for r in results:
        fname = os.path.basename(r["path"])
        print(
            f"{fname:<50} {r['predicted_class']:<8} "
            f"{r['confidence']:>10.4f}  {r['prob_io']:.4f}   {r['prob_nio']:.4f}"
        )

    # Optional: CSV speichern
    if args.output and results:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nErgebnisse gespeichert → {args.output}")


if __name__ == "__main__":
    main()
