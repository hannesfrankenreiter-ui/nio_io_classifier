"""
src/inference/predict.py
Inference-Modul: Einzelbild oder Ordner klassifizieren.

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
    image_size = config.get("image_size", 224)
    norm  = config.get("augmentation", {}).get("normalize", {})
    mean  = norm.get("mean", [0.485, 0.456, 0.406])
    std   = norm.get("std",  [0.229, 0.224, 0.225])
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def predict_single(model, transform, image_path: str) -> dict:
    """Klassifiziert ein einzelnes Bild. Gibt Dict mit Klasse + Konfidenz zurück."""
    image  = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # (1, C, H, W)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze(0)

    pred_idx = probs.argmax().item()

    return {
        "path":            image_path,
        "predicted_class": CLASSES[pred_idx],
        "confidence":      float(probs[pred_idx]),
        "prob_io":         float(probs[0]),
        "prob_nio":        float(probs[1]),
    }


def predict_batch(model, transform, folder: str) -> list:
    """Klassifiziert alle Bilder in einem Ordner."""
    paths = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
    )
    if not paths:
        print(f"[WARNUNG] Keine Bilder gefunden in: {folder}")
        return []
    return [predict_single(model, transform, p) for p in paths]


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
    args = parser.parse_args()

    # Config laden
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model     = load_model(args.checkpoint, config)
    transform = get_transform(config)

    # Eingabe auflösen
    if os.path.isdir(args.input):
        results = predict_batch(model, transform, args.input)
    elif os.path.isfile(args.input):
        results = [predict_single(model, transform, args.input)]
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