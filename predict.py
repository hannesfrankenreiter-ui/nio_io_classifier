#!/usr/bin/env python3
"""
predict.py  –  Inference-Einstiegspunkt (Wrapper um src/inference/predict.py)

Beispiele:
    # Einzelbild
    python predict.py --checkpoint outputs/runs/<run>/checkpoints/best.pt \
                      --config     outputs/runs/<run>/config.yaml \
                      --input      data/test/nio/bild001.png

    # Ganzer Ordner (alle Bilder werden klassifiziert)
    python predict.py --checkpoint outputs/runs/<run>/checkpoints/best.pt \
                      --config     outputs/runs/<run>/config.yaml \
                      --input      data/test/nio/ \
                      --output     outputs/runs/<run>/preds/inference_results.csv
"""
from src.inference.predict import main

if __name__ == "__main__":
    main()