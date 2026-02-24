# IO/NIO Binary Image Classifier
Kamerabasiertes Bildklassifikationssystem zur automatisierten Qualitätsbewertung von Bauteilchargen.

**Klassen:** `io` (in Ordnung) · `nio` (nicht in Ordnung)  
**Architektur:** ResNet18 / ResNet50 mit Transfer Learning  
**Framework:** PyTorch  

---

## Projektstruktur

```
io_nio_classifier/
├── configs/
│   └── default.yaml              ← Alle Hyperparameter
├── data/
│   ├── train/io/  train/nio/
│   ├── val/io/    val/nio/
│   └── test/io/   test/nio/
├── outputs/
│   └── runs/
│       └── 2026-02-18_21-30_resnet18_lr0.0001_bs16/
│           ├── config.yaml       ← Kopie der verwendeten Config
│           ├── train.log         ← Vollständiges Trainingsprotokoll
│           ├── checkpoints/
│           │   ├── best.pt       ← Bestes Modell (nach Val-Accuracy)
│           │   └── last.pt       ← Letzter Checkpoint
│           ├── metrics/
│           │   ├── class_info.json
│           │   ├── train_metrics.csv
│           │   ├── val_metrics.csv
│           │   ├── test_metrics.json
│           │   └── confusion_matrix.png
│           ├── preds/
│           │   └── example_predictions.csv
│           ├── tensorboard/      ← TensorBoard-Logs
│           └── xai/
│               └── integrated_gradients/
│                   └── img_0000.png
├── scripts/
│   └── create_dummy_data.py      ← Synthetisches Testdataset
├── src/
│   ├── dataset.py                ← Dataset, DataLoader, Transforms
│   ├── models/
│   │   └── resnet.py             ← ResNet18/50 Backbone
│   ├── training/
│   │   ├── trainer.py            ← Trainingsschleife
│   │   └── early_stopping.py
│   ├── evaluation/
│   │   └── evaluator.py          ← Metriken + Confusion Matrix
│   ├── inference/
│   │   └── predict.py            ← Inference CLI
│   ├── xai/
│   │   └── integrated_gradients.py
│   └── utils/
│       ├── reproducibility.py
│       ├── run_manager.py
│       └── logging_utils.py
├── tests/
│   └── test_pipeline.py
├── train.py                      ← Haupt-Einstiegspunkt
├── predict.py                    ← Inference-Wrapper
└── requirements.txt
```

---

## Setup

### 1. Virtual Environment erstellen
```bash
python -m venv venv

# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 2. Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

> **Hinweis:** Die `requirements.txt` enthält keine GPU-spezifische PyTorch-Version.  
> Für GPU-Support: https://pytorch.org/get-started/locally/ (CUDA-Version wählen)

---

## Schnellstart: Dummy-Daten (ohne echte Daten)

Testet die gesamte Pipeline mit synthetischen Bildern (Kreis=io, Rechteck=nio):

```bash
python train.py --dummy
```

Das erstellt automatisch ein Dummy-Dataset in `data/` und durchläuft Train → Eval → Konfusionsmatrix.

---

## Training mit echten Daten

```bash
# Mit Standard-Config:
python train.py

# Mit Custom-Config:
python train.py --config configs/default.yaml

# Run-Namen überschreiben:
python train.py --run_name mein_experiment_v1
```

---

## Wichtige Config-Optionen (`configs/default.yaml`)

| Parameter | Default | Beschreibung |
|---|---|---|
| `backbone` | `resnet18` | `resnet18` oder `resnet50` |
| `pretrained` | `true` | ImageNet-Vortraining |
| `freeze_backbone` | `false` | Nur Klassifikationskopf trainieren |
| `learning_rate` | `0.0001` | Lernrate |
| `batch_size` | `16` | Batch-Größe |
| `epochs` | `30` | Max. Trainings-Epochen |
| `use_class_weights` | `true` | Auto-Gewichtung bei Klassenimbalance |
| `use_weighted_sampler` | `false` | WeightedRandomSampler (Alternative) |
| `early_stopping` | `true` | Early Stopping aktiviert |
| `xai.enabled` | `false` | Integrated Gradients aktivieren |

### Klassenimbalance-Strategien

Es gibt zwei Ansätze – beide sind konfigurierbar:

**Option A: `use_class_weights: true`** (empfohlen)  
→ Gewichtet den Loss invers zur Klassenhäufigkeit  
→ Gewichte werden automatisch berechnet und in `metrics/class_info.json` gespeichert

**Option B: `use_weighted_sampler: true`**  
→ Oversampled die Minderheitenklasse im DataLoader  
→ Kann mit Option A kombiniert werden, ist aber oft redundant

---

## Inference

```bash
# Einzelbild
python predict.py \
  --checkpoint outputs/runs/<run>/checkpoints/best.pt \
  --config     outputs/runs/<run>/config.yaml \
  --input      pfad/zum/bild.png

# Ganzer Ordner + CSV-Export
python predict.py \
  --checkpoint outputs/runs/<run>/checkpoints/best.pt \
  --config     outputs/runs/<run>/config.yaml \
  --input      data/test/nio/ \
  --output     ergebnisse.csv
```

**Ausgabe-Beispiel:**
```
Datei                     Klasse     Konfidenz  prob_io  prob_nio
bauteil_001.png           nio          0.9732   0.0268    0.9732
bauteil_002.png           io           0.8851   0.8851    0.1149
```

---

## XAI – Integrated Gradients

Aktivierung in `configs/default.yaml`:
```yaml
xai:
  enabled: true
  n_samples: 10      # Anzahl Val-Bilder
  overlay: true      # Original + Heatmap nebeneinander
  ig_steps: 50       # Approximationsschritte (mehr = genauer, langsamer)
```

Heatmaps werden gespeichert unter: `outputs/runs/<run>/xai/integrated_gradients/`

---

## Tests ausführen

```bash
pytest tests/ -v
```

---

## TensorBoard

```bash
tensorboard --logdir outputs/runs/
```

Öffne im Browser: http://localhost:6006

---

## Reproduzierbarkeit

Alle Experimente sind durch `seed: 42` in der Config vollständig reproduzierbar.  
Seeds werden für Python, NumPy und PyTorch gesetzt.

---

## Hinweise für die Bachelorarbeit

- **`pos_label=1` = `nio`** in allen sklearn-Metriken (die industriell relevante Fehlerklasse)
- Die gespeicherte `config.yaml` im Run-Ordner dokumentiert exakt, welche Hyperparameter verwendet wurden
- `class_info.json` dokumentiert Klassenverteilung und Gewichte für jeden Lauf
- Für den Vergleich resnet18 vs. resnet50: einfach `backbone` in der Config tauschen und `--run_name` anpassen