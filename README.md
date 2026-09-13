# IO/NIO-Bildklassifikation für die Chargenprüfung

**Transfer Learning und erklärbare KI für die industrielle Qualitätsprüfung – ein Bachelorarbeitsprojekt mit PyTorch.**

## Projektüberblick

Dieses Projekt untersucht, wie sich die visuelle Qualitätsbewertung von Bauteilen durch eine
kamerabasierte IO/NIO-Klassifikation unterstützen lässt. Neben der Erkennung fehlerhafter Bauteile
steht die Nachvollziehbarkeit der Modellentscheidungen im Mittelpunkt.

Die entwickelte Pipeline verbindet Bildvorverarbeitung, den systematischen Vergleich vortrainierter
CNNs mit unterschiedlichen Fine-Tuning-Tiefen sowie Evaluation und Inferenz. Attributionskarten
zeigen entscheidungsrelevante Bildbereiche und lassen sich in die Originalaufnahmen zurückprojizieren.

**Klassen:** `io` (in Ordnung) · `nio` (nicht in Ordnung, die relevante Fehlerklasse)
**Backbones:** ResNet-18 · ResNet-50 · EfficientNet-B0 · ConvNeXt-Tiny (Transfer Learning, ImageNet-Gewichte)
**Erklärbarkeit:** Integrated Gradients · Saliency Map · Occlusion Map, inklusive Rückprojektion ins Originalbild

> **Hinweis zu den Daten:** Die Bachelorarbeit unterliegt einem Sperrvermerk. Aufnahmen der Bauteile,
> trainierte Modelle (`*.pt`) und Heatmaps sind deshalb **nicht** Teil dieses Repositorys.
> Der Ordner `data/` ist leer; die gesamte Pipeline lässt sich mit synthetischen Dummy-Daten und
> der Testsuite nachvollziehen. Die **Metriken und Vergleichstabellen der Versuchsläufe** liegen unter
> [`outputs/runs/`](outputs/runs/) – mit leerem Ordnergerüst dort, wo Modelle und Bilder hingehören.

Interaktive Modulübersicht: [Projektstruktur und Datenfluss](Projektstruktur_Datenfluss.html)
(herunterladen und im Browser öffnen).

---

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> `requirements.txt` enthält keine GPU-spezifische PyTorch-Version. Für CUDA-Unterstützung PyTorch
> passend zur eigenen CUDA-Version installieren: https://pytorch.org/get-started/locally/

---

## Schnellstart ohne echte Daten

Die Testsuite prüft Dataset, Modelle, Training, Evaluation, Inferenz, XAI und Rückprojektion mit
synthetischen Bildern:

```bash
pytest tests/ -v
```

Ein vollständiger Trainingslauf auf Dummy-Daten (Kreis = io, Rechteck = nio):

```bash
python train.py --dummy --config configs/dummy.yaml
```

`--dummy` schreibt die synthetischen Bilder nach `data_dir` der Config. Dafür eine Kopie von
`configs/default.yaml` als `configs/dummy.yaml` anlegen und darin mindestens setzen:

```yaml
data_dir: data          # Dummy-Bilder landen in data/train|val|test
epochs: 3               # für einen kurzen Durchlauf
xai:
  enabled: false        # IG auf allen Testbildern ist ohne GPU sehr langsam
```

Die Rückprojektion am Laufende wird ohne Originalaufnahmen automatisch mit einer Logzeile übersprungen.

---

## Projektstruktur

```
io_nio_classifier/
├── train.py                     ← Einstieg: Training → Evaluation → XAI → Rückprojektion
├── predict.py                   ← Einstieg: Inferenz auf Einzelbild oder Ordner
├── requirements.txt
├── Projektstruktur_Datenfluss.html ← interaktive Übersicht über Module und Verbindungen
├── configs/
│   ├── default.yaml             ← alle Hyperparameter (aktueller Stand)
│   ├── head_only.yaml           ← Kontrollversuch: nur der Klassifikationskopf trainiert
│   ├── stufe_2_basis.yaml       ← eingefrorene Basis-Config der Versuchsstufe 2
│   ├── stufe_1/                 ← 12 Laufconfigs Stufe 1 (Vollbilder)
│   └── stufe_2/                 ← 12 Laufconfigs Stufe 2 (Bauteil-Zuschnitte) + 800×320-Variante
├── data/                        ← leer (Sperrvermerk); erwartet train|val|test / io|nio
├── outputs/runs/                ← ein Ordner je Trainingslauf
│   ├── Stufe_1/                 ← 12 Läufe: Config, Metriken, Vorhersagen + Vergleichstabellen (.md)
│   └── Stufe_2/                 ← 13 Läufe, ebenso
├── src/
│   ├── dataset.py               ← IODataset, Transforms/Augmentierung, DataLoader
│   ├── preprocessing/
│   │   └── korb_zoom.py         ← Zuschnittgeometrie: Korbreferenz, Bauteilmaske, Fenster
│   ├── models/
│   │   └── resnet.py            ← build_model: vier Backbones, Freeze-Strategien, Kopf
│   ├── training/
│   │   ├── trainer.py           ← Trainingsschleife, Optimizer, Scheduler, Checkpoints
│   │   └── early_stopping.py
│   ├── evaluation/
│   │   └── evaluator.py         ← Metriken, Schwellenwertsuche, Konfusionsmatrix, ROC
│   ├── inference/
│   │   ├── predict.py           ← Einzel-/Batch-Inferenz, Schwellenwert, TTA
│   │   └── lauf.py              ← lade_lauf: Modell, Transform und Betriebspunkt eines Laufs
│   ├── xai/
│   │   ├── integrated_gradients.py ← IG, Saliency, Occlusion, Heatmap-Overlays
│   │   ├── metriken.py          ← Konzentrationsfaktor, Deletion/Insertion, Sanity-Check
│   │   ├── rueckprojektion.py   ← Attributionskarte zurück ins Originalbild
│   │   └── lauf_bilder.py       ← automatische Rückprojektion am Ende des Trainings
│   └── utils/
│       ├── reproducibility.py   ← Seeds (Python, NumPy, PyTorch, DataLoader-Worker)
│       ├── run_manager.py       ← Run-Ordner anlegen, Config sichern
│       ├── logging_utils.py
│       ├── plot_metrics.py      ← Trainings- und Validierungskurven
│       └── plot_style.py        ← einheitlicher Diagrammstil
├── scripts/                     ← Datenaufbereitung, Versuchsplan, Auswertung (siehe unten)
└── tests/                       ← pytest-Suite (151 Tests)
```

---

## Datenaufbereitung (echte Aufnahmen)

Die Originalaufnahmen (4096 × 3000 px) werden nicht direkt trainiert, sondern einmalig aufbereitet:

| Schritt | Befehl | Ergebnis |
|---|---|---|
| Korbreferenz bilden | `python scripts/korb_referenz.py erstellen data_stufe2` | Referenzbilder des leeren Gitterkorbs je Zeitfenster |
| Bauteil freistellen und zuschneiden | `python scripts/korb_referenz.py anwenden data_stufe2 --quadratisch` | quadratische Bauteil-Zuschnitte + Masken + `protokoll.csv` |
| Auf Trainingsgröße verkleinern | `python scripts/prepare_resized_data.py --source data_stufe2_freigestellt/zuschnitt --target data_stufe2_512 --size 512` | `data_stufe2_512/` |
| Zuschnitt nachweisen | `python scripts/verify_zuschnitt.py` | pixelgenauer Abgleich des Zuschnittfensters |

Wie ein Datensatz aus den Originalen entstanden ist, steht im `zuschnitt:`-Block der Config
(`verfahren: bauteil | fest | ganz`). Die Rückprojektion liest diesen Block, statt ihn zu erraten.

---

## Training

```bash
python train.py                                        # configs/default.yaml
python train.py --config configs/stufe_2/S2_12_convnext_tiny_vollstaendig.yaml
python train.py --run_name mein_experiment
```

Ablauf eines Laufs:

1. Config laden, Seeds setzen, Run-Ordner `outputs/runs/<timestamp>_<backbone>_lr<lr>_bs<bs>/` anlegen
2. DataLoader bauen, Klassengewichte aus dem Trainingsset berechnen
3. Modell bauen (Backbone + Freeze-Strategie + neuer Klassifikationskopf)
4. Training mit AMP, Gradient Clipping, Cosine-Scheduler, Early Stopping und `best.pt`
5. **Schwellenwert auf val** bestimmen (F1-Maximum) und **unverändert auf test** anwenden
6. Optional: ONNX-Export (`export_onnx: true`)
7. Optional: XAI-Heatmaps (`xai.enabled: true`)
8. Rückprojektion einiger Heatmaps ins Originalbild (`xai.rueckprojektion.enabled: true`)

`Strg+C` bricht das Training sauber ab und springt direkt in die Test-Evaluation des bisher besten Modells.

### Inhalt eines Run-Ordners

```
outputs/runs/<run>/
├── config.yaml                  ← exakte Kopie der verwendeten Config
├── train.log
├── checkpoints/   best.pt · last.pt
├── metrics/       train_metrics.csv · val_metrics.csv · val_metrics.json · test_metrics.json
│                  threshold.json · class_info.json
│                  confusion_matrix(_val).png · roc_curve(_val).png
│                  training_curves.png · val_curves.png
├── preds/         example_predictions(_val).csv
├── tensorboard/
├── xai/integrated_gradients/    ← Heatmaps (falls xai.enabled)
└── xai/rueckprojektion/<aufnahme>/  0_original.jpg + eine Datei je XAI-Methode
```

---

## Versuchsplan

Verglichen werden vier Backbones × drei Freeze-Strategien (volles 4 × 3-Raster):

| Freeze-Strategie | `freeze_backbone` |
|---|---|
| nur Kopf | `true` |
| letzter Block + Kopf | `"layer4"` (bei EfficientNet/ConvNeXt die jeweils letzte Stage) |
| vollständiges Fine-Tuning | `false` |

```bash
python scripts/run_versuchsplan.py --stufe 1 --dry-run                        # Plan anzeigen
python scripts/run_versuchsplan.py --stufe 1                                  # Vollbilder
python scripts/run_versuchsplan.py --stufe 2 --config configs/stufe_2_basis.yaml  # Bauteil-Zuschnitte
```

Der Treiber erzeugt je Lauf eine Config (`configs/stufe_<N>/`), startet `train.py` und schreibt die
Vergleichstabellen nach `outputs/runs/Stufe_<N>/`. Außer Backbone und Freeze-Strategie bleibt über alle
Läufe alles konstant. Auswahlkriterium: F1 (nio) bei Schwelle 0,5 auf test → ROC-AUC → Rechenzeit je Bild.

Ergebnisse der Arbeit:
[Stufe 1 (Vollbilder)](outputs/runs/Stufe_1/vergleich_gesamt.md) ·
[Stufe 2 (Bauteil-Zuschnitte)](outputs/runs/Stufe_2/vergleich_gesamt.md)

---

## Inferenz

```bash
# Einzelbild
python predict.py --checkpoint outputs/runs/<run>/checkpoints/best.pt \
                  --config     outputs/runs/<run>/config.yaml \
                  --input      pfad/zum/bild.png

# Ganzer Ordner mit CSV-Export und dem auf val bestimmten Betriebspunkt
python predict.py --checkpoint outputs/runs/<run>/checkpoints/best.pt \
                  --config     outputs/runs/<run>/config.yaml \
                  --input      data/test/nio/ \
                  --output     ergebnisse.csv \
                  --threshold  <wert aus metrics/threshold.json>
```

Entscheidungsregel: `prob_nio >= threshold → nio`. Ohne `--threshold` gilt `threshold` aus der Config (0,5).
Test-Time Augmentation lässt sich über `tta_enabled` / `tta_n` einschalten.

---

## Erklärbarkeit (XAI)

| Verfahren | Prinzip |
|---|---|
| Integrated Gradients | Gradienten entlang des Pfads von einer Baseline zum Bild, in Stützpunkt-Paketen gerechnet (`ig_step_batch`) |
| Saliency Map | Gradient der Klassenwahrscheinlichkeit nach dem Eingangsbild |
| Occlusion Map | Bildbereiche abdecken und die Änderung der Vorhersage messen |

```bash
# Heatmaps für einen abgeschlossenen Lauf nachträglich erzeugen
python scripts/run_xai_split.py --run outputs/runs/<run> --split test --auswahl vergleich --methoden all

# Heatmaps für alle Läufe einer Versuchsstufe
python scripts/run_xai_versuchsplan.py --stufe 2 --zufall 44 --max-fehler 6

# Quantitative Bewertung der Verfahren (Konzentrationsfaktor, Deletion/Insertion, Sanity-Check)
python scripts/xai_kennzahlen.py --stufe 2

# Verblindete qualitative Sichtung der Karten
python scripts/xai_sichtung.py --bogen --stufe 2
python scripts/xai_sichtung.py --auswerten --stufe 2

# Heatmap zurück ins Originalbild, an die Stelle, an der gezoomt wurde
python scripts/pipeline_original.py --run outputs/runs/<run> --zuschnitt-config configs/default.yaml --limit 6
```

---

## Auswertung

| Skript | Zweck |
|---|---|
| `scripts/threshold_analyse.py` | Precision/Recall/F1 über ein Schwellenraster; Übertragbarkeit des val-Betriebspunkts auf test |
| `scripts/benchmark_inferenz.py` | Inferenzzeit je Backbone unter kontrollierten Bedingungen (Median, Batch und Einzelbild) |
| `scripts/kap5_daten.py` | gemeinsame Datenbasis aller Ergebnistabellen und -diagramme |
| `scripts/thesis_kap5_abbildungen.py` | Diagramme des Ergebniskapitels (Verlustkurven, F1-Matrix, ROC, Konfusionsmatrizen …) |
| `scripts/thesis_kap5_abbildungen_matplotlib.py` | dieselben Diagramme im gerahmten Matplotlib-Stil |
| `scripts/xai_auswahl.py` | laufunabhängige Auswahl der zu erklärenden Bilder + Manifest |
| `scripts/build_part_review.py` | Zuordnung Aufnahme → physisches Bauteil über Zeitstempel-Segmente |

---

## Wichtige Config-Optionen (`configs/default.yaml`)

| Parameter | Default | Beschreibung |
|---|---|---|
| `backbone` | `resnet50` | `resnet18` · `resnet50` · `efficientnet_b0` · `convnext_tiny` |
| `freeze_backbone` | `layer4` | `true` = nur Kopf · `"layer4"` = letzter Block + Kopf · `false` = vollständig |
| `freeze_bn_stats` | `true` | BatchNorm eingefrorener Schichten bleibt im Training auf `eval()` |
| `learning_rate` / `backbone_learning_rate` | `1e-4` / `1e-5` | getrennte Lernraten für Kopf und entfrorenen Backbone |
| `optimizer` / `scheduler` | `adamw` / `cosine` | außerdem `adam`, `sgd` bzw. `step`, `plateau`, `none` |
| `image_size` | `512` | Zahl (quadratisch) oder `[Höhe, Breite]` |
| `use_class_weights` | `true` | Loss invers zur Klassenhäufigkeit gewichtet |
| `label_smoothing` | `0.05` | |
| `use_amp` / `grad_clip_max_norm` | `true` / `1.0` | Mixed Precision (nur CUDA), Gradient Clipping |
| `early_stopping_monitor` | `val_loss` | Metrik für Early Stopping **und** `best.pt` |
| `augmentation` | siehe Datei | RandomResizedCrop, Flip, Rotation, ColorJitter, GaussianBlur, RandomErasing |
| `data_dir` | `data_stufe2_512` | Datensatzordner mit `train|val|test / io|nio` |
| `zuschnitt` | `bauteil` | Herkunft des Datensatzes, gelesen von der Rückprojektion |
| `xai.method` | `[integrated_gradients, saliency]` | oder `occlusion` bzw. `all` |
| `xai.rueckprojektion` | `enabled: true` | Stil, Anzahl und Exportbreite der Rückprojektion |
| `seed` / `deterministic` | `123` / `false` | `deterministic: true` = bit-exakt, aber langsamer |

---

## Tests

```bash
pytest tests/ -v
```

| Datei | Tests | Inhalt |
|---|---|---|
| `test_pipeline.py` | 16 | Dataset, DataLoader, Modell, Early Stopping, Trainer, Evaluator, Inferenz |
| `test_edge_cases.py` | 21 | korrupte Bilder, leere Datensätze, Augmentierungen, Backbones, Grenzfälle |
| `test_training_stability.py` | 18 | BatchNorm-Freeze, layer4-Freeze, Optimizer-Gruppen, Scheduler, Schwellenwert |
| `test_xai.py` | 29 | IG, Saliency, Occlusion, Visualisierung, XAI-Kennzahlen |
| `test_rueckprojektion.py` | 36 | Skalierung, Rückprojektion, Überlagerung, Zuschnittplan |
| `test_xai_auswahl.py` | 31 | Bildauswahl und Manifest für die XAI-Auswertung |

---

## Monitoring

```bash
tensorboard --logdir outputs/runs/
```

---

## Methodische Hinweise

- **`pos_label = 1 = nio`** in allen sklearn-Metriken – die industriell relevante Fehlerklasse.
- **Schwellenwert:** wird ausschließlich auf val bestimmt (F1-Maximum) und unverändert auf test
  angewendet (`metrics/threshold.json`). Der Versuchsplan wählt zusätzlich bei fester Schwelle 0,5
  aus, weil sich der val-Betriebspunkt nachweislich nicht auf test überträgt.
- **Reproduzierbarkeit:** Jeder Lauf sichert seine `config.yaml`; `class_info.json` dokumentiert
  Klassenverteilung und Gewichte. Seeds gelten für Python, NumPy, PyTorch und DataLoader-Worker.
- **BatchNorm:** Ohne `freeze_bn_stats` driften `running_mean`/`running_var` eingefrorener Schichten
  trotz `requires_grad=False` weiter – sichtbar als springende Validierungskurve.
