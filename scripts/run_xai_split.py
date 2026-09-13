#!/usr/bin/env python3
"""
scripts/run_xai_split.py

Erzeugt XAI-Heatmaps für einen beliebigen Split (val | test) aus einem BEREITS
abgeschlossenen Run — ohne neu zu trainieren. Nutzt die im Run gespeicherte
`config.yaml` und `checkpoints/best.pt`, damit die Heatmaps exakt zu den
Metriken dieses Runs passen.

Hintergrund: `train.py` erklärt nur den einen Split aus `xai.split` und legt die
Bilder immer unter `<run>/xai/integrated_gradients/` ab. Dieses Skript schreibt
standardmäßig nach `<run>/xai/<split>/`, sodass val- und test-Heatmaps
nebeneinander bestehen können.

Zusätzlich entsteht ein `manifest.csv`, das jede Heatmap ihrem Quell-Dateinamen
sowie True-/Pred-Label zuordnet — `run_xai` selbst schreibt nur durchnummerierte
Dateien, was für die Thesis-Auswertung nicht reicht.

Zwei Betriebsarten (--auswahl):

  alle       ALLE Bilder des Splits (bzw. die ersten --n-samples), Dateinamen
             img_0000.png … Das ist das ursprüngliche Verhalten.

  vergleich  Eine kuratierte Auswahl: je Split die ersten und letzten k Bilder
             (laufunabhängig, also über alle Läufe vergleichbar) plus die
             Fehlklassifikationen dieses Laufs, gedeckelt. Sprechende Dateinamen.
             Mit --zufall N tritt an die Stelle von „erste/letzte k“ eine
             Zufallsstichprobe von N Bildern — ebenfalls laufunabhängig und in
             jedem Lauf dieselbe, nur breiter.
             Details und Begründung in scripts/xai_auswahl.py.

Beispiel:
    python scripts/run_xai_split.py --run outputs/runs/<run> --split val
    python scripts/run_xai_split.py --run outputs/runs/<run> --split test \\
        --auswahl vergleich --methoden all --resume
    python scripts/run_xai_split.py --run outputs/runs/<run> --split test \\
        --auswahl vergleich --zufall 44 --max-fehler 6 --methoden all
"""
import argparse
import csv
import logging
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xai_auswahl                                 # noqa: E402
from src.dataset import build_dataloaders          # noqa: E402
from src.models.resnet import build_model          # noqa: E402
from src.xai.integrated_gradients import CLASSES, run_xai  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description="XAI-Heatmaps für einen Split aus einem fertigen Run erzeugen."
    )
    p.add_argument("--run", required=True, help="Run-Verzeichnis (enthält config.yaml + checkpoints/)")
    p.add_argument("--split", default="val", choices=["val", "test"], help="Zu erklärender Split")
    p.add_argument("--checkpoint", default=None, help="Checkpoint (default: <run>/checkpoints/best.pt)")
    p.add_argument("--out", default=None, help="Zielordner (default: <run>/xai/<split>)")
    p.add_argument("--n-samples", default=None,
                   help="Anzahl Bilder überschreiben ('all' = alle). Default: Wert aus der Run-Config")
    p.add_argument("--device", default=None, help="cuda | cpu (default: automatisch)")
    p.add_argument("--resume", action="store_true",
                   help="Bereits vorhandene Heatmaps ueberspringen (abgebrochenen Lauf fortsetzen)")
    p.add_argument("--auswahl", default="alle", choices=["alle", "vergleich"],
                   help="alle = ganzer Split (Default) | vergleich = kuratierte Auswahl")
    p.add_argument("--max-fehler", type=int, default=xai_auswahl.MAX_FEHLER,
                   help="nur --auswahl vergleich: Deckel fuer Fehlklassifikationen")
    p.add_argument("--n-je-klasse", type=int, default=xai_auswahl.N_JE_KLASSE,
                   help="nur --auswahl vergleich: feste Vergleichsbilder je Klasse")
    p.add_argument("--zufall", type=int, default=0, metavar="N",
                   help="nur --auswahl vergleich: N zufaellige Basisbilder statt "
                        "erste/letzte k je Klasse. Fester Seed, in jedem Lauf "
                        "dieselben Bilder. 0 = aus")
    p.add_argument("--gespreizt", action="store_true",
                   help="nur --auswahl vergleich: gleichmaessig verteilte statt "
                        "Anfang/Ende (Anfang und Ende sind Serienaufnahmen desselben Teils)")
    p.add_argument("--methoden", nargs="+", default=None,
                   choices=["integrated_gradients", "saliency", "occlusion", "all"],
                   help="Ueberschreibt xai.method aus der Run-Config")
    return p.parse_args()


def setup_logger(log_path):
    logger = logging.getLogger("xai_split")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def write_manifest(model, loader, device, out_dir, logger):
    """Ordnet img_XXXX.png dem Quellbild zu und hält True-/Pred-Label fest.

    Setzt voraus, dass der Loader NICHT shuffelt (val/test in build_dataloaders):
    dann entspricht der laufende Index exakt der Nummerierung in run_xai.
    """
    samples = loader.dataset.samples
    preds = []
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            preds.extend(model(images.to(device)).argmax(dim=1).cpu().tolist())

    if len(preds) != len(samples):
        raise RuntimeError(f"Vorhersagen ({len(preds)}) != Datensatzgröße ({len(samples)})")

    manifest = os.path.join(out_dir, "manifest.csv")
    n_correct = 0
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["heatmap", "datei", "pfad", "true", "pred", "korrekt"])
        for i, ((path, label), pred) in enumerate(zip(samples, preds)):
            ok = label == pred
            n_correct += ok
            w.writerow([f"img_{i:04d}.png", os.path.basename(path), path,
                        CLASSES[label], CLASSES[pred], int(ok)])

    acc = n_correct / len(samples) if samples else 0.0
    logger.info(f"Manifest geschrieben: {manifest} ({len(samples)} Zeilen, "
                f"{n_correct} korrekt = {acc:.4f} Accuracy)")
    return manifest


def baue_vergleichsauswahl(args, run_dir, loader, out_dir, config, logger):
    """Kuratierte Auswahl: Teilmenge des Splits + sprechende Dateinamen.

    Gibt (loader_ueber_teilmenge, dateinamen) zurueck. run_xai bleibt dafuer
    unveraendert - ein Subset ist fuer sie ein ganz normaler Loader.

    Das Manifest entsteht aus preds/*.csv statt aus einem neuen Vorwaertslauf:
    keine Rechenzeit, und die Werte stimmen garantiert mit den Tabellen des
    Versuchsplans ueberein, statt eine zweite Wahrheitsquelle zu schaffen.
    """
    zeilen = xai_auswahl.lade_preds(run_dir, args.split)
    auswahl = xai_auswahl.waehle_bilder(
        zeilen, args.split, k=args.n_je_klasse, max_fehler=args.max_fehler,
        gespreizt=args.gespreizt, zufall=args.zufall,
    )

    n_fehler_gesamt = len(xai_auswahl.fehler_indizes(zeilen))
    n_fest = sum(a["fest"] for a in auswahl)
    n_fehler_gezeigt = sum(1 for a in auswahl if a["fehler"] and not a["fest"])
    if args.zufall > 0:
        art = f"zufaellig, Seed {xai_auswahl.ZUFALL_SEED}"
    elif args.gespreizt:
        art = "gespreizt"
    else:
        art = "Anfang/Ende"
    logger.info(
        f"Auswahl: {len(auswahl)} Bilder = {n_fest} Basis ({art}) + "
        f"{n_fehler_gezeigt} von {n_fehler_gesamt} Fehlklassifikationen "
        f"(Schwelle {xai_auswahl.SCHWELLE})"
    )

    xai_auswahl.schreibe_manifest(auswahl, zeilen, loader.dataset.samples,
                                  out_dir, os.path.basename(run_dir), args.split)
    logger.info(f"Manifest geschrieben: {os.path.join(out_dir, 'manifest.csv')}")

    teil = Subset(loader.dataset, [a["index"] for a in auswahl])
    # num_workers=0 bewusst: bei ~16 Bildern kostet das spawn von 8 Windows-Workern
    # (Run-Config sagt num_workers: 8) mehr, als die Parallelitaet einbringt.
    teil_loader = DataLoader(teil, batch_size=1, shuffle=False, num_workers=0)

    # Die Loader-Laenge regiert; kein zweiter Ort, an dem eine Anzahl steht.
    config["xai"]["n_samples"] = "all"
    return teil_loader, [a["heatmap"] for a in auswahl]


def main():
    args = parse_args()

    run_dir = args.run
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        sys.exit(f"[FEHLER] Keine config.yaml im Run-Verzeichnis: {cfg_path}")

    ckpt_path = args.checkpoint or os.path.join(run_dir, "checkpoints", "best.pt")
    if not os.path.isfile(ckpt_path):
        sys.exit(f"[FEHLER] Checkpoint nicht gefunden: {ckpt_path}")

    out_dir = args.out or os.path.join(run_dir, "xai", args.split)
    os.makedirs(out_dir, exist_ok=True)

    logger = setup_logger(os.path.join(out_dir, "xai.log"))

    with open(cfg_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    # Split + optionale Overrides in die XAI-Sektion spiegeln
    config.setdefault("xai", {})
    config["xai"]["split"] = args.split
    if args.n_samples is not None:
        config["xai"]["n_samples"] = (
            args.n_samples if args.n_samples.lower() == "all" else int(args.n_samples)
        )
    if args.methoden:
        # 'methods' hat in run_xai Vorrang vor 'method' - beide setzen, damit die
        # geerbte Run-Config nicht doch noch durchschlaegt.
        config["xai"]["methods"] = args.methoden
        config["xai"]["method"] = args.methoden

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    logger.info("=" * 65)
    logger.info(f"XAI-Nachlauf | Run: {run_dir}")
    logger.info(f"Split: {args.split} | Checkpoint: {os.path.basename(ckpt_path)} | Device: {device}")
    logger.info(f"Methoden: {config['xai'].get('method')} | n_samples: {config['xai'].get('n_samples')} "
                f"| ig_steps: {config['xai'].get('ig_steps')} | ig_step_batch: {config['xai'].get('ig_step_batch')}")
    logger.info(f"Ziel: {out_dir}")
    logger.info("=" * 65)

    _, val_loader, test_loader = build_dataloaders(config)
    loader = val_loader if args.split == "val" else test_loader
    logger.info(f"{args.split}: {len(loader.dataset)} Bilder")

    model = build_model(config)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    if "epoch" in ckpt:
        logger.info(f"Checkpoint aus Epoche {ckpt['epoch']}")

    # Die Attribution erklaert immer die VORHERGESAGTE Klasse (run_xai nutzt
    # argmax, entspricht Schwelle 0,5) - nicht die wahre. Fuer die Auswertung
    # festhalten, damit niemand die Bilder als "Erklaerung des Sollzustands" liest.
    logger.info("Attributionsziel: die vorhergesagte Klasse (argmax = Schwelle 0,5)")

    if args.auswahl == "vergleich":
        loader, dateinamen = baue_vergleichsauswahl(
            args, run_dir, loader, out_dir, config, logger
        )
    else:
        write_manifest(model, loader, device, out_dir, logger)
        dateinamen = None

    if args.resume:
        n_vorhanden = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
        logger.info(f"Resume-Modus: {n_vorhanden} vorhandene Heatmaps werden uebersprungen")

    logger.info("Starte XAI-Pass…")
    run_xai(model, loader, config, run_dir, logger=logger, out_dir=out_dir,
            skip_existing=args.resume, names=dateinamen)

    n = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
    logger.info("-" * 65)
    logger.info(f"Fertig: {n} Heatmaps in {out_dir}")


if __name__ == "__main__":
    main()
