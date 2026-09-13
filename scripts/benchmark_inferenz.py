#!/usr/bin/env python3
"""
scripts/benchmark_inferenz.py
Misst die reine Inferenzzeit je Backbone unter kontrollierten Bedingungen.

Warum ein eigenes Skript?
-------------------------
Die Spalte "Rechenzeit je Bild" der Versuchsplan-Tabellen stammt aus dem
Evaluator, der sie beilaeufig waehrend der Testauswertung eines Trainingslaufs
erhebt - also zu zwoelf verschiedenen Zeitpunkten, bei unterschiedlichem
GPU-Takt und Lastzustand. Auf einer Laptop-GPU ist das nicht vergleichbar.
Belegt an Werten, die sich physikalisch ausschliessen: dieselbe Architektur
misst je nach Lauf 150,82 / 147,74 / 41,17 ms (ResNet-50) bzw. 108,15 / 22,68 ms
(ConvNeXt-Tiny). Die Freeze-Strategie kann die Inferenzzeit nicht beeinflussen -
unter model.eval() und torch.no_grad() ist requires_grad bedeutungslos und die
Architektur identisch. Es sind also VIER Werte zu messen, nicht zwoelf.

Vorgehen: feste Zahl Aufwaermdurchlaeufe, danach n Wiederholungen, Median statt
Mittelwert (robust gegen einzelne Ausreisser durch Taktwechsel), Angabe der
Streuung. Zwei Betriebsarten:

  Batch  wie im Evaluator (batch_size aus der Config) - vergleichbar mit den
         bisherigen Zahlen.
  Einzel batch_size 1 - der realistische Fall fuer die Sichtpruefung, bei der
         ein Bauteil nach dem anderen bewertet wird.

Aufruf:
    python scripts/benchmark_inferenz.py
    python scripts/benchmark_inferenz.py --wiederholungen 50 --aufwaermen 10
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from run_versuchsplan import BACKBONES, schreibe_tabelle  # noqa: E402
from src.dataset import bildgroesse                       # noqa: E402
from src.models.resnet import build_model                 # noqa: E402


def messe(model, x, wiederholungen: int, aufwaermen: int) -> list:
    """Zeiten je Durchlauf in Sekunden. Aufwaermdurchlaeufe werden verworfen."""
    zeiten = []
    with torch.no_grad():
        for i in range(aufwaermen + wiederholungen):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if i >= aufwaermen:
                zeiten.append(dt)
    return zeiten


def main():
    p = argparse.ArgumentParser(description="Inferenzzeit je Backbone")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--stufe", type=int, default=1, help="Zielordner der Tabelle")
    p.add_argument("--wiederholungen", type=int, default=30)
    p.add_argument("--aufwaermen", type=int, default=10)
    args = p.parse_args()

    with open(os.path.join(PROJECT_ROOT, args.config), encoding="utf-8") as fh:
        basis = yaml.safe_load(fh)

    if not torch.cuda.is_available():
        sys.exit("[FEHLER] Keine CUDA-GPU verfuegbar - der Vergleich waere wertlos.")

    dev = torch.device("cuda")
    hoehe, breite = bildgroesse(basis)
    sz = f"{hoehe}x{breite}"
    bs = basis["batch_size"]

    print("=" * 72)
    print("INFERENZ-BENCHMARK")
    print("=" * 72)
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"Bildgroesse  : {sz}   Batch: {bs}   Praezision: fp32 (wie im Evaluator)")
    print(f"Messung      : {args.aufwaermen} Aufwaermdurchlaeufe, dann "
          f"{args.wiederholungen} Wiederholungen, Median")
    print("Hinweis      : Die Freeze-Strategie hat auf die Inferenzzeit keinen "
          "Einfluss.\n               Gemessen wird deshalb je Architektur, nicht je Lauf.\n")

    eintraege = []
    for bb in BACKBONES:
        cfg = dict(basis, backbone=bb, pretrained=False, freeze_backbone=False)
        model = build_model(cfg).to(dev).eval()

        e = {"backbone": bb}
        for modus, n in [("batch", bs), ("einzel", 1)]:
            x = torch.randn(n, 3, hoehe, breite, device=dev)
            zeiten = messe(model, x, args.wiederholungen, args.aufwaermen)
            je_bild = [1000.0 * t / n for t in zeiten]
            e[f"{modus}_median"] = statistics.median(je_bild)
            e[f"{modus}_min"] = min(je_bild)
            e[f"{modus}_stdev"] = statistics.stdev(je_bild) if len(je_bild) > 1 else 0.0

        e["params"] = sum(q.numel() for q in model.parameters())
        eintraege.append(e)
        print(f"  {bb:<17} Batch {e['batch_median']:>7.2f} ms/Bild   "
              f"Einzel {e['einzel_median']:>7.2f} ms/Bild   "
              f"(Streuung {e['batch_stdev']:.2f} / {e['einzel_stdev']:.2f})")

        del model, x
        torch.cuda.empty_cache()

    def _f(v, nk=2):
        return f"{v:.{nk}f}".replace(".", ",")

    print("\nTabelle:")
    schreibe_tabelle(
        args.stufe, sorted(eintraege, key=lambda e: e["batch_median"]),
        f"Versuchsstufe {args.stufe} – Inferenzzeit je Architektur",
        "inferenzzeit",
        [
            ("Backbone", lambda e: e["backbone"]),
            (f"Batch {bs} [ms/Bild]", lambda e: _f(e["batch_median"])),
            ("Einzelbild [ms]", lambda e: _f(e["einzel_median"])),
            ("Streuung Batch [ms]", lambda e: _f(e["batch_stdev"])),
            ("Schnellster Lauf [ms]", lambda e: _f(e["batch_min"])),
            ("Parameter gesamt",
             lambda e: f"{e['params']:,}".replace(",", ".")),
        ],
        kopfzeilen=[
            f"Median aus {args.wiederholungen} Wiederholungen nach "
            f"{args.aufwaermen} Aufwaermdurchlaeufen, {sz} px (HxB), fp32.  ",
            "**Je Architektur gemessen, nicht je Lauf**: unter `model.eval()` und "
            "`torch.no_grad()` hat die Freeze-Strategie keinen Einfluss auf die "
            "Inferenzzeit.  ",
            "Ersetzt die Spalte „Rechenzeit je Bild\" der Vergleichstabellen, die "
            "beilaeufig waehrend zwoelf verschiedener Trainingslaeufe erhoben wurde "
            "und dadurch ueberwiegend den GPU-Lastzustand misst.",
        ])

    ziel = os.path.join(PROJECT_ROOT, "outputs", "runs", f"Stufe_{args.stufe}",
                        "inferenzzeit.json")
    with open(ziel, "w", encoding="utf-8") as fh:
        json.dump({"gpu": torch.cuda.get_device_name(0), "image_size": sz,
                   "batch_size": bs, "wiederholungen": args.wiederholungen,
                   "aufwaermen": args.aufwaermen, "messungen": eintraege},
                  fh, indent=2, ensure_ascii=False)
    print(f"  Rohdaten → {os.path.relpath(ziel, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
