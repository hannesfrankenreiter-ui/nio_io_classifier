#!/usr/bin/env python3
"""
scripts/verify_zuschnitt.py

Beweist, dass sich die Zoomposition (das Zuschnittfenster im Originalbild) exakt
rekonstruieren laesst - und traegt sie danach in protokoll.csv nach.

Warum das der erste Schritt ist: die gesamte Rueckprojektion einer XAI-Heatmap ins
Originalbild steht und faellt mit dem Fenster (x0, y0, x1, y1). protokoll.csv speichert
bisher nur die Bauteil-Box und die Seitenlaenge, nicht das Fenster selbst. Rekonstruierbar
ist es (zuschnitt_fenster ist eine reine Funktion von Box, Bildgroesse und --quadratisch),
aber ein rekonstruierter Wert ist eine Behauptung, solange er nicht gegen die Wirklichkeit
geprueft ist.

Geprueft wird deshalb pixelweise: Fenster neu berechnen, aus dem Original schneiden, mit
der abgelegten Zuschnittdatei vergleichen. Abbruchkriterium ist eine maximale Abweichung
von 0 - nicht "klein", sondern null.

Zusaetzlich wird der feste Streifenzuschnitt (dataStufe2zugeschnitten) an einer Stichprobe
geprueft. Er ist geometrisch trivial, aber der Eintrag im spaeteren zuschnitt:-Block der
Config soll ebenfalls belegt und nicht geglaubt sein.

Aufruf:
    python scripts/verify_zuschnitt.py                 # alles, schreibt die Spalten nach
    python scripts/verify_zuschnitt.py --limit 100     # schneller Durchlauf
    python scripts/verify_zuschnitt.py --kein-schreiben
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

PROJEKT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJEKT))

from scripts.korb_referenz import zuschnitt_fenster  # noqa: E402

NEUE_SPALTEN = ["x0", "y0", "x1", "y1"]


def pruefe_zeile(zeile: dict, original_dir: Path, zuschnitt_dir: Path) -> dict:
    """Fenster neu berechnen, schneiden, mit der abgelegten Datei vergleichen."""
    rel = Path(*zeile["datei"].replace("\\", "/").split("/"))
    ergebnis = {"datei": zeile["datei"], "abweichung": None, "grund": "", "fenster": None}

    if int(zeile["breite"]) == 0:                      # kein Bauteil -> nichts geschrieben
        ergebnis["grund"] = "kein bauteil"
        return ergebnis

    bild = cv2.imread(str(original_dir / rel), cv2.IMREAD_COLOR)
    if bild is None:
        ergebnis["grund"] = "original nicht lesbar"
        return ergebnis
    gespeichert = cv2.imread(str(zuschnitt_dir / rel), cv2.IMREAD_COLOR)
    if gespeichert is None:
        ergebnis["grund"] = "zuschnitt nicht lesbar"
        return ergebnis

    kasten = (int(zeile["x"]), int(zeile["y"]), int(zeile["breite"]), int(zeile["hoehe"]))
    x0, y0, x1, y1 = zuschnitt_fenster(bild.shape[:2], kasten, quadratisch=True)
    eigen = bild[y0:y1, x0:x1]

    if eigen.shape != gespeichert.shape:
        ergebnis["grund"] = f"Form {eigen.shape} statt {gespeichert.shape}"
        return ergebnis

    ergebnis["abweichung"] = int(
        np.abs(eigen.astype(np.int16) - gespeichert.astype(np.int16)).max())
    ergebnis["fenster"] = (x0, y0, x1, y1)
    # Gegenprobe zur einzigen Fensterangabe, die heute in der CSV steht
    if int(zeile["zuschnitt_seite"]) != x1 - x0:
        ergebnis["grund"] = f"Seite {x1 - x0} statt {zeile['zuschnitt_seite']}"
    return ergebnis


def pruefe_bauteil(args) -> tuple[int, dict]:
    protokoll = PROJEKT / args.protokoll
    if not protokoll.exists():
        sys.exit(f"Protokoll nicht gefunden: {protokoll}")
    zeilen = list(csv.DictReader(protokoll.open(encoding="utf-8"), delimiter=";"))
    if args.limit:
        idx = np.linspace(0, len(zeilen) - 1, args.limit).round().astype(int)
        auswahl = [zeilen[i] for i in sorted(set(idx.tolist()))]
    else:
        auswahl = zeilen
    print(f"Verfahren 'bauteil': {len(auswahl)} von {len(zeilen)} Zeilen aus {protokoll.name}")

    original_dir = PROJEKT / args.original
    zuschnitt_dir = PROJEKT / args.zuschnitt
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        ergebnisse = list(pool.map(
            lambda z: pruefe_zeile(z, original_dir, zuschnitt_dir), auswahl))
    dauer = time.time() - t0

    geprueft = [e for e in ergebnisse if e["abweichung"] is not None and not e["grund"]]
    ohne_bauteil = [e for e in ergebnisse if e["grund"] == "kein bauteil"]
    probleme = [e for e in ergebnisse if e["grund"] and e["grund"] != "kein bauteil"]
    max_abw = max((e["abweichung"] for e in geprueft), default=None)

    print(f"  {len(geprueft)} verglichen, {len(ohne_bauteil)} ohne Bauteil, "
          f"{len(probleme)} Probleme  ({dauer:.1f} s, "
          f"{dauer / max(len(auswahl), 1):.2f} s/Bild)")
    print(f"  maximale Pixelabweichung: {max_abw}")
    for e in probleme[:15]:
        print(f"    {e['datei']}: {e['grund']}")

    fehler = len(probleme) + sum(1 for e in geprueft if e["abweichung"] != 0)
    fenster = {e["datei"]: e["fenster"] for e in geprueft if e["abweichung"] == 0}
    return fehler, fenster


def pruefe_fest(args) -> int:
    """Fester Streifenzuschnitt: Rechteck aus dem Protokoll gegen die abgelegten Bilder."""
    ziel = PROJEKT / args.fest
    protokoll = ziel / "zuschnitt_protokoll.json"
    if not protokoll.exists():
        print(f"Verfahren 'fest': uebersprungen, {protokoll} fehlt")
        return 0

    meta = json.loads(protokoll.read_text(encoding="utf-8"))
    b = meta["bereich"]
    x, y, breite, hoehe = int(b["x"]), int(b["y"]), int(b["breite"]), int(b["hoehe"])
    original_dir = PROJEKT / args.original
    print(f"Verfahren 'fest':   Bereich x={x} y={y} b={breite} h={hoehe} aus {protokoll.name}")

    dateien = sorted(ziel.rglob("*.png"))
    if not dateien:
        print("  keine Bilder gefunden")
        return 1
    idx = np.linspace(0, len(dateien) - 1, min(args.fest_stichprobe, len(dateien)))
    auswahl = [dateien[i] for i in sorted(set(idx.round().astype(int).tolist()))]

    abweichungen, probleme = [], []
    for pfad in auswahl:
        rel = pfad.relative_to(ziel)
        bild = cv2.imread(str(original_dir / rel), cv2.IMREAD_COLOR)
        gespeichert = cv2.imread(str(pfad), cv2.IMREAD_COLOR)
        if bild is None or gespeichert is None:
            probleme.append(f"{rel}: nicht lesbar")
            continue
        eigen = bild[y:y + hoehe, x:x + breite]
        if eigen.shape != gespeichert.shape:
            probleme.append(f"{rel}: Form {eigen.shape} statt {gespeichert.shape}")
            continue
        abweichungen.append(
            int(np.abs(eigen.astype(np.int16) - gespeichert.astype(np.int16)).max()))

    print(f"  {len(abweichungen)} von {len(dateien)} Bildern verglichen, "
          f"maximale Pixelabweichung: {max(abweichungen, default=None)}")
    for p in probleme[:10]:
        print(f"    {p}")
    return len(probleme) + sum(1 for a in abweichungen if a != 0)


def trage_fenster_nach(protokoll: Path, fenster: dict) -> None:
    """x0;y0;x1;y1 in protokoll.csv ergaenzen. Bilder werden nicht angefasst."""
    zeilen = list(csv.DictReader(protokoll.open(encoding="utf-8"), delimiter=";"))
    kopf = list(zeilen[0].keys())
    if any(s in kopf for s in NEUE_SPALTEN):
        print(f"  {protokoll.name} hat die Fensterspalten bereits - nichts zu tun")
        return

    fehlend = [z["datei"] for z in zeilen
               if int(z["breite"]) != 0 and z["datei"] not in fenster]
    if fehlend:
        print(f"  Nachtrag uebersprungen: fuer {len(fehlend)} Zeilen mit Bauteil liegt kein "
              f"geprueftes Fenster vor (--limit war gesetzt?)")
        return

    sicherung = protokoll.with_suffix(".csv.bak")
    shutil.copy(protokoll, sicherung)
    # Die Fenster stehen bei den uebrigen Geometriewerten, 'hinweis' bleibt letzte Spalte
    schnitt = kopf.index("zuschnitt_seite") + 1
    ziel_kopf = kopf[:schnitt] + NEUE_SPALTEN + kopf[schnitt:]
    for z in zeilen:
        x0, y0, x1, y1 = fenster.get(z["datei"]) or (0, 0, 0, 0)
        z.update({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
    with protokoll.open("w", newline="", encoding="utf-8") as f:
        schreiber = csv.DictWriter(f, fieldnames=ziel_kopf, delimiter=";")
        schreiber.writeheader()
        schreiber.writerows(zeilen)
    print(f"  Spalten {', '.join(NEUE_SPALTEN)} ergaenzt in {protokoll.name} "
          f"(Sicherung: {sicherung.name})")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Zuschnittfenster gegen die abgelegten Zuschnitte pixelweise pruefen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Aufruf:")[-1])
    p.add_argument("--protokoll", default="data_stufe2_freigestellt/protokoll.csv")
    p.add_argument("--original", default="data_stufe2")
    p.add_argument("--zuschnitt", default="data_stufe2_freigestellt/zuschnitt")
    p.add_argument("--fest", default="dataStufe2zugeschnitten",
                   help="Ordner des festen Streifenzuschnitts (leer = ueberspringen)")
    p.add_argument("--fest-stichprobe", type=int, default=20, dest="fest_stichprobe")
    p.add_argument("--limit", type=int, default=None,
                   help="nur N Zeilen, gleichmaessig ueber den Datensatz verteilt")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--kein-schreiben", action="store_true", dest="kein_schreiben",
                   help="protokoll.csv nicht um die Fensterspalten ergaenzen")
    args = p.parse_args()

    fehler_bauteil, fenster = pruefe_bauteil(args)
    print()
    fehler_fest = pruefe_fest(args) if args.fest else 0
    print()

    if fehler_bauteil or fehler_fest:
        print(f"FEHLGESCHLAGEN: {fehler_bauteil} Abweichungen bei 'bauteil', "
              f"{fehler_fest} bei 'fest'. Die Geometrie stimmt nicht - nicht weiterbauen.")
        return 1

    print("BESTANDEN: jedes gepruefte Fenster reproduziert den abgelegten Zuschnitt pixelgenau.")
    if not args.kein_schreiben:
        trage_fenster_nach(PROJEKT / args.protokoll, fenster)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
