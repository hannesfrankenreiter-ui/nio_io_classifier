#!/usr/bin/env python3
"""
scripts/pipeline_original.py

Vom Originalbild zur zurueckprojizierten Heatmap - fuer jeden Lauf und jeden Datensatz.

    Original (4096x3000) -> auf das Bauteil zoomen -> Zoomposition merken
    -> verkleinern -> CNN -> Klassifikation -> XAI
    -> Heatmap zurueck ins Original, an genau die Stelle, an der gezoomt wurde

Die Rueckprojektion gehoert zum Zoom. Sie laeuft bei 'bauteil' (das Fenster liegt je
Aufnahme woanders) und bei 'ganz' (das Fenster ist das ganze Bild), nicht bei 'fest': dort
ist der Ausschnitt fuer jede Aufnahme derselbe, die Uebersicht zeigte also bei jedem Bild
denselben Streifen an derselben Stelle und sagte nichts, was der Modelleingang nicht schon
sagt. Solche Laeufe schreiben nur zoom/ - Modelleingang und Karte darauf.

Was hier NICHT geraten wird: wie der Datensatz des Laufs aus den Originalen entstanden
ist. Das sagt der zuschnitt:-Block der Config (verfahren: bauteil | fest | ganz). Kuenftige
Laeufe tragen ihn selbst, weil run_manager.save_config die Config-Datei in den Lauf kopiert;
fuer aeltere Laeufe zeigt --zuschnitt-config auf die Config, mit der trainiert wurde.

Der Block wird beim Start gegen die Wirklichkeit geprueft: die ersten Bilder werden nach
seinen Angaben geschnitten, auf die Groesse der abgelegten Datensatzbilder gebracht und
pixelweise verglichen. Ein falscher Eintrag faellt damit sofort auf, statt still eine
verschobene Heatmap zu erzeugen.

Zwei Kontrollen laufen bei jedem Bild mit:

  Zuschnitt-Gleichheit (das Abnahmekriterium)
      Derselbe Vorwaertslauf noch einmal aus dem abgelegten Datensatzbild. Stimmt er, ist
      der aus dem Original berechnete Zuschnitt exakt der, auf dem trainiert wurde. Beide
      Seiten laufen in derselben Sitzung auf demselben Geraet, Rechengenauigkeit faellt
      also heraus - erwartet wird exakte Gleichheit.

  Abgleich mit preds/example_predictions.csv (nachrichtlich)
      Taugt NICHT als Abnahmekriterium: die Datei ist auf vier Stellen gerundet, und die
      Faltungen liefen auf der GPU mit TF32 (rund zehn Bit Mantisse). Beides zusammen
      begrenzt die erreichbare Uebereinstimmung auf etwa 1e-3, voellig unabhaengig davon,
      ob der Zuschnitt stimmt. Berichtet wird die Abweichung und ob die Klasse gleich
      bleibt.

Aufruf:
    python scripts/pipeline_original.py --run outputs/runs/korb_zuschnitt_512 \
           --zuschnitt-config configs/default.yaml --limit 6

    python scripts/pipeline_original.py --run outputs/runs/Stufe_2/S2_resnet18_vollstaendig_800x320 \
           --zuschnitt-config configs/stufe_2/S2_resnet18_vollstaendig_800x320.yaml --stil demo

    # nur die Klassifikation gegen preds/example_predictions.csv pruefen, ohne Bilder
    python scripts/pipeline_original.py --run outputs/runs/korb_zuschnitt_512 \
           --zuschnitt-config configs/default.yaml --pruefe-lauf
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

PROJEKT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJEKT))

from src.dataset import CLASSES, IODataset  # noqa: E402
from src.inference.lauf import lade_lauf  # noqa: E402
from src.preprocessing.korb_zoom import zuschnittplan_aus_config  # noqa: E402
from src.xai.lauf_bilder import (  # noqa: E402
    KURZ,
    METHODEN,
    als_pil,
    beschriften,
    karten_berechnen,
    modelleingang,
    tafel,
    uebersicht_zeichnen,
)
from src.xai.rueckprojektion import (  # noqa: E402
    anteil_in_maske,
    anzeigeskala,
    karte_ins_original,
    skala,
    STILE,
    ueberlagern,
)

# Abnahmekriterium fuer den Zuschnitt: der Vorwaertslauf aus dem Original muss dem aus dem
# abgelegten Datensatzbild entsprechen. Beide laufen in derselben Sitzung auf demselben
# Geraet, Rechengenauigkeit faellt also heraus - erwartet wird exakte Gleichheit.
ZUSCHNITT_GRENZE = 1e-6


# ──────────────────────────────────────────────────────────────
#  Vorverarbeitung: Zuschnitt genau wie im Training
# ──────────────────────────────────────────────────────────────





def selbsttest(plan, lauf, eintraege, data_dir: Path, anzahl: int = 5) -> dict:
    """Zuschnitt-Block gegen die abgelegten Datensatzbilder pruefen.

    Geschnitten wird nach dem Block, dann auf die Groesse der abgelegten Datei gebracht
    und pixelweise verglichen. Beides ist genau eine bilineare Skalierung mit demselben
    Verfahren, die Bilder muessen also identisch sein - nicht "aehnlich".
    """
    ergebnis = {"geprueft": 0, "max_abweichung": None, "probleme": []}
    for rel, _ in eintraege[:anzahl]:
        gespeichert_pfad = data_dir / rel
        original_pfad = plan.original / rel if plan.original else None
        if original_pfad is None or not original_pfad.exists():
            ergebnis["probleme"].append(f"{rel}: Original nicht gefunden ({original_pfad})")
            continue
        bild = cv2.imread(str(original_pfad), cv2.IMREAD_COLOR)
        gespeichert = cv2.imread(str(gespeichert_pfad), cv2.IMREAD_COLOR)
        if bild is None or gespeichert is None:
            ergebnis["probleme"].append(f"{rel}: nicht lesbar")
            continue

        zoom = plan.fenster(bild)
        if not zoom.auswertbar:
            ergebnis["probleme"].append(f"{rel}: {zoom.hinweis}")
            continue
        x0, y0, x1, y1 = zoom.fenster
        h, w = gespeichert.shape[:2]
        eigen = modelleingang(bild[y0:y1, x0:x1], (h, w))
        abw = int(np.abs(eigen.astype(np.int16) - gespeichert.astype(np.int16)).max())
        ergebnis["geprueft"] += 1
        ergebnis["max_abweichung"] = max(ergebnis["max_abweichung"] or 0, abw)
    return ergebnis


# ──────────────────────────────────────────────────────────────
#  XAI
# ──────────────────────────────────────────────────────────────



# ──────────────────────────────────────────────────────────────
#  Darstellung
# ──────────────────────────────────────────────────────────────







# ──────────────────────────────────────────────────────────────
#  Datensatz des Laufs
# ──────────────────────────────────────────────────────────────

def eintraege_des_laufs(data_dir: Path, split: str) -> list[tuple[Path, str]]:
    """(relativer Pfad, Klasse) in der Reihenfolge von IODataset.

    Iteriert wird ueber den Datensatz des Laufs, nicht ueber den Originalordner: die
    beiden sind nicht deckungsgleich (data/test/nio hat 162 Bilder, data_512/test/nio
    nur 138). Die Reihenfolge - erst alle io, dann alle nio, je alphabetisch - ist
    zugleich der Index in preds/example_predictions.csv.
    """
    ds = IODataset(str(data_dir / split))
    return [(Path(p).relative_to(data_dir), CLASSES[label]) for p, label in ds.samples]


def eintraege_aus_eingabe(eingabe: Path) -> list[tuple[Path, str]]:
    """Freie Bilder: eine Datei oder ein Ordner (rekursiv). Klasse unbekannt."""
    if eingabe.is_file():
        return [(Path(eingabe.name), "")]
    treffer = sorted(p for p in eingabe.rglob("*")
                     if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    return [(p.relative_to(eingabe), "") for p in treffer]


def lade_lauf_vorhersagen(lauf, split: str) -> dict[int, float]:
    """prob_nio je Index aus preds/example_predictions[_val].csv."""
    name = "example_predictions.csv" if split == "test" else f"example_predictions_{split}.csv"
    pfad = lauf.verzeichnis / "preds" / name
    if not pfad.exists():
        return {}
    return {int(z["index"]): float(z["prob_nio"])
            for z in csv.DictReader(pfad.open(encoding="utf-8"))}


# ──────────────────────────────────────────────────────────────
#  Hauptlauf
# ──────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="XAI-Heatmap ins Originalbild zurueckprojizieren",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Aufruf:")[-1])
    p.add_argument("--run", required=True, help="Lauf-Verzeichnis (mit config.yaml, checkpoints/)")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--zuschnitt-config", default=None, dest="zuschnitt_config",
                   help="Config, aus der der zuschnitt:-Block genommen wird "
                        "(fuer Laeufe, die ihn noch nicht selbst tragen)")
    p.add_argument("--verfahren", default=None, choices=["bauteil", "fest", "ganz"])
    p.add_argument("--original", default=None, help="Ordner mit den Originalaufnahmen")
    p.add_argument("--referenzen", default=None, help="Korb-Referenzordner (nur 'bauteil')")
    p.add_argument("--suchbereich", default=None, metavar="X,Y,B,H")
    p.add_argument("--bereich", default=None, metavar="X,Y,B,H", help="nur 'fest'")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--input", default=None, help="freie Bilddatei oder -ordner statt --split")
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--nur", default=None,
                   help="nur diese Dateinamen rendern, komma-getrennt oder als Textdatei "
                        "mit einem Namen je Zeile. Anders als --input bleiben Label und "
                        "Index des Datensatzes erhalten.")
    p.add_argument("--stil", default="lauf", choices=sorted(STILE),
                   help="lauf = Farben wie die Heatmaps im Lauf (Standard), "
                        "thesis/demo = pixelweise Deckkraft auf ungedimmtem Foto")
    p.add_argument("--methoden", default=",".join(METHODEN),
                   help=f"komma-getrennt aus {', '.join(METHODEN)}")
    p.add_argument("--gamma", type=float, default=None, help="Anzeige-Gamma (Stil thesis)")
    p.add_argument("--kappung", type=float, default=None, help="Quantil in %% (Stil demo)")
    p.add_argument("--anzeige-breite", type=int, default=2000, dest="anzeige_breite")
    p.add_argument("--karten", action="store_true", help="Rohkarten als .npz mitschreiben")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--pruefe-lauf", action="store_true", dest="pruefe_lauf",
                   help="nur Vorhersagen gegen preds/ vergleichen, keine Bilder schreiben")
    p.add_argument("--kein-selbsttest", action="store_true", dest="kein_selbsttest")
    args = p.parse_args()

    methoden = [m.strip() for m in args.methoden.split(",") if m.strip()]
    unbekannt = [m for m in methoden if m not in METHODEN]
    if unbekannt:
        sys.exit(f"Unbekannte Methode(n): {unbekannt}. Erlaubt: {', '.join(METHODEN)}")

    # ---------------------------------------------------------- Lauf
    lauf = lade_lauf(args.run, args.checkpoint, args.device, args.threshold)
    print(f"Lauf      {lauf.name}  ({lauf.config['backbone']}, "
          f"freeze={lauf.config['freeze_backbone']}, Eingang "
          f"{lauf.groesse[0]}x{lauf.groesse[1]}, Epoche {lauf.epoche})")
    print(f"Schwelle  {lauf.schwelle:.4f}  aus {lauf.schwelle_quelle}")

    # ---------------------------------------------------------- Zuschnitt
    block = dict(lauf.zuschnitt)
    herkunft = "config.yaml des Laufs" if block else "-"
    if args.zuschnitt_config:
        fremd = yaml.safe_load(Path(args.zuschnitt_config).read_text(encoding="utf-8"))
        if not fremd.get("zuschnitt"):
            sys.exit(f"{args.zuschnitt_config} hat keinen zuschnitt:-Block")
        block, herkunft = dict(fremd["zuschnitt"]), args.zuschnitt_config
    zahlen = lambda t: [int(v) for v in t.split(",")] if t else None  # noqa: E731
    try:
        plan = zuschnittplan_aus_config(
            {"zuschnitt": block}, wurzel=PROJEKT,
            verfahren=args.verfahren, original=args.original, referenzen=args.referenzen,
            suchbereich=zahlen(args.suchbereich), bereich=zahlen(args.bereich))
    except ValueError as fehler:
        sys.exit(str(fehler))
    print(f"Zuschnitt {plan.verfahren}  (aus {herkunft})"
          + (f", Referenzen {plan.referenzen_dir.name}" if plan.verfahren == "bauteil" else "")
          + (f", Bereich {list(plan.bereich)}" if plan.verfahren == "fest" else ""))

    # Siehe Kopf: beim festen Ausschnitt ist das Fenster keine Information, deshalb keine
    # Uebersicht im Original. Der zuschnitt:-Block wird trotzdem gebraucht - aus ihm
    # entsteht der Modelleingang, und der Selbsttest prueft ihn gegen den Datensatz.
    rueckprojektion = plan.verfahren != "fest"

    # ---------------------------------------------------------- Bilder
    data_dir = PROJEKT / lauf.config["data_dir"]
    if args.input:
        eingabe = Path(args.input)
        eintraege = eintraege_aus_eingabe(eingabe)
        quelle_dir = eingabe if eingabe.is_dir() else eingabe.parent
        split = "-"
    else:
        if not data_dir.is_dir():
            sys.exit(f"Datensatz des Laufs nicht gefunden: {data_dir}")
        split = args.split
        eintraege = eintraege_des_laufs(data_dir, split)
        quelle_dir = plan.original
    if not eintraege:
        sys.exit("Keine Bilder gefunden")
    print(f"Bilder    {len(eintraege)} in {quelle_dir}  (Split {split})")

    # ---------------------------------------------------------- Selbsttest
    test = {"uebersprungen": True}
    if not args.kein_selbsttest and not args.input:
        test = selbsttest(plan, lauf, eintraege, data_dir)
        if test["probleme"] or test["max_abweichung"] not in (0, None):
            print(f"\nSELBSTTEST FEHLGESCHLAGEN: der zuschnitt:-Block passt nicht zu "
                  f"{lauf.config['data_dir']}.")
            print(f"  maximale Pixelabweichung: {test['max_abweichung']}")
            for m in test["probleme"][:5]:
                print(f"  {m}")
            print("  Block pruefen (verfahren/original/bereich/referenzen) oder, wenn die "
                  "Abweichung erklaerbar ist, --kein-selbsttest setzen.")
            return 1
        print(f"Selbsttest {test['geprueft']} Bilder geschnitten und mit "
              f"{lauf.config['data_dir']} verglichen: Abweichung {test['max_abweichung']}")

    if args.nur:
        quelle = Path(args.nur)
        namen = {Path(n.strip()).name for n in
                 (quelle.read_text(encoding="utf-8").splitlines() if quelle.exists()
                  else args.nur.split(",")) if n.strip()}
        auswahl = [(i, e) for i, e in enumerate(eintraege) if e[0].name in namen]
        fehlend = namen - {e[0].name for _, e in auswahl}
        if fehlend:
            sys.exit(f"Nicht im Split '{split}' gefunden: {sorted(fehlend)}")
    elif args.limit:
        idx = np.linspace(0, len(eintraege) - 1, args.limit).round().astype(int)
        auswahl = [(i, eintraege[i]) for i in sorted(set(idx.tolist()))]
    else:
        auswahl = list(enumerate(eintraege))

    lauf_preds = lade_lauf_vorhersagen(lauf, split) if not args.input else {}
    if args.pruefe_lauf and not lauf_preds:
        sys.exit(f"Keine Vorhersagen zum Vergleichen in {lauf.verzeichnis / 'preds'}")

    # ---------------------------------------------------------- Ausgabeordner
    aus_dir = Path(args.out) if args.out else PROJEKT / "outputs" / "pipeline" / \
        f"{lauf.name}_{split}"
    if not args.pruefe_lauf:
        if rueckprojektion:
            for m in methoden:
                (aus_dir / "overlay" / m).mkdir(parents=True, exist_ok=True)
        (aus_dir / "zoom").mkdir(parents=True, exist_ok=True)
        if args.karten:
            (aus_dir / "karten").mkdir(parents=True, exist_ok=True)
        print(f"Ausgabe   {aus_dir}")
    print(f"Methoden  {', '.join(methoden)}   Stil {args.stil}"
          + ("" if rueckprojektion else "   (fester Ausschnitt: keine "
                                        "Rueckprojektion, nur zoom/)") + "\n")

    # ---------------------------------------------------------- Durchlauf
    zeilen, abweichungen, zuschnitt_abw, klassen_gleich = [], [], [], []
    t0 = time.time()
    for n, (index, (rel, klasse)) in enumerate(auswahl, 1):
        zeile = {"index": index, "datei": str(rel), "split": split, "klasse_wahr": klasse,
                 "verfahren": plan.verfahren, "hinweis": ""}
        original_pfad = (quelle_dir / rel) if quelle_dir else None
        bild = cv2.imread(str(original_pfad), cv2.IMREAD_COLOR) if original_pfad else None
        if bild is None:
            zeile["hinweis"] = f"Original nicht lesbar ({original_pfad})"
            zeilen.append(zeile)
            continue

        H, W = bild.shape[:2]
        zoom = plan.fenster(bild)
        zeile.update(referenz=zoom.referenz, restdifferenz=round(zoom.restdifferenz, 1))
        if not zoom.auswertbar:
            # Kein Rueckfall auf einen festen Ausschnitt: sonst stuende im Ergebnis ein
            # Zuschnitt, den das Modell nie in dieser Form gesehen hat.
            zeile["hinweis"] = zoom.hinweis
            zeilen.append(zeile)
            continue

        x0, y0, x1, y1 = zoom.fenster
        zuschnitt = bild[y0:y1, x0:x1]
        sx, sy = skala(zoom.fenster, lauf.groesse)
        zeile.update(x0=x0, y0=y0, x1=x1, y1=y1, breite=x1 - x0, hoehe=y1 - y0,
                     px_je_kartenpixel_x=round(sx, 3), px_je_kartenpixel_y=round(sy, 3))

        tensor = lauf.transform(als_pil(zuschnitt)).unsqueeze(0).to(lauf.device)
        with torch.no_grad():
            wahrsch = torch.softmax(lauf.model(tensor), dim=1).squeeze(0)
        p_io, p_nio = float(wahrsch[0]), float(wahrsch[1])
        vorhersage = "nio" if p_nio >= lauf.schwelle else "io"
        zeile.update(prob_io=round(p_io, 6), prob_nio=round(p_nio, 6),
                     schwelle=round(lauf.schwelle, 6), vorhersage=vorhersage,
                     korrekt=int(vorhersage == klasse) if klasse else "")

        # Der entscheidende Nachweis: derselbe Vorwaertslauf aus dem abgelegten
        # Datensatzbild. Stimmt er, ist der aus dem Original berechnete Zuschnitt exakt
        # der, auf dem trainiert wurde - unabhaengig von Rechengenauigkeit und Geraet,
        # weil beide Seiten in derselben Sitzung auf demselben Geraet laufen.
        datensatz_pfad = (data_dir / rel) if not args.input else None
        if datensatz_pfad is not None and datensatz_pfad.exists():
            t_ref = lauf.transform(
                Image.open(datensatz_pfad).convert("RGB")).unsqueeze(0).to(lauf.device)
            with torch.no_grad():
                p_ref = float(torch.softmax(lauf.model(t_ref), dim=1)[0, 1])
            zuschnitt_abw.append(abs(p_ref - p_nio))
            zeile["bitgleich_zum_datensatz"] = int(torch.equal(tensor, t_ref))
            zeile["abweichung_zuschnitt"] = round(abs(p_ref - p_nio), 10)

        if index in lauf_preds:
            abw = abs(lauf_preds[index] - p_nio)
            abweichungen.append(abw)
            klasse_lauf = "nio" if lauf_preds[index] >= lauf.schwelle else "io"
            klassen_gleich.append(klasse_lauf == vorhersage)
            zeile["abweichung_zum_lauf"] = round(abw, 8)

        if args.pruefe_lauf:
            zeilen.append(zeile)
            if n % 25 == 0 or n == len(auswahl):
                print(f"  [{n}/{len(auswahl)}]  {(time.time() - t0) / n:.2f} s/Bild", flush=True)
            continue

        # ------------------------------------------------------ XAI
        ziel = 1 if vorhersage == "nio" else 0
        karten = karten_berechnen(lauf.model, lauf.xai, tensor, ziel, methoden)
        eingang = modelleingang(zuschnitt, lauf.groesse)
        zoom_tafeln = [beschriften(eingang, f"Modelleingang {lauf.groesse[1]}x{lauf.groesse[0]}")]

        for m in methoden:
            karte, leer = karten[m]["karte"], karten[m]["leer"]
            zeile[f"{m}_leer"] = int(leer)
            # Attributionsanteil auf dem Bauteil, im Fenster gerechnet: der Flaechenanteil
            # bezoege sich sonst auf das ganze Bild und waere nicht vergleichbar.
            if zoom.maske is not None:
                gross = karte_ins_original(karte, zoom.fenster, (H, W))
                anteil, flaeche = anteil_in_maske(gross[y0:y1, x0:x1], zoom.maske[y0:y1, x0:x1])
                zeile[f"{m}_anteil_bauteil"] = "" if np.isnan(anteil) else round(anteil, 4)
                zeile["maskenflaeche"] = round(flaeche, 4)
            else:
                zeile[f"{m}_anteil_bauteil"] = ""

            anzeige, kennwert = anzeigeskala(karte, args.stil, args.gamma, args.kappung)
            zeile[f"{m}_kennwert"] = round(kennwert, 4)
            zoom_tafeln.append(beschriften(
                ueberlagern(eingang, anzeige, args.stil),
                f"{KURZ[m]}" + ("  (leer)" if leer else "")))

            if leer or not rueckprojektion:
                # Leere Karte: ein schwarzes Overlay als "Ergebnis" waere eine Luege, der
                # Befund steht in der CSV. Fester Ausschnitt: die Uebersicht entfaellt
                # (siehe Kopf), die Karte steht auf dem Modelleingang in der zoom/-Tafel.
                continue
            titel = (f"{rel.name}   {vorhersage.upper()}  p(nio)={p_nio:.3f}  "
                     f"(tau={lauf.schwelle:.3f})   {KURZ[m]}")
            uebersicht, _ = uebersicht_zeichnen(bild, karte, zoom, args.stil, args.gamma,
                                                args.kappung, args.anzeige_breite, titel)
            ziel_datei = aus_dir / "overlay" / m / rel.with_suffix(".jpg")
            ziel_datei.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ziel_datei), uebersicht, [cv2.IMWRITE_JPEG_QUALITY, 90])

        zoom_datei = aus_dir / "zoom" / rel.with_suffix(".jpg")
        zoom_datei.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(zoom_datei), tafel(zoom_tafeln), [cv2.IMWRITE_JPEG_QUALITY, 92])

        if args.karten:
            karten_datei = aus_dir / "karten" / rel.with_suffix(".npz")
            karten_datei.parent.mkdir(parents=True, exist_ok=True)
            inhalt = {m: karten[m]["karte"] for m in methoden}
            if karten.get("integrated_gradients", {}).get("roh") is not None:
                inhalt["integrated_gradients_signiert"] = karten["integrated_gradients"]["roh"]
            np.savez_compressed(karten_datei, **inhalt)

        zeilen.append(zeile)
        if n % 10 == 0 or n == len(auswahl):
            print(f"  [{n}/{len(auswahl)}]  {(time.time() - t0) / n:.2f} s/Bild", flush=True)

    # ---------------------------------------------------------- Bilanz
    dauer = time.time() - t0
    auswertbar = [z for z in zeilen if not z["hinweis"]]
    print(f"\nFertig in {dauer / 60:.1f} min  ({dauer / max(len(auswahl), 1):.2f} s/Bild)")
    print(f"  auswertbar: {len(auswertbar)}/{len(zeilen)}")
    for z in [z for z in zeilen if z["hinweis"]][:10]:
        print(f"    {z['datei']}: {z['hinweis']}")

    if zuschnitt_abw:
        groesste = max(zuschnitt_abw)
        bitgleich = sum(z.get("bitgleich_zum_datensatz", 0) for z in auswertbar)
        urteil = "IDENTISCH" if groesste <= ZUSCHNITT_GRENZE else "ABWEICHUNG"
        print(f"  Zuschnitt gegen {lauf.config['data_dir']}: {len(zuschnitt_abw)} Bilder, "
              f"{bitgleich} davon bitgleicher Modelleingang,")
        print(f"    groesste Abweichung in prob_nio {groesste:.2e}  -> {urteil}")
    if abweichungen:
        gleich = sum(klassen_gleich)
        print(f"  Abgleich mit preds/: {len(abweichungen)} Bilder, groesste Abweichung "
              f"in prob_nio {max(abweichungen):.2e}, Klasse gleich in {gleich}/"
              f"{len(klassen_gleich)}")
        print("    (nachrichtlich: preds/ ist auf 4 Stellen gerundet und wurde mit "
              "TF32-Faltungen auf der GPU")
        print("     gerechnet - beides begrenzt die erreichbare Uebereinstimmung auf "
              "etwa 1e-3.)")
    for m in ([] if args.pruefe_lauf else methoden):
        leere = sum(1 for z in auswertbar if z.get(f"{m}_leer"))
        anteile = [z[f"{m}_anteil_bauteil"] for z in auswertbar
                   if isinstance(z.get(f"{m}_anteil_bauteil"), float)]
        text = f"  {KURZ[m]:10s} leere Karten {leere}/{len(auswertbar)}"
        if anteile:
            flaechen = [z["maskenflaeche"] for z in auswertbar if "maskenflaeche" in z]
            text += (f",  Attribution auf dem Bauteil {np.mean(anteile) * 100:.1f} % "
                     f"(Flaechenanteil {np.mean(flaechen) * 100:.1f} %)")
        print(text)

    # ---------------------------------------------------------- Schreiben
    if not zeilen:
        return 0
    aus_dir.mkdir(parents=True, exist_ok=True)
    spalten = list(dict.fromkeys(s for z in zeilen for s in z))
    spalten = [s for s in spalten if s != "hinweis"] + ["hinweis"]
    with (aus_dir / "ergebnisse.csv").open("w", newline="", encoding="utf-8") as f:
        schreiber = csv.DictWriter(f, fieldnames=spalten, delimiter=";", restval="")
        schreiber.writeheader()
        schreiber.writerows(zeilen)

    (aus_dir / "parameter.json").write_text(json.dumps({
        "zeitpunkt": datetime.now().isoformat(timespec="seconds"),
        "lauf": lauf.als_dict(),
        "zuschnitt": dict(plan.als_dict(), herkunft=herkunft),
        "selbsttest": test,
        "split": split, "bilder": len(zeilen), "methoden": methoden,
        "stil": args.stil, "anzeige": dict(STILE[args.stil]),
        "gamma": args.gamma, "kappung": args.kappung,
        "anzeige_breite": args.anzeige_breite,
        "ig_steps": lauf.xai.get("ig_steps", 50),
        "ig_step_batch": lauf.xai.get("ig_step_batch", 8),
        "occ_patch_size": lauf.xai.get("occ_patch_size", 64),
        "groesste_abweichung_zuschnitt": max(zuschnitt_abw) if zuschnitt_abw else None,
        "groesste_abweichung_zum_lauf": max(abweichungen) if abweichungen else None,
        "klasse_wie_lauf": f"{sum(klassen_gleich)}/{len(klassen_gleich)}"
                           if klassen_gleich else None,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  {aus_dir / 'ergebnisse.csv'}")

    if zuschnitt_abw and max(zuschnitt_abw) > ZUSCHNITT_GRENZE:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
