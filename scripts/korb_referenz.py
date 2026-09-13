"""
Gitterkorb-Referenzbild und Hintergrundabzug
============================================
Kamera und Gitterkorb stehen fest, nur das Bauteil wechselt Position. Der
pixelweise Median über viele Aufnahmen ist deshalb der *leere Korb* – ein
Leerbild muss nicht extra aufgenommen werden. Die Differenz zwischen Aufnahme
und Referenz enthält dann praktisch nur noch das Bauteil.

Gemessen an data_stufe2 löscht sich das Gitter dabei fast vollständig aus:
der Median der Absolutdifferenz zweier Aufnahmen liegt bei 1 Grauwert.
Übrig bleiben Glanzlichter auf dem polierten Draht (die je nach
Bauteilposition anders funkeln) und der Schatten des Bauteils auf dem Gitter.
Beides wird morphologisch abgefangen.

Warum das nötig ist: Ein Bauteil belegt nur 0,3–2,5 % der Bildfläche. Beim
Skalieren von 4096×3000 auf die CNN-Eingangsgröße schrumpft es auf wenige
Dutzend Pixel, während die Gitterteilung unter die Nyquist-Grenze fällt und
ein kräftiges Moiré über das ganze Bild legt. Das Gitter ist damit das
kontrastreichste Signal im Bild – das Netz greift zwangsläufig danach.

Mehrere Referenzen statt einer: Der Korb steht nicht die ganze Session über
exakt gleich – beim Bestücken wird er gelegentlich verschoben und nimmt dann
eine neue Stellung ein. Referenzen werden deshalb je Zeitfenster gebildet
(Default 10 min) und decken so die vorkommenden Korbstellungen ab. Beim
Anwenden wird nicht die zeitlich passende, sondern die am besten passende
Referenz gewählt (kleinste Restdifferenz) – nie nach Klasse, sonst wanderte
das Label in die Vorverarbeitung. Gemessen: bei verschobenem Korb sinkt die
Restdifferenz dadurch von 24 auf 1 Grauwert.

Ablauf:
    1. erstellen  – Referenzbilder je Zeitfenster aus einem Bilderordner
    2. anwenden   – Differenz, Bauteilmaske, Zuschnitt in Originalauflösung

Zuschnittformat: --quadratisch liefert quadratische Ausschnitte. Das ist für
das anschließende Skalieren auf image_size wichtig – die Bauteil-Boxen haben
Seitenverhältnisse bis 4,9, und ein direktes Quetschen aufs Quadrat würde
jedes Bild anders verzerren. Das Quadrat wird mit echtem Hintergrund gefüllt
(Fenster wird bei Bildrand nach innen geschoben), nicht mit schwarzen Balken:
künstliche Ränder wären für ein CNN ein eigenes Merkmal.

Der Eingabeordner wird ausschließlich gelesen; alle Ausgaben landen daneben.

Voraussetzungen:
    pip install numpy opencv-python

Aufruf:
    python scripts/korb_referenz.py erstellen data_stufe2
    python scripts/korb_referenz.py anwenden  data_stufe2 --limit 20 --vorschau
    python scripts/korb_referenz.py anwenden  data_stufe2 --quadratisch
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Die Zuschnittgeometrie liegt seit 19.08.2026 in src/preprocessing/korb_zoom.py, damit
# Datensatzerzeugung (dieses Skript) und Inferenz (scripts/pipeline_original.py) dieselbe
# Funktion benutzen und nicht auseinanderdriften koennen. Hier bleibt nur, was allein zur
# Referenzerstellung gehoert.
from src.preprocessing.korb_zoom import (  # noqa: E402
    MAX_RESTDIFF,
    SUCHBEREICH,
    bauteil_maske,
    beste_referenz,
    lade_referenzen,
    zuschnitt_fenster,
)

# ──────────────────────────────────────────────────────────────
#  KONFIGURIERBARE PARAMETER
# ──────────────────────────────────────────────────────────────

FENSTER_MIN      = 10    # Breite eines Zeitfensters in Minuten
BILDER_PRO_REF   = 41    # Aufnahmen je Referenz (ungerade → echter Median ohne Mittelung)
STREIFEN_HOEHE   = 250   # zeilenweise Medianbildung, begrenzt den Speicherbedarf

ZEIT_MUSTER = re.compile(r"_(\d{8})_(\d{2})(\d{2})(\d{2})_")


# ──────────────────────────────────────────────────────────────
#  HILFSFUNKTIONEN
# ──────────────────────────────────────────────────────────────

def aufnahmezeit(pfad: Path) -> int | None:
    """Sekunden seit Mitternacht aus dem Dateinamen, None wenn kein Zeitstempel."""
    m = ZEIT_MUSTER.search(pfad.name)
    if not m:
        return None
    _, h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s)


def hhmmss(sekunden: int) -> str:
    return f"{sekunden // 3600:02d}{sekunden % 3600 // 60:02d}{sekunden % 60:02d}"


def bilder_sammeln(ordner: Path) -> list[tuple[int, Path]]:
    """Alle PNGs mit Zeitstempel, nach Aufnahmezeit sortiert."""
    treffer, ohne_zeit = [], 0
    for p in sorted(ordner.rglob("*.png")):
        t = aufnahmezeit(p)
        if t is None:
            ohne_zeit += 1
        else:
            treffer.append((t, p))
    if ohne_zeit:
        print(f"  Hinweis: {ohne_zeit} Datei(en) ohne Zeitstempel im Namen – übersprungen")
    treffer.sort()
    return treffer


def median_bild(pfade: list[Path]) -> np.ndarray:
    """Pixelweiser Median über die Bilder, streifenweise gegen Speicherdruck.

    np.partition arbeitet direkt auf uint8 – anders als np.median, das eine
    float64-Kopie des gesamten Stapels anlegt.
    """
    def lesen(p: Path) -> np.ndarray | None:
        return cv2.imread(str(p), cv2.IMREAD_COLOR)

    with ThreadPoolExecutor(max_workers=8) as ex:
        bilder = [b for b in ex.map(lesen, pfade) if b is not None]
    if not bilder:
        raise RuntimeError("kein Bild lesbar")

    formen = {b.shape for b in bilder}
    if len(formen) > 1:
        raise RuntimeError(f"uneinheitliche Bildgrößen: {formen}")

    stapel = np.stack(bilder)
    del bilder
    mitte = len(stapel) // 2
    h = stapel.shape[1]
    referenz = np.empty(stapel.shape[1:], dtype=np.uint8)
    for y in range(0, h, STREIFEN_HOEHE):
        y1 = min(y + STREIFEN_HOEHE, h)
        referenz[y:y1] = np.partition(stapel[:, y:y1], mitte, axis=0)[mitte]
    return referenz



# ──────────────────────────────────────────────────────────────
#  BEFEHL: erstellen
# ──────────────────────────────────────────────────────────────

def befehl_erstellen(args) -> int:
    eingabe = Path(args.eingabe)
    if not eingabe.is_dir():
        sys.exit(f"Eingabeordner nicht gefunden: {eingabe}")
    ausgabe = Path(args.ausgabe) if args.ausgabe else eingabe.parent / (eingabe.name + "_referenz")
    ausgabe.mkdir(parents=True, exist_ok=True)

    bilder = bilder_sammeln(eingabe)
    if not bilder:
        sys.exit(f"Keine PNGs mit Zeitstempel in {eingabe}")
    print(f"{len(bilder)} Aufnahmen von {hhmmss(bilder[0][0])} bis {hhmmss(bilder[-1][0])}")

    fenster_sek = args.fenster * 60
    gruppen: dict[int, list[tuple[int, Path]]] = {}
    for t, p in bilder:
        gruppen.setdefault(t // fenster_sek, []).append((t, p))

    index, t_start = [], time.time()
    for nr, (schluessel, gruppe) in enumerate(sorted(gruppen.items()), 1):
        von, bis = gruppe[0][0], gruppe[-1][0]
        # gleichmäßig über das Fenster verteilt auswählen: streut die
        # Bauteilpositionen maximal, damit kein Bauteil im Median überlebt
        anzahl = min(args.pro_fenster, len(gruppe))
        idx = np.linspace(0, len(gruppe) - 1, anzahl).round().astype(int)
        auswahl = [gruppe[i][1] for i in sorted(set(idx.tolist()))]

        if len(auswahl) < args.min_bilder:
            print(f"  [{nr}/{len(gruppen)}] {hhmmss(von)}–{hhmmss(bis)}: nur {len(auswahl)} Bilder "
                  f"(< {args.min_bilder}) – Fenster übersprungen")
            continue

        t0 = time.time()
        referenz = median_bild(auswahl)
        datei = f"referenz_{hhmmss(schluessel * fenster_sek)}.png"
        cv2.imwrite(str(ausgabe / datei), referenz)

        klassen = sorted({p.parent.name for _, p in gruppe})
        index.append({
            "datei": datei,
            "von_sek": von, "bis_sek": bis,
            "von": hhmmss(von), "bis": hhmmss(bis),
            "bilder_im_fenster": len(gruppe),
            "bilder_fuer_median": len(auswahl),
            "klassen_im_fenster": klassen,
        })
        print(f"  [{nr}/{len(gruppen)}] {hhmmss(von)}–{hhmmss(bis)}: "
              f"{len(auswahl)} von {len(gruppe)} Bildern → {datei}  ({time.time() - t0:.1f} s)"
              f"   Klassen: {', '.join(klassen)}")

    if not index:
        sys.exit("Kein Fenster hatte genug Bilder – --fenster erhöhen oder --min-bilder senken")

    (ausgabe / "index.json").write_text(json.dumps({
        "quelle": str(eingabe),
        "fenster_minuten": args.fenster,
        "bilder_pro_referenz": args.pro_fenster,
        "referenzen": index,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nFertig: {len(index)} Referenzbilder in {(time.time() - t_start) / 60:.1f} min")
    print(f"  Ausgabe: {ausgabe}")

    rein = [r for r in index if len(r["klassen_im_fenster"]) == 1]
    if rein:
        print(f"\n  ACHTUNG: {len(rein)} von {len(index)} Zeitfenstern enthalten nur eine Klasse.")
        print("  Die Klassen sind also zeitlich getrennt aufgenommen – jede Drift über die")
        print("  Session hinweg korreliert damit perfekt mit dem Label.")
    return 0


# ──────────────────────────────────────────────────────────────
#  BEFEHL: anwenden
# ──────────────────────────────────────────────────────────────

def vorschau_bild(bild, referenz, maske, kasten, bereich, pfad: Path) -> None:
    """Montage: Original | Differenz | Freistellung mit Such- und Zuschnittrahmen."""
    d = cv2.cvtColor(cv2.absdiff(bild, referenz), cv2.COLOR_BGR2GRAY)
    frei = bild.copy()
    frei[maske == 0] = 0
    sx, sy, sb, sh = bereich
    cv2.rectangle(frei, (sx, sy), (sx + sb, sy + sh), (0, 190, 0), 4)
    if kasten[2]:
        x, y, b, h = kasten
        cv2.rectangle(frei, (x, y), (x + b, y + h), (0, 0, 255), 8)
    breite = 640
    hoehe = int(bild.shape[0] * breite / bild.shape[1])
    kacheln = [cv2.resize(im, (breite, hoehe), interpolation=cv2.INTER_AREA) for im in
               (bild, cv2.cvtColor(d, cv2.COLOR_GRAY2BGR), frei)]
    cv2.imwrite(str(pfad), cv2.hconcat(kacheln))


def befehl_anwenden(args) -> int:
    eingabe = Path(args.eingabe)
    if not eingabe.is_dir():
        sys.exit(f"Eingabeordner nicht gefunden: {eingabe}")
    ref_dir = Path(args.referenz) if args.referenz else eingabe.parent / (eingabe.name + "_referenz")
    try:
        referenzen, index = lade_referenzen(ref_dir)
    except (FileNotFoundError, RuntimeError) as fehler:
        sys.exit(str(fehler))
    print(f"{len(index)} Referenzbilder geladen aus {ref_dir}")

    ausgabe = Path(args.ausgabe) if args.ausgabe else eingabe.parent / (eingabe.name + "_freigestellt")
    out_crop = ausgabe / "zuschnitt"
    out_maske = ausgabe / "masken"
    out_vorschau = ausgabe / "vorschau" if args.vorschau else None

    bilder = bilder_sammeln(eingabe)
    if args.limit:
        # gleichmäßig über den ganzen Zeitraum, nicht nur die ersten N
        idx = np.linspace(0, len(bilder) - 1, args.limit).round().astype(int)
        bilder = [bilder[i] for i in sorted(set(idx.tolist()))]
    print(f"{len(bilder)} Bilder werden verarbeitet\n")

    bereich = tuple(int(v) for v in args.suchbereich.split(",")) if args.suchbereich else SUCHBEREICH
    print(f"Suchbereich (x, y, b, h): {bereich}\n")

    protokoll, fehlschlaege, t_start = [], [], time.time()
    for i, (zeit, pfad) in enumerate(bilder, 1):
        rel = pfad.relative_to(eingabe)
        bild = cv2.imread(str(pfad), cv2.IMREAD_COLOR)
        if bild is None:
            fehlschlaege.append((str(rel), "nicht lesbar"))
            continue

        r, restdiff = beste_referenz(bild, referenzen, bereich)
        if restdiff > MAX_RESTDIFF:
            # Korb steht in einer Stellung, für die keine Referenz existiert –
            # die Differenz wäre voller Gitter und der Zuschnitt wertlos
            fehlschlaege.append((str(rel), f"keine passende Referenz (Restdifferenz {restdiff:.0f})"))
            maske = np.zeros(bild.shape[:2], np.uint8)
            x = y = b = h = 0
            fuellgrad = 0.0
        else:
            maske, (x, y, b, h), fuellgrad = bauteil_maske(bild, referenzen[r], bereich)

        seite = anteil = 0
        x0 = y0 = x1 = y1 = 0
        if b == 0:
            if restdiff <= MAX_RESTDIFF:
                fehlschlaege.append((str(rel), "kein Bauteil gefunden"))
            grund = "KEIN BAUTEIL"
        else:
            x0, y0, x1, y1 = zuschnitt_fenster(bild.shape[:2], (x, y, b, h), args.quadratisch)
            for ziel, inhalt in ((out_crop, bild), (out_maske, maske)):
                (ziel / rel.parent).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(ziel / rel), inhalt[y0:y1, x0:x1])
            seite = x1 - x0
            # Anteil, den das Bauteil im fertigen Zuschnitt einnimmt – bei
            # länglichen Teilen füllt das Quadrat sich überwiegend mit Gitter
            anteil = 100 * float(np.count_nonzero(maske[y0:y1, x0:x1])) / ((y1 - y0) * (x1 - x0))
            grund = ""

        protokoll.append({
            "datei": str(rel), "klasse": pfad.parent.name, "zeit": hhmmss(zeit),
            "referenz": index[r]["datei"], "restdifferenz": round(restdiff, 1),
            "x": x, "y": y, "breite": b, "hoehe": h,
            "zuschnitt_seite": seite,
            # Das tatsaechlich geschnittene Fenster. Ohne diese vier Spalten muesste jede
            # Rueckprojektion es aus Box und Bildgroesse neu ableiten - rekonstruierbar,
            # aber eine zweite Wahrheit neben der geschriebenen Datei.
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "bauteil_im_zuschnitt_prozent": round(anteil, 2),
            "flaeche_prozent": round(100 * float(np.count_nonzero(maske)) / maske.size, 3),
            "fuellgrad": round(fuellgrad, 3), "hinweis": grund,
        })

        if out_vorschau:
            (out_vorschau / rel.parent).mkdir(parents=True, exist_ok=True)
            vorschau_bild(bild, referenzen[r], maske, (x, y, b, h), bereich, out_vorschau / rel)

        if i % 25 == 0 or i == len(bilder):
            print(f"  [{i}/{len(bilder)}]  {(time.time() - t_start) / i:.2f} s/Bild"
                  f"   Fehlschläge: {len(fehlschlaege)}", flush=True)

    ausgabe.mkdir(parents=True, exist_ok=True)
    with (ausgabe / "protokoll.csv").open("w", newline="", encoding="utf-8") as f:
        schreiber = csv.DictWriter(f, fieldnames=list(protokoll[0].keys()), delimiter=";")
        schreiber.writeheader()
        schreiber.writerows(protokoll)

    gefunden = [p for p in protokoll if p["breite"]]
    print(f"\nFertig in {(time.time() - t_start) / 60:.1f} min")
    print(f"  Bauteil gefunden: {len(gefunden)}/{len(protokoll)} "
          f"({100 * len(gefunden) / max(len(protokoll), 1):.1f} %)")
    if gefunden:
        breiten = np.array([p["breite"] for p in gefunden])
        hoehen = np.array([p["hoehe"] for p in gefunden])
        flaechen = np.array([p["flaeche_prozent"] for p in gefunden])
        restdiff = np.array([p["restdifferenz"] for p in gefunden])
        print(f"  Bauteil-Box:    {breiten.min()}–{breiten.max()} × {hoehen.min()}–{hoehen.max()} px "
              f"(Median {int(np.median(breiten))} × {int(np.median(hoehen))})")
        print(f"  Bauteilfläche:  {flaechen.min():.2f}–{flaechen.max():.2f} % "
              f"(Median {np.median(flaechen):.2f} %)")
        print(f"  Restdifferenz:  Median {np.median(restdiff):.1f}, max {restdiff.max():.0f} Grauwerte")

        if args.quadratisch:
            seiten = np.array([p["zuschnitt_seite"] for p in gefunden])
            anteile = np.array([p["bauteil_im_zuschnitt_prozent"] for p in gefunden])
            print(f"  Quadratzuschnitt: {seiten.min()}–{seiten.max()} px "
                  f"(Median {int(np.median(seiten))}) – Skalenspreizung "
                  f"{np.percentile(seiten, 95) / np.percentile(seiten, 5):.1f}× (5.–95. Perz.)")
            print(f"  Bauteil füllt davon: Median {np.median(anteile):.1f} %, "
                  f"5. Perzentil {np.percentile(anteile, 5):.1f} % (Rest ist Gitter)")

        # Bauteile am Rand des Suchbereichs sind womöglich angeschnitten. Am
        # unteren Bildrand ist das die Kamera, nicht der Suchbereich – nur die
        # drei künstlichen Kanten melden.
        sx, sy, sb, sh = bereich
        H, W = referenzen[0].shape[:2]
        angeschnitten = [p for p in gefunden
                         if (sx > 0 and p["x"] <= sx + 5)
                         or (sy > 0 and p["y"] <= sy + 5)
                         or (sx + sb < W and p["x"] + p["breite"] >= sx + sb - 5)
                         or (sy + sh < H and p["y"] + p["hoehe"] >= sy + sh - 5)]
        if angeschnitten:
            print(f"  ACHTUNG: {len(angeschnitten)} Bauteil(e) berühren die Grenze des "
                  f"Suchbereichs – ggf. angeschnitten, --suchbereich vergrößern")
    print(f"  Zuschnitte: {out_crop}")
    print(f"  Masken:     {out_maske}")
    if out_vorschau:
        print(f"  Vorschau:   {out_vorschau}")
    print(f"  Protokoll:  {ausgabe / 'protokoll.csv'}")
    if fehlschlaege:
        print(f"\n{len(fehlschlaege)} Fehlschlag/Fehlschläge (erste 15):")
        for name, grund in fehlschlaege[:15]:
            print(f"    {name}: {grund}")
    return 0


# ──────────────────────────────────────────────────────────────
#  HAUPTPROGRAMM
# ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gitterkorb als Referenzbild abziehen und Bauteil freistellen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Aufruf:")[-1])
    unter = parser.add_subparsers(dest="befehl", required=True)

    p1 = unter.add_parser("erstellen", help="Referenzbilder je Zeitfenster berechnen")
    p1.add_argument("eingabe", help="Bilderordner (wird nur gelesen, rekursiv)")
    p1.add_argument("--ausgabe", default=None, help="Zielordner (Default: <eingabe>_referenz)")
    p1.add_argument("--fenster", type=int, default=FENSTER_MIN,
                    help=f"Breite eines Zeitfensters in Minuten (Default: {FENSTER_MIN})")
    p1.add_argument("--pro-fenster", type=int, default=BILDER_PRO_REF, dest="pro_fenster",
                    help=f"Aufnahmen je Referenz (Default: {BILDER_PRO_REF})")
    p1.add_argument("--min-bilder", type=int, default=15, dest="min_bilder",
                    help="Fenster mit weniger Bildern überspringen (Default: 15)")
    p1.set_defaults(funktion=befehl_erstellen)

    p2 = unter.add_parser("anwenden", help="Hintergrund abziehen, Bauteil zuschneiden")
    p2.add_argument("eingabe", help="Bilderordner (wird nur gelesen, rekursiv)")
    p2.add_argument("--referenz", default=None, help="Referenzordner (Default: <eingabe>_referenz)")
    p2.add_argument("--ausgabe", default=None, help="Zielordner (Default: <eingabe>_freigestellt)")
    p2.add_argument("--limit", type=int, default=None,
                    help="nur N Bilder, gleichmäßig über den Zeitraum verteilt (Testlauf)")
    p2.add_argument("--suchbereich", default=None, metavar="X,Y,B,H",
                    help=f"Korbboden im Originalbild (Default: {','.join(map(str, SUCHBEREICH))})")
    p2.add_argument("--quadratisch", action="store_true",
                    help="quadratische Zuschnitte statt Bounding-Box-Format – verhindert, dass "
                         "das spätere Skalieren auf image_size jedes Bild anders verzerrt")
    p2.add_argument("--vorschau", action="store_true",
                    help="Montagen Original|Differenz|Freistellung mitschreiben")
    p2.set_defaults(funktion=befehl_anwenden)

    args = parser.parse_args()
    return args.funktion(args)


if __name__ == "__main__":
    raise SystemExit(main())
