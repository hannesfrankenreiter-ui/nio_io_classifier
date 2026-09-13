#!/usr/bin/env python3
"""
scripts/xai_auswahl.py
Bestimmt, WELCHE Bilder eines Splits erklaert werden - und schreibt das Manifest.

Warum ueberhaupt eine Auswahl?
------------------------------
`run_xai` erklaert die ERSTEN n_samples Bilder in Loader-Reihenfolge
(src/xai/integrated_gradients.py). Die Klassen liegen im Datensatz sortiert vor
(io zuerst, siehe src/dataset.py), also liefert "n_samples: 20" zwanzig io-Bilder
und kein einziges nio. Fuer "ein paar Bilder je Lauf" ist das unbrauchbar.

Die Auswahl hier besteht aus zwei Teilen:

  BASIS  eine laufunabhaengige Menge von Vergleichsbildern. Zwei Varianten:
         - Anfang/Ende (Default): die ersten k und die letzten k Bilder. Weil die
           Klassen sortiert sind, sind das k io und k nio.
         - Zufall (zufall=n, Stufe 2): n zufaellig gezogene Bilder, klassenweise
           gleich verteilt. Fuer eine breitere Stichprobe als 2k Bilder, ohne
           gleich den ganzen Split (400 Bilder, ~100 min) zu rechnen.
         Beide Varianten sind laufunabhaengig - dasselbe Bauteil liegt in allen
         Laeufen unter demselben Dateinamen und laesst sich damit ueber Backbones
         und Freeze-Strategien hinweg direkt vergleichen.

  FEHLER die Fehlklassifikationen dieses Laufs, gedeckelt auf max_fehler. Die
         zeigen, worauf das Modell schaut, wenn es danebenliegt.

ACHTUNG Schwellenwert
---------------------
Die Spalte "correct" in preds/*.csv gilt fuer den auf VAL bestimmten
Betriebspunkt (z.B. 0,209), nicht fuer 0,5. Der Versuchsplan waehlt aber bei
FESTER Schwelle 0,5 aus. Beispiel S1_01/test: 35 Fehler laut Spalte, aber nur 9
bei 0,5. Deshalb wird hier immer aus prob_nio neu gerechnet und die Spalte
"correct" bewusst ignoriert.

ACHTUNG Serienaufnahmen
-----------------------
Anfang und Ende eines Splits sind Serienaufnahmen desselben Bauteils im
Sekundenabstand (val/io: 31 s ueber die ersten fuenf, val/nio: 20 s ueber die
letzten fuenf). Die zehn festen Bilder zeigen damit faktisch zwei Bauteile.
`gespreizt=True` waehlt stattdessen k gleichmaessig ueber jede Klasse verteilte
Indizes - genauso deterministisch, genauso laufunabhaengig, aber 2k
verschiedene Bauteile. Default ist Anfang/Ende (so entschieden).

Das Modul ist bewusst TORCH-FREI: es rechnet nur auf den bereits abgelegten
Wahrscheinlichkeiten. Dadurch ist es in Millisekunden testbar und der Treiber
kann seinen --dry-run ohne Torch-Import fahren.
"""
import csv
import os
import random

# Entscheidungsschwelle des Versuchsplans. NICHT die val-optimale Schwelle aus
# metrics/threshold.json - siehe Modulkopf.
SCHWELLE = 0.5
N_JE_KLASSE = 5      # feste Vergleichsbilder je Klasse und Split
MAX_FEHLER = 6       # Deckel fuer Fehlklassifikationen je Lauf und Split
N_ZUFALL = 44        # zufaellige Basisbilder je Split (44 + 6 Fehler = 50)
ZUFALL_SEED = 123    # wie config seed - eine Zahl, an der der Zufall haengt

CLASSES = ["io", "nio"]


# ================================================================
#  Einlesen
# ================================================================
def preds_pfad(run_dir: str, split: str) -> str:
    """Der Evaluator legt test unter example_predictions.csv ab, val mit Suffix."""
    name = "example_predictions.csv" if split == "test" else "example_predictions_val.csv"
    return os.path.join(run_dir, "preds", name)


def lade_preds(run_dir: str, split: str) -> list:
    """Liest preds/*.csv. Gibt eine nach 'index' aufsteigend sortierte Liste.

    Der Index ist die Position im unshuffelten Loader und damit direkt der Index
    in dataset.samples - darauf beruht die gesamte Zuordnung.
    """
    pfad = preds_pfad(run_dir, split)
    with open(pfad, encoding="utf-8") as fh:
        roh = list(csv.DictReader(fh))

    zeilen = [{
        "index":    int(z["index"]),
        "true":     z["true_label"],
        "prob_io":  float(z["prob_io"]),
        "prob_nio": float(z["prob_nio"]),
    } for z in roh]
    zeilen.sort(key=lambda z: z["index"])

    erwartet = list(range(len(zeilen)))
    if [z["index"] for z in zeilen] != erwartet:
        raise ValueError(
            f"{pfad}: Indizes sind nicht lueckenlos 0..{len(zeilen) - 1}. "
            "Ohne das stimmt die Zuordnung zu dataset.samples nicht."
        )
    return zeilen


# ================================================================
#  Auswahl
# ================================================================
def _gespreizte_indizes(start: int, n: int, k: int) -> list:
    """k gleichmaessig ueber [start, start+n) verteilte Indizes, Raender inklusive."""
    if k <= 1 or n <= 1:
        return [start] if n else []
    k = min(k, n)
    return [start + round(i * (n - 1) / (k - 1)) for i in range(k)]


def feste_indizes(zeilen: list, k: int = N_JE_KLASSE, gespreizt: bool = False) -> list:
    """Die laufunabhaengigen Vergleichsbilder.

    gespreizt=False (Default): die ersten k und die letzten k des Splits.
    gespreizt=True: k gleichmaessig verteilte Indizes je Klasse.
    """
    n = len(zeilen)
    if n == 0:
        return []

    if not gespreizt:
        k = min(k, n)
        return sorted(set(list(range(k)) + list(range(max(0, n - k), n))))

    # Klassengrenzen aus den Labels ableiten statt anzunehmen - robuster,
    # falls sich die Sortierung des Datensatzes je aendert.
    idx = {c: [z["index"] for z in zeilen if z["true"] == c] for c in CLASSES}
    gewaehlt = []
    for c in CLASSES:
        if idx[c]:
            gewaehlt += _gespreizte_indizes(idx[c][0], len(idx[c]), k)
    return sorted(set(gewaehlt))


def zufalls_indizes(zeilen: list, n: int, split: str,
                    seed: int = ZUFALL_SEED) -> list:
    """n zufaellig gezogene Indizes, klassenweise gleich verteilt.

    LAUFUNABHAENGIG: der Seed haengt allein an (seed, split), nicht am Lauf und
    nicht am Backbone. Damit zeigen alle zwoelf Laeufe einer Stufe dieselben
    Bauteile unter denselben Dateinamen - genau die Eigenschaft, die
    feste_indizes hat, nur mit einer breiteren Stichprobe.

    Der Split geht in den Seed ein, damit val und test nicht dieselben
    Positionen ziehen; die Splits sind verschieden lang und verschieden besetzt,
    identische Positionen waeren reine Zufallskopplung ohne Aussage.

    Klassenweise gezogen statt global, weil ein globaler Griff bei n=44 aus 400
    Bildern die Klassen nur im Mittel ausgleicht - hier sind es garantiert
    n//2 je Klasse (der Rest geht an die groessere Klasse).
    """
    if n <= 0 or not zeilen:
        return []

    idx = {c: [z["index"] for z in zeilen if z["true"] == c] for c in CLASSES}
    # Reihenfolge der Klassen mit den meisten Bildern zuerst: der ungerade Rest
    # landet dort, wo er am wenigsten ins Gewicht faellt.
    rest = n
    gewaehlt = []
    for i, c in enumerate(CLASSES):
        # Was noch offen ist, gleichmaessig auf die noch offenen Klassen. Ein
        # ungerades n gibt das Extrabild der ersten Klasse; hat eine Klasse
        # weniger Bilder als ihr Anteil, faellt der Rest automatisch an die
        # naechste, weil 'rest' erst danach verringert wird.
        offen = len(CLASSES) - i
        anteil = min(len(idx[c]), rest // offen + (1 if rest % offen else 0))
        # Eigener Generator je Klasse: ein gemeinsamer wuerde die Ziehung der
        # zweiten Klasse davon abhaengig machen, wie viele Bilder die erste hat.
        rng = random.Random(f"{seed}|{split}|{c}")
        gewaehlt += rng.sample(idx[c], anteil)
        rest -= anteil
    return sorted(gewaehlt)


def basis_indizes(zeilen: list, k: int = N_JE_KLASSE, gespreizt: bool = False,
                  zufall: int = 0, split: str = "", seed: int = ZUFALL_SEED) -> list:
    """Die laufunabhaengige Basismenge - Zufall schlaegt Anfang/Ende."""
    if zufall > 0:
        return zufalls_indizes(zeilen, zufall, split, seed)
    return feste_indizes(zeilen, k, gespreizt)


def ist_fehler(zeile: dict, schwelle: float = SCHWELLE) -> bool:
    return (zeile["prob_nio"] >= schwelle) != (zeile["true"] == "nio")


def vorhersage(zeile: dict, schwelle: float = SCHWELLE) -> str:
    return "nio" if zeile["prob_nio"] >= schwelle else "io"


def konfidenz(zeile: dict, schwelle: float = SCHWELLE) -> float:
    """Wahrscheinlichkeit der VORHERGESAGTEN Klasse.

    Bei einem Fehler ist das die Sicherheit, mit der das Modell danebenlag - je
    hoeher, desto lehrreicher die Heatmap.
    """
    return zeile["prob_nio"] if vorhersage(zeile, schwelle) == "nio" else zeile["prob_io"]


def fehler_indizes(zeilen: list, schwelle: float = SCHWELLE) -> list:
    return [z["index"] for z in zeilen if ist_fehler(z, schwelle)]


def heatmap_name(split: str, index: int, true: str, fest: bool) -> str:
    """Dateiname der Heatmap.

    Bewusst OHNE die Vorhersage: die variiert je Lauf, und dann laegen dieselben
    festen Bilder in verschiedenen Laeufen unter verschiedenen Namen - genau die
    Vergleichbarkeit waere hin. Auch ohne Quelldateinamen (Pfadlaenge unter
    OneDrive). Beides steht im Manifest.
    """
    return f"{split}_i{index:04d}_{true}_{'fest' if fest else 'fehler'}.png"


def waehle_bilder(zeilen: list, split: str, k: int = N_JE_KLASSE,
                  max_fehler: int = MAX_FEHLER, schwelle: float = SCHWELLE,
                  gespreizt: bool = False, zufall: int = 0,
                  seed: int = ZUFALL_SEED) -> list:
    """Die zu erklaerenden Bilder, aufsteigend nach Index.

    zufall > 0 ersetzt die Basismenge "erste/letzte k je Klasse" durch eine
    Zufallsstichprobe dieser Groesse (siehe zufalls_indizes); 0 = altes
    Verhalten, damit die Stufe-1-Auswahl reproduzierbar bleibt.

    Ein Basisbild, das zugleich falsch klassifiziert ist, erscheint EINMAL
    (mit fest=1 und fehler=1) und verbraucht KEIN Fehlerkontingent - sonst
    haenge der Dateiname der Vergleichsbilder wieder am Lauf.
    """
    nach_index = {z["index"]: z for z in zeilen}
    fest = set(basis_indizes(zeilen, k, gespreizt, zufall, split, seed))

    # Die konfidentesten Fehler zuerst. Tie-Break nach Index, weil die CSV nur
    # vier Nachkommastellen hat und Gleichstaende real vorkommen.
    kandidaten = [i for i in fehler_indizes(zeilen, schwelle) if i not in fest]
    kandidaten.sort(key=lambda i: (-konfidenz(nach_index[i], schwelle), i))
    gewaehlte_fehler = set(kandidaten[:max_fehler])

    auswahl = []
    for i in sorted(fest | gewaehlte_fehler):
        z = nach_index[i]
        auswahl.append({
            "index":     i,
            "split":     split,
            "true":      z["true"],
            "pred":      vorhersage(z, schwelle),
            "prob_io":   z["prob_io"],
            "prob_nio":  z["prob_nio"],
            "konfidenz": konfidenz(z, schwelle),
            "fest":      int(i in fest),
            "fehler":    int(ist_fehler(z, schwelle)),
            "heatmap":   heatmap_name(split, i, z["true"], i in fest),
        })
    return auswahl


# ================================================================
#  Manifest
# ================================================================
MANIFEST_KOPF = ["heatmap", "index", "split", "datei", "pfad", "true", "pred",
                 "prob_io", "prob_nio", "konfidenz", "fest", "fehler", "korrekt"]


def schreibe_manifest(auswahl: list, zeilen: list, samples: list, out_dir: str,
                      run_name: str, split: str) -> str:
    """Schreibt manifest.csv und prueft vorher die Index-Kopplung.

    samples: dataset.samples, also [(pfad, label), ...] in Loader-Reihenfolge.

    Die Pruefung ist der einzige Weg, den stillen Totalausfall zu bemerken, bei
    dem CSV-Index und Datensatz-Index auseinanderlaufen - etwa weil seit dem
    Training ein Bild in data_512/ dazugekommen oder verschwunden ist. Ohne sie
    bekaeme man klaglos Heatmaps zu den falschen Bildern.
    """
    if len(zeilen) != len(samples):
        raise RuntimeError(
            f"{run_name}/{split}: preds-CSV hat {len(zeilen)} Zeilen, der Datensatz "
            f"aber {len(samples)} Bilder. Hat sich data_512/ seit dem Training "
            "geaendert? Die Indizes waeren dann verschoben."
        )
    for z in zeilen:
        soll = CLASSES[samples[z["index"]][1]]
        if z["true"] != soll:
            raise RuntimeError(
                f"{run_name}/{split}: Index {z['index']} ist laut CSV '{z['true']}', "
                f"laut Datensatz aber '{soll}'. Index-Kopplung gebrochen."
            )

    os.makedirs(out_dir, exist_ok=True)
    pfad = os.path.join(out_dir, "manifest.csv")
    with open(pfad, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(MANIFEST_KOPF)
        for a in auswahl:
            quelle = samples[a["index"]][0]
            w.writerow([a["heatmap"], a["index"], a["split"], os.path.basename(quelle),
                        quelle, a["true"], a["pred"],
                        f"{a['prob_io']:.4f}", f"{a['prob_nio']:.4f}",
                        f"{a['konfidenz']:.4f}", a["fest"], a["fehler"],
                        int(not a["fehler"])])
    return pfad


def lies_manifest(out_dir: str) -> list:
    pfad = os.path.join(out_dir, "manifest.csv")
    if not os.path.isfile(pfad):
        return []
    with open(pfad, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def manifest_vollstaendig(out_dir: str) -> tuple:
    """(fertig, vorhanden, erwartet) - gezaehlt wird gegen das MANIFEST.

    Nicht gegen die Dateizahl im Ordner: aendert man max_fehler, bleiben nicht
    mehr ausgewaehlte PNGs als Waisen liegen, und ein blosses Abzaehlen der
    Dateien wuerde den Ordner faelschlich als fertig melden.
    """
    eintraege = lies_manifest(out_dir)
    if not eintraege:
        return (False, 0, 0)
    vorhanden = sum(1 for e in eintraege
                    if os.path.isfile(os.path.join(out_dir, e["heatmap"])))
    return (vorhanden == len(eintraege), vorhanden, len(eintraege))
