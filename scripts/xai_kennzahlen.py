#!/usr/bin/env python3
"""
scripts/xai_kennzahlen.py
Quantitative Auswertung der Erklaerbarkeitsverfahren fuer das Ergebniskapitel,
plus die Heatmap-Galerie in Druckqualitaet.

Gerechnet wird auf dem besten Modell einer Versuchsstufe. Der Umfang steht auf
`--umfang`: `alle` misst den ganzen Testanteil (Vorgabe seit 08/2026), `fest`
bildet den alten Stand aus sechs Vergleichsbildern nach. Sechs Aufnahmen
reichten fuer eine Groessenordnung, nicht fuer eine Streuung - und bei der
Occlusion waren davon regelmaessig nur zwei bis drei Karten belegt.

Erhoben werden fuenf Groessen. Die ersten beiden brauchen eine Bauteilmaske, die
uebrigen nicht - das ist Absicht: an der Maske haengt sonst die ganze Aussage.

1. Konzentrationsfaktor  (`src/xai/metriken.py`)
   Anteil der Attributionsmasse auf dem Bauteil, geteilt durch den
   Flaechenanteil der Maske. 1,0 heisst gleichverteilte Karte. Der Rohanteil
   bleibt daneben stehen, ist ohne den Bezugswert aber nicht lesbar: 36,5 %
   klingen nach wenig und sind das 2,6-Fache einer gleichverteilten Karte.

   Die Maske kommt aus drei Quellen, in dieser Reihenfolge:
     1. der Ordner aus `zuschnitt.masken` der Run-Config - echte, beim
        Freistellen referenzbasiert erzeugte Masken (Stufe 2).
     2. eine Otsu-Schwelle je Aufnahme (Stufe 1, dunkles Bauteil auf hellem
        Grund).
     3. `--maske heuristik`: die abgeloesten dunkelsten 24 % der Flaeche. Nur
        noch fuer den Vergleich; sie liefert per Konstruktion immer denselben
        Flaechenanteil und ist damit blind gegenueber der Bauteilgroesse.

   Welcher Weg gegriffen hat, steht je Aufnahme im Bericht. Auf den
   Bauteil-Zuschnitten der Stufe 2 traefe eine Helligkeitsheuristik das
   Korbgitter statt des Blechs; vor dem Fix vom 20.08.2026 kamen daher Anteile
   UNTER Gleichverteilung heraus - ein Befund ueber die Maske, der wie einer
   ueber das Modell aussah.

2. Maskensensitivitaet
   Derselbe Faktor auf um +-8 und +-16 Pixel verformten Masken. Beziffert, wie
   stark das Ergebnis an der Maskengrenze haengt, statt es zuzusichern.

3. Deletion und Insertion  - ohne Maske
   Hoechstattribuierte Pixel zuerst durch die Baseline ersetzen (Deletion) bzw.
   von der Baseline aus einsetzen (Insertion) und die Wahrscheinlichkeit der
   vorhergesagten Klasse verfolgen. Bezug ist die zufaellige Reihenfolge, die
   mitgemessen wird. Prueft am Modell selbst, ob die markierten Pixel die
   Entscheidung tragen.

4. Completeness (Sundararajan et al. 2017)  - ohne Maske
   Die Summe der signierten Attributionen muss der Differenz der
   Modellausgaben zwischen Eingabe und Baseline entsprechen. Gemessen mit der im
   Betrieb verwendeten Stuetzpunktzahl ueber alle Aufnahmen und mit einer
   deutlich hoeheren ueber eine Teilmenge, weil der Diskretisierungsfehler der
   rechten Riemann-Summe mit O(1/m) faellt - erst der Vergleich beider Werte
   zeigt, dass eine verbleibende Abweichung von der Schrittzahl und nicht von
   der Implementierung stammt. `tests/test_xai.py::test_completeness_axiom`
   prueft dieselbe Eigenschaft automatisiert mit 5 % Toleranz.

5. Sanity-Check (Adebayo et al. 2018)  - ohne Maske
   Gewichte schichtweise randomisieren und die Karte gegen die des trainierten
   Modells korrelieren. Bleibt die Korrelation hoch, erklaert das Verfahren
   nicht das Modell, sondern zeigt Bildkanten.

Ausgabe:
    outputs/runs/Stufe_<N>/xai_kennzahlen.json  (+ .md)
    outputs/runs/Stufe_<N>/xai_kennzahlen.csv   (eine Zeile je Aufnahme und
                                                 Verfahren - fuer die Diagramme)
    outputs/runs/Stufe_<N>/xai_karten.npz        (Zwischenspeicher der Karten)
    thesis/abbildungen/kap5/<praefix>_io.png    (praefix: --galerie-praefix)
    thesis/abbildungen/kap5/<praefix>_nio.png

Zwischengespeichert werden nur die Karten der sechs Galeriebilder. Ueber den
ganzen Testanteil waere die npz rund 800 MB gross, und die Galerie hat sechs
Zeilen. `--nur-galerie` zeichnet daraus neu, ohne das Modell anzufassen.

Die JSON-, MD-, CSV- und NPZ-Ausgaben liegen je Versuchsstufe getrennt, die
beiden Galerien nicht - sie landen alle im selben Abbildungsordner der Arbeit.
Ohne --galerie-praefix wuerde eine zweite Versuchsstufe die Galerien der ersten
ueberschreiben. Default bleibt "galerie", damit Stufe 1 unveraendert bleibt.

Aufruf:
    python scripts/xai_kennzahlen.py --stufe 1
    python scripts/xai_kennzahlen.py --stufe 2 --galerie-praefix galerie_stufe2
    python scripts/xai_kennzahlen.py --umfang fest --ohne-di --ohne-sanity
    python scripts/xai_kennzahlen.py --nur-galerie
"""
import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    # line_buffering: der Lauf ueber den vollen Testanteil dauert Stunden. Ohne
    # Zeilenpufferung steht in einer umgeleiteten Ausgabe stundenlang nichts,
    # und ein haengender Lauf ist von einem langsamen nicht zu unterscheiden.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from kap5_daten import (BACKBONE_NAME, FREEZE_NAME, de, lade_inferenzzeit,  # noqa: E402
                        lade_laeufe, lade_manifest, rangfolge)
from src.dataset import build_dataloaders  # noqa: E402
from src.models.resnet import build_model  # noqa: E402
from src.xai.integrated_gradients import (_normalize,  # noqa: E402
                                          integrated_gradients,
                                          integrated_gradients_signed, occlusion_map,
                                          saliency_map)
from src.xai.metriken import (deletion_insertion, konzentrationsfaktor,  # noqa: E402
                              maskensensitivitaet, otsu_maske,
                              randomisiere_schichten, sanity_spearman)

# ---- Messparameter --------------------------------------------------------
N_JE_KLASSE = 3          # Galeriebilder je Klasse (Vorgabe aus Abschnitt 4.4.3)
STEPS_FEIN = 256         # Kontrollwert fuer die Completeness (wie im pytest)
MASKE_ANTEIL = 0.24      # Flaechenanteil der abgeloesten Quantilheuristik
TOLERANZ = 0.05          # Toleranz des automatisierten Completeness-Tests

# Umfang: bis August 2026 wurde auf sechs Aufnahmen gemessen. Das reichte, um
# eine Groessenordnung zu nennen, nicht um eine Streuung anzugeben - und bei
# Occlusion waren davon regelmaessig nur zwei bis drei Karten ueberhaupt belegt.
# Der volle Testanteil (311 bzw. 400 Aufnahmen) macht daraus eine Verteilung.
STICHPROBE_JE_KLASSE = 30   # fuer --umfang stichprobe

# Maskensensitivitaet: Erosion und Dilatation in Pixeln. Der Wertebereich ist an
# der Bildgroesse 512 orientiert - 16 Pixel sind gut 3 % der Kantenlaenge und
# damit deutlich mehr, als eine Maske typischerweise danebenliegt.
SENS_STUFEN = (-16, -8, 0, 8, 16)

# Deletion/Insertion: 20 Schritte ergeben eine Kurve, die sich im Diagramm noch
# als Kurve liest, und kosten je Karte 42 Vorwaertsdurchlaeufe.
#
# Anders als der Konzentrationsfaktor laeuft die Messung NICHT ueber den ganzen
# Testanteil, sondern ueber DI_N Aufnahmen je Klasse. Grund ist der Preis: auf
# dieser Maschine (RTX PRO 4000 Blackwell Laptop) kostet ein Vorwaertslauf bei
# 512x512 rund 67 ms, vier Karten mal zwei Richtungen mal 21 Stuetzstellen also
# gut 17 s je Aufnahme - ueber 400 Aufnahmen waeren das knapp zwei Stunden
# zusaetzlich. Berichtet wird von dieser Kennzahl eine ueber Aufnahmen
# gemittelte Kurve; dafuer traegt eine geschichtete Teilmenge dieser Groesse
# genauso weit, waehrend der Konzentrationsfaktor als Verteilung berichtet wird
# und deshalb den vollen Umfang braucht.
DI_SCHRITTE = 20
DI_BATCH = 8
DI_N = 40                # je Klasse

# Sanity-Check nach Adebayo et al.: teuer, weil je Stufe ein eigenes Modell und
# je Aufnahme eine eigene IG-Berechnung noetig ist. Adebayo et al. arbeiten
# ebenfalls auf wenigen Aufnahmen - die Aussage ist qualitativ (bricht die
# Korrelation ein?), nicht verteilungsbasiert.
SANITY_N = 12
SANITY_STUFEN = (1, 2, 4, 8, 16)

# Strukturierte Sichtung: die Karten der ausgewaehlten Aufnahmen werden waehrend
# der Messung als einzelne, unbeschriftete Tafeln abgelegt. Sie im selben Lauf
# mitzuschreiben kostet nichts - die Karten liegen ohnehin im Speicher -, waehrend
# ein eigener Durchlauf noch einmal Stunden brauchte.
#
# Eine Tafel je Aufnahme UND Verfahren, mit zufaelligem Dateinamen: der Bewerter
# soll nicht wissen, welches Verfahren er gerade sieht. Sonst bewertet er die
# Erwartung an das Verfahren mit, und die Sichtung belegte nur noch, was man
# ohnehin vermutet hat. Die Zuordnung steht allein in `schluessel.csv`.
SICHTUNG_N = 20          # Aufnahmen je Klasse
SICHTUNG_GAMMA = 0.30    # wie DARSTELLUNGS_GAMMA - dieselbe Anzeigekennlinie

# ---- Darstellung ----------------------------------------------------------
BREITE_ZOLL = 7.48       # 19 cm; im Dokument auf 16 cm skaliert
DPI = 300
# Die Heatmap wird mit pixelweiser Deckkraft ueberlagert: Bereiche ohne
# Relevanz lassen die Aufnahme unveraendert stehen, statt sie flaechig
# abzudunkeln. Bei konstanter Deckkraft verschluckt die dunkle Seite der
# Farbskala sonst das ganze Bild, weil der weit ueberwiegende Teil der Pixel
# nahe null liegt.
#
# DARSTELLUNGS_GAMMA hebt zusaetzlich Farbe UND Deckkraft der mittleren Werte
# an. Es ist eine monotone Umskalierung derselben Karte: null bleibt null, eins
# bleibt eins, die Reihenfolge der Werte aendert sich nicht. Ohne sie bleiben
# die Karten der Integrated Gradients im Druck praktisch unsichtbar, weil ihre
# Attribution auf wenigen Pixeln sitzt und die Normierung auf [0, 1] alles
# uebrige nahe null druekt. Der Wert wird empirisch aus den Quantilen der
# tatsaechlichen Karten bestimmt (siehe Ausgabe des Skripts).
DARSTELLUNGS_GAMMA = 0.30
ALPHA_MAX = 0.92
METHODEN = ["integrated_gradients", "saliency", "occlusion"]
METHODEN_NAME = {
    "integrated_gradients": "Integrated Gradients",
    "saliency": "Saliency Map",
    "occlusion": "Occlusion Map",
}


def bauteilmaske(bild: np.ndarray) -> np.ndarray:
    """
    Abgeloeste Quantilheuristik: die dunkelsten `MASKE_ANTEIL` der Flaeche.

    Steht nur noch fuer den Vergleich in `--maske heuristik` zur Verfuegung.
    Ihr Konstruktionsfehler ist, dass sie per Definition immer denselben
    Flaechenanteil liefert - wie gross das Bauteil in der Aufnahme wirklich ist,
    geht nicht ein. Der Flaechenanteil ist aber die Bezugsgroesse des
    Konzentrationsfaktors, also genau die Zahl, auf die es ankommt.
    Ersatz ist `otsu_maske` aus `src/xai/metriken.py`.
    """
    grau = bild.mean(axis=2)
    grenze = np.quantile(grau, MASKE_ANTEIL)
    return grau <= grenze


def maskenwurzel(config: dict) -> str:
    """Ordner der echten Bauteilmasken aus dem zuschnitt:-Block, sonst None."""
    pfad = (config.get("zuschnitt") or {}).get("masken")
    if not pfad:
        return None
    voll = os.path.join(PROJECT_ROOT, pfad)
    return voll if os.path.isdir(voll) else None


def echte_maske(wurzel: str, quellpfad: str, form) -> np.ndarray:
    """Abgelegte Bauteilmaske zu einem Datensatzbild, auf `form` gebracht.

    quellpfad ist der Pfad aus dem Manifest, z.B. data_stufe2_512/test/io/x.png.
    Die Maske liegt unter <wurzel>/test/io/x.png - also die letzten drei
    Bestandteile des Pfades. None, wenn es sie nicht gibt.

    NEAREST beim Skalieren: eine Maske ist binaer, jede Interpolation erzeugt
    Zwischenwerte und damit einen weichgezeichneten Rand, dessen Breite in den
    Flaechenanteil eingeht.
    """
    if not wurzel:
        return None
    teile = os.path.normpath(quellpfad).split(os.sep)[-3:]
    pfad = os.path.join(wurzel, *teile)
    if not os.path.isfile(pfad):
        return None
    bild = Image.open(pfad).convert("L")
    if bild.size != (form[1], form[0]):
        bild = bild.resize((form[1], form[0]), Image.NEAREST)
    return np.array(bild) > 127


def hole_maske(masken_wurzel, quellpfad, original, modus):
    """(Maske, Quelle, Diagnose) - abgelegte Maske, Otsu oder die alte Heuristik.

    Reihenfolge: abgelegte Masken haben immer Vorrang, weil sie beim
    Freistellen referenzbasiert entstanden sind und nicht aus dem Bild geraten
    werden muessen. Erst wenn es keine gibt (Stufe 1, `verfahren: ganz`), greift
    `--maske`: `otsu` ermittelt die Schwelle aus dem Bild selbst, `heuristik`
    ist der abgeloeste Weg und bleibt nur fuer den Vergleich erreichbar.
    """
    maske = echte_maske(masken_wurzel, quellpfad, original.shape[:2])
    if maske is not None:
        return maske, "abgelegt", {"anteil": float(maske.mean()), "plausibel": True}
    if modus == "heuristik":
        maske = bauteilmaske(original)
        return maske, "heuristik", {"anteil": float(maske.mean()), "plausibel": True}
    maske, diag = otsu_maske(original, dunkles_bauteil=True)
    return maske, "otsu", diag


def completeness(model, x, ziel: int, steps: int, step_batch: int, attr=None):
    """Summe der signierten Attributionen gegen F(x) - F(x') am Klassen-Logit.

    `attr` nimmt eine bereits gerechnete signierte Attribution entgegen. Die
    Messschleife braucht dieselbe Groesse ohnehin fuer die Heatmap - ohne diesen
    Weg liefe die 50-Schritt-Pfadintegration je Aufnahme zweimal, und ueber den
    vollen Testanteil ist das rund ein Drittel der gesamten Rechenzeit.
    """
    if attr is None:
        attr = integrated_gradients_signed(model, x, ziel, steps=steps, step_batch=step_batch)
    with torch.no_grad():
        delta_f = (model(x)[0, ziel] - model(torch.zeros_like(x))[0, ziel]).item()
    summe = float(attr.sum())
    abweichung = abs(summe - delta_f)
    return {
        "steps": steps,
        "delta_f": delta_f,
        "summe_ig": summe,
        "abweichung": abweichung,
        "relative_abweichung": abweichung / abs(delta_f) if abs(delta_f) > 1e-12 else float("nan"),
    }


def lade_vorhersagen(lauf, dataset):
    """Eine Zeile je Bild des Splits: Index, Datei, Pfad, wahre Klasse, Vorhersage.

    Das Manifest des XAI-Nachlaufs kennt nur die rund 50 ausgewaehlten Bilder.
    Fuer den vollen Testanteil ist die Vorhersagedatei die Quelle; ihre
    Zeilennummer ist der Datensatzindex. `xai_auswahl.schreibe_manifest` prueft
    diese Kopplung beim Erzeugen des Manifests hart gegen `dataset.samples`,
    weshalb sie hier vorausgesetzt werden darf - der Wahrheitsgehalt wird
    trotzdem je Zeile gegen das Label des Datensatzes geprueft.
    """
    import csv

    pfad = os.path.join(lauf["run_dir"], "preds", "example_predictions.csv")
    with open(pfad, encoding="utf-8") as f:
        rohe = list(csv.DictReader(f))
    if len(rohe) != len(dataset.samples):
        raise RuntimeError(f"Vorhersagedatei hat {len(rohe)} Zeilen, der Split aber "
                           f"{len(dataset.samples)} Bilder - Index waere bedeutungslos")
    zeilen = []
    for i, z in enumerate(rohe):
        quelle, label = dataset.samples[i]
        klasse = "nio" if label == 1 else "io"
        if z["true_label"] != klasse:
            raise RuntimeError(f"Zeile {i}: Vorhersagedatei sagt {z['true_label']}, "
                               f"Datensatz sagt {klasse}")
        zeilen.append({
            "index": i,
            "datei": os.path.basename(quelle),
            "pfad": quelle,
            "true": klasse,
            "pred": z["predicted_label"],
            "prob_nio": float(z["prob_nio"]),
        })
    return zeilen


def waehle_messbilder(zeilen, manifest, umfang):
    """Welche Bilder gemessen werden - `alle`, `stichprobe` oder `fest`.

    `fest` bildet den Stand bis August 2026 nach: die ersten drei festen
    Vergleichsbilder je Klasse. `stichprobe` zieht mit festem Seed gleich viele
    je Klasse, damit ein Zwischenstand reproduzierbar bleibt. Die Reihenfolge
    ist in allen Faellen nach Index sortiert, sodass die CSV zwischen Laeufen
    zeilenweise vergleichbar bleibt.
    """
    if umfang == "alle":
        return list(zeilen)
    if umfang == "stichprobe":
        rng = np.random.default_rng(123)          # fester Seed, siehe xai_auswahl.py
        gewaehlt = []
        for klasse in ("io", "nio"):
            kandidaten = [z for z in zeilen if z["true"] == klasse]
            n = min(STICHPROBE_JE_KLASSE, len(kandidaten))
            idx = rng.choice(len(kandidaten), size=n, replace=False)
            gewaehlt += [kandidaten[int(i)] for i in idx]
        return sorted(gewaehlt, key=lambda z: z["index"])
    nach_index = {z["index"]: z for z in zeilen}
    gewaehlt = []
    for klasse in ("io", "nio"):
        fest = [m for m in manifest if m["fest"] == "1" and m["true"] == klasse]
        if len(fest) < N_JE_KLASSE:
            raise RuntimeError(f"Zu wenige feste Vergleichsbilder fuer {klasse}: {len(fest)}")
        gewaehlt += [nach_index[int(m["index"])] for m in fest[:N_JE_KLASSE]]
    return gewaehlt


def galerie(bilder, klasse: str, pfad: str, gamma: float = DARSTELLUNGS_GAMMA):
    """Heatmap-Galerie: eine Zeile je Bild, Spalten Original und drei Verfahren."""
    n = len(bilder)
    fig, axes = plt.subplots(n, 4, figsize=(BREITE_ZOLL, BREITE_ZOLL / 4 * n + 0.45))
    axes = np.atleast_2d(axes)

    for zeile, eintrag in enumerate(bilder):
        original = eintrag["original"]
        panels = [("Original", None)] + [(METHODEN_NAME[m], eintrag["karten"][m]) for m in METHODEN]
        for spalte, (titel, karte) in enumerate(panels):
            ax = axes[zeile, spalte]
            ax.imshow(original)
            if karte is not None:
                gezeigt = np.clip(karte, 0.0, 1.0) ** gamma
                ax.imshow(gezeigt, cmap="inferno", vmin=0.0, vmax=1.0,
                          alpha=gezeigt * ALPHA_MAX)
            if zeile == 0:
                ax.set_title(titel, fontsize=9.5, pad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        # Zeilenbeschriftung unter die Zeile statt an den linken Rand: als
        # y-Achsenbeschriftung frisst sie rund ein Viertel der Bildbreite, die
        # den Aufnahmen selbst fehlt.
        axes[zeile, 0].text(
            0.0, -0.035,
            f"{eintrag['datei_kurz']}   p(n.i.O.) = {de(eintrag['p_nio'], 3)}",
            transform=axes[zeile, 0].transAxes, ha="left", va="top", fontsize=8)

    fig.suptitle(f"Klasse {klasse}", fontsize=11, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.subplots_adjust(wspace=0.02, hspace=0.16)
    fig.savefig(pfad, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return pfad


def speichere_karten(pfad, galerie_daten):
    """Originale und Attributionskarten zwischenspeichern (float16 reicht fuer die Anzeige)."""
    ablage = {}
    for klasse, eintraege in galerie_daten.items():
        for i, e in enumerate(eintraege):
            ablage[f"{klasse}_{i}_original"] = e["original"].astype(np.float16)
            for m in METHODEN:
                ablage[f"{klasse}_{i}_{m}"] = e["karten"][m].astype(np.float16)
            ablage[f"{klasse}_{i}_meta"] = np.array([e["datei_kurz"], str(e["p_nio"])])
    np.savez_compressed(pfad, **ablage)


def lade_karten(pfad):
    d = np.load(pfad, allow_pickle=False)
    galerie_daten = {"io": [], "nio": []}
    for klasse in galerie_daten:
        i = 0
        while f"{klasse}_{i}_original" in d:
            galerie_daten[klasse].append({
                "original": d[f"{klasse}_{i}_original"].astype(np.float32),
                "karten": {m: d[f"{klasse}_{i}_{m}"].astype(np.float32) for m in METHODEN},
                "datei_kurz": str(d[f"{klasse}_{i}_meta"][0]),
                "p_nio": float(d[f"{klasse}_{i}_meta"][1]),
            })
            i += 1
    return galerie_daten


def quantile_bericht(galerie_daten):
    """Quantile der Attributionskarten - Grundlage fuer die Wahl des Darstellungs-Gammas."""
    print("  Quantile der Attributionskarten (ueber alle sechs Aufnahmen):")
    for m in METHODEN:
        werte = np.concatenate([e["karten"][m].ravel()
                                for eintraege in galerie_daten.values() for e in eintraege])
        q = np.quantile(werte, [0.5, 0.9, 0.99, 0.999])
        print(f"    {METHODEN_NAME[m]:22s} p50={q[0]:.4f}  p90={q[1]:.4f}  "
              f"p99={q[2]:.4f}  p99,9={q[3]:.4f}")


def main():
    ap = argparse.ArgumentParser(description="XAI-Kennzahlen und Galerie fuer Kapitel 5")
    ap.add_argument("--stufe", type=int, default=1)
    ap.add_argument("--run", default=None,
                    help="Laufverzeichnis; ohne Angabe das nach 4.5.2 beste Modell")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--nur-galerie", action="store_true",
                    help="Karten aus dem Zwischenspeicher neu zeichnen, nichts neu rechnen")
    ap.add_argument("--gamma", type=float, default=DARSTELLUNGS_GAMMA,
                    help="Darstellungs-Gamma der Heatmap-Ueberlagerung")
    ap.add_argument("--umfang", default="alle", choices=["alle", "stichprobe", "fest"],
                    help="alle = ganzer Split (Vorgabe), stichprobe = "
                         f"{STICHPROBE_JE_KLASSE} je Klasse, fest = die sechs "
                         "Vergleichsbilder wie bis August 2026")
    ap.add_argument("--maske", default="otsu", choices=["otsu", "heuristik"],
                    help="Ersatzweg, wenn die Run-Config keinen Maskenordner nennt")
    ap.add_argument("--ohne-di", action="store_true", dest="ohne_di",
                    help="Deletion/Insertion auslassen (spart rund die Haelfte der Zeit)")
    ap.add_argument("--sichtung-n", type=int, default=SICHTUNG_N, dest="sichtung_n",
                    help=f"Aufnahmen je Klasse fuer die Sichtungstafeln (Vorgabe "
                         f"{SICHTUNG_N}, 0 schaltet sie ab)")
    ap.add_argument("--di-n", type=int, default=DI_N, dest="di_n",
                    help=f"Aufnahmen je Klasse fuer Deletion/Insertion (Vorgabe {DI_N})")
    ap.add_argument("--ohne-sanity", action="store_true", dest="ohne_sanity",
                    help="Sanity-Check nach Adebayo auslassen")
    ap.add_argument("--galerie-praefix", default="galerie", dest="galerie_praefix",
                    help="Dateinamen der Galerien: <praefix>_io.png / <praefix>_nio.png. "
                         "Default 'galerie' = die Bilder der Stufe 1; fuer eine weitere "
                         "Versuchsstufe einen eigenen Praefix setzen, sonst wird "
                         "ueberschrieben")
    args = ap.parse_args()

    stufen_dir = os.path.join(PROJECT_ROOT, "outputs", "runs", f"Stufe_{args.stufe}")
    abb_dir = os.path.join(PROJECT_ROOT, "thesis", "abbildungen", "kap5")
    os.makedirs(abb_dir, exist_ok=True)
    npz_pfad = os.path.join(stufen_dir, "xai_karten.npz")

    if args.nur_galerie:
        # Aus dem Zwischenspeicher: Bericht neu aggregieren und Galerie neu
        # zeichnen, ohne das Modell anzufassen. Die Messwerte je Bild bleiben
        # unveraendert, nur ihre Aggregation und die Darstellung werden erneuert.
        if not os.path.isfile(npz_pfad):
            sys.exit(f"[FEHLER] Zwischenspeicher fehlt: {npz_pfad}")
        json_pfad = os.path.join(stufen_dir, "xai_kennzahlen.json")
        if not os.path.isfile(json_pfad):
            sys.exit(f"[FEHLER] Messwerte fehlen: {json_pfad}")
        with open(json_pfad, encoding="utf-8") as f:
            bericht = json.load(f)
        bericht["zusammenfassung"] = fasse_zusammen(bericht["bilder"])
        schreibe_bericht(stufen_dir, bericht, args.stufe)

        galerie_daten = lade_karten(npz_pfad)
        quantile_bericht(galerie_daten)
        for klasse, bez in (("io", "i.O."), ("nio", "n.i.O.")):
            datei = f"{args.galerie_praefix}_{klasse}.png"
            p = galerie(galerie_daten[klasse], bez, os.path.join(abb_dir, datei), args.gamma)
            print(f"  geschrieben: {os.path.relpath(p, PROJECT_ROOT)}  (Gamma {args.gamma})")
        drucke_zusammenfassung(bericht["zusammenfassung"], bericht["ig_steps_betrieb"])
        return

    laeufe = lade_laeufe(stufen_dir)
    zeiten = lade_inferenzzeit(stufen_dir)
    if args.run:
        lauf = next(l for l in laeufe if l["run"] == os.path.basename(args.run.rstrip("/\\")))
    else:
        lauf = rangfolge(laeufe, zeiten)[0]
    print(f"Lauf: {lauf['run']}  ({BACKBONE_NAME[lauf['backbone']]}, "
          f"{FREEZE_NAME[lauf['freeze']]})")

    # ---- Modell und Daten -------------------------------------------------
    with open(os.path.join(lauf["run_dir"], "config.yaml"), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    xai_cfg = config.get("xai", {})
    ig_steps = int(xai_cfg.get("ig_steps", 50))
    step_batch = int(xai_cfg.get("ig_step_batch", 8))
    patch = int(xai_cfg.get("occ_patch_size", 64))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        # Dieselben Faltungseinstellungen wie im Training und in der Auswertung
        # (vgl. CLAUDE.md). `benchmark` lohnt hier besonders, weil ueber den
        # ganzen Lauf immer dieselbe Eingabegroesse anliegt und cudnn den
        # Algorithmus damit einmal waehlt statt je Aufruf.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    _, val_loader, test_loader = build_dataloaders(config)
    dataset = (val_loader if args.split == "val" else test_loader).dataset

    model = build_model(config)
    ckpt = torch.load(os.path.join(lauf["run_dir"], "checkpoints", "best.pt"),
                      map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Checkpoint aus Epoche {ckpt.get('epoch')} | Device: {device} | "
          f"ig_steps: {ig_steps}, occ_patch_size: {patch}")

    # Echte Bauteilmasken, falls die Run-Config welche nennt. Sonst entsteht die
    # Maske je Aufnahme neu - per Otsu-Schwelle, siehe `hole_maske`.
    masken_wurzel = maskenwurzel(config)
    print("Bauteilmaske: " + (f"abgelegte Masken aus "
          f"{os.path.relpath(masken_wurzel, PROJECT_ROOT)}" if masken_wurzel
          else ("Otsu-Schwelle je Aufnahme" if args.maske == "otsu"
                else f"Quantilheuristik (dunkelste {MASKE_ANTEIL:.0%})")))

    norm = config.get("augmentation", {}).get("normalize", {})
    mean = torch.tensor(norm.get("mean", [0.485, 0.456, 0.406])).view(3, 1, 1)
    std = torch.tensor(norm.get("std", [0.229, 0.224, 0.225])).view(3, 1, 1)

    # ---- Auswahl ----------------------------------------------------------
    manifest = lade_manifest(lauf, args.split)
    alle_zeilen = lade_vorhersagen(lauf, dataset)
    auswahl = waehle_messbilder(alle_zeilen, manifest, args.umfang)
    # Die Galerie zeigt weiterhin genau die sechs festen Vergleichsbilder, damit
    # Bild und Zahl dieselben Bauteile meinen (README kap5, Abschnitt 1). Nur
    # deren Karten kommen in den Zwischenspeicher: ueber den ganzen Testanteil
    # waere die npz rund 800 MB gross und nutzlos, weil die Galerie sechs Zeilen hat.
    galerie_zeilen = waehle_messbilder(alle_zeilen, manifest, "fest")
    galerie_index = {z["index"]: z["true"] for z in galerie_zeilen}
    # Die sechs Galeriebilder gehoeren immer zur Messung. Bei --umfang alle sind
    # sie ohnehin dabei; bei --umfang stichprobe zieht die Zufallsauswahl sie
    # nicht zwingend, und die Galerie bliebe ohne sie unvollstaendig.
    bekannt = {z["index"] for z in auswahl}
    auswahl = sorted(auswahl + [z for z in galerie_zeilen if z["index"] not in bekannt],
                     key=lambda z: z["index"])
    # Completeness mit 256 Stuetzpunkten und der Sanity-Check laufen nur auf
    # einer Teilmenge - beide kosten je Aufnahme ein Vielfaches der uebrigen
    # Messung, und beide beantworten eine Frage nach der Groessenordnung.
    teilmenge = {z["index"] for z in galerie_zeilen}
    for z in auswahl:
        if len(teilmenge) >= SANITY_N:
            break
        teilmenge.add(z["index"])
    # Deletion/Insertion auf einer geschichteten Teilmenge - siehe DI_N.
    di_menge = set(teilmenge)
    rng_di = np.random.default_rng(4711)
    for klasse in ("io", "nio"):
        kandidaten = [z["index"] for z in auswahl if z["true"] == klasse]
        n = min(args.di_n, len(kandidaten))
        di_menge |= {int(kandidaten[i]) for i in
                     rng_di.choice(len(kandidaten), size=n, replace=False)}
    # Sichtungsmenge: geschichtet, fester Seed, plus alle Fehlklassifikationen -
    # gerade die tragen die qualitative Aussage und sind zu selten, um sie dem
    # Zufall zu ueberlassen.
    sichtung_menge = set()
    if args.sichtung_n:
        rng_s = np.random.default_rng(90210)
        for klasse in ("io", "nio"):
            kandidaten = [z["index"] for z in auswahl if z["true"] == klasse]
            n = min(args.sichtung_n, len(kandidaten))
            sichtung_menge |= {int(kandidaten[i]) for i in
                               rng_s.choice(len(kandidaten), size=n, replace=False)}
        sichtung_menge |= {z["index"] for z in auswahl if z["true"] != z["pred"]}
    print(f"Umfang: {args.umfang} -> {len(auswahl)} Aufnahmen "
          f"({sum(1 for z in auswahl if z['true'] == 'io')} i.O. / "
          f"{sum(1 for z in auswahl if z['true'] == 'nio')} n.i.O.); "
          f"Deletion/Insertion auf {len(di_menge)}, Feinmessung auf {len(teilmenge)}, "
          f"Sichtung auf {len(sichtung_menge)}")

    # ---- Messung je Bild --------------------------------------------------
    ergebnisse = []
    galerie_daten = {"io": [], "nio": []}
    sanity_karten = {}
    sichtung_dir = os.path.join(PROJECT_ROOT, "outputs", "xai_sichtung",
                                f"stufe{args.stufe}")
    sichtung_schluessel = []
    if sichtung_menge:
        os.makedirs(os.path.join(sichtung_dir, "bilder"), exist_ok=True)
    rng_tafel = np.random.default_rng(31415)
    rng = np.random.default_rng(20260826)
    begonnen = datetime.now()
    for lfd, zeile in enumerate(auswahl, start=1):
        idx = zeile["index"]
        tensor, _ = dataset[idx]
        x = tensor.unsqueeze(0).to(device)
        fein = idx in teilmenge

        with torch.no_grad():
            logits = model(x)
        ziel = int(logits.argmax(dim=1).item())   # erklaert wird die Vorhersage

        # Die signierte Attribution einmal rechnen und beides daraus ableiten:
        # die Heatmap (Betrag, auf [0, 1] normiert - genau das, was
        # `integrated_gradients` tut) und die Completeness, die die Vorzeichen
        # braucht. `_normalize` wird bewusst aus dem XAI-Modul geholt statt hier
        # nachgebaut, damit Kennzahl und Laufausgabe dieselbe Normierung haben.
        ig_signiert = integrated_gradients_signed(model, x, ziel, steps=ig_steps,
                                                  step_batch=step_batch)
        karten = {
            "integrated_gradients": _normalize(ig_signiert.abs().detach().cpu().numpy()),
            "saliency": saliency_map(model, x, ziel),
            "occlusion": occlusion_map(model, x, ziel, patch_size=patch),
        }

        original = (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        maske, maskenquelle, diagnose = hole_maske(masken_wurzel, zeile["pfad"],
                                                   original, args.maske)

        # Konzentrationsfaktor: Massenanteil geteilt durch Flaechenanteil. Der
        # Massenanteil bleibt fuer die Rueckwaertsvertraeglichkeit im Bericht -
        # `build_kap5_payload.py` liest ihn unter diesem Namen.
        anteil, faktor = {}, {}
        for m in METHODEN:
            f, a, _fl = konzentrationsfaktor(karten[m], maske)
            anteil[m], faktor[m] = a, f

        sens = {m: maskensensitivitaet(karten[m], maske, SENS_STUFEN) for m in METHODEN}

        di = {}
        if not args.ohne_di and idx in di_menge:
            for m in METHODEN:
                di[m] = deletion_insertion(model, x, karten[m], ziel,
                                           schritte=DI_SCHRITTE, batch=DI_BATCH, rng=rng)
            # Zufallsreihenfolge als Bezugswert: ohne sie sagt eine Deletion-AUC
            # nichts: sie haengt auch davon ab, wie empfindlich das Modell auf
            # das blosse Ersetzen von Pixeln reagiert.
            di["zufall"] = deletion_insertion(model, x, rng.random(karten["saliency"].shape),
                                              ziel, schritte=DI_SCHRITTE, batch=DI_BATCH,
                                              rng=rng)

        eintrag = {
            "index": idx,
            "datei": zeile["datei"],
            "wahre_klasse": zeile["true"],
            "vorhersage": zeile["pred"],
            "p_nio": float(zeile["prob_nio"]),
            "ziel_klasse": ziel,
            "maskenanteil": float(maske.mean()),
            "maskenquelle": maskenquelle,
            "maske_plausibel": bool(diagnose.get("plausibel", True)),
            "completeness": {
                "betrieb": completeness(model, x, ziel, ig_steps, step_batch,
                                        attr=ig_signiert),
            },
            "attributionsanteil": anteil,
            "konzentrationsfaktor": faktor,
            "maskensensitivitaet": sens,
            "deletion_insertion": di,
        }
        if fein:
            eintrag["completeness"]["fein"] = completeness(model, x, ziel, STEPS_FEIN,
                                                           step_batch)
        ergebnisse.append(eintrag)

        if idx in galerie_index:
            galerie_daten[galerie_index[idx]].append({
                "original": original,
                "karten": karten,
                "datei_kurz": zeile["datei"].replace("sick_midicam2_", "").replace(".png", ""),
                "p_nio": float(zeile["prob_nio"]),
            })
        if idx in sichtung_menge:
            # Ein Fehlschlag beim Zeichnen darf einen mehrstuendigen Messlauf
            # nicht beenden - die Tafeln sind Beiwerk, die Kennzahlen sind es nicht.
            try:
                for m in METHODEN:
                    tafel_id = f"{rng_tafel.integers(10**9, 10**10)}"
                    sichtungstafel(original, karten[m],
                                   os.path.join(sichtung_dir, "bilder", f"{tafel_id}.png"))
                    sichtung_schluessel.append({
                        "tafel": tafel_id, "index": idx, "datei": zeile["datei"],
                        "verfahren": m, "wahre_klasse": zeile["true"],
                        "vorhersage": zeile["pred"], "p_nio": zeile["prob_nio"],
                    })
            except Exception as fehler:                      # noqa: BLE001
                print(f"    [WARNUNG] Sichtungstafel fuer Index {idx} nicht "
                      f"geschrieben: {fehler}")

        if fein and len(sanity_karten) < SANITY_N:
            sanity_karten[idx] = {"ziel": ziel, "karten": {m: karten[m] for m in METHODEN}}

        if lfd <= 8 or lfd % 25 == 0 or lfd == len(auswahl):
            vergangen = (datetime.now() - begonnen).total_seconds()
            rest = vergangen / lfd * (len(auswahl) - lfd)
            print(f"  [{lfd:4d}/{len(auswahl)}] Index {idx:4d} ({zeile['true']}): "
                  f"Faktor IG {faktor['integrated_gradients']:.2f} | "
                  f"Sal {faktor['saliency']:.2f} | Occ {faktor['occlusion']:.2f} | "
                  f"Maske {eintrag['maskenanteil']:.3f} ({maskenquelle})"
                  + (f" | Rest ~{rest / 60:.0f} min" if lfd < len(auswahl) else ""))

    if sichtung_schluessel:
        import csv as _csv
        with open(os.path.join(sichtung_dir, "schluessel.csv"), "w", encoding="utf-8",
                  newline="") as f:
            schreiber = _csv.DictWriter(f, fieldnames=list(sichtung_schluessel[0]),
                                        delimiter=";")
            schreiber.writeheader()
            schreiber.writerows(sichtung_schluessel)
        print(f"  Sichtungstafeln: {len(sichtung_schluessel)} in "
              f"{os.path.relpath(sichtung_dir, PROJECT_ROOT)}")

    sanity = {}
    if not args.ohne_sanity and sanity_karten:
        sanity = messe_sanity(config, lauf, dataset, sanity_karten, device,
                              ig_steps, step_batch, patch)

    zusammenfassung = fasse_zusammen(ergebnisse)

    bericht = baue_bericht(args, lauf, ckpt.get("epoch"), device, ig_steps, step_batch,
                           patch, ergebnisse, zusammenfassung)
    bericht["sanity"] = sanity
    json_pfad, md_pfad = schreibe_bericht(stufen_dir, bericht, args.stufe)
    csv_pfad = schreibe_csv(stufen_dir, ergebnisse)

    # ---- Galerie ----------------------------------------------------------
    speichere_karten(npz_pfad, galerie_daten)
    quantile_bericht(galerie_daten)
    pfade = [
        galerie(galerie_daten["io"], "i.O.",
                os.path.join(abb_dir, f"{args.galerie_praefix}_io.png"), args.gamma),
        galerie(galerie_daten["nio"], "n.i.O.",
                os.path.join(abb_dir, f"{args.galerie_praefix}_nio.png"), args.gamma),
    ]
    drucke_zusammenfassung(zusammenfassung, ig_steps)
    for p in [json_pfad, md_pfad, csv_pfad, npz_pfad] + pfade:
        print(f"  geschrieben: {os.path.relpath(p, PROJECT_ROOT)}")


def sichtungstafel(original, karte, pfad, gamma=SICHTUNG_GAMMA):
    """Eine unbeschriftete Tafel: links die Aufnahme, rechts dieselbe mit Karte.

    Ohne Titel, ohne Verfahrensnamen, ohne Dateinamen im Bild - der Bewerter
    soll die Karte sehen und sonst nichts. Die Darstellung entspricht der der
    Galerie (pixelweise Deckkraft, `karte**gamma`), damit Sichtung und
    abgebildete Heatmaps denselben Eindruck ergeben.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
    for ax in axes:
        ax.imshow(original)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    gezeigt = np.clip(karte, 0.0, 1.0) ** gamma
    axes[1].imshow(gezeigt, cmap="inferno", vmin=0.0, vmax=1.0, alpha=gezeigt * ALPHA_MAX)
    fig.tight_layout(pad=0.2)
    fig.subplots_adjust(wspace=0.02)
    fig.savefig(pfad, dpi=110, bbox_inches="tight")
    plt.close(fig)


def messe_sanity(config, lauf, dataset, sanity_karten, device, ig_steps, step_batch, patch):
    """Sanity-Check nach Adebayo et al. (2018), kaskadierende Gewichtsrandomisierung.

    Fuer jede Randomisierungsstufe entsteht ein frisches Modell aus dem
    Checkpoint, dessen letzte k gewichtstragende Schichten neu initialisiert
    werden. Verglichen wird die Karte dieses Modells mit der des trainierten -
    ueber die Rangkorrelation, weil die absoluten Betraege nach der
    Randomisierung ohnehin nicht mehr vergleichbar sind.

    Ein Verfahren, dessen Korrelation hoch bleibt, erklaert nicht das Modell: es
    zeigt eine Eigenschaft des Bildes. Genau dieser Fall ist bei Adebayo et al.
    fuer mehrere gaengige Verfahren eingetreten und der Grund, warum der Test
    hier mitlaeuft.
    """
    print(f"\nSanity-Check (Adebayo et al.): {len(sanity_karten)} Aufnahmen, "
          f"Stufen {list(SANITY_STUFEN)}")
    aus = {m: {} for m in METHODEN}
    for stufe in SANITY_STUFEN:
        modell = build_model(config)
        ckpt = torch.load(os.path.join(lauf["run_dir"], "checkpoints", "best.pt"),
                          map_location="cpu")
        modell.load_state_dict(ckpt["model_state_dict"])
        modell, gesamt = randomisiere_schichten(modell, stufe, seed=stufe)
        modell.to(device).eval()

        werte = {m: [] for m in METHODEN}
        for idx, daten in sanity_karten.items():
            tensor, _ = dataset[idx]
            x = tensor.unsqueeze(0).to(device)
            ziel = daten["ziel"]
            neu = {
                "integrated_gradients": integrated_gradients(modell, x, ziel, steps=ig_steps,
                                                             step_batch=step_batch),
                "saliency": saliency_map(modell, x, ziel),
                "occlusion": occlusion_map(modell, x, ziel, patch_size=patch),
            }
            for m in METHODEN:
                werte[m].append(sanity_spearman(daten["karten"][m], neu[m])["betrag"])
        for m in METHODEN:
            aus[m][str(stufe)] = _agg(werte[m])
        print(f"  {stufe:2d} von {gesamt} Schichten randomisiert: " + " | ".join(
            f"{METHODEN_NAME[m][:3]} {aus[m][str(stufe)]['mittel']:.3f}" for m in METHODEN))
        del modell
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"stufen": list(SANITY_STUFEN), "n": len(sanity_karten), "betrag": aus}


def schreibe_csv(stufen_dir, ergebnisse):
    """Eine Zeile je Aufnahme und Verfahren - Datengrundlage der Diagramme.

    Die JSON traegt dieselben Zahlen, ist aber verschachtelt; fuer Boxplots und
    Kurven ist eine flache Tabelle die brauchbarere Form. Semikolon und
    Dezimalpunkt: die Datei wird von `kap5_daten.py` gelesen, nicht von Excel -
    das deutsche Komma entsteht erst beim Setzen des Textes ueber `de()`.
    """
    import csv

    pfad = os.path.join(stufen_dir, "xai_kennzahlen.csv")
    spalten = (["index", "datei", "wahre_klasse", "vorhersage", "p_nio", "ziel_klasse",
                "maskenanteil", "maskenquelle", "maske_plausibel", "verfahren",
                "attributionsanteil", "konzentrationsfaktor", "karte_belegt"]
               + [f"faktor_{s}" for s in SENS_STUFEN]
               + ["deletion_auc", "insertion_auc"])
    with open(pfad, "w", encoding="utf-8", newline="") as f:
        schreiber = csv.DictWriter(f, fieldnames=spalten, delimiter=";")
        schreiber.writeheader()
        for e in ergebnisse:
            for m in METHODEN:
                anteil = e["attributionsanteil"][m]
                zeile = {
                    "index": e["index"], "datei": e["datei"],
                    "wahre_klasse": e["wahre_klasse"], "vorhersage": e["vorhersage"],
                    "p_nio": e["p_nio"], "ziel_klasse": e["ziel_klasse"],
                    "maskenanteil": e["maskenanteil"], "maskenquelle": e["maskenquelle"],
                    "maske_plausibel": int(e.get("maske_plausibel", True)),
                    "verfahren": m,
                    "attributionsanteil": anteil,
                    "konzentrationsfaktor": e["konzentrationsfaktor"][m],
                    "karte_belegt": int(anteil == anteil),
                }
                for s in SENS_STUFEN:
                    zeile[f"faktor_{s}"] = e["maskensensitivitaet"][m][str(s)]["faktor"]
                di = e.get("deletion_insertion", {}).get(m)
                zeile["deletion_auc"] = di["deletion_auc"] if di else ""
                zeile["insertion_auc"] = di["insertion_auc"] if di else ""
                schreiber.writerow(zeile)
    return pfad


def _agg(werte):
    """
    Aggregat ueber die Bilder, unbelegte Karten ausgenommen.

    Eine Occlusion-Karte kann vollstaendig null sein - dann senkt kein einziger
    abgedeckter Bildbereich die Wahrscheinlichkeit der vorhergesagten Klasse,
    und ein Anteil auf dem Bauteil ist nicht definiert. Diese Faelle werden
    gezaehlt statt stillschweigend mitgemittelt.
    """
    arr = np.array(werte, dtype=float)
    gueltig = arr[~np.isnan(arr)]
    if gueltig.size == 0:
        return {"mittel": float("nan"), "median": float("nan"), "q25": float("nan"),
                "q75": float("nan"), "std": float("nan"), "min": float("nan"),
                "max": float("nan"), "n_gueltig": 0, "n_gesamt": int(arr.size)}
    # Median und Quartile stehen neben dem Mittelwert, weil die Verteilungen
    # rechtsschief sind: bei der Completeness trug eine einzelne Aufnahme
    # frueher gut die Haelfte des Mittelwerts. Ueber sechs Aufnahmen liess sich
    # das nur anmerken, ueber den ganzen Testanteil laesst es sich berichten.
    return {"mittel": float(gueltig.mean()), "median": float(np.median(gueltig)),
            "q25": float(np.quantile(gueltig, 0.25)), "q75": float(np.quantile(gueltig, 0.75)),
            "std": float(gueltig.std(ddof=1)) if gueltig.size > 1 else 0.0,
            "min": float(gueltig.min()), "max": float(gueltig.max()),
            "n_gueltig": int(gueltig.size), "n_gesamt": int(arr.size)}


def _hole(e, *pfad, default=float("nan")):
    """Verschachtelten Wert holen; fehlt er, NaN. Haelt `--nur-galerie` auf
    aelteren Berichten lauffaehig, die die neuen Kennzahlen nicht kennen."""
    for teil in pfad:
        if not isinstance(e, dict) or teil not in e:
            return default
        e = e[teil]
    return e


def fasse_zusammen(ergebnisse):
    zusammen = {
        "n": len(ergebnisse),
        "n_io": sum(1 for e in ergebnisse if e["wahre_klasse"] == "io"),
        "n_nio": sum(1 for e in ergebnisse if e["wahre_klasse"] == "nio"),
        "completeness": {
            stufe: _agg([_hole(e, "completeness", stufe, "relative_abweichung")
                         for e in ergebnisse])
            for stufe in ("betrieb", "fein")
        },
        "attributionsanteil": {
            m: _agg([e["attributionsanteil"][m] for e in ergebnisse]) for m in METHODEN
        },
        "konzentrationsfaktor": {
            m: _agg([_hole(e, "konzentrationsfaktor", m) for e in ergebnisse])
            for m in METHODEN
        },
        "maskenanteil": _agg([e["maskenanteil"] for e in ergebnisse]),
        "maskenquellen": {q: sum(1 for e in ergebnisse if e.get("maskenquelle") == q)
                          for q in ("abgelegt", "otsu", "heuristik")},
        "masken_unplausibel": sum(1 for e in ergebnisse
                                  if not e.get("maske_plausibel", True)),
        # Nach Klasse getrennt, weil sich die Verfahren dort unterschiedlich
        # verhalten: die Occlusion-Karte ist auf i.O.-Aufnahmen regelmaessig leer,
        # auf n.i.O.-Aufnahmen nicht. Ein gemeinsamer Mittelwert verwischt das.
        "je_klasse": {
            klasse: {
                m: _agg([_hole(e, "konzentrationsfaktor", m) for e in ergebnisse
                         if e["wahre_klasse"] == klasse])
                for m in METHODEN
            } for klasse in ("io", "nio")
        },
    }
    zusammen["maskensensitivitaet"] = {
        m: {str(s): _agg([_hole(e, "maskensensitivitaet", m, str(s), "faktor")
                          for e in ergebnisse])
            for s in SENS_STUFEN}
        for m in METHODEN
    }
    zusammen["deletion_insertion"] = {
        schluessel: {
            groesse: _agg([_hole(e, "deletion_insertion", schluessel, groesse)
                           for e in ergebnisse])
            for groesse in ("deletion_auc", "insertion_auc")
        } for schluessel in METHODEN + ["zufall"]
    }
    zusammen["kurven"] = _mittlere_kurven(ergebnisse)
    return zusammen


def _mittlere_kurven(ergebnisse):
    """Ueber alle Aufnahmen gemittelte Deletion- und Insertion-Kurven.

    Grundlage des Kurvendiagramms in Kapitel 6. Gemittelt wird ueber Aufnahmen,
    nicht ueber Verfahren; die Stuetzstellen sind bei allen dieselben, weil
    `deletion_insertion` sie fest ueber `np.linspace` legt.
    """
    aus = {}
    for schluessel in METHODEN + ["zufall"]:
        vorhanden = [e["deletion_insertion"][schluessel] for e in ergebnisse
                     if schluessel in e.get("deletion_insertion", {})]
        if not vorhanden:
            continue
        aus[schluessel] = {
            "anteile": vorhanden[0]["anteile"],
            "deletion": np.mean([v["deletion_kurve"] for v in vorhanden], axis=0).tolist(),
            "insertion": np.mean([v["insertion_kurve"] for v in vorhanden], axis=0).tolist(),
            "n": len(vorhanden),
        }
    return aus


def baue_bericht(args, lauf, epoche, device, ig_steps, step_batch, patch,
                 ergebnisse, zusammenfassung):
    return {
        "erzeugt": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "lauf": lauf["run"],
        "backbone": lauf["backbone"],
        "freeze": lauf["freeze"],
        "checkpoint_epoche": epoche,
        "split": args.split,
        "device": str(device),
        "ig_steps_betrieb": ig_steps,
        "ig_steps_fein": STEPS_FEIN,
        "ig_step_batch": step_batch,
        "occ_patch_size": patch,
        "maskenanteil_soll": MASKE_ANTEIL,
        "toleranz_test": TOLERANZ,
        "umfang": args.umfang,
        "maske_ersatzweg": args.maske,
        "sens_stufen": list(SENS_STUFEN),
        "di_schritte": DI_SCHRITTE,
        "di_n_je_klasse": args.di_n,
        "attributionsziel": "die vorhergesagte Klasse (argmax, entspricht Schwelle 0,5)",
        "bilder": ergebnisse,
        "zusammenfassung": zusammenfassung,
    }


def _pz(wert, stellen=1):
    return "—" if wert != wert else de(wert * 100, stellen) + " %"


def schreibe_bericht(stufen_dir, bericht, stufe):
    ergebnisse = bericht["bilder"]
    zusammenfassung = bericht["zusammenfassung"]
    ig_steps = bericht["ig_steps_betrieb"]

    json_pfad = os.path.join(stufen_dir, "xai_kennzahlen.json")
    with open(json_pfad, "w", encoding="utf-8") as f:
        json.dump(bericht, f, indent=1, ensure_ascii=False)

    z = zusammenfassung["completeness"]
    masken = zusammenfassung["maskenanteil"]
    quellen = ", ".join(f"{n}× {q}" for q, n in zusammenfassung["maskenquellen"].items() if n)
    zeilen = [
        f"# Versuchsstufe {stufe} – XAI-Kennzahlen",
        "",
        f"Erzeugt: {bericht['erzeugt']}  ",
        f"Lauf: `{bericht['lauf']}`, Checkpoint aus Epoche {bericht['checkpoint_epoche']}, "
        f"Split {bericht['split']}.  ",
        f"Umfang `{bericht.get('umfang', 'fest')}`: {zusammenfassung['n']} Aufnahmen "
        f"({zusammenfassung['n_io']} i.O. / {zusammenfassung['n_nio']} n.i.O.). "
        f"Attributionsziel: {bericht['attributionsziel']}.  ",
        f"Bauteilmaske: {quellen}; Flächenanteil {_pz(masken['mittel'])} im Mittel "
        f"({_pz(masken['min'])} … {_pz(masken['max'])}), "
        f"{zusammenfassung['masken_unplausibel']} als unplausibel markiert.",
        "",
        "## Konzentrationsfaktor",
        "",
        "Massenanteil geteilt durch Flächenanteil der Maske. 1,0 = gleichverteilte Karte.",
        "",
        "| Verfahren | Belegte Karten | Median | Q25 – Q75 | Mittel | Min – Max | "
        "davon i.O. | davon n.i.O. |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in METHODEN:
        k = zusammenfassung["konzentrationsfaktor"][m]
        io = zusammenfassung["je_klasse"]["io"][m]
        nio = zusammenfassung["je_klasse"]["nio"][m]
        zeilen.append(
            f"| {METHODEN_NAME[m]} | {k['n_gueltig']} von {k['n_gesamt']} | "
            f"{de(k['median'], 2)} | {de(k['q25'], 2)} – {de(k['q75'], 2)} | "
            f"{de(k['mittel'], 2)} | {de(k['min'], 2)} – {de(k['max'], 2)} | "
            f"{de(io['median'], 2)} ({io['n_gueltig']}) | "
            f"{de(nio['median'], 2)} ({nio['n_gueltig']}) |")
    zeilen += ["", "„Belegte Karten“ zählt die Aufnahmen, für die das Verfahren überhaupt "
                   "Attribution ausweist; eine durchgehend leere Karte trägt keinen Anteil "
                   "bei und wird nicht mitgemittelt.",
               "", "## Maskensensitivität", "",
               "Konzentrationsfaktor (Median) auf erodierter bzw. dilatierter Maske. "
               "Zeigt, wie stark das Ergebnis an der Maskengrenze hängt.", "",
               "| Verfahren | " + " | ".join(f"{s:+d} px" if s else "unverändert"
                                             for s in SENS_STUFEN) + " |",
               "|---|" + "---|" * len(SENS_STUFEN)]
    for m in METHODEN:
        s = zusammenfassung["maskensensitivitaet"][m]
        zeilen.append(f"| {METHODEN_NAME[m]} | "
                      + " | ".join(de(s[str(k)]["median"], 2) for k in SENS_STUFEN) + " |")

    di = zusammenfassung["deletion_insertion"]
    if any(di[k]["deletion_auc"]["n_gueltig"] for k in di):
        zeilen += ["", "## Deletion und Insertion", "",
                   f"Fläche unter der Wahrscheinlichkeitskurve über {DI_SCHRITTE} Schritten, "
                   "ohne Bauteilmaske gerechnet. Deletion: **klein** ist gut. "
                   "Insertion: **groß** ist gut. Bezug ist die zufällige Reihenfolge.", "",
                   "| Reihenfolge | Deletion-AUC | Insertion-AUC |", "|---|---|---|"]
        for k in METHODEN + ["zufall"]:
            name = METHODEN_NAME.get(k, "Zufällige Reihenfolge")
            d, i = di[k]["deletion_auc"], di[k]["insertion_auc"]
            if not d["n_gueltig"]:
                continue
            zeilen.append(f"| {name} | {de(d['median'], 3)} | {de(i['median'], 3)} |")

    zeilen += ["", "## Completeness der Integrated Gradients", "",
               f"Mittelwert {de(z['betrieb']['mittel'], 4)}, Median "
               f"{de(z['betrieb']['median'], 4)}, Maximum {de(z['betrieb']['max'], 4)} "
               f"bei {ig_steps} Stützpunkten über {z['betrieb']['n_gueltig']} Aufnahmen. "
               f"Bei {STEPS_FEIN} Stützpunkten (Teilmenge von {z['fein']['n_gueltig']}): "
               f"Mittelwert {de(z['fein']['mittel'], 4)}, Maximum {de(z['fein']['max'], 4)}. "
               f"Toleranz des automatisierten Tests: {de(TOLERANZ, 2)}."]

    sanity = bericht.get("sanity") or {}
    if sanity:
        zeilen += ["", "## Sanity-Check (Adebayo et al. 2018)", "",
                   f"Betrag der Rangkorrelation gegen die Karte des trainierten Modells, "
                   f"gemittelt über {sanity['n']} Aufnahmen. Ein Wert nahe null heißt: "
                   f"die Karte hängt am Modell.", "",
                   "| Verfahren | " + " | ".join(f"{s} Schichten" for s in sanity["stufen"])
                   + " |", "|---|" + "---|" * len(sanity["stufen"])]
        for m in METHODEN:
            zeilen.append(f"| {METHODEN_NAME[m]} | " + " | ".join(
                de(sanity["betrag"][m][str(s)]["mittel"], 3) for s in sanity["stufen"]) + " |")

    md_pfad = os.path.join(stufen_dir, "xai_kennzahlen.md")
    with open(md_pfad, "w", encoding="utf-8") as f:
        f.write("\n".join(zeilen) + "\n")
    return json_pfad, md_pfad


def drucke_zusammenfassung(zusammenfassung, ig_steps):
    z = zusammenfassung["completeness"]
    print()
    print(f"Aufnahmen: {zusammenfassung['n']} "
          f"({zusammenfassung['n_io']} i.O. / {zusammenfassung['n_nio']} n.i.O.), "
          f"Maskenanteil {_pz(zusammenfassung['maskenanteil']['mittel'])}")
    print(f"Completeness (rel. Abweichung): Mittel {de(z['betrieb']['mittel'], 4)} "
          f"bei {ig_steps} Stützpunkten, {de(z['fein']['mittel'], 4)} bei {STEPS_FEIN} "
          f"(n = {z['fein']['n_gueltig']})")
    for m in METHODEN:
        k = zusammenfassung["konzentrationsfaktor"][m]
        a = zusammenfassung["attributionsanteil"][m]
        print(f"Konzentration {METHODEN_NAME[m]:22s}: Median {de(k['median'], 2)}× "
              f"(Q25–Q75 {de(k['q25'], 2)}–{de(k['q75'], 2)}), "
              f"Anteil {_pz(a['median'])}, belegt {k['n_gueltig']}/{k['n_gesamt']}")
    di = zusammenfassung["deletion_insertion"]
    for k in METHODEN + ["zufall"]:
        d = di[k]["deletion_auc"]
        if d["n_gueltig"]:
            print(f"Deletion-AUC  {METHODEN_NAME.get(k, 'Zufall'):22s}: "
                  f"{de(d['median'], 3)}  |  Insertion-AUC "
                  f"{de(di[k]['insertion_auc']['median'], 3)}")


if __name__ == "__main__":
    main()
