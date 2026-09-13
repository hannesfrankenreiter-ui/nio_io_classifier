#!/usr/bin/env python3
"""
scripts/kap5_daten.py
Gemeinsame Datenbasis des Ergebniskapitels: liest die abgelegten Ergebnisse
einer Versuchsstufe ein und stellt sie einheitlich bereit. Wird von
`thesis_kap5_abbildungen.py` (Diagramme) und `build_kap5_payload.py`
(Word-Inhalte) genutzt, damit Abbildungen und Tabellen garantiert dieselben
Zahlen zeigen.

Kein torch - ausschliesslich JSON, CSV und die Logdateien der Laeufe.
"""
import csv
import json
import os
import re

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- Anzeigenamen ---------------------------------------------------------
BACKBONE_NAME = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny": "ConvNeXt-Tiny",
}
# Reihenfolge = zunehmende Tiefe des Fine-Tunings
FREEZE_ORDER = ["nurkopf", "letzterblock", "vollstaendig"]
FREEZE_NAME = {
    "nurkopf": "nur Kopf",
    "letzterblock": "letzter Block",
    "vollstaendig": "vollständig",
}


def de(wert, stellen: int = 4) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{wert:.{stellen}f}".replace(".", ",").replace("-", "−")


def de_tausend(wert: int) -> str:
    """Ganzzahl mit Punkt als Tausendertrennzeichen."""
    return f"{wert:,}".replace(",", ".")


def _zahl(s: str) -> float:
    """Deutsches Dezimalkomma aus den erzeugten CSV-Tabellen zurueckwandeln."""
    return float(s.replace(",", "."))


def _beste_epoche(run_dir: str) -> int:
    """
    Epoche des gesicherten Modellstands.

    Das Loss-Minimum der val-Kurve taugt dafuer NICHT: `EarlyStopping` und die
    best.pt-Auswahl verlangen eine Verbesserung um mehr als `delta` (0,001).
    Eine spaetere Epoche mit minimal kleinerem Loss wird deshalb nicht mehr
    gesichert - bei S1_05 und S1_12 weichen Loss-Minimum und Checkpoint
    auseinander. Massgeblich ist der Wert, den der XAI-Nachlauf direkt aus
    `best.pt` gelesen und protokolliert hat.
    """
    for split in ("test", "val"):
        log = os.path.join(run_dir, "xai", split, "xai.log")
        if not os.path.isfile(log):
            continue
        with open(log, encoding="utf-8", errors="replace") as f:
            treffer = re.search(r"Checkpoint aus Epoche (\d+)", f.read())
        if treffer:
            return int(treffer.group(1))
    raise FileNotFoundError(f"Checkpoint-Epoche nicht ermittelbar: {run_dir}")


def _laufzeit(run_dir: str):
    """
    Wanduhrzeit des Trainings in Sekunden.

    Rueckgabe (sekunden, quelle). `metrics/laufzeit.json` fehlt bei zwei
    Laeufen; dort werden ersatzweise die Zeitstempel des Trainingslogs
    herangezogen. Die Herkunft wird mitgegeben, damit sie in der Tabelle
    ausgewiesen werden kann.
    """
    pfad = os.path.join(run_dir, "metrics", "laufzeit.json")
    if os.path.isfile(pfad):
        with open(pfad, encoding="utf-8") as f:
            return float(json.load(f)["wanduhr_sekunden"]), "laufzeit.json"

    stempel = []
    with open(os.path.join(run_dir, "train.log"), encoding="utf-8", errors="replace") as f:
        for zeile in f:
            treffer = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", zeile)
            if treffer:
                stempel.append(treffer.group(1))
    if len(stempel) < 2:
        raise ValueError(f"Laufzeit nicht ermittelbar: {run_dir}")
    from datetime import datetime
    fmt = "%Y-%m-%d %H:%M:%S"
    dauer = datetime.strptime(stempel[-1], fmt) - datetime.strptime(stempel[0], fmt)
    return dauer.total_seconds(), "train.log"


def hhmmss(sekunden: float) -> str:
    s = int(round(sekunden))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def lade_laeufe(stufen_dir: str):
    """Sammelt je Lauf Konfiguration, Kennzahlen, Trainingsverlauf und Vorhersagen."""
    laeufe = []
    for name in sorted(os.listdir(stufen_dir)):
        run_dir = os.path.join(stufen_dir, name)
        test_json = os.path.join(run_dir, "metrics", "test_metrics.json")
        if not os.path.isfile(test_json):
            continue

        # Laufname des Versuchsplans: S<stufe>_<nr>_<backbone>_<freeze>
        #
        # Die laufende Nummer ist das Erkennungsmerkmal. Neben den Planlaeufen
        # liegen im Stufenordner auch Einzellaeufe von Hand (Stufe 2 etwa
        # S2_resnet18_vollstaendig_800x320, ein Vorversuch auf einem anderen
        # Zuschnitt). Die gehoeren nicht in den Vergleich und werden
        # uebersprungen - frueher riss ein solcher Ordner die ganze
        # Kapitel-5-Auswertung ab.
        treffer = re.fullmatch(r"S\d+_\d+_(.+)", name)
        if not treffer:
            print(f"  [uebersprungen] {name}: kein Lauf des Versuchsplans "
                  "(Schema S<stufe>_<nr>_<backbone>_<freeze>)")
            continue
        backbone, freeze = treffer.group(1).rsplit("_", 1)
        # Hier wird bewusst weiter hart abgebrochen: der Ordner traegt eine
        # Laufnummer, ist also als Planlauf gemeint - dann ist ein unbekanntes
        # Backbone ein Fehler und keine Fremddatei.
        if backbone not in BACKBONE_NAME or freeze not in FREEZE_NAME:
            raise ValueError(f"Laufname nicht deutbar: {name}")

        def _json(*teile):
            with open(os.path.join(run_dir, *teile), encoding="utf-8") as f:
                return json.load(f)

        test = _json("metrics", "test_metrics.json")
        val = _json("metrics", "val_metrics.json")
        schwelle = _json("metrics", "threshold.json")
        transfer = _json("metrics", "threshold_transfer.json")

        y_true, p_nio = [], []
        with open(os.path.join(run_dir, "preds", "example_predictions.csv"),
                  newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                y_true.append(1 if row["true_label"] == "nio" else 0)
                p_nio.append(float(row["prob_nio"]))

        val_ep, val_f1, val_loss = [], [], []
        with open(os.path.join(run_dir, "metrics", "val_metrics.csv"),
                  newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                val_ep.append(int(row["epoch"]))
                val_f1.append(float(row["f1"]))
                val_loss.append(float(row["loss"]))

        train_ep, train_loss = [], []
        with open(os.path.join(run_dir, "metrics", "train_metrics.csv"),
                  newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                train_ep.append(int(row["epoch"]))
                train_loss.append(float(row["loss"]))

        sekunden, zeitquelle = _laufzeit(run_dir)
        laeufe.append({
            "run": name,
            "nummer": name.split("_")[1],
            "backbone": backbone,
            "freeze": freeze,
            "test": test,
            "val": val,
            "f1_05": test["metrics_at_0.5"]["f1"],
            "schwelle": schwelle["threshold"],
            "transfer": transfer,
            "y_true": np.array(y_true),
            "p_nio": np.array(p_nio),
            "val_ep": val_ep,
            "val_f1": val_f1,
            "val_loss": val_loss,
            "train_ep": train_ep,
            "train_loss": train_loss,
            "epochen": len(val_ep),
            "beste_epoche": _beste_epoche(run_dir),
            "sekunden": sekunden,
            "zeitquelle": zeitquelle,
            "run_dir": run_dir,
        })
    return laeufe


def lade_inferenzzeit(stufen_dir: str):
    with open(os.path.join(stufen_dir, "inferenzzeit.json"), encoding="utf-8") as f:
        d = json.load(f)
    return {e["backbone"]: e for e in d["messungen"]}


def lade_sweep(lauf):
    """Schwellenraster eines Laufs (tau, Precision, Recall, F1, TP/FP/FN/TN auf test)."""
    zeilen = []
    with open(os.path.join(lauf["run_dir"], "metrics", "threshold_sweep.csv"),
              newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            zeilen.append({
                "tau": _zahl(row["Schwelle"]),
                "precision": _zahl(row["test Precision"]),
                "recall": _zahl(row["test Recall"]),
                "f1": _zahl(row["test F1"]),
                "tp": int(row["test TP"]), "fp": int(row["test FP"]),
                "fn": int(row["test FN"]), "tn": int(row["test TN"]),
            })
    return zeilen


METHODEN = ["integrated_gradients", "saliency", "occlusion"]
METHODEN_NAME = {
    "integrated_gradients": "Integrated Gradients",
    "saliency": "Saliency Map",
    "occlusion": "Occlusion Map",
    "zufall": "Zufällige Reihenfolge",
}


def lade_xai_kennzahlen(stufen_dir: str):
    """XAI-Kennzahlen einer Versuchsstufe: Zusammenfassung, Kurven, Zeilen je Aufnahme.

    Erzeugt von `scripts/xai_kennzahlen.py`. Die JSON traegt die Aggregate und
    die gemittelten Kurven, die CSV eine flache Zeile je Aufnahme und Verfahren -
    Letztere ist die Grundlage der Boxplots, weil dafuer die Einzelwerte noetig
    sind und nicht ihre Kennzahlen.

    Damit gilt fuer die XAI-Abschnitte dieselbe Regel wie fuer das uebrige
    Kapitel: Abbildung und Tabelle lesen dieselbe Datei, also koennen sie nicht
    auseinanderlaufen.
    """
    json_pfad = os.path.join(stufen_dir, "xai_kennzahlen.json")
    with open(json_pfad, encoding="utf-8") as f:
        bericht = json.load(f)

    csv_pfad = os.path.join(stufen_dir, "xai_kennzahlen.csv")
    zeilen = []
    if os.path.isfile(csv_pfad):
        with open(csv_pfad, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                eintrag = dict(row)
                for schluessel in ("konzentrationsfaktor", "attributionsanteil",
                                   "maskenanteil", "p_nio", "deletion_auc",
                                   "insertion_auc"):
                    eintrag[schluessel] = _float_oder_nan(row.get(schluessel))
                eintrag["index"] = int(row["index"])
                eintrag["karte_belegt"] = row["karte_belegt"] == "1"
                zeilen.append(eintrag)
    bericht["zeilen"] = zeilen
    return bericht


def _float_oder_nan(s):
    """Leere Zelle oder NaN-Text zu float('nan') - beides kommt in der CSV vor.

    Leer steht fuer "nicht gemessen" (Deletion/Insertion laeuft auf einer
    Teilmenge), `nan` fuer "gemessen, aber nicht definiert" (leere Karte). Beide
    duerfen nicht als Zahl in einen Mittelwert geraten, deshalb dieselbe
    Behandlung an dieser Stelle - unterschieden wird ueber `karte_belegt`.
    """
    if s is None or s == "":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def di_je_klasse(bericht, groesse: str = "deletion_auc"):
    """Deletion- bzw. Insertion-Flaeche je Verfahren UND Klasse (Median).

    Der ueber beide Klassen gemittelte Wert ist bei dieser Kennzahl irrefuehrend,
    weil sie zwei verschiedene Regime mittelt: Auf einer n.i.O.-Aufnahme gibt es
    eine oertlich begrenzte Evidenz, die eine Karte treffen kann; auf einer
    i.O.-Aufnahme gibt es keine - die Abwesenheit eines Fehlers ist keine Stelle
    im Bild. Entsprechend liegen dort alle Reihenfolgen einschliesslich der
    zufaelligen dicht beieinander.

    Rueckgabe: {verfahren: {"io": median, "nio": median, "n_io": .., "n_nio": ..}}
    """
    aus = {}
    for m in METHODEN + ["zufall"]:
        eintrag = {}
        for klasse in ("io", "nio"):
            werte = [b["deletion_insertion"][m][groesse] for b in bericht["bilder"]
                     if b["wahre_klasse"] == klasse
                     and m in b.get("deletion_insertion", {})]
            werte = [w for w in werte if w == w]
            eintrag[klasse] = float(np.median(werte)) if werte else float("nan")
            eintrag[f"n_{klasse}"] = len(werte)
        aus[m] = eintrag
    return aus


def di_kurven_je_klasse(bericht):
    """Ueber die Aufnahmen gemittelte Deletion-Kurven, getrennt nach Klasse."""
    aus = {}
    for klasse in ("io", "nio"):
        aus[klasse] = {}
        for m in METHODEN + ["zufall"]:
            vorhanden = [b["deletion_insertion"][m] for b in bericht["bilder"]
                         if b["wahre_klasse"] == klasse
                         and m in b.get("deletion_insertion", {})]
            if not vorhanden:
                continue
            aus[klasse][m] = {
                "anteile": vorhanden[0]["anteile"],
                "deletion": np.mean([v["deletion_kurve"] for v in vorhanden],
                                    axis=0).tolist(),
                "insertion": np.mean([v["insertion_kurve"] for v in vorhanden],
                                     axis=0).tolist(),
                "n": len(vorhanden),
            }
    return aus


def completeness_absolut(bericht):
    """Absolute Abweichung der Completeness - ohne den Nenner der relativen.

    Die relative Abweichung teilt durch die Differenz der Modellausgaben. Liegt
    die nahe null - bei einzelnen Aufnahmen ist sie es -, wird aus einem
    winzigen absoluten Fehler ein riesiger relativer. Der Mittelwert der
    relativen Abweichung haengt dadurch an wenigen Aufnahmen; die absolute
    Groesse hat dieses Problem nicht.
    """
    werte = sorted(abs(b["completeness"]["betrieb"]["abweichung"])
                   for b in bericht["bilder"])
    nenner = sorted(abs(b["completeness"]["betrieb"]["delta_f"])
                    for b in bericht["bilder"])
    return {
        "median": float(np.median(werte)),
        "q75": float(np.quantile(werte, 0.75)),
        "max": float(werte[-1]),
        "nenner_median": float(np.median(nenner)),
        "nenner_min": float(nenner[0]),
        "n": len(werte),
    }


def werte_je_verfahren(bericht, spalte: str, verfahren: str, klasse: str = None):
    """Einzelwerte einer Spalte fuer ein Verfahren, ohne die undefinierten.

    Fuer Boxplots: matplotlib zeichnet NaN nicht weg, es verschiebt die Quartile.
    """
    aus = [z[spalte] for z in bericht["zeilen"]
           if z["verfahren"] == verfahren
           and (klasse is None or z["wahre_klasse"] == klasse)]
    return [w for w in aus if w == w]


def lade_manifest(lauf, split: str = "test"):
    """XAI-Manifest eines Laufs: Dateiname je Bildindex plus Vorhersagedaten."""
    pfad = os.path.join(lauf["run_dir"], "xai", split, "manifest.csv")
    with open(pfad, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def konfusion(lauf, tau: float = 0.5):
    """
    Konfusionsmatrix [[TN, FP], [FN, TP]] auf dem Testanteil.

    Vergleich mit `>=` wie im uebrigen Projekt (`src/evaluation/evaluator.py`,
    `scripts/threshold_analyse.py`). Mit `>` weicht das Ergebnis genau dann ab,
    wenn eine Wahrscheinlichkeit exakt auf der Schwelle liegt - was bei den aus
    den Vorhersagen abgeleiteten Schwellen der Regelfall ist.
    """
    pred = (lauf["p_nio"] >= tau).astype(int)
    y = lauf["y_true"]
    return np.array([
        [int(((y == 0) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())],
        [int(((y == 1) & (pred == 0)).sum()), int(((y == 1) & (pred == 1)).sum())],
    ])


def rangfolge(laeufe, zeiten):
    """
    Sortiert nach dem in Abschnitt 4.5.2 vorab festgelegten Auswahlkriterium:
    F1 (n.i.O.) bei tau = 0,5 auf test, dann ROC-AUC, dann Rechenzeit je Bild.
    """
    return sorted(laeufe, key=lambda l: (-l["f1_05"], -l["test"]["roc_auc"],
                                         zeiten[l["backbone"]]["batch_median"]))
