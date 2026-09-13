#!/usr/bin/env python3
"""
scripts/threshold_analyse.py
Betriebspunkt-Analyse aus den abgelegten Vorhersagen eines Trainingslaufs.

Erzeugt zwei Dinge, die der Evaluator selbst nicht liefert:

1. SCHWELLENWERT-TABELLE (metrics/threshold_sweep.csv|md)
   Precision, Recall, F1 und Konfusionsmatrix der n.i.O.-Klasse ueber ein Raster
   von Schwellenwerten, getrennt fuer val und test. Das ist die in Abschnitt
   4.5.3 zugesagte Darstellung des "erreichbaren Spielraums".

2. UEBERTRAGBARKEIT val -> test (metrics/threshold_transfer.json)
   Wo liegt das F1-Optimum auf val, wo auf test, und was kostet es, den auf val
   bestimmten Betriebspunkt unveraendert auf test anzuwenden? Zusaetzlich die
   Breite des val-Plateaus und die Spanne der test-F1-Werte darueber - sie zeigt,
   wie stark die Rangfolge allein von der Schwellenwahl abhaengen kann.

Warum eigenstaendig und nicht im Evaluator: die Analyse braucht nur die bereits
gespeicherten Wahrscheinlichkeiten. So laesst sie sich auf ALLE Laeufe anwenden,
auch auf laengst abgeschlossene, ohne ein einziges Modell neu zu rechnen und ohne
die Vergleichbarkeit laufender Experimente zu beruehren.

Aufruf:
    python scripts/threshold_analyse.py --run outputs/runs/Stufe_1/S1_01_...
    python scripts/threshold_analyse.py --stufe 1      # alle Laeufe der Stufe
"""
import argparse
import csv
import glob
import json
import os
import sys

# Windows waehlt fuer eine umgeleitete stdout cp1252. Die Zusammenfassung unten
# schreibt ein 'tau', und ohne diese Zeile reisst ein einziges Zeichen die ganze
# Auswertung ab - passiert am 19.08.2026 nach der Analyse aller zwoelf Stufe-2-
# Laeufe, kurz vor dem Schreiben von threshold_transfer.md. Dieselbe Absicherung
# steht in run_versuchsplan.py und run_xai_versuchsplan.py.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raster der untersuchten Schwellenwerte fuer die Tabelle
RASTER = [round(x, 2) for x in np.arange(0.05, 1.00, 0.05)]
# Feines Raster fuer die Optimumsuche
FEIN = np.linspace(0.005, 0.995, 199)
# Ein Lauf gilt als "auf dem Plateau", wenn er hoechstens so weit unter dem Maximum liegt
PLATEAU_TOLERANZ = 0.005


def _fmt(v, nk=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{nk}f}".replace(".", ",")


def lade_vorhersagen(pfad: str):
    """Liest (labels, prob_nio) aus einer example_predictions-CSV."""
    y, p = [], []
    with open(pfad, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            y.append(1 if r["true_label"] == "nio" else 0)
            p.append(float(r["prob_nio"]))
    return np.array(y), np.array(p)


def kennzahlen(y, p, t):
    """Kennzahlen der n.i.O.-Klasse bei Schwelle t, inkl. Konfusionsmatrix."""
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "threshold": float(t),
        "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall":    float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "f1":        float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "accuracy":  float((pred == y).mean()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def optimum(y, p):
    """F1-Maximum ueber das feine Raster. Gibt (schwelle, f1) zurueck."""
    werte = np.array([f1_score(y, p >= t, pos_label=1, zero_division=0) for t in FEIN])
    i = int(werte.argmax())
    return float(FEIN[i]), float(werte[i]), werte


# Recall-Stufen, fuer die der Preis in Fehlalarmen ausgewiesen wird
RECALL_ZIELE = [1.00, 0.99, 0.98, 0.95]


def fehlalarme_je_recall(y, p, ziele=RECALL_ZIELE):
    """Wie viele Fehlalarme kostet ein geforderter Mindest-Recall?

    Das ist der betrieblich entscheidende Vergleich: In der Sichtpruefung wiegt
    ein durchgelassener Defekt schwerer als ein Fehlalarm. Gefragt ist deshalb
    nicht "welches Modell hat den besten F1", sondern "wie viele einwandfreie
    Bauteile muss ich faelschlich aussortieren, damit (nahezu) kein Defekt
    durchrutscht".

    Betriebspunktfrei: die noetige Schwelle ergibt sich aus den Daten und wird
    nicht gewaehlt. Der Wert umgeht damit sowohl die Willkuer der festen
    Schwelle 0,5 als auch die Unzuverlaessigkeit der auf val bestimmten Schwelle.

    ACHTUNG bei recall = 1,0: dieser Wert ist ein Minimum ueber alle n.i.O.-
    Bilder und haengt damit an einem EINZIGEN, schwierigsten Bild. Bei
    EfficientNet-B0 sprang er dadurch von 11 auf 34 Fehlalarme, nur wegen einer
    Aufnahme mit prob_nio = 0,12. Fuer Rangfolgen deshalb die Stufe 0,99
    verwenden; 1,00 nur als Zusatzangabe berichten.
    """
    if not (y == 1).any() or not (y == 0).any():
        return None
    nio_sortiert = np.sort(p[y == 1])[::-1]   # absteigend
    n_nio, n_io = len(nio_sortiert), int((y == 0).sum())

    ergebnis = {}
    for z in ziele:
        noetig = int(np.ceil(z * n_nio))          # so viele n.i.O. muessen erkannt werden
        schwelle = float(nio_sortiert[noetig - 1])
        fp = int((p[y == 0] >= schwelle).sum())
        ergebnis[f"{z:.2f}"] = {
            "schwelle": schwelle,
            "fehlalarme": fp,
            "fehlalarmquote": fp / n_io,
            "uebersehene_defekte": n_nio - noetig,
        }
    ergebnis["n_io"] = n_io
    ergebnis["n_nio"] = n_nio
    return ergebnis


def analysiere(run_dir: str) -> dict:
    """Wertet einen Lauf aus. None, wenn die noetigen Dateien fehlen."""
    voll = run_dir if os.path.isabs(run_dir) else os.path.join(PROJECT_ROOT, run_dir)
    p_test = os.path.join(voll, "preds", "example_predictions.csv")
    p_val = os.path.join(voll, "preds", "example_predictions_val.csv")
    if not (os.path.exists(p_test) and os.path.exists(p_val)):
        return None

    yv, pv = lade_vorhersagen(p_val)
    yt, pt = lade_vorhersagen(p_test)

    t_val, f1_val, kurve_val = optimum(yv, pv)
    t_test, f1_test, _ = optimum(yt, pt)

    # Die tatsaechlich im Training bestimmte Schwelle - kann leicht vom Raster
    # abweichen, weil find_best_threshold die echten Ausgabewerte absucht.
    tj = os.path.join(voll, "metrics", "threshold.json")
    t_angewandt = t_val
    if os.path.exists(tj):
        with open(tj, encoding="utf-8") as f:
            t_angewandt = float(json.load(f).get("threshold", t_val))

    # Wie breit ist das val-Optimum, und was deckt es auf test ab?
    auf_plateau = kurve_val >= (f1_val - PLATEAU_TOLERANZ)
    f1t_auf_plateau = np.array(
        [f1_score(yt, pt >= t, pos_label=1, zero_division=0) for t in FEIN[auf_plateau]]
    )

    bericht = {
        "run": os.path.basename(voll.rstrip(os.sep)),
        "val": {
            "optimum_schwelle": t_val,
            "optimum_f1": f1_val,
            "plateau_von": float(FEIN[auf_plateau].min()),
            "plateau_bis": float(FEIN[auf_plateau].max()),
            "roc_auc": float(roc_auc_score(yv, pv)) if len(np.unique(yv)) > 1 else None,
        },
        "test": {
            "optimum_schwelle": t_test,
            "optimum_f1": f1_test,
            "f1_bei_val_schwelle": kennzahlen(yt, pt, t_angewandt)["f1"],
            "f1_bei_0.5": kennzahlen(yt, pt, 0.5)["f1"],
            "roc_auc": float(roc_auc_score(yt, pt)) if len(np.unique(yt)) > 1 else None,
            "average_precision": float(average_precision_score(yt, pt)),
            "f1_spanne_ueber_val_plateau": [float(f1t_auf_plateau.min()),
                                            float(f1t_auf_plateau.max())],
            "fehlalarme_je_recall": fehlalarme_je_recall(yt, pt),
        },
        "angewandte_schwelle": t_angewandt,
        "kosten_der_val_schwelle": (kennzahlen(yt, pt, t_angewandt)["f1"]
                                    - kennzahlen(yt, pt, 0.5)["f1"]),
        "hinweis": (
            "Negative 'kosten_der_val_schwelle' bedeuten: der auf val bestimmte "
            "Betriebspunkt liefert auf test einen SCHLECHTEREN F1 als die feste "
            "Schwelle 0,5. Ursache ist das Split-Design - val und test tragen je "
            "einen exklusiven Fehlertyp und sind daher nicht austauschbar."
        ),
    }

    # ---- Ausgabe schreiben ---------------------------------------
    mdir = os.path.join(voll, "metrics")
    with open(os.path.join(mdir, "threshold_transfer.json"), "w", encoding="utf-8") as f:
        json.dump(bericht, f, indent=2, ensure_ascii=False)

    zeilen = []
    for t in RASTER:
        v, s = kennzahlen(yv, pv, t), kennzahlen(yt, pt, t)
        zeilen.append([
            _fmt(t, 2),
            _fmt(v["precision"]), _fmt(v["recall"]), _fmt(v["f1"]),
            _fmt(s["precision"]), _fmt(s["recall"]), _fmt(s["f1"]),
            str(s["tp"]), str(s["fp"]), str(s["fn"]), str(s["tn"]),
        ])
    kopf = ["Schwelle", "val Precision", "val Recall", "val F1",
            "test Precision", "test Recall", "test F1",
            "test TP", "test FP", "test FN", "test TN"]

    with open(os.path.join(mdir, "threshold_sweep.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(kopf)
        w.writerows(zeilen)

    md = [f"# Schwellenwertanalyse – {bericht['run']}", "",
          "Kennzahlen der n.i.O.-Klasse über das Schwellenwertraster. "
          "Grundlage sind die im Lauf abgelegten Wahrscheinlichkeiten; "
          "es wurde nichts neu berechnet.", "",
          f"- val-Optimum: **τ = {_fmt(t_val, 3)}** (F1 = {_fmt(f1_val)}), "
          f"Plateau τ = {_fmt(bericht['val']['plateau_von'], 3)} … "
          f"{_fmt(bericht['val']['plateau_bis'], 3)}",
          f"- test-Optimum: **τ = {_fmt(t_test, 3)}** (F1 = {_fmt(f1_test)}) "
          f"— nicht nutzbar, nur als Obergrenze",
          f"- angewandte Schwelle (auf val bestimmt): **τ = {_fmt(t_angewandt, 3)}** "
          f"→ test-F1 = {_fmt(bericht['test']['f1_bei_val_schwelle'])}",
          f"- feste Schwelle 0,5 → test-F1 = {_fmt(bericht['test']['f1_bei_0.5'])}",
          f"- **Differenz: {_fmt(bericht['kosten_der_val_schwelle'])}**",
          f"- test-F1 über das gesamte val-Plateau: "
          f"{_fmt(bericht['test']['f1_spanne_ueber_val_plateau'][0])} … "
          f"{_fmt(bericht['test']['f1_spanne_ueber_val_plateau'][1])}",
          ""]
    fr = bericht["test"].get("fehlalarme_je_recall")
    if fr:
        md += ["## Preis eines geforderten Mindest-Recalls", "",
               "Wie viele einwandfreie Bauteile werden fälschlich aussortiert, "
               f"damit der geforderte Anteil der {fr['n_nio']} Defekte erkannt wird? "
               "Betriebspunktfrei — die Schwelle ergibt sich aus den Daten.", "",
               "| Mindest-Recall | nötige Schwelle | Fehlalarme | von i.O. | übersehene Defekte |",
               "|---|---|---|---|---|"]
        for z in RECALL_ZIELE:
            e = fr[f"{z:.2f}"]
            md.append(f"| {_fmt(z, 2)} | {_fmt(e['schwelle'], 3)} | "
                      f"{e['fehlalarme']} | {fr['n_io']} | {e['uebersehene_defekte']} |")
        md += ["", "> Die Stufe 1,00 hängt an einem einzigen, schwierigsten Bild und "
               "schwankt dadurch stark. Für Rangfolgen die Stufe 0,99 verwenden.", ""]
    md += [
          "| " + " | ".join(kopf) + " |",
          "|" + "|".join(["---"] * len(kopf)) + "|"]
    md += ["| " + " | ".join(z) + " |" for z in zeilen]
    with open(os.path.join(mdir, "threshold_sweep.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    return bericht


def main():
    p = argparse.ArgumentParser(description="Betriebspunkt-Analyse aus abgelegten Vorhersagen")
    p.add_argument("--run", default=None, help="Einzelner Run-Ordner")
    p.add_argument("--stufe", type=int, default=None, help="Alle Laeufe einer Versuchsstufe")
    args = p.parse_args()

    if args.run:
        ordner = [args.run]
    elif args.stufe is not None:
        muster = os.path.join(PROJECT_ROOT, "outputs", "runs",
                              f"Stufe_{args.stufe}", "*")
        ordner = sorted(d for d in glob.glob(muster) if os.path.isdir(d))
    else:
        p.error("--run oder --stufe angeben")

    berichte = []
    for d in ordner:
        b = analysiere(d)
        if b is None:
            continue
        berichte.append(b)
        print(f"{b['run']:<42} val τ={_fmt(b['val']['optimum_schwelle'], 3)}  "
              f"test τ={_fmt(b['test']['optimum_schwelle'], 3)}  "
              f"test-F1 @val={_fmt(b['test']['f1_bei_val_schwelle'])}  "
              f"@0,5={_fmt(b['test']['f1_bei_0.5'])}  "
              f"Diff={_fmt(b['kosten_der_val_schwelle'])}")

    if not berichte:
        print("Keine auswertbaren Laeufe gefunden.")
        return

    # ---- Sammeltabelle ueber alle Laeufe -------------------------
    if args.stufe is not None:
        zdir = os.path.join(PROJECT_ROOT, "outputs", "runs", f"Stufe_{args.stufe}")
        kopf = (["Lauf", "val-Optimum τ", "test-Optimum τ", "test-F1 @val-Schwelle",
                 "test-F1 @0,5", "Differenz", "test-F1-Spanne über val-Plateau"]
                + [f"Fehlalarme @R≥{z:.2f}".replace(".", ",") for z in RECALL_ZIELE])

        def _fa(b, z):
            fr = b["test"].get("fehlalarme_je_recall")
            return str(fr[f"{z:.2f}"]["fehlalarme"]) if fr else "—"

        zeilen = [[
            b["run"],
            _fmt(b["val"]["optimum_schwelle"], 3),
            _fmt(b["test"]["optimum_schwelle"], 3),
            _fmt(b["test"]["f1_bei_val_schwelle"]),
            _fmt(b["test"]["f1_bei_0.5"]),
            _fmt(b["kosten_der_val_schwelle"]),
            f"{_fmt(b['test']['f1_spanne_ueber_val_plateau'][0])} … "
            f"{_fmt(b['test']['f1_spanne_ueber_val_plateau'][1])}",
        ] + [_fa(b, z) for z in RECALL_ZIELE] for b in berichte]

        md = [f"# Übertragbarkeit des Betriebspunkts val → test (Stufe {args.stufe})", "",
              "Der Schwellenwert wird auf **val** F1-optimal bestimmt und unverändert "
              "auf **test** angewendet (Abschnitt 4.5.3). Die Tabelle zeigt, was diese "
              "Übertragung kostet.", "",
              "Eine negative Differenz bedeutet: der angepasste Betriebspunkt ist auf "
              "test **schlechter** als die feste Schwelle 0,5. Ursache ist das "
              "Split-Design — val und test tragen je einen exklusiven Fehlertyp und "
              "sind damit bewusst nicht austauschbar.", "",
              "| " + " | ".join(kopf) + " |",
              "|" + "|".join(["---"] * len(kopf)) + "|"]
        md += ["| " + " | ".join(z) + " |" for z in zeilen]

        with open(os.path.join(zdir, "threshold_transfer.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(md) + "\n")
        with open(os.path.join(zdir, "threshold_transfer.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(kopf)
            w.writerows(zeilen)
        print(f"\nSammeltabelle → outputs/runs/Stufe_{args.stufe}/threshold_transfer.md / .csv")


if __name__ == "__main__":
    main()
