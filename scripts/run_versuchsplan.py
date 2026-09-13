#!/usr/bin/env python3
"""
scripts/run_versuchsplan.py
Faehrt den Versuchsplan einer Versuchsstufe automatisch ab.

AUFBAU (zweiteilig, volles Raster)
----------------------------------
Schritt 1 – Backbone-Vergleich bei einheitlicher Konfiguration
    Vier Backbones, alle mit demselben Rezept (configs/default.yaml) und
    derselben Freeze-Strategie "letzter Block + Kopf". Variiert wird
    ausschliesslich das Backbone. (Bereits gelaufen, Laeufe 01–04.)

Schritt 2 – Freeze-Strategie, fuer JEDES Backbone
    nur Kopf | letzter Block + Kopf | vollstaendiges Fine-Tuning
    "letzter Block + Kopf" stammt aus Schritt 1 und wird wiederverwendet,
    die beiden anderen Varianten werden je Backbone neu trainiert.

    Bewusst das volle Raster (4 Backbones x 3 Strategien) statt nur des
    Siegerbackbones: die beste Freeze-Strategie muss nicht fuer alle
    Architekturen dieselbe sein. Ein ResNet-18 mit 11 M Parametern vertraegt
    vollstaendiges Fine-Tuning auf einem kleinen Datensatz anders als ein
    ConvNeXt-Tiny mit 28 M. Wuerde man die Freeze-Strategie nur am Sieger aus
    Schritt 1 variieren, waere die Endauswahl auf die Kombination beschraenkt,
    die unter der willkuerlich vorgegebenen Startstrategie gewonnen hat.
    Erst das volle Raster erlaubt die Aussage, welche Kombination aus Backbone
    und Freeze-Strategie insgesamt die beste ist.

KRITERIEN
---------
Auswahl: F1 (n.i.O.) bei Schwelle 0,5 auf TEST -> ROC-AUC -> Rechenzeit je Bild

  Die feste Schwelle 0,5 statt des auf val F1-optimalen Betriebspunkts ist eine
  bewusste Abweichung vom urspruenglichen Plan. Grund: val und test tragen je
  einen EXKLUSIVEN Fehlertyp und sind damit nicht austauschbar; der auf val
  optimale Betriebspunkt uebertraegt sich nachweislich nicht (t_val ~ 0,21 gegen
  t_test = 0,31…0,61; Kosten auf test 0,037…0,066 F1). Bei fester Schwelle wird
  nichts angepasst, also kann auch nichts falsch uebertragen werden.
  Die Tabellen fuehren die Rangfolgen nach F1@val-Schwelle, ROC-AUC und
  Fehlalarmen bei gefordertem Mindest-Recall nachrichtlich mit.

KONSTANT ueber alle Laeufe: Datensatz und Aufteilung, Seed, Bildgroesse,
Batchgroesse, Augmentierung, Klassengewichte, Lernraten, Weight Decay,
Abbruchkriterium.
VARIIERT: Backbone (Schritt 1) und Freeze-Strategie (Schritt 2).

Aufruf:
    python scripts/run_versuchsplan.py --stufe 1
    python scripts/run_versuchsplan.py --stufe 1 --dry-run
    python scripts/run_versuchsplan.py --stufe 1 --nur-schritt 2
    python scripts/run_versuchsplan.py --stufe 1 --backbone resnet18 resnet50
    python scripts/run_versuchsplan.py --stufe 1 --nur-tabelle

Wiederaufnahme: Laeufe mit vorhandener metrics/test_metrics.json werden
uebersprungen. Ein Abbruch kostet hoechstens den gerade laufenden Lauf.
"""
import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Windows waehlt fuer eine umgeleitete stdout cp1252 statt UTF-8. Der Treiber
# wird ueblicherweise nach tee/Datei gepiped, und ein einziges '→' in einer
# Statusmeldung reisst dann den ganzen Lauf ab - genau das ist am 10.08. nach
# dem zwoelften Trainingslauf passiert, kurz vor dem Schreiben der Tabellen.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Betriebspunktfreie Kennzahlen kommen aus threshold_analyse - eine Quelle,
# damit Vergleichstabelle und Schwellenwertanalyse nicht auseinanderlaufen.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from threshold_analyse import analysiere as _schwellenanalyse  # noqa: E402

# Reihenfolge bestimmt die Laufnummern. NICHT umsortieren – die Laeufe 01–04
# aus Schritt 1 liegen unter genau diesen Namen bereits auf der Platte.
BACKBONES = ["resnet18", "resnet50", "efficientnet_b0", "convnext_tiny"]

FREEZE_VARIANTEN = {"nurkopf": True, "letzterblock": "layer4", "vollstaendig": False}
FREEZE_LABEL = {
    "nurkopf":      "nur Kopf",
    "letzterblock": "letzter Block + Kopf",
    "vollstaendig": "vollstaendiges Fine-Tuning",
}
# Aufsteigende Zahl trainierbarer Parameter – Lesereihenfolge in den Tabellen.
FREEZE_REIHENFOLGE = ["nurkopf", "letzterblock", "vollstaendig"]

# Freeze-Strategie der Schritt-1-Laeufe. Diese Laeufe sind zugleich die
# mittlere Variante von Schritt 2 und werden dort wiederverwendet.
SCHRITT_1_FREEZE = "letzterblock"
# In Schritt 2 zusaetzlich zu trainieren.
SCHRITT_2_NEU = ["nurkopf", "vollstaendig"]


def laufplan(stufe: int) -> list:
    """Alle Laeufe des Versuchsplans in Ausfuehrungsreihenfolge.

    Die Nummerierung ist fortlaufend ueber beide Schritte, damit ein Ordnername
    den Lauf eindeutig benennt. Schritt 1 belegt 01–04 (bereits vorhanden),
    Schritt 2 die Nummern ab 05.
    """
    plan, nr = [], 0
    for bb in BACKBONES:
        nr += 1
        plan.append({"nr": nr, "backbone": bb, "freeze_key": SCHRITT_1_FREEZE,
                     "schritt": 1,
                     "run_name": f"S{stufe}_{nr:02d}_{bb}_{SCHRITT_1_FREEZE}"})
    for bb in BACKBONES:
        for fk in SCHRITT_2_NEU:
            nr += 1
            plan.append({"nr": nr, "backbone": bb, "freeze_key": fk,
                         "schritt": 2,
                         "run_name": f"S{stufe}_{nr:02d}_{bb}_{fk}"})
    return plan


# ================================================================
#  Config-Erzeugung
# ================================================================
def lade_basis_config(pfad: str) -> dict:
    with open(pfad, encoding="utf-8") as f:
        return yaml.safe_load(f)


def baue_config(basis: dict, backbone: str, freeze_key: str) -> dict:
    """Basis-Config mit genau zwei Aenderungen: Backbone und Freeze-Strategie.

    Alles Weitere bleibt wie in der Basis-Config – nur so ist der Vergleich
    ueber alle zwoelf Laeufe hinweg sauber.
    """
    cfg = copy.deepcopy(basis)
    cfg["backbone"] = backbone
    cfg["freeze_backbone"] = FREEZE_VARIANTEN[freeze_key]

    # XAI auf dem ganzen Split kostet ~2 h je Lauf (400 Bilder x ~15 s) und
    # liefert keine Groesse des Versuchsplans. Die Heatmaps entstehen spaeter
    # gezielt ueber scripts/run_xai_versuchsplan.py, dort auf einer Auswahl.
    #
    # Der Unterblock xai.rueckprojektion bleibt BEWUSST unberuehrt: er hat in
    # train.py seinen eigenen Schalter und kostet nur die zwoelf Bilder aus
    # xai.rueckprojektion.n_samples. Damit bekommt jeder Vergleichslauf seine
    # Rueckprojektion ins Originalbild, ohne die zwei Stunden zu bezahlen.
    cfg["xai"] = copy.deepcopy(cfg.get("xai", {}))
    cfg["xai"]["enabled"] = False
    cfg["run_name"] = None
    return cfg


def schreibe_config(cfg: dict, pfad: str, kopf: str) -> None:
    os.makedirs(os.path.dirname(pfad), exist_ok=True)
    with open(pfad, "w", encoding="utf-8") as f:
        f.write(kopf)
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False,
                       allow_unicode=True)


# ================================================================
#  Einzellauf
# ================================================================
def run_dir_bauen(stufe: int, run_name: str) -> str:
    return os.path.join("outputs", "runs", f"Stufe_{stufe}", run_name)


def ist_fertig(run_dir: str) -> bool:
    return os.path.exists(os.path.join(PROJECT_ROOT, run_dir, "metrics",
                                       "test_metrics.json"))


def starte_lauf(stufe: int, lauf: dict, basis: dict, dry_run: bool,
                basis_pfad: str = "configs/default.yaml") -> str:
    """Erzeugt Config + startet train.py. Gibt den run_dir zurueck."""
    run_name = lauf["run_name"]
    backbone = lauf["backbone"]
    freeze_key = lauf["freeze_key"]

    run_dir = run_dir_bauen(stufe, run_name)
    if ist_fertig(run_dir):
        print(f"[SKIP] {run_name} – test_metrics.json existiert bereits.")
        return run_dir

    cfg = baue_config(basis, backbone, freeze_key)
    kopf = (
        "# ============================================================\n"
        f"#  Versuchsstufe {stufe} – Lauf {lauf['nr']:02d} (Schritt {lauf['schritt']})\n"
        "# ============================================================\n"
        f"#  Backbone        : {backbone}\n"
        f"#  Freeze-Strategie: {FREEZE_LABEL[freeze_key]}\n"
        "#\n"
        "#  Automatisch erzeugt von scripts/run_versuchsplan.py aus\n"
        f"#  {basis_pfad}. Variiert sind ausschliesslich 'backbone'\n"
        "#  und 'freeze_backbone'; XAI ist fuer die Vergleichslaeufe aus,\n"
        "#  die Rueckprojektion (xai.rueckprojektion) bleibt an.\n"
        "#  NICHT von Hand editieren – wird beim naechsten Lauf ueberschrieben.\n"
        "# ============================================================\n"
    )
    cfg_pfad = os.path.join(PROJECT_ROOT, "configs", f"stufe_{stufe}", f"{run_name}.yaml")
    schreibe_config(cfg, cfg_pfad, kopf)

    rel_cfg = os.path.relpath(cfg_pfad, PROJECT_ROOT)
    cmd = [sys.executable, "train.py", "--config", rel_cfg,
           "--run_name", os.path.join(f"Stufe_{stufe}", run_name)]

    print("\n" + "=" * 70)
    print(f"[LAUF {lauf['nr']:02d}] {run_name}   (Schritt {lauf['schritt']})")
    print(f"  Backbone : {backbone}")
    print(f"  Freeze   : {FREEZE_LABEL[freeze_key]}")
    print(f"  Start    : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70, flush=True)

    if dry_run:
        print("  [DRY-RUN] " + " ".join(cmd))
        return run_dir

    t0 = time.time()
    res = subprocess.run(cmd, cwd=PROJECT_ROOT)
    dauer = time.time() - t0
    if res.returncode != 0:
        raise RuntimeError(
            f"Lauf {run_name} mit Exit-Code {res.returncode} abgebrochen. "
            f"Log: {os.path.join(run_dir, 'train.log')}"
        )
    with open(os.path.join(PROJECT_ROOT, run_dir, "metrics", "laufzeit.json"),
              "w", encoding="utf-8") as f:
        json.dump({"wanduhr_sekunden": round(dauer, 1),
                   "wanduhr_hhmmss": time.strftime("%H:%M:%S", time.gmtime(dauer))},
                  f, indent=2)
    print(f"[FERTIG] {run_name} nach {time.strftime('%H:%M:%S', time.gmtime(dauer))}")
    return run_dir


# ================================================================
#  Auswertung
# ================================================================
_BENCH_CACHE = {}


def benchmark_inferenzzeiten(stufe: int) -> dict:
    """{backbone: ms je Bild} aus scripts/benchmark_inferenz.py, sonst {}.

    Die im Evaluator beilaeufig erhobene Rechenzeit ist nicht vergleichbar: sie
    entsteht zu zwoelf verschiedenen Zeitpunkten bei wechselndem GPU-Takt. Belegt
    an Werten, die sich ausschliessen - ResNet-50 misst je nach Lauf 41,17 /
    147,74 / 150,82 ms, obwohl die Freeze-Strategie die Inferenz nicht beruehrt.
    Liegt die Benchmark-Messung vor, wird sie bevorzugt: ein Wert je Architektur,
    Median aus 30 Wiederholungen nach Aufwaermen.
    """
    if stufe in _BENCH_CACHE:
        return _BENCH_CACHE[stufe]
    pfad = os.path.join(PROJECT_ROOT, "outputs", "runs", f"Stufe_{stufe}",
                        "inferenzzeit.json")
    werte = {}
    if os.path.exists(pfad):
        with open(pfad, encoding="utf-8") as f:
            daten = json.load(f)
        werte = {m["backbone"]: m["batch_median"] for m in daten.get("messungen", [])}
    _BENCH_CACHE[stufe] = werte
    return werte


def _laufzeit_aus_log(logpfad: str):
    if not os.path.exists(logpfad):
        return None
    stempel = []
    with open(logpfad, encoding="utf-8", errors="replace") as f:
        for zeile in f:
            try:
                stempel.append(datetime.strptime(zeile[:19], "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                continue
    if len(stempel) < 2:
        return None
    return time.strftime("%H:%M:%S",
                         time.gmtime((stempel[-1] - stempel[0]).total_seconds()))


def _bei_05(m: dict) -> dict:
    """Werte bei fester Schwelle 0,5. Der Evaluator legt "metrics_at_0.5" nur an,
    wenn die angewendete Schwelle abweicht; sonst sind es die Hauptfelder."""
    return m.get("metrics_at_0.5") or {
        "f1": m.get("f1"), "precision": m.get("precision"),
        "recall": m.get("recall"), "accuracy": m.get("accuracy"),
    }


def lies_ergebnis(run_dir: str, **meta) -> dict:
    """Sammelt die Messgroessen eines Laufs. None, wenn der Lauf fehlt."""
    voll = os.path.join(PROJECT_ROOT, run_dir)
    mpfad = os.path.join(voll, "metrics", "test_metrics.json")
    if not os.path.exists(mpfad):
        return None
    with open(mpfad, encoding="utf-8") as f:
        m = json.load(f)
    t05 = _bei_05(m)

    e = dict(meta)
    e.update({
        "run_dir":       run_dir,
        "run_name":      os.path.basename(run_dir),
        "f1_05":         t05.get("f1"),
        "precision_05":  t05.get("precision"),
        "recall_05":     t05.get("recall"),
        "f1_nio":        m.get("f1"),
        "roc_auc":       m.get("roc_auc"),
        "ap":            m.get("average_precision"),
        "n_params":      m.get("n_params_trainable"),
        "ms_pro_bild":   m.get("inference_ms_per_image"),
        "threshold":     m.get("threshold"),
    })

    # Kontrollierte Messung schlaegt die beilaeufige, siehe benchmark_inferenzzeiten
    bench = benchmark_inferenzzeiten(meta.get("stufe", 1))
    e["ms_im_lauf"] = e["ms_pro_bild"]
    if e.get("backbone") in bench:
        e["ms_pro_bild"] = bench[e["backbone"]]
        e["ms_gemessen"] = "benchmark"
    else:
        e["ms_gemessen"] = "im Lauf"

    # VAL – nachrichtlich. Die Auswahl laeuft ueber test, siehe Modulkopf.
    vpfad = os.path.join(voll, "metrics", "val_metrics.json")
    if os.path.exists(vpfad):
        with open(vpfad, encoding="utf-8") as f:
            vm = json.load(f)
        v05 = _bei_05(vm)
        e["val_f1_05"] = v05.get("f1")
        e["val_roc_auc"] = vm.get("roc_auc")

    lpfad = os.path.join(voll, "metrics", "laufzeit.json")
    if os.path.exists(lpfad):
        with open(lpfad, encoding="utf-8") as f:
            e["laufzeit"] = json.load(f).get("wanduhr_hhmmss")
    else:
        e["laufzeit"] = _laufzeit_aus_log(os.path.join(voll, "train.log"))

    cpfad = os.path.join(voll, "metrics", "val_metrics.csv")
    if os.path.exists(cpfad):
        with open(cpfad, encoding="utf-8") as f:
            e["epochen"] = max(0, sum(1 for _ in f) - 1)

    # Betriebspunktfreie Kennzahlen (Fehlalarme je gefordertem Mindest-Recall).
    # Rechnet nur auf den abgelegten Wahrscheinlichkeiten, kostet nichts.
    try:
        sa = _schwellenanalyse(run_dir)
        fr = sa["test"].get("fehlalarme_je_recall") if sa else None
        if fr:
            e["fa_r099"] = fr["0.99"]["fehlalarme"]
            e["fa_r100"] = fr["1.00"]["fehlalarme"]
            e["fa_r095"] = fr["0.95"]["fehlalarme"]
            e["n_io_test"] = fr["n_io"]
    except Exception as exc:                     # nie den Vergleich blockieren
        print(f"  [Hinweis] Schwellenanalyse fuer {os.path.basename(run_dir)} "
              f"uebersprungen: {exc}")
    return e


def _ab(w):
    return -(w if w is not None else -1)


def _rz(e):
    return e["ms_pro_bild"] if e["ms_pro_bild"] is not None else float("inf")


def sortier_test(e):
    """AUSWAHL auf test: F1@0,5 -> ROC-AUC -> Rechenzeit je Bild."""
    return (_ab(e["f1_05"]), _ab(e["roc_auc"]), _rz(e))


def sortier_valschwelle(e):
    return (_ab(e["f1_nio"]), _ab(e["roc_auc"]), _rz(e))


def sortier_roc(e):
    return (_ab(e["roc_auc"]), _ab(e["ap"]), _rz(e))


def sortier_fehlalarme(e):
    """Betrieblich: wenigste Fehlalarme bei mindestens 99 % erkannten Defekten.

    Betriebspunktfrei und bildet die Kostenasymmetrie der Sichtpruefung ab
    (ein durchgelassener Defekt wiegt schwerer als ein Fehlalarm). Bewusst
    Stufe 0,99 statt 1,00: Letztere haengt an einem einzigen schwierigsten
    Bild und schwankt dadurch stark.
    """
    fa = e.get("fa_r099")
    return (fa if fa is not None else float("inf"), _ab(e["roc_auc"]), _rz(e))


RANGFOLGEN = {
    "Rang (Fehlalarme @R≥0,99)": sortier_fehlalarme,
    "Rang (ROC-AUC)":            sortier_roc,
    "Rang (F1@0,5)":             sortier_test,
    "Rang (F1@val-Schw.)":       sortier_valschwelle,
}


def _fmt(v, nk=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nk}f}".replace(".", ",")
    return str(v)


def schreibe_tabelle(stufe, eintraege, titel, dateiname, spalten, kopfzeilen=()):
    """spalten: Liste (Ueberschrift, Funktion(eintrag) -> str)."""
    zdir = os.path.join(PROJECT_ROOT, "outputs", "runs", f"Stufe_{stufe}")
    os.makedirs(zdir, exist_ok=True)
    da = [e for e in eintraege if e]
    if not da:
        return None

    kopf = [k for k, _ in spalten]
    zeilen = [[fn(e) for _, fn in spalten] for e in da]

    md = [f"# {titel}", "", f"Erzeugt: {datetime.now():%Y-%m-%d %H:%M}  "]
    md += list(kopfzeilen) + ["",
                              "| " + " | ".join(kopf) + " |",
                              "|" + "|".join(["---"] * len(kopf)) + "|"]
    md += ["| " + " | ".join(z) + " |" for z in zeilen]

    md_pfad = os.path.join(zdir, dateiname + ".md")
    with open(md_pfad, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(zdir, dateiname + ".csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(kopf)
        w.writerows(zeilen)
    print(f"  Tabelle → {dateiname}.md / .csv")
    return md_pfad


def vergleichsspalten(mit_freeze=False, rangindex=None):
    sp = []
    if rangindex:
        for name in RANGFOLGEN:
            sp.append((name, lambda e, n=name: str(rangindex[n][e["run_name"]])))
    sp.append(("Backbone", lambda e: e["backbone"]))
    if mit_freeze:
        sp.append(("Freeze-Strategie", lambda e: FREEZE_LABEL[e["freeze_key"]]))
    sp += [
        ("F1 (n.i.O.) @0,5", lambda e: _fmt(e["f1_05"])),
        ("Precision @0,5", lambda e: _fmt(e["precision_05"])),
        ("Recall @0,5",   lambda e: _fmt(e["recall_05"])),
        ("Fehlalarme @R≥0,99",
         lambda e: (f"{e['fa_r099']} von {e['n_io_test']}"
                    if e.get("fa_r099") is not None else "—")),
        ("Fehlalarme @R=1,00",
         lambda e: str(e["fa_r100"]) if e.get("fa_r100") is not None else "—"),
        ("ROC-AUC",       lambda e: _fmt(e["roc_auc"])),
        ("Average Precision", lambda e: _fmt(e["ap"])),
        ("F1 @val-Schwelle", lambda e: _fmt(e["f1_nio"])),
        ("val-Schwelle",  lambda e: _fmt(e["threshold"], 3)),
        ("val-F1 @0,5",   lambda e: _fmt(e.get("val_f1_05"))),
        ("Trainierbare Parameter",
         lambda e: f"{e['n_params']:,}".replace(",", ".") if e["n_params"] else "—"),
        ("Rechenzeit je Bild [ms]", lambda e: _fmt(e["ms_pro_bild"], 2)),
        ("Quelle Rechenzeit", lambda e: e.get("ms_gemessen", "im Lauf")),
        ("Epochen",  lambda e: str(e.get("epochen", "—"))),
        ("Laufzeit", lambda e: e.get("laufzeit") or "—"),
        ("Run",      lambda e: f"`{e['run_name']}`"),
    ]
    return sp


def rangindex_bauen(eintraege):
    return {n: {e["run_name"]: i + 1 for i, e in enumerate(sorted(eintraege, key=k))}
            for n, k in RANGFOLGEN.items()}


# ================================================================
#  Hauptablauf
# ================================================================
def main():
    p = argparse.ArgumentParser(description="Versuchsplan Backbone-/Freeze-Vergleich")
    p.add_argument("--stufe", type=int, default=1)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--nur-schritt", type=int, choices=[1, 2], default=None)
    p.add_argument("--backbone", nargs="+", default=None, choices=BACKBONES,
                   help="Nur diese Backbones fahren (Standard: alle)")
    p.add_argument("--nur-tabelle", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    stufe = args.stufe
    basis = lade_basis_config(os.path.join(PROJECT_ROOT, args.config))
    backbones = args.backbone or BACKBONES

    print("=" * 70)
    print(f"VERSUCHSPLAN – Versuchsstufe {stufe}")
    print("=" * 70)
    print(f"Basis-Config : {args.config}")
    print(f"Datensatz    : {basis.get('data_dir')}   Bildgroesse: {basis.get('image_size')}")
    print(f"Seed         : {basis.get('seed')}   Batch: {basis.get('batch_size')}   "
          f"Epochen: {basis.get('epochs')}")
    print(f"Optimierer   : {basis.get('optimizer')}  lr_kopf={basis.get('learning_rate')}  "
          f"lr_backbone={basis.get('backbone_learning_rate')}  "
          f"wd={basis.get('weight_decay')}")
    print(f"Abbruch      : {basis.get('early_stopping_monitor')}, "
          f"patience={basis.get('early_stopping_patience')}")
    print(f"Raster       : {len(backbones)} Backbones x "
          f"{len(FREEZE_REIHENFOLGE)} Freeze-Strategien")
    print("Auswahl      : F1@0,5 auf test -> ROC-AUC -> Rechenzeit je Bild")
    print(f"Ziel         : outputs/runs/Stufe_{stufe}/\n")

    plan = [l for l in laufplan(stufe) if l["backbone"] in backbones]
    if args.nur_schritt:
        plan = [l for l in plan if l["schritt"] == args.nur_schritt]

    # ---- Laeufe abfahren -----------------------------------------
    erg = []
    for lauf in plan:
        if args.nur_tabelle:
            rd = run_dir_bauen(stufe, lauf["run_name"])
        else:
            rd = starte_lauf(stufe, lauf, basis, args.dry_run, args.config)
        e = lies_ergebnis(rd, backbone=lauf["backbone"],
                          freeze_key=lauf["freeze_key"],
                          schritt=lauf["schritt"], nr=lauf["nr"], stufe=stufe)
        if e:
            erg.append(e)

    if not erg:
        print("\nKeine ausgewerteten Laeufe – keine Tabellen erzeugt.")
        return

    # ---- Schritt 1: Backbone-Vergleich ---------------------------
    s1 = [e for e in erg if e["freeze_key"] == SCHRITT_1_FREEZE]
    if s1:
        print("\nSchritt 1 – Backbone-Vergleich:")
        schreibe_tabelle(
            stufe, sorted(s1, key=sortier_test),
            f"Versuchsstufe {stufe} – Schritt 1: Backbone-Vergleich",
            "vergleich_schritt_1_backbone",
            vergleichsspalten(rangindex=rangindex_bauen(s1)),
            kopfzeilen=[
                "Alle Backbones mit **identischer Konfiguration** "
                f"(`{args.config}`) und der Freeze-Strategie "
                f"„{FREEZE_LABEL[SCHRITT_1_FREEZE]}\". Variiert ist "
                "ausschliesslich das Backbone.  ",
                "**Auswahlkriterium: F1 (n.i.O.) @0,5** → ROC-AUC → "
                "Rechenzeit je Bild.",
            ])

    # ---- Schritt 2: Freeze-Raster, je Backbone -------------------
    ri = rangindex_bauen(erg)
    nach_backbone = {bb: [e for e in erg if e["backbone"] == bb] for bb in backbones}
    vollstaendig = {bb: v for bb, v in nach_backbone.items()
                    if len(v) == len(FREEZE_REIHENFOLGE)}

    def gruppiert(e):
        return (backbones.index(e["backbone"]),
                FREEZE_REIHENFOLGE.index(e["freeze_key"]))

    if any(e["freeze_key"] != SCHRITT_1_FREEZE for e in erg):
        print("\nSchritt 2 – Freeze-Strategie je Backbone:")
        schreibe_tabelle(
            stufe, sorted(erg, key=gruppiert),
            f"Versuchsstufe {stufe} – Schritt 2: Freeze-Strategie je Backbone",
            "vergleich_schritt_2_freeze",
            vergleichsspalten(mit_freeze=True, rangindex=ri),
            kopfzeilen=[
                "Volles Raster: **jedes** Backbone mit allen drei "
                "Freeze-Strategien, nach Backbone gruppiert.  ",
                f"Die Zeilen „{FREEZE_LABEL[SCHRITT_1_FREEZE]}\" sind die "
                "Laeufe aus Schritt 1 – identische Konfiguration, deshalb "
                "wiederverwendet statt wiederholt.  ",
                "Die Rangspalten beziehen sich auf **alle** Laeufe der Tabelle, "
                "nicht auf die jeweilige Backbone-Gruppe.",
            ])

    # ---- Beste Freeze-Strategie je Backbone ----------------------
    if vollstaendig:
        print("\nBeste Freeze-Strategie je Backbone (auf test, F1@0,5):")
        for bb in backbones:
            if bb not in vollstaendig:
                offen = len(nach_backbone.get(bb, []))
                print(f"  {bb:<16} [offen] erst {offen} von "
                      f"{len(FREEZE_REIHENFOLGE)} Varianten fertig.")
                continue
            b = sorted(vollstaendig[bb], key=sortier_test)[0]
            print(f"  {bb:<16} {FREEZE_LABEL[b['freeze_key']]:<28} "
                  f"F1@0,5={_fmt(b['f1_05'])}  ROC-AUC={_fmt(b['roc_auc'])}")

    # ---- Gesamtrangliste -----------------------------------------
    print("\nGesamtrangliste:")
    schreibe_tabelle(
        stufe, sorted(erg, key=sortier_test),
        f"Versuchsstufe {stufe} – Gesamtrangliste aller Kombinationen",
        "vergleich_gesamt",
        vergleichsspalten(mit_freeze=True, rangindex=ri),
        kopfzeilen=[
            "Alle Kombinationen aus Backbone und Freeze-Strategie, sortiert "
            "nach dem Auswahlkriterium.  ",
            "**Auswahlkriterium: F1 (n.i.O.) @0,5** → ROC-AUC → "
            "Rechenzeit je Bild.",
        ])

    sieger = sorted(erg, key=sortier_test)[0]
    fehlend = [l["run_name"] for l in laufplan(stufe)
               if l["backbone"] in backbones
               and l["run_name"] not in {e["run_name"] for e in erg}]

    print("\n" + "=" * 70)
    print(f"Versuchsstufe {stufe}: {len(erg)} von "
          f"{len(backbones) * len(FREEZE_REIHENFOLGE)} Laeufen ausgewertet.")
    if fehlend:
        print("Es fehlen noch: " + ", ".join(fehlend))
        print("Die Auswahl unten ist damit VORLAEUFIG.")
    print(f"Beste Kombination: {sieger['backbone']} + "
          f"{FREEZE_LABEL[sieger['freeze_key']]}")
    print(f"  F1 (n.i.O.) @0,5 = {_fmt(sieger['f1_05'])}   "
          f"ROC-AUC = {_fmt(sieger['roc_auc'])}   "
          f"{_fmt(sieger['ms_pro_bild'], 2)} ms je Bild")
    print(f"  Lauf: {sieger['run_name']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
