#!/usr/bin/env python3
"""
scripts/run_xai_versuchsplan.py
Erzeugt XAI-Heatmaps fuer ALLE Laeufe einer Versuchsstufe - je Lauf fuer val und
test, immer mit dem besten Modell (checkpoints/best.pt).

Je Lauf und Split werden erklaert:
  - eine laufunabhaengige Basismenge: entweder die ersten und letzten k Bilder
    des Splits (Default) oder mit --zufall N eine Zufallsstichprobe von N
    Bildern, klassenweise gleich verteilt. Beide haengen nicht vom Lauf ab,
    dasselbe Bauteil liegt also in allen Laeufen unter demselben Dateinamen und
    laesst sich ueber Backbones und Freeze-Strategien hinweg direkt vergleichen.
  - bis zu --max-fehler Fehlklassifikationen dieses Laufs (Schwelle 0,5, also
    das Auswahlkriterium des Versuchsplans), die konfidentesten zuerst.

Stufe 2 faehrt --zufall 44 --max-fehler 6, also 50 Bilder je Lauf und Split.
Der ganze Split waeren 400 Bilder und rund 100 min je Lauf und Split.

Die Laufnamen kommen aus scripts/run_versuchsplan.py (laufplan) - sie werden
hier NICHT dupliziert.

Aufruf:
    python scripts/run_xai_versuchsplan.py --stufe 1
    python scripts/run_xai_versuchsplan.py --stufe 1 --dry-run
    python scripts/run_xai_versuchsplan.py --stufe 1 --splits test
    python scripts/run_xai_versuchsplan.py --stufe 1 --nur-lauf S1_06_resnet18_vollstaendig
    python scripts/run_xai_versuchsplan.py --stufe 2 --zufall 44 --max-fehler 6

Wiederaufnahme: Ist das Manifest eines Ordners vollstaendig und liegt jede darin
genannte Heatmap auf der Platte, wird der Ordner uebersprungen. Ein unfertiger
Ordner wird per --resume im Unterprozess fortgesetzt, es wird also nur das
Fehlende gerechnet. Bei ~12 s je Bild lohnt sich das.

Hinweis: nicht parallel zum Training starten - beide konkurrieren um die GPU.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Siehe run_versuchsplan.py: umgeleitete stdout ist unter Windows cp1252, ein
# '≈' oder '→' in einer Statusmeldung wuerde den Lauf sonst abbrechen.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import xai_auswahl  # noqa: E402
from run_versuchsplan import (  # noqa: E402
    FREEZE_LABEL,
    laufplan,
    run_dir_bauen,
    schreibe_tabelle,
)

SPLITS = ["val", "test"]
# Sekunden je Bild: IG (50 Stuetzpunkte) + Saliency ~10 s gemessen, Occlusion
# bei patch_size 64 und 512 px 64 zusaetzliche Vorwaertsdurchlaeufe.
SEK_JE_BILD = 12


def zustand(run_dir: str, split: str, out_dir: str) -> tuple:
    """(status, auswahl_oder_None). status: bereit | fertig | kein_checkpoint | keine_preds"""
    voll = os.path.join(PROJECT_ROOT, run_dir)
    if not os.path.isfile(os.path.join(voll, "checkpoints", "best.pt")):
        return ("kein_checkpoint", None)
    if not os.path.isfile(xai_auswahl.preds_pfad(voll, split)):
        return ("keine_preds", None)

    fertig, _, _ = xai_auswahl.manifest_vollstaendig(os.path.join(PROJECT_ROOT, out_dir))
    return ("fertig" if fertig else "bereit", None)


def main():
    p = argparse.ArgumentParser(description="XAI-Heatmaps fuer eine ganze Versuchsstufe")
    p.add_argument("--stufe", type=int, default=1)
    p.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    p.add_argument("--nur-lauf", nargs="+", default=None,
                   help="Nur diese Laufnamen (Default: alle der Stufe)")
    p.add_argument("--max-fehler", type=int, default=xai_auswahl.MAX_FEHLER)
    p.add_argument("--n-je-klasse", type=int, default=xai_auswahl.N_JE_KLASSE)
    p.add_argument("--zufall", type=int, default=0, metavar="N",
                   help="N zufaellige Basisbilder je Split statt erste/letzte k "
                        "je Klasse. Fester Seed, in jedem Lauf dieselben Bilder "
                        "(Stufe 2: 44). 0 = aus")
    p.add_argument("--gespreizt", action="store_true",
                   help="gleichmaessig verteilte statt Anfang/Ende")
    p.add_argument("--methoden", nargs="+", default=["all"],
                   choices=["integrated_gradients", "saliency", "occlusion", "all"])
    p.add_argument("--neu", action="store_true",
                   help="fertige Ordner nicht ueberspringen, sondern neu rechnen")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    stufe = args.stufe
    plan = laufplan(stufe)
    if args.nur_lauf:
        plan = [l for l in plan if l["run_name"] in args.nur_lauf]

    print("=" * 72)
    print(f"XAI-NACHLAUF – Versuchsstufe {stufe}")
    print("=" * 72)
    print(f"Laeufe    : {len(plan)}   Splits: {', '.join(args.splits)}")
    if args.zufall > 0:
        basis_text = (f"{args.zufall} zufaellig je Split "
                      f"(Seed {xai_auswahl.ZUFALL_SEED}, laufunabhaengig)")
    else:
        basis_text = (f"{args.n_je_klasse} je Klasse "
                      f"({'gespreizt' if args.gespreizt else 'Anfang/Ende'})")
    print(f"Auswahl   : {basis_text} + bis zu {args.max_fehler} Fehler je Split")
    print(f"Verfahren : {', '.join(args.methoden)}")
    print(f"Modell    : checkpoints/best.pt")
    print(f"Schwelle  : {xai_auswahl.SCHWELLE} (Auswahlkriterium des Versuchsplans)\n")

    eintraege = []
    zu_rechnen = []

    for lauf in plan:
        run_dir = run_dir_bauen(stufe, lauf["run_name"])
        for split in args.splits:
            out_dir = os.path.join(run_dir, "xai", split)
            status, _ = zustand(run_dir, split, out_dir)

            e = {
                "run_name": lauf["run_name"],
                "backbone": lauf["backbone"],
                "freeze_key": lauf["freeze_key"],
                "split": split,
                "run_dir": run_dir,
                "out_dir": out_dir,
                "status": status,
                "n_fest": 0, "n_fehler_gesamt": 0, "n_fehler_gezeigt": 0, "n_bilder": 0,
            }

            if status in ("bereit", "fertig"):
                voll = os.path.join(PROJECT_ROOT, run_dir)
                zeilen = xai_auswahl.lade_preds(voll, split)
                auswahl = xai_auswahl.waehle_bilder(
                    zeilen, split, k=args.n_je_klasse, max_fehler=args.max_fehler,
                    gespreizt=args.gespreizt, zufall=args.zufall)
                e["n_fest"] = sum(a["fest"] for a in auswahl)
                e["n_fehler_gesamt"] = len(xai_auswahl.fehler_indizes(zeilen))
                e["n_fehler_gezeigt"] = sum(1 for a in auswahl
                                            if a["fehler"] and not a["fest"])
                e["n_bilder"] = len(auswahl)
                if status == "bereit" or args.neu:
                    zu_rechnen.append(e)

            eintraege.append(e)
            marke = {"bereit": "[TODO]", "fertig": "[FERTIG]",
                     "kein_checkpoint": "[FEHLT] kein best.pt",
                     "keine_preds": "[FEHLT] keine preds"}[status]
            print(f"  {lauf['run_name']:<34} {split:<5} {marke:<22} "
                  f"{e['n_bilder']} Bilder "
                  f"({e['n_fehler_gezeigt']}/{e['n_fehler_gesamt']} Fehler)")

    gesamt = sum(e["n_bilder"] for e in zu_rechnen)
    print(f"\nZu rechnen: {len(zu_rechnen)} Einheiten, {gesamt} Bilder "
          f"≈ {gesamt * SEK_JE_BILD / 60:.0f} min")

    # ---- Ausfuehren -------------------------------------------------
    if not args.dry_run:
        for i, e in enumerate(zu_rechnen, start=1):
            print("\n" + "=" * 72)
            print(f"[{i}/{len(zu_rechnen)}] {e['run_name']} – {e['split']} "
                  f"({e['n_bilder']} Bilder)   {datetime.now():%H:%M:%S}")
            print("=" * 72, flush=True)

            cmd = [sys.executable, "scripts/run_xai_split.py",
                   "--run", e["run_dir"], "--split", e["split"],
                   "--auswahl", "vergleich",
                   "--max-fehler", str(args.max_fehler),
                   "--n-je-klasse", str(args.n_je_klasse),
                   "--methoden", *args.methoden,
                   "--resume"]
            if args.zufall > 0:
                cmd += ["--zufall", str(args.zufall)]
            if args.gespreizt:
                cmd.append("--gespreizt")

            t0 = time.time()
            # Unterprozess statt Schleife im selben Prozess: der CUDA-Kontext wird
            # zwischen den vier Architekturen sauber freigegeben, und ein Absturz
            # reisst nicht die uebrigen Einheiten mit.
            res = subprocess.run(cmd, cwd=PROJECT_ROOT)
            dauer = time.time() - t0
            if res.returncode != 0:
                e["status"] = "fehler"
                print(f"[FEHLER] {e['run_name']}/{e['split']} "
                      f"Exit-Code {res.returncode} – wird uebersprungen.")
                continue
            fertig, vorhanden, erwartet = xai_auswahl.manifest_vollstaendig(
                os.path.join(PROJECT_ROOT, e["out_dir"]))
            e["status"] = "fertig" if fertig else "unvollstaendig"
            print(f"[OK] {vorhanden}/{erwartet} Heatmaps nach "
                  f"{time.strftime('%H:%M:%S', time.gmtime(dauer))}")

    # ---- Uebersicht -------------------------------------------------
    print("\nUebersicht:")
    schreibe_tabelle(
        stufe, eintraege,
        f"Versuchsstufe {stufe} – XAI-Heatmaps je Lauf und Split",
        "xai_uebersicht",
        [
            ("Lauf",     lambda e: f"`{e['run_name']}`"),
            ("Backbone", lambda e: e["backbone"]),
            ("Freeze-Strategie", lambda e: FREEZE_LABEL[e["freeze_key"]]),
            ("Split",    lambda e: e["split"]),
            ("Vergleichsbilder (Basis)", lambda e: str(e["n_fest"])),
            ("Fehler gesamt",  lambda e: str(e["n_fehler_gesamt"])),
            ("Fehler gezeigt", lambda e: str(e["n_fehler_gezeigt"])),
            ("Heatmaps", lambda e: str(e["n_bilder"])),
            ("Status",   lambda e: e["status"]),
        ],
        kopfzeilen=[
            f"Je Lauf und Split die laufunabhaengigen Vergleichsbilder ({basis_text}) "
            f"plus bis zu {args.max_fehler} Fehlklassifikationen.  ",
            f"Fehler bestimmt bei **Schwelle {xai_auswahl.SCHWELLE}** – dem "
            "Auswahlkriterium des Versuchsplans, nicht der auf val bestimmten "
            "Schwelle aus `metrics/threshold.json`.  ",
            "Erklaert wird jeweils die **vorhergesagte** Klasse, nicht die wahre.",
        ])

    n_fertig = sum(1 for e in eintraege if e["status"] == "fertig")
    print("\n" + "=" * 72)
    print(f"{n_fertig} von {len(eintraege)} Einheiten fertig.")
    print("=" * 72)


if __name__ == "__main__":
    main()
