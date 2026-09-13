#!/usr/bin/env python3
"""
scripts/xai_sichtung.py
Strukturierte Sichtung der Attributionskarten - der qualitative Teil des
XAI-Abschnitts in Kapitel 6.

Warum es diesen Schritt gibt: Die Arbeit darf keine Aufnahmen des Bauteils
zeigen. Damit faellt die uebliche Belegform des qualitativen Teils weg - "man
sieht in Abbildung X, dass die Attribution auf der Fehlstelle liegt". Statt
diesen Beleg ersatzlos zu streichen, wird er in eine nachpruefbare Zahl
ueberfuehrt: Ein Bewerter ordnet jede Karte einem vorab festgelegten
Kategorienschema zu, und berichtet wird die Haeufigkeit je Kategorie. Das ist
eine qualitative Bewertung, die ohne Bildmaterial auskommt.

Zwei Vorkehrungen halten die Sichtung von der Erwartung frei:

- **Verblindung.** Die Tafeln tragen keinen Verfahrensnamen und keinen
  Dateinamen, nur eine zufaellige Nummer. Wer weiss, dass er gerade eine
  Occlusion-Karte sieht, bewertet die Erwartung an das Verfahren mit.
- **Vorab festgelegtes Schema.** Die Kategorien stehen in diesem Modul und
  werden vor der Sichtung festgelegt, nicht aus dem Gesehenen abgeleitet.
  Andernfalls waere jede Karte irgendeiner passenden Kategorie zuzuordnen und
  die Auswertung saegte nur die eigene Erwartung nach.

Die Tafeln entstehen nicht hier, sondern in `scripts/xai_kennzahlen.py`
waehrend der Messung - dort liegen die Karten ohnehin im Speicher. Dieses Modul
baut daraus den Bewertungsbogen und wertet ihn aus.

Ablauf:
    1. python scripts/xai_kennzahlen.py --stufe 2      (erzeugt die Tafeln)
    2. python scripts/xai_sichtung.py --bogen --stufe 2
    3. outputs/xai_sichtung/stufe2/bogen.html im Browser oeffnen, bewerten,
       am Ende "Bewertung sichern" - die Datei landet im Download-Ordner und
       gehoert als sichtung.csv neben den Bogen
    4. python scripts/xai_sichtung.py --auswerten --stufe 2

Ausgabe von Schritt 4:
    outputs/xai_sichtung/stufe<N>/sichtung_ergebnis.json  (+ .md)
"""
import argparse
import csv
import html
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from kap5_daten import METHODEN, METHODEN_NAME, de  # noqa: E402

# ---- Das Kategorienschema -------------------------------------------------
# Vorab festgelegt (26.08.2026). Die Reihenfolge ist die des Bogens und die der
# Ergebnistabelle. Die ersten beiden Kategorien liegen auf dem Bauteil und sind
# fachlich brauchbar, die dritte liegt darauf ohne erkennbaren Bezug, die
# letzten drei sind Befunde gegen die Nachvollziehbarkeit.
KATEGORIEN = [
    ("defekt", "Schwerpunkt auf einer sichtbaren Fehlstelle"),
    ("kante", "Schwerpunkt auf Bauteilkante oder Silhouette"),
    ("flaeche", "auf der Bauteilfläche, ohne erkennbaren Bezug zu einer Fehlstelle"),
    ("hintergrund", "überwiegend außerhalb des Bauteils"),
    ("diffus", "über das ganze Bild verteilt, kein Schwerpunkt"),
    ("leer", "keine Attribution erkennbar"),
]
AUF_BAUTEIL = {"defekt", "kante", "flaeche"}
SEED_REIHENFOLGE = 815   # feste Mischung, damit der Bogen reproduzierbar ist


def stufen_dir(stufe: int) -> str:
    return os.path.join(PROJECT_ROOT, "outputs", "xai_sichtung", f"stufe{stufe}")


def lade_schluessel(ordner: str):
    pfad = os.path.join(ordner, "schluessel.csv")
    if not os.path.isfile(pfad):
        sys.exit(f"[FEHLER] {pfad} fehlt. Erst `python scripts/xai_kennzahlen.py "
                 f"--stufe <N>` laufen lassen - die Tafeln entstehen dort.")
    with open(pfad, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=";"))


def baue_bogen(stufe: int) -> str:
    ordner = stufen_dir(stufe)
    schluessel = lade_schluessel(ordner)
    reihenfolge = list(schluessel)
    random.Random(SEED_REIHENFOLGE).shuffle(reihenfolge)

    knoepfe = "".join(
        f'<label><input type="radio" name="{{tafel}}" value="{k}">'
        f'<span><b>{i + 1}</b> {html.escape(bez)}</span></label>'
        for i, (k, bez) in enumerate(KATEGORIEN))

    tafeln = []
    for nr, z in enumerate(reihenfolge, start=1):
        tafeln.append(f'''
<section class="tafel" id="t{z['tafel']}" data-tafel="{z['tafel']}">
  <h2>{nr} von {len(reihenfolge)}<span class="id">Tafel {z['tafel']}</span></h2>
  <img src="bilder/{z['tafel']}.png" alt="Attributionskarte" loading="lazy">
  <div class="wahl">{knoepfe.format(tafel=z['tafel'])}</div>
</section>''')

    legende = "".join(f"<li><b>{i + 1}</b> <code>{k}</code> – {html.escape(bez)}</li>"
                      for i, (k, bez) in enumerate(KATEGORIEN))

    seite = f'''<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<title>Sichtung der Attributionskarten – Versuchsstufe {stufe}</title>
<style>
 body {{ font-family: "Segoe UI", system-ui, sans-serif; max-width: 62rem;
        margin: 0 auto; padding: 1.5rem 1.5rem 8rem; color: #25282E; }}
 h1 {{ font-size: 1.35rem; }}
 .hinweis {{ background:#F4F6FA; border-left:4px solid #0F2DB3; padding:.9rem 1.1rem;
             border-radius:4px; line-height:1.55; }}
 .hinweis ul {{ margin:.5rem 0 0; padding-left:1.2rem; }}
 .hinweis li {{ margin:.15rem 0; }}
 code {{ background:#E9ECF3; padding:.05rem .3rem; border-radius:3px; }}
 .tafel {{ border-top:1px solid #E0E0E0; padding-top:1.1rem; margin-top:1.6rem; }}
 .tafel h2 {{ font-size:.95rem; font-weight:600; color:#4C525B; margin:0 0 .5rem;
              display:flex; justify-content:space-between; }}
 .tafel .id {{ font-weight:400; color:#8A919B; }}
 .tafel img {{ width:100%; height:auto; border:1px solid #E0E0E0; border-radius:4px; }}
 .wahl {{ display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.6rem; }}
 .wahl label {{ display:flex; align-items:center; gap:.4rem; padding:.35rem .6rem;
                border:1px solid #D6DAE2; border-radius:999px; cursor:pointer;
                font-size:.86rem; }}
 .wahl label:has(input:checked) {{ background:#0F2DB3; color:#fff; border-color:#0F2DB3; }}
 .tafel.fertig h2 {{ color:#2A9D4E; }}
 .leiste {{ position:fixed; left:0; right:0; bottom:0; background:#fff;
            border-top:1px solid #E0E0E0; padding:.7rem 1.5rem; display:flex;
            gap:1rem; align-items:center; justify-content:center; }}
 button {{ font:inherit; padding:.5rem 1.1rem; border-radius:6px; border:1px solid #0F2DB3;
           background:#0F2DB3; color:#fff; cursor:pointer; }}
 button.zweit {{ background:#fff; color:#0F2DB3; }}
 #stand {{ font-variant-numeric:tabular-nums; }}
</style></head><body>
<h1>Sichtung der Attributionskarten – Versuchsstufe {stufe}</h1>
<div class="hinweis">
<p>Links die Aufnahme, rechts dieselbe Aufnahme mit der Attributionskarte.
Welche Kategorie beschreibt am besten, <b>wo die Karte ihren Schwerpunkt hat</b>?</p>
<ul>{legende}</ul>
<p>Welches Verfahren eine Tafel zeigt, steht bewusst nicht dabei – die Zuordnung
erfolgt erst bei der Auswertung. Die Ziffern <b>1</b>–<b>6</b> wählen per Tastatur,
sobald eine Tafel im Bild ist. Der Stand wird laufend im Browser gesichert; am
Ende <b>Bewertung sichern</b> drücken und die Datei als
<code>sichtung.csv</code> neben diese Seite legen.</p>
</div>
{''.join(tafeln)}
<div class="leiste">
  <span id="stand">0 von {len(reihenfolge)} bewertet</span>
  <button onclick="sichern()">Bewertung sichern</button>
  <button class="zweit" onclick="naechste()">Nächste offene Tafel</button>
</div>
<script>
const SPEICHER = "sichtung_stufe{stufe}";
const gesamt = {len(reihenfolge)};

function laden() {{
  let d = {{}};
  try {{ d = JSON.parse(localStorage.getItem(SPEICHER) || "{{}}"); }} catch (e) {{ d = {{}}; }}
  for (const [tafel, wert] of Object.entries(d)) {{
    const el = document.querySelector(`input[name="${{tafel}}"][value="${{wert}}"]`);
    if (el) {{ el.checked = true; el.closest(".tafel").classList.add("fertig"); }}
  }}
  stand();
}}
function sammeln() {{
  const d = {{}};
  document.querySelectorAll(".tafel").forEach(s => {{
    const g = s.querySelector("input:checked");
    if (g) d[s.dataset.tafel] = g.value;
  }});
  return d;
}}
function stand() {{
  const n = Object.keys(sammeln()).length;
  document.getElementById("stand").textContent = n + " von " + gesamt + " bewertet";
}}
document.addEventListener("change", e => {{
  if (e.target.type !== "radio") return;
  e.target.closest(".tafel").classList.add("fertig");
  try {{ localStorage.setItem(SPEICHER, JSON.stringify(sammeln())); }} catch (e2) {{}}
  stand();
}});
// Ziffern 1-6 bewerten die Tafel, die gerade im Bild steht.
const REIHE = {json.dumps([k for k, _ in KATEGORIEN])};
document.addEventListener("keydown", e => {{
  const i = parseInt(e.key, 10);
  if (!(i >= 1 && i <= REIHE.length)) return;
  const mitte = window.innerHeight / 2;
  let beste = null, abstand = Infinity;
  document.querySelectorAll(".tafel").forEach(s => {{
    const r = s.getBoundingClientRect();
    const d = Math.abs((r.top + r.bottom) / 2 - mitte);
    if (d < abstand) {{ abstand = d; beste = s; }}
  }});
  if (!beste) return;
  const el = beste.querySelector(`input[value="${{REIHE[i - 1]}}"]`);
  if (el) {{ el.checked = true; el.dispatchEvent(new Event("change", {{bubbles: true}})); }}
}});
function naechste() {{
  for (const s of document.querySelectorAll(".tafel")) {{
    if (!s.querySelector("input:checked")) {{ s.scrollIntoView({{behavior: "smooth", block: "center"}}); return; }}
  }}
  alert("Alle Tafeln sind bewertet.");
}}
function sichern() {{
  const d = sammeln();
  const zeilen = ["tafel;kategorie"];
  for (const [t, w] of Object.entries(d)) zeilen.push(t + ";" + w);
  const blob = new Blob([zeilen.join("\\n") + "\\n"], {{type: "text/csv;charset=utf-8"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "sichtung.csv";
  a.click();
  URL.revokeObjectURL(a.href);
}}
laden();
</script>
</body></html>'''

    pfad = os.path.join(ordner, "bogen.html")
    with open(pfad, "w", encoding="utf-8") as f:
        f.write(seite)
    return pfad, len(reihenfolge)


def werte_aus(stufe: int):
    ordner = stufen_dir(stufe)
    schluessel = {z["tafel"]: z for z in lade_schluessel(ordner)}

    pfad = os.path.join(ordner, "sichtung.csv")
    if not os.path.isfile(pfad):
        sys.exit(f"[FEHLER] {pfad} fehlt. Den Bogen im Browser bewerten, "
                 f"'Bewertung sichern' druecken und die Datei hierher legen.")
    with open(pfad, newline="", encoding="utf-8-sig") as f:
        bewertet = {z["tafel"]: z["kategorie"] for z in csv.DictReader(f, delimiter=";")}

    gueltig = {k for k, _ in KATEGORIEN}
    unbekannt = {w for w in bewertet.values() if w not in gueltig}
    if unbekannt:
        sys.exit(f"[FEHLER] Unbekannte Kategorien in sichtung.csv: {sorted(unbekannt)}")
    fremd = set(bewertet) - set(schluessel)
    if fremd:
        sys.exit(f"[FEHLER] {len(fremd)} Tafeln aus sichtung.csv stehen nicht im "
                 f"Schluessel - gehoert die Datei zu dieser Versuchsstufe?")

    # Je Verfahren zaehlen. Offen gebliebene Tafeln werden NICHT als Kategorie
    # gefuehrt, sondern gesondert ausgewiesen: eine nicht bewertete Karte ist
    # keine Beobachtung, und sie stillschweigend unter "diffus" zu buchen
    # verschoebe das Ergebnis.
    je_verfahren = {m: Counter() for m in METHODEN}
    je_klasse = {m: {"io": Counter(), "nio": Counter()} for m in METHODEN}
    offen = {m: 0 for m in METHODEN}
    for tafel, z in schluessel.items():
        m = z["verfahren"]
        if tafel in bewertet:
            je_verfahren[m][bewertet[tafel]] += 1
            je_klasse[m][z["wahre_klasse"]][bewertet[tafel]] += 1
        else:
            offen[m] += 1

    ergebnis = {
        "erzeugt": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "stufe": stufe,
        "kategorien": [{"schluessel": k, "bezeichnung": b} for k, b in KATEGORIEN],
        "n_tafeln": len(schluessel),
        "n_bewertet": len(bewertet),
        "je_verfahren": {m: dict(c) for m, c in je_verfahren.items()},
        "je_klasse": {m: {k: dict(c) for k, c in d.items()} for m, d in je_klasse.items()},
        "offen": offen,
        "anteil_auf_bauteil": {
            m: (sum(c[k] for k in AUF_BAUTEIL) / sum(c.values())) if sum(c.values()) else None
            for m, c in je_verfahren.items()
        },
    }

    with open(os.path.join(ordner, "sichtung_ergebnis.json"), "w", encoding="utf-8") as f:
        json.dump(ergebnis, f, indent=1, ensure_ascii=False)

    zeilen = [
        f"# Versuchsstufe {stufe} – Sichtung der Attributionskarten", "",
        f"Erzeugt: {ergebnis['erzeugt']}  ",
        f"{ergebnis['n_bewertet']} von {ergebnis['n_tafeln']} Tafeln bewertet, "
        f"verblindet und in fester Zufallsreihenfolge.", "",
        "| Kategorie | " + " | ".join(METHODEN_NAME[m] for m in METHODEN) + " |",
        "|---|" + "---|" * len(METHODEN),
    ]
    for k, bez in KATEGORIEN:
        werte = []
        for m in METHODEN:
            n = je_verfahren[m][k]
            gesamt = sum(je_verfahren[m].values())
            werte.append(f"{n} ({de(100 * n / gesamt, 1)} %)" if gesamt else "—")
        zeilen.append(f"| {bez} | " + " | ".join(werte) + " |")
    zeilen += ["", "| Auf dem Bauteil (Fehlstelle, Kante oder Fläche) | "
               + " | ".join(
                   f"{de(100 * ergebnis['anteil_auf_bauteil'][m], 1)} %"
                   if ergebnis["anteil_auf_bauteil"][m] is not None else "—"
                   for m in METHODEN) + " |"]
    md_pfad = os.path.join(ordner, "sichtung_ergebnis.md")
    with open(md_pfad, "w", encoding="utf-8") as f:
        f.write("\n".join(zeilen) + "\n")
    return ergebnis, md_pfad


def main():
    ap = argparse.ArgumentParser(description="Strukturierte Sichtung der Attributionskarten")
    ap.add_argument("--stufe", type=int, default=1)
    ap.add_argument("--bogen", action="store_true", help="Bewertungsbogen erzeugen")
    ap.add_argument("--auswerten", action="store_true", help="sichtung.csv auswerten")
    args = ap.parse_args()

    if not (args.bogen or args.auswerten):
        ap.error("--bogen oder --auswerten angeben")

    if args.bogen:
        pfad, n = baue_bogen(args.stufe)
        print(f"Bogen mit {n} Tafeln: {os.path.relpath(pfad, PROJECT_ROOT)}")
        print("  Im Browser oeffnen, bewerten, 'Bewertung sichern' druecken und die "
              "Datei als sichtung.csv daneben legen.")

    if args.auswerten:
        ergebnis, md = werte_aus(args.stufe)
        print(f"{ergebnis['n_bewertet']} von {ergebnis['n_tafeln']} Tafeln bewertet")
        for m in METHODEN:
            anteil = ergebnis["anteil_auf_bauteil"][m]
            offen = ergebnis["offen"][m]
            print(f"  {METHODEN_NAME[m]:22s} auf dem Bauteil: "
                  + (f"{de(100 * anteil, 1)} %" if anteil is not None else "—")
                  + (f"   ({offen} offen)" if offen else ""))
        print(f"  geschrieben: {os.path.relpath(md, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
