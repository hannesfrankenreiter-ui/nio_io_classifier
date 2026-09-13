"""
tests/test_xai_auswahl.py
Tests der XAI-Bildauswahl. Torch-frei, laeuft in Millisekunden.

Der wichtigste Test ist test_fehler_bei_05_nicht_laut_correct_spalte: die Spalte
"correct" in preds/*.csv gilt fuer den auf val bestimmten Betriebspunkt, der
Versuchsplan waehlt aber bei fester Schwelle 0,5 aus. Wer die bequeme Spalte
benutzt, erklaert die falschen Bilder.
"""
import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

import xai_auswahl as xa  # noqa: E402


def mache_zeilen(n_io=8, n_nio=6, prob_nio_io=0.1, prob_nio_nio=0.9):
    """Ein Split wie im echten Datensatz: erst alle io, dann alle nio."""
    zeilen = []
    for i in range(n_io):
        zeilen.append({"index": i, "true": "io",
                       "prob_io": 1 - prob_nio_io, "prob_nio": prob_nio_io})
    for j in range(n_nio):
        zeilen.append({"index": n_io + j, "true": "nio",
                       "prob_io": 1 - prob_nio_nio, "prob_nio": prob_nio_nio})
    return zeilen


# ================================================================
#  Feste Vergleichsbilder
# ================================================================
def test_feste_indizes_sind_anfang_und_ende():
    zeilen = mache_zeilen(n_io=8, n_nio=6)          # 14 Bilder, Indizes 0..13
    assert xa.feste_indizes(zeilen, k=3) == [0, 1, 2, 11, 12, 13]


def test_feste_indizes_treffen_beide_klassen():
    zeilen = mache_zeilen(n_io=8, n_nio=6)
    idx = xa.feste_indizes(zeilen, k=3)
    klassen = {zeilen[i]["true"] for i in idx}
    assert klassen == {"io", "nio"}


def test_feste_indizes_reproduzierbar():
    zeilen = mache_zeilen()
    assert xa.feste_indizes(zeilen, k=5) == xa.feste_indizes(zeilen, k=5)


def test_feste_indizes_haengen_nicht_vom_lauf_ab():
    """Andere Wahrscheinlichkeiten (= anderer Lauf), gleiche feste Auswahl."""
    a = mache_zeilen(prob_nio_io=0.02, prob_nio_nio=0.99)
    b = mache_zeilen(prob_nio_io=0.40, prob_nio_nio=0.55)
    assert xa.feste_indizes(a, k=5) == xa.feste_indizes(b, k=5)


def test_gespreizt_enthaelt_raender_und_mehr_bauteile():
    zeilen = mache_zeilen(n_io=100, n_nio=100)
    idx = xa.feste_indizes(zeilen, k=5, gespreizt=True)
    assert len(idx) == 10
    assert idx[0] == 0 and idx[4] == 99        # Raender io
    assert idx[5] == 100 and idx[9] == 199     # Raender nio
    # gespreizt streut, Anfang/Ende klumpt
    assert idx != xa.feste_indizes(zeilen, k=5, gespreizt=False)


def test_feste_indizes_bei_winzigem_split():
    zeilen = mache_zeilen(n_io=2, n_nio=1)      # 3 Bilder, k=5 > n
    assert xa.feste_indizes(zeilen, k=5) == [0, 1, 2]


# ================================================================
#  Fehlerbestimmung - der kritische Teil
# ================================================================
def test_fehler_bei_05_nicht_laut_correct_spalte():
    """prob_nio=0,30 bei true=nio ist bei der val-Schwelle 0,21 KORREKT,
    bei der Auswahlschwelle 0,5 aber ein Fehler. Genau diese Zeile muss als
    Fehler gelten - sonst wird an der falschen Schwelle ausgewaehlt."""
    z = {"index": 0, "true": "nio", "prob_io": 0.70, "prob_nio": 0.30}
    assert xa.ist_fehler(z) is True
    assert xa.vorhersage(z) == "io"


def test_fehler_beide_richtungen():
    fp = {"index": 0, "true": "io", "prob_io": 0.2, "prob_nio": 0.8}
    fn = {"index": 1, "true": "nio", "prob_io": 0.8, "prob_nio": 0.2}
    ok = {"index": 2, "true": "io", "prob_io": 0.9, "prob_nio": 0.1}
    assert xa.ist_fehler(fp) and xa.ist_fehler(fn) and not xa.ist_fehler(ok)


def test_schwelle_genau_getroffen_zaehlt_als_nio():
    """prob_nio >= schwelle, nicht >. Muss zur Evaluator-Konvention passen."""
    z = {"index": 0, "true": "nio", "prob_io": 0.5, "prob_nio": 0.5}
    assert xa.vorhersage(z) == "nio"
    assert not xa.ist_fehler(z)


def test_konfidenz_ist_die_der_vorhergesagten_klasse():
    z = {"index": 0, "true": "nio", "prob_io": 0.93, "prob_nio": 0.07}
    assert xa.konfidenz(z) == pytest.approx(0.93)


# ================================================================
#  Zusammenspiel fest + Fehler
# ================================================================
def test_deckel_waehlt_die_konfidentesten_fehler():
    zeilen = mache_zeilen(n_io=30, n_nio=4)
    # Indizes 10..19 sind io, werden aber als nio vorhergesagt (10 Fehler)
    for i, p in zip(range(10, 20), [0.60, 0.99, 0.70, 0.95, 0.65,
                                    0.98, 0.80, 0.90, 0.75, 0.85]):
        zeilen[i]["prob_nio"] = p
        zeilen[i]["prob_io"] = 1 - p

    auswahl = xa.waehle_bilder(zeilen, "val", k=3, max_fehler=4)
    fehler = [a["index"] for a in auswahl if a["fehler"] and not a["fest"]]
    assert fehler == [11, 15, 13, 17] or sorted(fehler) == sorted([11, 15, 13, 17])
    assert len(fehler) == 4


def test_deckel_tie_break_nach_index():
    zeilen = mache_zeilen(n_io=20, n_nio=4)
    for i in (10, 11, 12):                       # identische Konfidenz
        zeilen[i]["prob_nio"] = 0.90
        zeilen[i]["prob_io"] = 0.10
    auswahl = xa.waehle_bilder(zeilen, "val", k=2, max_fehler=2)
    fehler = sorted(a["index"] for a in auswahl if a["fehler"] and not a["fest"])
    assert fehler == [10, 11]                    # kleinster Index gewinnt


def test_festes_bild_das_zugleich_fehler_ist_erscheint_einmal():
    zeilen = mache_zeilen(n_io=8, n_nio=6)
    zeilen[0]["prob_nio"] = 0.95                 # Index 0 ist fest UND falsch
    zeilen[0]["prob_io"] = 0.05

    auswahl = xa.waehle_bilder(zeilen, "val", k=3, max_fehler=6)
    treffer = [a for a in auswahl if a["index"] == 0]
    assert len(treffer) == 1
    assert treffer[0]["fest"] == 1 and treffer[0]["fehler"] == 1


def test_festes_fehlerbild_verbraucht_kein_kontingent():
    zeilen = mache_zeilen(n_io=20, n_nio=6)
    zeilen[0]["prob_nio"] = 0.99                 # fest und falsch
    zeilen[0]["prob_io"] = 0.01
    for i in (10, 11):                           # zwei weitere, nicht feste Fehler
        zeilen[i]["prob_nio"] = 0.90
        zeilen[i]["prob_io"] = 0.10

    auswahl = xa.waehle_bilder(zeilen, "val", k=3, max_fehler=2)
    zusatz = [a["index"] for a in auswahl if a["fehler"] and not a["fest"]]
    assert zusatz == [10, 11]                    # volle zwei, trotz Fehler bei 0


def test_auswahl_aufsteigend_und_ohne_duplikate():
    zeilen = mache_zeilen(n_io=20, n_nio=10)
    for i in (5, 12, 18):
        zeilen[i]["prob_nio"] = 0.9
        zeilen[i]["prob_io"] = 0.1
    idx = [a["index"] for a in xa.waehle_bilder(zeilen, "test", k=4, max_fehler=3)]
    assert idx == sorted(idx)
    assert len(idx) == len(set(idx))


def test_dateiname_ohne_vorhersage():
    """Der Name darf nicht von der Vorhersage abhaengen - sonst laegen dieselben
    festen Bilder in verschiedenen Laeufen unter verschiedenen Namen."""
    a = xa.heatmap_name("val", 42, "nio", True)
    assert a == "val_i0042_nio_fest.png"
    assert "io_" not in a.replace("nio_", "")     # kein pred im Namen
    assert xa.heatmap_name("test", 7, "io", False) == "test_i0007_io_fehler.png"


def test_feste_namen_gleich_bei_verschiedenen_laeufen():
    a = mache_zeilen(prob_nio_io=0.02, prob_nio_nio=0.99)
    b = mache_zeilen(prob_nio_io=0.45, prob_nio_nio=0.55)
    na = [x["heatmap"] for x in xa.waehle_bilder(a, "val", k=3, max_fehler=0) if x["fest"]]
    nb = [x["heatmap"] for x in xa.waehle_bilder(b, "val", k=3, max_fehler=0) if x["fest"]]
    assert na == nb


# ================================================================
#  Manifest
# ================================================================
def test_manifest_prueft_index_kopplung(tmp_path):
    zeilen = mache_zeilen(n_io=4, n_nio=2)
    samples = [(f"data/val/io/{i}.png", 0) for i in range(4)]
    samples += [(f"data/val/nio/{i}.png", 1) for i in range(2)]
    auswahl = xa.waehle_bilder(zeilen, "val", k=2, max_fehler=0)

    pfad = xa.schreibe_manifest(auswahl, zeilen, samples, str(tmp_path), "R", "val")
    with open(pfad, encoding="utf-8") as fh:
        gelesen = list(csv.DictReader(fh))
    assert len(gelesen) == len(auswahl)
    assert gelesen[0]["true"] == "io"


def test_manifest_erkennt_verschobene_indizes(tmp_path):
    """Ein Bild mehr im Datensatz als in der CSV -> muss knallen, nicht stillschweigen."""
    zeilen = mache_zeilen(n_io=4, n_nio=2)
    samples = [(f"data/val/io/{i}.png", 0) for i in range(5)]   # eins zu viel
    samples += [(f"data/val/nio/{i}.png", 1) for i in range(2)]
    auswahl = xa.waehle_bilder(zeilen, "val", k=2, max_fehler=0)
    with pytest.raises(RuntimeError, match="geaendert"):
        xa.schreibe_manifest(auswahl, zeilen, samples, str(tmp_path), "R", "val")


def test_manifest_erkennt_vertauschte_labels(tmp_path):
    zeilen = mache_zeilen(n_io=4, n_nio=2)
    samples = [(f"data/val/io/{i}.png", 0) for i in range(3)]
    samples += [(f"data/val/nio/{i}.png", 1) for i in range(3)]  # Grenze verschoben
    auswahl = xa.waehle_bilder(zeilen, "val", k=2, max_fehler=0)
    with pytest.raises(RuntimeError, match="Index-Kopplung"):
        xa.schreibe_manifest(auswahl, zeilen, samples, str(tmp_path), "R", "val")


def test_manifest_vollstaendig(tmp_path):
    zeilen = mache_zeilen(n_io=4, n_nio=2)
    samples = [(f"data/val/io/{i}.png", 0) for i in range(4)]
    samples += [(f"data/val/nio/{i}.png", 1) for i in range(2)]
    auswahl = xa.waehle_bilder(zeilen, "val", k=2, max_fehler=0)
    xa.schreibe_manifest(auswahl, zeilen, samples, str(tmp_path), "R", "val")

    fertig, vorhanden, erwartet = xa.manifest_vollstaendig(str(tmp_path))
    assert not fertig and vorhanden == 0 and erwartet == len(auswahl)

    for a in auswahl:
        (tmp_path / a["heatmap"]).write_bytes(b"x")
    fertig, vorhanden, erwartet = xa.manifest_vollstaendig(str(tmp_path))
    assert fertig and vorhanden == erwartet == len(auswahl)


def test_manifest_vollstaendig_ohne_manifest(tmp_path):
    assert xa.manifest_vollstaendig(str(tmp_path)) == (False, 0, 0)


# ================================================================
#  Einlesen echter CSVs
# ================================================================
def test_lade_preds_erkennt_luecken(tmp_path):
    run = tmp_path / "run"
    (run / "preds").mkdir(parents=True)
    with open(run / "preds" / "example_predictions.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "true_label", "predicted_label", "prob_io", "prob_nio", "correct"])
        w.writerow([0, "io", "io", 0.9, 0.1, 1])
        w.writerow([2, "io", "io", 0.9, 0.1, 1])          # Index 1 fehlt
    with pytest.raises(ValueError, match="lueckenlos"):
        xa.lade_preds(str(run), "test")


def test_preds_pfad_split_konvention():
    assert xa.preds_pfad("r", "test").endswith("example_predictions.csv")
    assert xa.preds_pfad("r", "val").endswith("example_predictions_val.csv")


# ================================================================
#  Zufallsauswahl (Stufe 2: 44 zufaellig + bis zu 6 Fehler = 50)
# ================================================================
def test_zufall_ist_laufunabhaengig():
    """Zwei Aufrufe liefern dieselben Indizes.

    Darauf beruht die ganze Vergleichbarkeit: nur wenn alle zwoelf Laeufe
    dieselben Bauteile erklaeren, sagt ein Unterschied zwischen zwei Heatmaps
    etwas ueber die Backbones und nicht ueber die Bildauswahl.
    """
    zeilen = mache_zeilen(n_io=200, n_nio=200)
    assert xa.zufalls_indizes(zeilen, 44, "test") == xa.zufalls_indizes(zeilen, 44, "test")


def test_zufall_klassenweise_gleich_verteilt():
    zeilen = mache_zeilen(n_io=200, n_nio=200)
    idx = xa.zufalls_indizes(zeilen, 44, "test")
    assert len(idx) == len(set(idx)) == 44
    assert sum(1 for i in idx if i < 200) == 22
    assert sum(1 for i in idx if i >= 200) == 22
    assert idx == sorted(idx)


def test_zufall_split_trennt_die_ziehung():
    """val und test duerfen nicht dieselben Positionen ziehen."""
    zeilen = mache_zeilen(n_io=200, n_nio=200)
    assert xa.zufalls_indizes(zeilen, 44, "val") != xa.zufalls_indizes(zeilen, 44, "test")


def test_zufall_ungerade_und_kleine_klasse():
    """Ungerades n und eine Klasse, die ihren Anteil nicht fuellen kann."""
    zeilen = mache_zeilen(n_io=3, n_nio=200)
    idx = xa.zufalls_indizes(zeilen, 45, "test")
    assert len(idx) == 45
    assert sum(1 for i in idx if i < 3) == 3        # alle io, mehr gibt es nicht
    assert sum(1 for i in idx if i >= 3) == 42      # der Rest faellt an nio


def test_zufall_mehr_als_vorhanden():
    zeilen = mache_zeilen(n_io=4, n_nio=4)
    assert len(xa.zufalls_indizes(zeilen, 99, "test")) == 8
    assert xa.zufalls_indizes(zeilen, 0, "test") == []


def test_waehle_bilder_zufall_plus_fehler():
    """44 Basis + Fehler, ohne Doppelung und ohne verbrauchtes Kontingent."""
    # nio-Bilder werden als io vorhergesagt -> alle 200 sind Fehler bei 0,5
    zeilen = mache_zeilen(n_io=200, n_nio=200, prob_nio_nio=0.2)
    auswahl = xa.waehle_bilder(zeilen, "test", max_fehler=6, zufall=44)

    indizes = [a["index"] for a in auswahl]
    assert len(indizes) == len(set(indizes)) == 50
    assert indizes == sorted(indizes)

    basis = {a["index"] for a in auswahl if a["fest"]}
    assert basis == set(xa.zufalls_indizes(zeilen, 44, "test"))
    # die 6 zusaetzlichen sind Fehler AUSSERHALB der Basis
    zusatz = [a for a in auswahl if not a["fest"]]
    assert len(zusatz) == 6
    assert all(a["fehler"] and a["index"] not in basis for a in zusatz)


def test_zufall_null_ist_altes_verhalten():
    """zufall=0 muss die Stufe-1-Auswahl bit-genau reproduzieren."""
    zeilen = mache_zeilen(n_io=20, n_nio=20)
    alt = xa.waehle_bilder(zeilen, "test", k=5, max_fehler=6)
    neu = xa.waehle_bilder(zeilen, "test", k=5, max_fehler=6, zufall=0)
    assert alt == neu
    assert {a["index"] for a in alt if a["fest"]} == set(xa.feste_indizes(zeilen, 5))
