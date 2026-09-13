"""
Tests fuer die Rueckprojektion einer Attributionskarte ins Originalbild.

Die Kernfrage ist immer dieselbe: landet ein bekannter Punkt der Karte an der Stelle im
Original, an der er hingehoert? Geprueft wird ueber den Schwerpunkt der zurueckprojizierten
Masse - der ist gegen die bilineare Interpolation robust, waehrend ein Argmax bei starker
Streckung auf einem von vier gleich hellen Nachbarn landen kann.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def schwerpunkt(karte: np.ndarray) -> tuple[float, float]:
    """(x, y) der Massenmitte einer Karte, in Pixelkoordinaten mit Halbpixel-Mitte."""
    y, x = np.nonzero(karte)
    w = karte[y, x].astype(np.float64)
    return float((x + 0.5) @ w / w.sum()), float((y + 0.5) @ w / w.sum())


def punktkarte(hoehe: int, breite: int, zeile: int, spalte: int) -> np.ndarray:
    karte = np.zeros((hoehe, breite), np.float32)
    karte[zeile, spalte] = 1.0
    return karte


# ================================================================
#  Skalenfaktoren
# ================================================================
class TestSkala:

    def test_quadratisch_ist_isotrop(self):
        """Quadratisches Fenster auf quadratische Karte: sx == sy."""
        from src.xai.rueckprojektion import skala
        sx, sy = skala((2355, 2114, 3221, 2980), (512, 512))
        assert sx == sy == pytest.approx(866 / 512), f"sx={sx}, sy={sy}"

    def test_stufe2_ist_anisotrop(self):
        """Der reale Fall S2: 2999x1200-Streifen auf 320x800 - sx und sy sind verschieden.

        Wer hier einen gemeinsamen Faktor annimmt, verschiebt die Karte.
        """
        from src.xai.rueckprojektion import skala
        sx, sy = skala((523, 1694, 523 + 2999, 1694 + 1200), (320, 800))
        assert sx == pytest.approx(2999 / 800), f"sx={sx}"
        assert sy == pytest.approx(1200 / 320), f"sy={sy}"
        assert sx != sy, "S2 ist anisotrop, der Test faengt genau das ab"

    def test_leere_kartengroesse_wird_abgelehnt(self):
        from src.xai.rueckprojektion import skala
        with pytest.raises(ValueError, match="positiv"):
            skala((0, 0, 10, 10), (0, 5))


# ================================================================
#  Rueckprojektion in Originalaufloesung
# ================================================================
class TestKarteInsOriginal:

    @pytest.mark.parametrize("zeile,spalte", [(2, 3), (4, 4), (5, 2)])
    def test_punkt_landet_an_der_erwarteten_stelle(self, zeile, spalte):
        """Ein heller Punkt bei (a, b) muss bei (x0 + a*sx, y0 + b*sy) ankommen.

        Nur Punkte im Inneren der Karte: am Rand klemmt die bilineare Interpolation die
        Quellkoordinate, wodurch aussen ein Plateau statt einer abfallenden Flanke steht.
        Der Schwerpunkt ist dort systematisch nach innen verschoben - das prueft
        test_randpunkt_bleibt_am_rand mit der passenden Messgroesse.
        """
        from src.xai.rueckprojektion import karte_ins_original, skala
        fenster = (100, 200, 356, 456)          # 256 x 256
        quelle = (800, 1000)
        karte = punktkarte(8, 8, zeile, spalte)

        aus = karte_ins_original(karte, fenster, quelle)
        assert aus.shape == quelle, f"Erwartet {quelle}, erhalten {aus.shape}"

        sx, sy = skala(fenster, karte.shape)
        soll_x = fenster[0] + (spalte + 0.5) * sx
        soll_y = fenster[1] + (zeile + 0.5) * sy
        ist_x, ist_y = schwerpunkt(aus)
        assert ist_x == pytest.approx(soll_x, abs=1.0), f"x: {ist_x:.2f} statt {soll_x:.2f}"
        assert ist_y == pytest.approx(soll_y, abs=1.0), f"y: {ist_y:.2f} statt {soll_y:.2f}"

    @pytest.mark.parametrize("zeile,spalte", [(0, 0), (7, 7), (0, 7)])
    def test_randpunkt_bleibt_am_rand(self, zeile, spalte):
        """Ein Punkt am Kartenrand muss in der zugehoerigen Ecke des Fensters landen.

        Gemessen wird hier das Maximum und die Ausdehnung der Masse, nicht der
        Schwerpunkt: durch das Klemmen am Bildrand ist die Verteilung asymmetrisch, der
        Schwerpunkt liegt dann um bis zu sx/2 innen. Die Lage selbst stimmt trotzdem.
        """
        from src.xai.rueckprojektion import karte_ins_original, skala
        fenster = (100, 200, 356, 456)          # 256 x 256
        karte = punktkarte(8, 8, zeile, spalte)
        aus = karte_ins_original(karte, fenster, (800, 1000))

        sx, sy = skala(fenster, karte.shape)
        soll_x = fenster[0] + (spalte + 0.5) * sx
        soll_y = fenster[1] + (zeile + 0.5) * sy
        y, x = np.nonzero(aus == aus.max())
        assert abs(float(x.mean()) - soll_x) <= sx, f"x: {x.mean():.1f} statt ~{soll_x:.1f}"
        assert abs(float(y.mean()) - soll_y) <= sy, f"y: {y.mean():.1f} statt ~{soll_y:.1f}"
        # Die gesamte Masse bleibt im erwarteten Umkreis von einem Kartenpixel
        ys, xs = np.nonzero(aus)
        assert xs.min() >= soll_x - sx and xs.max() <= soll_x + sx, "Masse zu weit gestreut"
        assert ys.min() >= soll_y - sy and ys.max() <= soll_y + sy, "Masse zu weit gestreut"

    def test_anisotrop_getrennte_faktoren(self):
        """Nicht-quadratisches Fenster auf nicht-quadratische Karte (der S2-Fall)."""
        from src.xai.rueckprojektion import karte_ins_original, skala
        fenster = (523, 1694, 523 + 2999, 1694 + 1200)
        quelle = (3000, 4096)
        karte = punktkarte(320, 800, 100, 600)

        aus = karte_ins_original(karte, fenster, quelle)
        sx, sy = skala(fenster, karte.shape)
        ist_x, ist_y = schwerpunkt(aus)
        assert ist_x == pytest.approx(523 + 600.5 * sx, abs=2.0), f"x={ist_x:.1f}"
        assert ist_y == pytest.approx(1694 + 100.5 * sy, abs=2.0), f"y={ist_y:.1f}"

    def test_ausserhalb_des_fensters_bleibt_null(self):
        """Wo das Modell nichts gesehen hat, wird nichts behauptet."""
        from src.xai.rueckprojektion import karte_ins_original
        fenster = (100, 200, 356, 456)
        aus = karte_ins_original(np.ones((16, 16), np.float32), fenster, (800, 1000))
        x0, y0, x1, y1 = fenster
        assert aus[y0:y1, x0:x1].min() > 0.9, "im Fenster muss die Karte stehen"
        aussen = aus.copy()
        aussen[y0:y1, x0:x1] = 0
        assert aussen.max() == 0.0, f"ausserhalb ist {aussen.max()}, erwartet 0"

    def test_fenster_am_bildrand(self):
        """Fenster in der Ecke: Groesse und Lage muessen trotzdem stimmen."""
        from src.xai.rueckprojektion import karte_ins_original
        quelle = (3000, 4096)
        aus = karte_ins_original(np.ones((64, 64), np.float32), (0, 0, 233, 233), quelle)
        assert aus.shape == quelle
        assert aus[0, 0] > 0.9 and aus[232, 232] > 0.9
        assert aus[233, 233] == 0.0, "einen Pixel ausserhalb muss es null sein"

    def test_fenster_kleiner_als_die_karte(self):
        """Kleinste reale Zuschnittseite 233 px gegen 512er-Karte: Faktor 0,455."""
        from src.xai.rueckprojektion import karte_ins_original
        aus = karte_ins_original(np.ones((512, 512), np.float32),
                                 (1000, 900, 1233, 1133), (3000, 4096))
        assert aus[900:1133, 1000:1233].min() > 0.9
        assert float(aus.sum()) == pytest.approx(233 * 233, rel=0.01)

    def test_fenster_groesser_als_die_karte(self):
        """Groesste reale Zuschnittseite 2108 px gegen 512er-Karte: Faktor 4,12."""
        from src.xai.rueckprojektion import karte_ins_original
        aus = karte_ins_original(np.ones((512, 512), np.float32),
                                 (500, 400, 2608, 2508), (3000, 4096))
        assert aus[400:2508, 500:2608].min() > 0.9
        assert float(aus.sum()) == pytest.approx(2108 * 2108, rel=0.01)

    def test_nullkarte_bleibt_null(self):
        """Occlusion kann komplett null sein - das darf nicht knallen."""
        from src.xai.rueckprojektion import karte_ins_original
        aus = karte_ins_original(np.zeros((512, 512), np.float32),
                                 (100, 100, 612, 612), (3000, 4096))
        assert np.isfinite(aus).all() and aus.max() == 0.0

    def test_fenster_ausserhalb_des_bildes_wird_abgelehnt(self):
        from src.xai.rueckprojektion import karte_ins_original
        with pytest.raises(ValueError, match="Fenster"):
            karte_ins_original(np.ones((8, 8), np.float32), (900, 900, 950, 950), (100, 100))


# ================================================================
#  Rueckprojektion in Anzeigeaufloesung
# ================================================================
class TestKarteInAnzeige:

    def test_lage_bleibt_relativ_gleich(self):
        """Derselbe Punkt, nur in kleinerer Aufloesung - die relative Lage haelt."""
        from src.xai.rueckprojektion import karte_in_anzeige, karte_ins_original
        fenster, quelle, breite = (523, 1694, 3522, 2894), (3000, 4096), 2000
        karte = punktkarte(320, 800, 100, 600)

        gross = karte_ins_original(karte, fenster, quelle)
        klein = karte_in_anzeige(karte, fenster, quelle, breite)
        assert klein.shape == (round(3000 * breite / 4096), breite), klein.shape

        gx, gy = schwerpunkt(gross)
        kx, ky = schwerpunkt(klein)
        f = breite / quelle[1]
        assert kx == pytest.approx(gx * f, abs=2.0), f"x: {kx:.1f} statt {gx * f:.1f}"
        assert ky == pytest.approx(gy * f, abs=2.0), f"y: {ky:.1f} statt {gy * f:.1f}"

    def test_duenne_karte_ueberlebt_das_verkleinern(self):
        """Der Grund fuer INTER_AREA: eine duenn besetzte Karte darf nicht verschwinden.

        Ein einzelner Punkt in einer 512er-Karte, projiziert in ein grosses Fenster und
        dann auf Anzeigebreite gebracht, muss noch Masse tragen.
        """
        from src.xai.rueckprojektion import karte_in_anzeige
        karte = punktkarte(512, 512, 256, 256)
        klein = karte_in_anzeige(karte, (500, 400, 2608, 2508), (3000, 4096), 2000)
        assert klein.sum() > 0, "die Karte ist beim Verkleinern verschwunden"


# ================================================================
#  Anzeigeskala
# ================================================================
class TestAnzeigeskala:

    def test_gamma_ist_monoton_und_haelt_die_endpunkte(self):
        from src.xai.rueckprojektion import anzeigeskala
        karte = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
        aus, kennwert = anzeigeskala(karte, "thesis")
        assert kennwert == pytest.approx(0.30)
        assert aus.min() == 0.0 and aus.max() == pytest.approx(1.0)
        flach = aus.ravel()
        assert np.all(np.diff(flach) >= -1e-6), "Gamma muss monoton sein"

    def test_gamma_hebt_kleine_werte(self):
        """Der Zweck: kleine Attributionen sichtbar machen, ohne die Ordnung zu aendern."""
        from src.xai.rueckprojektion import anzeigeskala
        aus, _ = anzeigeskala(np.array([[0.006]], np.float32), "thesis")
        assert aus[0, 0] > 0.2, f"0,006 bleibt bei {aus[0, 0]:.3f} unsichtbar"

    def test_quantil_kappt_den_langen_schwanz(self):
        from src.xai.rueckprojektion import anzeigeskala
        karte = np.full((100, 100), 0.01, np.float32)
        karte[0, 0] = 1.0
        aus, grenze = anzeigeskala(karte, "demo", kappung=99.0)
        assert grenze == pytest.approx(0.01, abs=1e-6), f"Grenze {grenze}"
        assert aus[0, 0] == pytest.approx(1.0), "der Ausreisser wird gekappt, nicht entfernt"

    def test_leere_karte_ohne_division_durch_null(self):
        """Die leere Occlusion-Karte laeuft durch beide Kennlinien."""
        from src.xai.rueckprojektion import anzeigeskala
        leer = np.zeros((32, 32), np.float32)
        for stil in ("thesis", "demo"):
            aus, _ = anzeigeskala(leer, stil)
            assert np.isfinite(aus).all(), f"{stil}: nicht-endliche Werte"
            assert aus.max() == 0.0, f"{stil}: leere Karte wird nicht leer angezeigt"

    def test_lauf_laesst_die_karte_unveraendert(self):
        """Stil 'lauf' skaliert die Anzeige nicht - die Karte geht roh in die Farbe."""
        from src.xai.rueckprojektion import anzeigeskala
        karte = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
        aus, kennwert = anzeigeskala(karte, "lauf")
        assert np.array_equal(aus, karte) and kennwert == 1.0

    def test_unbekannter_stil(self):
        from src.xai.rueckprojektion import anzeigeskala
        with pytest.raises(ValueError, match="Stil"):
            anzeigeskala(np.zeros((4, 4), np.float32), "bunt")


# ================================================================
#  Ueberlagerung
# ================================================================
class TestUeberlagern:

    @pytest.mark.parametrize("stil", ["thesis", "demo"])
    def test_deckkraft_laesst_unattribuierte_stellen_unveraendert(self, stil):
        """Bei pixelweiser Deckkraft gibt es keine Abdunklung ohne Grund."""
        from src.xai.rueckprojektion import ueberlagern
        bild = np.full((16, 16, 3), 200, np.uint8)
        anzeige = np.zeros((16, 16), np.float32)
        anzeige[8, 8] = 1.0
        aus = ueberlagern(bild, anzeige, stil)
        assert (aus[0, 0] == 200).all(), f"unattribuiert veraendert: {aus[0, 0]}"
        assert not (aus[8, 8] == 200).all(), "attribuiert unveraendert"

    def test_lauf_blendet_flaechig_wie_run_xai(self):
        """Stil 'lauf' muss die Ueberlagerung aus integrated_gradients.py:378-379 treffen:
        imshow(bild, alpha=0.4) und darueber imshow(karte, inferno, alpha=0.7) auf
        weissem Achsengrund. Gerechnet:
            0,7*inferno(karte) + 0,3*(0,4*bild + 0,6*255)
        Der weisse Grund scheint dabei mit 18 % durch - deshalb ist die dunkelste Stelle
        nicht schwarz. Das gehoert zum Aussehen und wird nicht wegkorrigiert.
        """
        import matplotlib
        from src.xai.rueckprojektion import ueberlagern
        bild = np.full((8, 8, 3), 200, np.uint8)
        anzeige = np.zeros((8, 8), np.float32)
        anzeige[4, 4] = 1.0

        farbe = matplotlib.colormaps["inferno"](anzeige)[..., 2::-1] * 255.0
        soll = 0.7 * farbe + 0.3 * (0.4 * bild.astype(float) + 0.6 * 255.0)
        aus = ueberlagern(bild, anzeige, "lauf")
        assert np.abs(aus.astype(float) - soll).max() <= 1.0, (
            f"groesste Abweichung {np.abs(aus.astype(float) - soll).max():.2f}")
        # Das Foto wird flaechig abgedunkelt, auch wo nichts attribuiert ist
        assert aus[0, 0].max() < 200, f"nicht abgedunkelt: {aus[0, 0]}"

    def test_unbekannter_stil(self):
        from src.xai.rueckprojektion import ueberlagern
        with pytest.raises(ValueError, match="Stil"):
            ueberlagern(np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8), np.float32), "bunt")

    def test_groessen_muessen_passen(self):
        from src.xai.rueckprojektion import ueberlagern
        with pytest.raises(ValueError, match="Groesse"):
            ueberlagern(np.zeros((8, 8, 3), np.uint8), np.zeros((4, 4), np.float32))


# ================================================================
#  Attributionsanteil auf dem Bauteil
# ================================================================
class TestAnteilInMaske:

    def test_konzentrierte_karte_liegt_ueber_dem_flaechenanteil(self):
        from src.xai.rueckprojektion import anteil_in_maske
        karte = np.zeros((100, 100), np.float32)
        karte[40:60, 40:60] = 1.0
        maske = np.zeros((100, 100), np.uint8)
        maske[40:60, 40:60] = 255
        anteil, flaeche = anteil_in_maske(karte, maske)
        assert anteil == pytest.approx(1.0) and flaeche == pytest.approx(0.04)

    def test_gleichverteilte_karte_trifft_den_flaechenanteil(self):
        """Reines Rauschen: der Massenanteil entspricht dem Flaechenanteil."""
        from src.xai.rueckprojektion import anteil_in_maske
        maske = np.zeros((100, 100), np.uint8)
        maske[:24] = 255
        anteil, flaeche = anteil_in_maske(np.ones((100, 100), np.float32), maske)
        assert anteil == pytest.approx(flaeche, abs=1e-6)

    def test_leere_karte_ergibt_nan_statt_null(self):
        """Bei leerer Karte ist kein Anteil definiert - nan, nicht 0, damit es auffaellt."""
        from src.xai.rueckprojektion import anteil_in_maske
        anteil, flaeche = anteil_in_maske(np.zeros((10, 10), np.float32),
                                          np.ones((10, 10), np.uint8))
        assert np.isnan(anteil) and flaeche == pytest.approx(1.0)


# ================================================================
#  Zuschnittplan: Groessen muessen zusammenpassen
# ================================================================
class TestZuschnittplan:
    """Der Zuschnitt ist auf die Aufnahmegroesse geeicht (hier 4096x3000).

    Passt ein Bild nicht dazu, muss das als lesbarer Hinweis herauskommen und nicht als
    OpenCV-Meldung ueber Array-Groessen - sonst sucht man den Fehler an der falschen Stelle.
    """

    def _plan(self, tmp_path, referenzgroesse=(3000, 4096)):
        import cv2
        import json
        from src.preprocessing.korb_zoom import Zuschnittplan
        ref = tmp_path / "ref"
        ref.mkdir()
        cv2.imwrite(str(ref / "referenz_000000.png"),
                    np.zeros((*referenzgroesse, 3), np.uint8))
        (ref / "index.json").write_text(
            json.dumps({"referenzen": [{"datei": "referenz_000000.png"}]}), encoding="utf-8")
        return Zuschnittplan("bauteil", original=tmp_path, referenzen=ref)

    def test_abweichende_bildgroesse_meldet_sich(self, tmp_path):
        plan = self._plan(tmp_path)
        zoom = plan.fenster(np.zeros((1500, 2048, 3), np.uint8))
        assert not zoom.auswertbar
        assert "2048x1500" in zoom.hinweis and "4096x3000" in zoom.hinweis, zoom.hinweis

    def test_suchbereich_ausserhalb_meldet_sich(self, tmp_path):
        from src.preprocessing.korb_zoom import Zuschnittplan
        plan = self._plan(tmp_path, referenzgroesse=(600, 800))
        plan.suchbereich = (0, 0, 3100, 1800)
        zoom = plan.fenster(np.zeros((600, 800, 3), np.uint8))
        assert not zoom.auswertbar and "ragt ueber" in zoom.hinweis, zoom.hinweis

    def test_ganz_und_fest_brauchen_keine_referenzen(self):
        from src.preprocessing.korb_zoom import Zuschnittplan
        bild = np.zeros((3000, 4096, 3), np.uint8)
        assert Zuschnittplan("ganz").fenster(bild).fenster == (0, 0, 4096, 3000)
        fest = Zuschnittplan("fest", bereich=[523, 1694, 2999, 1200]).fenster(bild)
        assert fest.fenster == (523, 1694, 3522, 2894) and fest.auswertbar

    def test_fest_bereich_zu_gross_meldet_sich(self):
        from src.preprocessing.korb_zoom import Zuschnittplan
        zoom = Zuschnittplan("fest", bereich=[0, 0, 5000, 1200]).fenster(
            np.zeros((3000, 4096, 3), np.uint8))
        assert not zoom.auswertbar and "passt nicht" in zoom.hinweis, zoom.hinweis


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
