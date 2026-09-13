"""
Rueckprojektion einer Attributionskarte ins Originalbild
=======================================================
Der Weg Original -> Zuschnitt -> Skalierung ist eine reine affine Abbildung: eine
Verschiebung um (x0, y0) und eine Streckung. Keine Rotation, kein Zufall - der
Eval-Transform besteht nur aus T.Resize, die Augmentierungen greifen ausschliesslich im
Trainingszweig. Die Umkehrung ist deshalb verlustfrei und besteht aus zwei Schritten:
Karte auf Fenstergroesse skalieren, an (x0, y0) einsetzen.

    Hinweg:   X_modell = (X_orig - x0) / sx        sx = (x1 - x0) / W_modell
    Rueckweg: X_orig   = x0 + X_modell * sx        sy = (y1 - y0) / H_modell

sx und sy werden getrennt gefuehrt. Bei quadratischem Zuschnitt auf quadratischen
Eingang sind sie gleich, aber der Lauf S2 skaliert einen 2999x1200-Streifen auf 320x800 -
wer dort Isotropie annimmt, verschiebt die Karte.

Umgekehrt wird nur die Geometrie, nicht die Pixelentstehung: ein Kartenpixel entspricht
sx bzw. sy Originalpixeln (im Datensatz 0,46 bis 4,12). Das gehoert in jede
Bildunterschrift. Bilineares Hochskalieren ist zulaessig, zusaetzliches Glaetten nicht -
dieselbe Linie gilt im Projekt schon bei der Saliency Map.
"""
from __future__ import annotations

from typing import Optional

import cv2
import matplotlib
import numpy as np

# Eine Implementierung der Min-Max-Normierung, nicht zwei: _normalize behandelt den
# konstanten Fall (Differenz < 1e-8 -> Nullen statt NaN), auf den sich die leere
# Occlusion-Karte verlaesst.
from src.xai.integrated_gradients import _normalize as normieren  # noqa: F401

# "inferno" ist im Projekt die faktische Konvention fuer Attributionskarten
# (integrated_gradients.py, xai_kennzahlen.py). Hier steht sie einmal als Konstante.
KARTE_CMAP = "inferno"

# Anzeigestile. Zwei Dinge unterscheiden sie: die Kennlinie (wie Attributionswerte auf
# Anzeigewerte abgebildet werden) und die Blendung (wie die Karte auf das Foto kommt).
#
#   blend "flaeche"   feste Alphas ueber weissem Grund - genau das, was run_xai
#                     (src/xai/integrated_gradients.py:378-379) fuer die Heatmaps eines
#                     Laufs tut. Das Foto wird dabei flaechig abgedunkelt; die dunkelste
#                     Stelle landet nicht bei 0, weil der weisse Achsengrund mit
#                     (1-alpha_karte)*(1-alpha_bild) = 18 % durchscheint. Beides gehoert
#                     zum Aussehen dazu und wird nicht nachtraeglich aufgehellt.
#   blend "deckkraft" pixelweise Deckkraft, Foto bleibt in voller Helligkeit. Wo nichts
#                     attribuiert ist, bleibt das Bild unveraendert.
STILE = {
    # Wie die Heatmaps im Lauf selbst - damit Rueckprojektion und xai/ desselben Laufs
    # dieselben Farben zeigen. Ohne Anzeigekennlinie, die Karte geht roh hinein.
    "lauf": {"blend": "flaeche", "kennlinie": "roh",
             "alpha_bild": 0.40, "alpha_karte": 0.70, "grund": 255.0},
    # Konvention der Thesis-Galerien (xai_kennzahlen.py, thesis/abbildungen/kap5/README.md)
    "thesis": {"blend": "deckkraft", "kennlinie": "gamma", "gamma": 0.30, "deckkraft": 0.92},
    # Praesentationsfolie: linear bis zum Quantil, darueber gekappt
    "demo": {"blend": "deckkraft", "kennlinie": "quantil", "kappung": 99.5, "deckkraft": 0.90},
}


def skala(fenster: tuple[int, int, int, int],
          kartengroesse: tuple[int, int]) -> tuple[float, float]:
    """Originalpixel je Kartenpixel, getrennt fuer x und y.

    `kartengroesse` ist (Hoehe, Breite) der Karte - die Reihenfolge von numpy.shape.
    """
    x0, y0, x1, y1 = fenster
    h, w = kartengroesse
    if w <= 0 or h <= 0:
        raise ValueError(f"Kartengroesse muss positiv sein, bekam {kartengroesse}")
    return (x1 - x0) / w, (y1 - y0) / h


def _skalieren(karte: np.ndarray, breite: int, hoehe: int) -> np.ndarray:
    """Karte auf (breite, hoehe) bringen. Beim Verkleinern Flaechenmittel.

    INTER_AREA beim Verkleinern ist kein Schoenheitsfehler: eine duenn besetzte
    Attributionskarte verschwindet sonst fast vollstaendig, weil INTER_LINEAR beim
    starken Verkleinern zwischen den besetzten Punkten abtastet.
    """
    h, w = karte.shape[:2]
    verfahren = cv2.INTER_AREA if (breite < w or hoehe < h) else cv2.INTER_LINEAR
    return cv2.resize(karte.astype(np.float32), (breite, hoehe), interpolation=verfahren)


def karte_ins_original(karte: np.ndarray, fenster: tuple[int, int, int, int],
                       quellgroesse: tuple[int, int]) -> np.ndarray:
    """Karte in Originalaufloesung an die Stelle setzen, an der gezoomt wurde.

    `quellgroesse` ist (Hoehe, Breite) des Originals. Ausserhalb des Fensters bleibt die
    Leinwand null - dort hat das Modell nichts gesehen, also wird dort auch nichts
    behauptet.
    """
    H, W = quellgroesse
    x0, y0, x1, y1 = (int(v) for v in fenster)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Fenster {fenster} liegt nicht im Bild {W}x{H}")

    leinwand = np.zeros((H, W), np.float32)
    leinwand[y0:y1, x0:x1] = _skalieren(karte, x1 - x0, y1 - y0)
    return leinwand


def karte_in_anzeige(karte: np.ndarray, fenster: tuple[int, int, int, int],
                     quellgroesse: tuple[int, int], breite: int) -> np.ndarray:
    """Wie karte_ins_original, aber direkt in der Aufloesung der Anzeige.

    Der Umweg ueber die volle Aufloesung mit anschliessendem Verkleinern des fertigen
    Bildes mittelt die duenn besetzte Karte weg. Deshalb wird gleich in
    Anzeigekoordinaten gerechnet - das Ergebnis ist dasselbe Bild, nur sichtbar.
    """
    H, W = quellgroesse
    faktor = breite / W
    hoehe = round(H * faktor)
    x0, y0, x1, y1 = fenster
    X0, Y0 = round(x0 * faktor), round(y0 * faktor)
    X1, Y1 = min(breite, round(x1 * faktor)), min(hoehe, round(y1 * faktor))
    if X1 <= X0 or Y1 <= Y0:                       # Fenster kleiner als ein Anzeigepixel
        X1, Y1 = min(breite, X0 + 1), min(hoehe, Y0 + 1)

    leinwand = np.zeros((hoehe, breite), np.float32)
    leinwand[Y0:Y1, X0:X1] = _skalieren(karte, X1 - X0, Y1 - Y0)
    return leinwand


def anzeigeskala(karte: np.ndarray, stil: str = "thesis",
                 gamma: Optional[float] = None,
                 kappung: Optional[float] = None) -> tuple[np.ndarray, float]:
    """Karte aus [0,1] auf die Anzeige umskalieren. Rueckgabe: (Anzeige, Kennwert).

    Beides sind monotone Umskalierungen der ANZEIGE, keine Glaettung: die raeumliche
    Verteilung bleibt unveraendert, null bleibt null, eins bleibt eins. Der Kennwert
    (Gamma bzw. der Kappungswert) gehoert in die Bildunterschrift.

    Noetig ist das, weil die Karten stark langschwaenzig sind - bei Integrated Gradients
    liegt der Median um 0,006 und das 99,9-%-Quantil um 0,29. Ungekappt bliebe die
    Deckkraft fast ueberall bei null und auf dem hellen Foto waere nichts zu sehen.
    """
    if stil not in STILE:
        raise ValueError(f"Stil muss einer von {sorted(STILE)} sein, nicht {stil!r}")
    k = np.clip(np.asarray(karte, dtype=np.float32), 0.0, 1.0)
    p = STILE[stil]

    if p["kennlinie"] == "roh":
        return k, 1.0                 # 1,0 = keine Anzeigeskalierung

    if p["kennlinie"] == "gamma":
        g = float(gamma if gamma is not None else p["gamma"])
        return np.power(k, g, dtype=np.float32), g

    q = float(kappung if kappung is not None else p["kappung"])
    grenze = float(np.percentile(k, q))
    if grenze <= 1e-8:            # leere Karte (Occlusion kann komplett null sein)
        return np.zeros_like(k), grenze
    return np.clip(k / grenze, 0.0, 1.0).astype(np.float32), grenze


def _farben(anzeige: np.ndarray) -> np.ndarray:
    """Anzeigewerte [0,1] -> BGR als float32 in [0,255]."""
    rgba = matplotlib.colormaps[KARTE_CMAP](np.clip(anzeige, 0.0, 1.0))
    return (rgba[..., 2::-1] * 255.0).astype(np.float32)


def einfaerben(anzeige: np.ndarray) -> np.ndarray:
    """Anzeigewerte [0,1] -> BGR-Bild in der Projekt-Colormap."""
    return _farben(anzeige).astype(np.uint8)


def ueberlagern(bild: np.ndarray, anzeige: np.ndarray, stil: str = "lauf") -> np.ndarray:
    """Karte ueber ein BGR-Bild legen, nach der Blendung des gewaehlten Stils.

    "flaeche" bildet die Ueberlagerung von run_xai nach: erst das Foto mit alpha_bild
    ueber weissem Grund, darauf die Karte mit alpha_karte. Das dunkelt die ganze Aufnahme
    flaechig ab - genau so sehen die Heatmaps im Lauf aus, und genau so bleibt es
    (keine nachtraegliche Aufhellung).

    "deckkraft" laesst das Foto in voller Helligkeit und macht die Deckkraft der Karte
    proportional zum Anzeigewert; wo nichts attribuiert ist, bleibt das Bild unveraendert.
    """
    if stil not in STILE:
        raise ValueError(f"Stil muss einer von {sorted(STILE)} sein, nicht {stil!r}")
    if bild.shape[:2] != anzeige.shape[:2]:
        raise ValueError(f"Bild {bild.shape[:2]} und Karte {anzeige.shape[:2]} "
                         f"haben verschiedene Groesse")
    p = STILE[stil]
    foto = bild.astype(np.float32)
    farbe = _farben(anzeige)

    if p["blend"] == "flaeche":
        unten = p["alpha_bild"] * foto + (1.0 - p["alpha_bild"]) * float(p["grund"])
        aus = p["alpha_karte"] * farbe + (1.0 - p["alpha_karte"]) * unten
    else:
        a = (np.clip(anzeige, 0.0, 1.0) * float(p["deckkraft"]))[..., None]
        aus = foto * (1.0 - a) + farbe * a
    return np.clip(aus, 0, 255).astype(np.uint8)


def breite_auf(bild: np.ndarray, breite: int) -> np.ndarray:
    """Bild auf eine Zielbreite bringen, Seitenverhaeltnis erhalten."""
    h, w = bild.shape[:2]
    if w == breite:
        return bild
    verfahren = cv2.INTER_AREA if breite < w else cv2.INTER_LINEAR
    return cv2.resize(bild, (breite, max(1, round(h * breite / w))), interpolation=verfahren)


def anteil_in_maske(karte: np.ndarray, maske: np.ndarray) -> tuple[float, float]:
    """(Anteil der Attributionsmasse in der Maske, Flaechenanteil der Maske).

    Berichtet wird der Anteil, nicht das Verhaeltnis Bauteil zu Hintergrund: die
    Occlusion-Karte ist im Hintergrund exakt null, der Quotient waere nicht definiert.
    Aussagekraeftig ist der Vergleich beider Rueckgabewerte - liegt der Massenanteil auf
    Hoehe des Flaechenanteils, zeigt die Karte nur Rauschen.
    """
    if karte.shape[:2] != maske.shape[:2]:
        raise ValueError(f"Karte {karte.shape[:2]} und Maske {maske.shape[:2]} "
                         f"haben verschiedene Groesse")
    drin = maske > 0
    gesamt = float(np.abs(karte).sum())
    flaeche = float(drin.mean())
    if gesamt <= 1e-12:                      # leere Karte: kein Anteil definierbar
        return float("nan"), flaeche
    return float(np.abs(karte)[drin].sum() / gesamt), flaeche
