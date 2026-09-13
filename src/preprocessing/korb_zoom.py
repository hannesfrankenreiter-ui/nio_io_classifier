"""
Bauteil-Zoom: vom Originalbild zum Modelleingang
================================================
Hier steht die Geometrie der Vorverarbeitung - der Weg vom 4096x3000-Korbbild zu dem
Ausschnitt, den das Netz tatsaechlich sieht. Bis 19.08.2026 lag diese Logik allein in
`scripts/korb_referenz.py`. Sie ist hierher verschoben (nicht kopiert), damit
Datensatzerzeugung und Inferenz garantiert dieselbe Funktion benutzen: sobald es zwei
Fassungen gaebe, koennten sie auseinanderdriften, und dann zeigte eine zurueckprojizierte
Heatmap an die falsche Stelle, ohne dass es jemandem auffiele.

`scripts/korb_referenz.py` importiert die Funktionen von hier; seine CLI ist unveraendert.

Drei Verfahren, wie ein Datensatz aus den Originalen entsteht - welches gilt, sagt der
`zuschnitt:`-Block der Config, nicht eine Erkennung zur Laufzeit:

    bauteil   je Bild neu: Korbreferenz abziehen, groesste Komponente, quadratisches
              Fenster um deren Bounding-Box            (data_stufe2_512)
    fest      ein Rechteck fuer alle Aufnahmen         (dataStufe2zugeschnitten_800x320)
    ganz      kein Zuschnitt, das ganze Bild           (data_512)

Alle drei liefern dasselbe: ein Fenster (x0, y0, x1, y1) im Original. Genau dieses Fenster
ist die "gemerkte Zoomposition", ueber die eine Attributionskarte spaeter zurueckfindet.

Voraussetzungen: pip install numpy opencv-python
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Projektwurzel: src/preprocessing/korb_zoom.py -> zwei Ebenen hoch
WURZEL = Path(__file__).resolve().parents[2]

# ──────────────────────────────────────────────────────────────
#  KONFIGURIERBARE PARAMETER
# ──────────────────────────────────────────────────────────────

# Suchbereich (x, y, breite, höhe) im Originalbild: der Korbboden, auf dem
# Bauteile liegen können. Empirisch aus 45 verlässlichen Detektionen bestimmt
# (Hüllbox x 713–3063, y 1379–3000), mit Rand. Seitenwände und Deckenstreben
# bleiben draußen – dort sitzen die größten Restdifferenzen, aber nie ein Bauteil.
SUCHBEREICH      = (350, 1200, 3100, 1800)

# Referenzwahl
REF_STICHPROBE   = 4     # Unterabtastung beim Vergleich aller Referenzen (Tempo)
MAX_RESTDIFF     = 6     # passt keine Referenz besser als das, gilt das Bild als Fehlschlag

# Bauteilerkennung im Differenzbild
UNSCHAERFE       = 9     # Gaußkernel vor der Schwelle (glättet Sensorrauschen)
SCHWELLE         = 25    # Grauwert-Differenz ab der ein Pixel als Bauteil gilt
OEFFNEN_PX       = 11    # MORPH_OPEN – entfernt vereinzelte Glanzlicht-Pixel
SCHLIESSEN_PX    = 51    # MORPH_CLOSE – zieht das Bauteil zu einer Fläche zusammen
MIN_FLAECHE      = 4000  # kleinere Komponenten sind Funkeln, kein Bauteil
MAX_FLAECHE_ANT  = 0.12  # größere Anteile am Suchbereich sind verschobener Korb, kein Bauteil
# Füllgrad = Fläche / achsenparallele Bounding-Box, nur als Schutz gegen völlig
# entartete Formen. Bewusst niedrig: an 163 Stichproben gemessen trennt der
# Füllgrad Bauteil und Glanzlichtrest praktisch nicht (Verteilungen fast
# deckungsgleich), er verwirft aber bei 0,18 rund 4 % der echten Bauteile –
# vor allem diagonal liegende Lochstreifen, die ihre Bounding-Box kaum füllen.
# Die eigentliche Auswahl leisten die Flächengrenzen und "größte Komponente".
MIN_FUELLGRAD    = 0.10
RAND_PX          = 40    # Sicherheitsrand um die Bounding-Box beim Zuschnitt

VERFAHREN = ("bauteil", "fest", "ganz")


# ──────────────────────────────────────────────────────────────
#  REFERENZEN UND BAUTEILERKENNUNG
# ──────────────────────────────────────────────────────────────

def lade_referenzen(ordner) -> tuple[list[np.ndarray], list[dict]]:
    """Referenzbilder eines Korb-Referenzordners laden.

    Erwartet `index.json` neben den PNGs, so wie `korb_referenz.py erstellen` es
    schreibt. Rueckgabe: (Bilder, Index-Eintraege) in derselben Reihenfolge.
    """
    ordner = Path(ordner)
    index_datei = ordner / "index.json"
    if not index_datei.exists():
        raise FileNotFoundError(
            f"Keine Referenzen gefunden: {index_datei}\n"
            f"Zuerst 'python scripts/korb_referenz.py erstellen <bilderordner>' ausfuehren.")
    index = json.loads(index_datei.read_text(encoding="utf-8"))["referenzen"]
    referenzen = [cv2.imread(str(ordner / r["datei"]), cv2.IMREAD_COLOR) for r in index]
    fehlend = [r["datei"] for r, b in zip(index, referenzen) if b is None]
    if fehlend:
        raise RuntimeError(f"Referenzbild(er) nicht lesbar: {', '.join(fehlend)}")
    return referenzen, index


def beste_referenz(bild: np.ndarray, referenzen: list[np.ndarray],
                   bereich: tuple[int, int, int, int]) -> tuple[int, float]:
    """Referenz mit der kleinsten Restdifferenz im Suchbereich.

    Der Median ist robust gegen die wenigen Prozent Bauteilfläche – das Bauteil
    beeinflusst die Wahl also nicht, nur die Korbstellung tut es.
    Rückgabe: (index, median_restdifferenz).
    """
    x, y, b, h = bereich
    s = REF_STICHPROBE
    aus = bild[y:y + h:s, x:x + b:s]
    werte = [float(np.median(cv2.absdiff(aus, r[y:y + h:s, x:x + b:s]))) for r in referenzen]
    i = int(np.argmin(werte))
    return i, werte[i]


def bauteil_maske(bild: np.ndarray, referenz: np.ndarray,
                  bereich: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple, float]:
    """Differenz zur Referenz → Bauteilmaske, Bounding-Box, Füllgrad.

    Gesucht wird nur innerhalb von `bereich`; Maske und Box werden aber in
    Koordinaten des Gesamtbilds zurückgegeben. Bei Fehlschlag ist die Box
    (0, 0, 0, 0) und der Füllgrad 0.
    """
    bx, by, bb, bh = bereich
    d = cv2.cvtColor(cv2.absdiff(bild[by:by + bh, bx:bx + bb],
                                 referenz[by:by + bh, bx:bx + bb]), cv2.COLOR_BGR2GRAY)
    d = cv2.GaussianBlur(d, (UNSCHAERFE, UNSCHAERFE), 0)
    roh = (d > SCHWELLE).astype(np.uint8) * 255

    k_auf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OEFFNEN_PX, OEFFNEN_PX))
    k_zu = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SCHLIESSEN_PX, SCHLIESSEN_PX))
    roh = cv2.morphologyEx(roh, cv2.MORPH_OPEN, k_auf)
    roh = cv2.morphologyEx(roh, cv2.MORPH_CLOSE, k_zu)

    maske = np.zeros(bild.shape[:2], np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(roh, connectivity=8)
    max_flaeche = MAX_FLAECHE_ANT * roh.size
    beste, beste_flaeche, bester_fuellgrad = None, 0, 0.0
    for i in range(1, n):
        x, y, b, h, flaeche = stats[i, :5]
        if not MIN_FLAECHE <= flaeche <= max_flaeche:   # Funkeln bzw. verschobener Korb
            continue
        fuellgrad = flaeche / float(b * h)
        if fuellgrad < MIN_FUELLGRAD:                   # entartete Form, kein Bauteil
            continue
        if flaeche > beste_flaeche:
            beste, beste_flaeche, bester_fuellgrad = i, flaeche, fuellgrad

    if beste is None:
        return maske, (0, 0, 0, 0), 0.0

    # Gitter, das durch die Bauteilkontur scheint, mitfüllen
    teil = (labels == beste).astype(np.uint8) * 255
    konturen, _ = cv2.findContours(teil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    teil[:] = 0
    cv2.drawContours(teil, konturen, -1, 255, cv2.FILLED)
    maske[by:by + bh, bx:bx + bb] = teil

    x, y, b, h = (int(v) for v in stats[beste, :4])
    return maske, (x + bx, y + by, b, h), bester_fuellgrad


def zuschnitt_fenster(form: tuple[int, int], kasten: tuple[int, int, int, int],
                      quadratisch: bool) -> tuple[int, int, int, int]:
    """Zuschnittfenster (x0, y0, x1, y1) um die Bauteil-Box, inkl. Sicherheitsrand.

    Quadratisch: Seitenlänge = längere Kante + 2×Rand, um den Boxmittelpunkt
    zentriert. Ragt das Fenster über den Bildrand, wird es nach innen
    verschoben statt beschnitten – so bleibt es quadratisch und enthält echten
    Hintergrund statt künstlicher schwarzer Balken. Künstliche Ränder wären
    für ein CNN ein eigenes Merkmal; echtes Gitter ist es nicht.
    """
    H, W = form
    x, y, b, h = kasten
    if not quadratisch:
        return (max(0, x - RAND_PX), max(0, y - RAND_PX),
                min(W, x + b + RAND_PX), min(H, y + h + RAND_PX))

    seite = min(max(b, h) + 2 * RAND_PX, H, W)
    x0 = int(round(x + b / 2 - seite / 2))
    y0 = int(round(y + h / 2 - seite / 2))
    x0 = max(0, min(x0, W - seite))
    y0 = max(0, min(y0, H - seite))
    return x0, y0, x0 + seite, y0 + seite


# ──────────────────────────────────────────────────────────────
#  ZUSCHNITTPLAN: EIN FENSTER, DREI VERFAHREN
# ──────────────────────────────────────────────────────────────

@dataclass
class Zoom:
    """Die gemerkte Zoomposition eines Bildes, plus was dabei anfiel."""
    fenster: tuple[int, int, int, int]              # x0, y0, x1, y1 im Original
    verfahren: str
    referenz: str = ""                              # nur bei 'bauteil'
    restdifferenz: float = 0.0                      # nur bei 'bauteil'
    kasten: tuple[int, int, int, int] = (0, 0, 0, 0)
    fuellgrad: float = 0.0
    maske: Optional[np.ndarray] = field(default=None, repr=False)
    hinweis: str = ""                               # leer = auswertbar

    @property
    def auswertbar(self) -> bool:
        return not self.hinweis

    @property
    def groesse(self) -> tuple[int, int]:
        """(Breite, Hoehe) des Fensters."""
        x0, y0, x1, y1 = self.fenster
        return x1 - x0, y1 - y0


class Zuschnittplan:
    """Liefert zu jedem Originalbild das Fenster, das der Datensatz benutzt hat.

    Die Referenzbilder werden erst beim ersten Bild geladen (17 x 17 MB) - so kostet
    ein Plan fuer 'fest' oder 'ganz' nichts.
    """

    def __init__(self, verfahren: str, original=None, referenzen=None,
                 suchbereich=SUCHBEREICH, quadratisch: bool = True,
                 bereich=None, masken=None, wurzel: Path = WURZEL):
        if verfahren not in VERFAHREN:
            raise ValueError(f"zuschnitt.verfahren muss eines von {VERFAHREN} sein, "
                             f"nicht {verfahren!r}")
        self.verfahren = verfahren
        self.wurzel = Path(wurzel)
        self.original = self._pfad(original)
        self.referenzen_dir = self._pfad(referenzen)
        self.masken_dir = self._pfad(masken)
        self.suchbereich = tuple(int(v) for v in suchbereich)
        self.quadratisch = bool(quadratisch)
        self.bereich = tuple(int(v) for v in bereich) if bereich else None
        self._referenzen: Optional[list[np.ndarray]] = None
        self._index: Optional[list[dict]] = None

        if verfahren == "bauteil" and self.referenzen_dir is None:
            raise ValueError("zuschnitt.verfahren 'bauteil' braucht zuschnitt.referenzen "
                             "(Ordner mit den Korb-Referenzbildern)")
        if verfahren == "fest":
            if self.bereich is None or len(self.bereich) != 4:
                raise ValueError("zuschnitt.verfahren 'fest' braucht zuschnitt.bereich "
                                 "als [x, y, breite, hoehe]")

    def _pfad(self, wert) -> Optional[Path]:
        if not wert:
            return None
        p = Path(wert)
        return p if p.is_absolute() else self.wurzel / p

    def _laden(self) -> list[np.ndarray]:
        if self._referenzen is None:
            self._referenzen, self._index = lade_referenzen(self.referenzen_dir)
        return self._referenzen

    def fenster(self, bild: np.ndarray) -> Zoom:
        """Fenster fuer ein Originalbild. Bei Fehlschlag traegt Zoom.hinweis den Grund."""
        H, W = bild.shape[:2]

        if self.verfahren == "ganz":
            return Zoom(fenster=(0, 0, W, H), verfahren="ganz")

        if self.verfahren == "fest":
            x, y, b, h = self.bereich
            if x + b > W or y + h > H:
                return Zoom(fenster=(0, 0, W, H), verfahren="fest",
                            hinweis=f"Bereich {x},{y},{b},{h} passt nicht auf {W}x{H}")
            return Zoom(fenster=(x, y, x + b, y + h), verfahren="fest")

        # 'bauteil': Referenz waehlen, Bauteil finden, Fenster legen
        referenzen = self._laden()
        # Die Differenz wird pixelweise gebildet - eine abweichende Bildgroesse laesst
        # OpenCV sonst mit einer Meldung ueber Array-Groessen abbrechen, aus der niemand
        # ablesen kann, was zu tun ist. Ausserdem sind SUCHBEREICH und die Referenzen
        # auf die Aufnahmegroesse geeicht (hier 4096x3000).
        rh, rw = referenzen[0].shape[:2]
        if (H, W) != (rh, rw):
            return Zoom(fenster=(0, 0, W, H), verfahren="bauteil",
                        hinweis=f"Bildgroesse {W}x{H} passt nicht zu den Referenzen "
                                f"{rw}x{rh} in {self.referenzen_dir}")
        if self.suchbereich[0] + self.suchbereich[2] > W \
                or self.suchbereich[1] + self.suchbereich[3] > H:
            return Zoom(fenster=(0, 0, W, H), verfahren="bauteil",
                        hinweis=f"Suchbereich {list(self.suchbereich)} ragt ueber "
                                f"{W}x{H} hinaus")

        r, restdiff = beste_referenz(bild, referenzen, self.suchbereich)
        name = self._index[r]["datei"]
        if restdiff > MAX_RESTDIFF:
            # Korb steht in einer Stellung, fuer die keine Referenz existiert - die
            # Differenz waere voller Gitter und der Zuschnitt wertlos
            return Zoom(fenster=(0, 0, W, H), verfahren="bauteil", referenz=name,
                        restdifferenz=restdiff,
                        hinweis=f"keine passende Referenz (Restdifferenz {restdiff:.0f})")

        maske, kasten, fuellgrad = bauteil_maske(bild, referenzen[r], self.suchbereich)
        if kasten[2] == 0:
            return Zoom(fenster=(0, 0, W, H), verfahren="bauteil", referenz=name,
                        restdifferenz=restdiff, hinweis="kein Bauteil gefunden")

        return Zoom(fenster=zuschnitt_fenster((H, W), kasten, self.quadratisch),
                    verfahren="bauteil", referenz=name, restdifferenz=restdiff,
                    kasten=kasten, fuellgrad=fuellgrad, maske=maske)

    def als_dict(self) -> dict:
        """Der Plan zum Mitschreiben in parameter.json."""
        d = {"verfahren": self.verfahren,
             "original": str(self.original) if self.original else None}
        if self.verfahren == "bauteil":
            d.update(referenzen=str(self.referenzen_dir), suchbereich=list(self.suchbereich),
                     quadratisch=self.quadratisch,
                     masken=str(self.masken_dir) if self.masken_dir else None)
        if self.verfahren == "fest":
            d["bereich"] = list(self.bereich)
        return d


def zuschnittplan_aus_config(config: dict, wurzel: Path = WURZEL, **ueberschreibungen
                             ) -> Zuschnittplan:
    """Zuschnittplan aus dem `zuschnitt:`-Block einer Config bauen.

    Der Block sagt, wie `data_dir` aus den Originalaufnahmen entstanden ist. Er wird
    bewusst gelesen statt geraten: eine Erkennung zur Laufzeit muesste aus Ordnernamen
    schliessen und laege irgendwann still falsch. Fehlt der Block, ist das ein Fehler
    mit klarer Meldung - kein stiller Rueckfall auf 'ganz'.

    `ueberschreibungen` (aus CLI-Argumenten) gewinnen gegen den Block; None wird ignoriert.
    """
    block = dict(config.get("zuschnitt") or {})
    block.update({k: v for k, v in ueberschreibungen.items() if v is not None})
    if "verfahren" not in block:
        raise ValueError(
            "Die Config hat keinen 'zuschnitt:'-Block und es wurde keiner uebergeben.\n"
            "Entweder --zuschnitt-config <config.yaml> angeben oder --verfahren setzen.\n"
            "Der Block beschreibt, wie data_dir aus den Originalen entstanden ist:\n"
            "  zuschnitt:\n"
            "    verfahren: bauteil | fest | ganz\n"
            "    original:  data_stufe2")
    erlaubt = {"verfahren", "original", "referenzen", "suchbereich", "quadratisch",
               "bereich", "masken"}
    unbekannt = set(block) - erlaubt
    if unbekannt:
        raise ValueError(f"Unbekannte Schluessel im zuschnitt:-Block: {sorted(unbekannt)}")
    return Zuschnittplan(wurzel=wurzel, **block)
