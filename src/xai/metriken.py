"""
src/xai/metriken.py
Kennzahlen zur Bewertung von Attributionskarten.

Bis August 2026 ruhte die Bewertung der Erklaerbarkeitsverfahren auf einer
einzigen Groesse: dem Anteil der Attributionsmasse, der auf das Bauteil faellt.
Diese Groesse hat zwei Schwaechen, beide belegt in
`thesis/abbildungen/kap5/README.md`.

Erstens haengt sie an der Guete der Bauteilmaske. Vor dem Fix vom 20.08.2026
lieferte die Helligkeitsheuristik auf den Bauteil-Zuschnitten Werte UNTER
Gleichverteilung (Occlusion 0,51x) - gemessen wurde dort der Hintergrund, nicht
das Bauteil. Der Zahlenwert sah aus wie ein Befund ueber das Modell und war
einer ueber die Maske.

Zweitens ist der nackte Anteil ohne Bezugsgroesse nicht lesbar: 36,5 % klingt
nach wenig und sind das 2,6-Fache einer gleichverteilten Karte, weil das
Bauteil nur 13,9 % der Flaeche bedeckt.

Dieses Modul beantwortet beides:

- `konzentrationsfaktor` normiert den Anteil auf den Flaechenanteil. 1,0 heisst
  Gleichverteilung, unabhaengig davon, wie gross das Bauteil im Bild ist. Erst
  damit sind Stufe 1 (Vollbild) und Stufe 2 (Zuschnitt) vergleichbar.
- `maskensensitivitaet` beziffert die verbleibende Maskenabhaengigkeit, statt
  sie zu verschweigen: dieselbe Messung auf erodierten und dilatierten Masken.
- `deletion_insertion` kommt ohne jede Maske aus. Sie prueft am Modell selbst,
  ob die hoch attribuierten Pixel die Entscheidung tragen, und traegt die
  Aussage auch dann, wenn jemand die Maske anzweifelt.
- `sanity_spearman` gehoert zum Test von Adebayo et al. (2018): erklaert die
  Karte das Modell, oder zeigt sie nur Bildkanten?
- `otsu_maske` ersetzt die feste Quantilheuristik fuer die Vollbilder der
  Stufe 1 durch eine Schwelle, die sich am Bild selbst orientiert.

Die Funktionen sind frei von Projektwissen und ohne GPU testbar; einzig
`deletion_insertion` braucht ein Modell. Geprueft in `tests/test_xai.py`.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from src.xai.rueckprojektion import anteil_in_maske

# ---- Otsu-Maske -----------------------------------------------------------
# Die Vollbilder der Stufe 1 zeigen ein dunkles Bauteil auf hellem Grund - der
# Lehrbuchfall fuer Otsu. Die abgeloeste Heuristik nahm stattdessen die
# dunkelsten 24 % der Flaeche und war damit blind gegenueber der tatsaechlichen
# Bauteilgroesse: sie lieferte per Konstruktion immer 24 %, egal wie gross das
# Teil im Bild lag.
SCHLIESSEN = 9        # Kernel, um Bohrungen und Reflexe im Blech zu schliessen
OEFFNEN = 5           # Kernel, um Staub und Einzelpixel zu entfernen
MIN_ANTEIL = 0.01     # darunter gilt die Maske als fehlgeschlagen
MAX_ANTEIL = 0.90     # darueber ebenso (Otsu hat dann den Grund erwischt)
RAND_BREITE = 2       # Streifen, der als "beruehrt den Bildrand" gilt

# Untergrundnormierung. Die Aufnahmen der Stufe 1 zeigen das Bauteil vor einer
# hellen Wand, die aber NICHT gleichmaessig ausgeleuchtet ist: eine Bildhaelfte
# liegt im Schatten, dazu kommt Vignettierung und am unteren Rand die Tischkante.
# Eine einzige globale Schwelle - und Otsu ist genau das - schiebt die
# beschattete Wandhaelfte auf die Bauteilseite. Gemessen ueber den Testanteil
# beruehrte die Maske dadurch im Median rund 38 % des Bildrandes, in einem Fall
# bedeckte sie das ganze Bild.
#
# Abhilfe: den Untergrund per morphologischem Schliessen schaetzen (ein Kernel,
# der groesser ist als die schmale Ausdehnung des Bauteils, verschluckt das
# Bauteil und laesst die Wand stehen) und das Bild dadurch teilen. Danach ist die
# Wand ueberall gleich hell und die Schwelle trennt Bauteil von Wand.
#
# UNTERGRUND_KERNEL = 61 statt 101: beide fuehren zu einem Randkontakt-Median
# von 0,000, aber der groessere Kernel zog bei einzelnen Aufnahmen das dunkle
# Band am unteren Bildrand mit hinein (Maximum 0,269 gegenueber 0,082).
UNTERGRUND_KERNEL = 61


def untergrund_normieren(g8: np.ndarray, kernel: int = UNTERGRUND_KERNEL) -> np.ndarray:
    """Helligkeitsverlauf des Untergrunds herausrechnen (siehe UNTERGRUND_KERNEL)."""
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    hintergrund = cv2.morphologyEx(g8, cv2.MORPH_CLOSE, kern)
    verhaeltnis = g8.astype(np.float32) / np.maximum(hintergrund.astype(np.float32), 1.0)
    # Faktor 200 statt 255: das Verhaeltnis liegt auf der Wand bei 1,0, und ein
    # Ausreisser knapp darueber soll nicht in die Saettigung laufen.
    return np.clip(verhaeltnis * 200.0, 0, 255).astype(np.uint8)


def otsu_maske(bild: np.ndarray, dunkles_bauteil: bool = True,
               rand_ausschliessen: bool = False, normieren: bool = True):
    """Bauteilmaske ueber eine Otsu-Schwelle.

    bild: (H, W, 3) in [0, 1] oder (H, W) - das zurueckgerechnete Originalbild.
    dunkles_bauteil: True, wenn das Bauteil dunkler als der Untergrund ist.
    rand_ausschliessen: Komponenten verwerfen, die den Bildrand beruehren.
        Standardmaessig AUS. Der Gedanke war, die Tischkante am unteren Bildrand
        loszuwerden; nachdem die Untergrundnormierung das Problem an der Wurzel
        loest, richtet die Regel nur noch Schaden an: das Bauteil reicht auf
        einem Teil der Aufnahmen selbst bis an den Bildrand, und dort verwarf
        sie genau die richtige Komponente (bei 58 von 311 Aufnahmen blieb eine
        Maske unter 1 % der Flaeche uebrig).
    normieren: Helligkeitsverlauf des Untergrunds vorher herausrechnen.

    Ablauf: Untergrundnormierung -> Otsu-Schwelle -> morphologisches Schliessen
    und Oeffnen -> groesste Komponente -> Konturfuellung.

    Die Normierung traegt den Fortschritt gegenueber der abgeloesten
    Quantilheuristik; die Begruendung steht bei `UNTERGRUND_KERNEL`. Sie
    verhindert denselben Fehler, der auf den Zuschnitten der
    Stufe 2 schon einmal aufgetreten ist (vgl. `thesis/abbildungen/kap5/README.md`,
    Fix vom 20.08.2026): eine Helligkeitsregel, die statt des Bauteils den
    Untergrund trifft und deren Ergebnis wie eine Aussage ueber das Modell aussieht.

    Die Konturfuellung danach ist noetig, weil Ausschnitte im Blech sonst als
    Loch in der Maske stehen bleiben - sie zeigen die helle Wand dahinter und
    fielen auf die Untergrundseite der Schwelle. Der Flaechenanteil ist die
    Bezugsgroesse des Konzentrationsfaktors, ein Loch darin verschoebe ihn.

    Rueckgabe: (Maske als bool-Array, Diagnose-Dict). Die Diagnose traegt
    `plausibel`; ein False heisst nicht "Programmfehler", sondern "diese
    Aufnahme von Hand ansehen". Fehlschlaege werden protokolliert statt still
    verworfen - so wie `korb_zoom.Zoom.hinweis` es fuer die referenzbasierte
    Maske haelt.
    """
    grau = bild.mean(axis=2) if bild.ndim == 3 else bild
    if float(grau.max()) <= 1.0 + 1e-6:
        g8 = np.clip(grau * 255.0, 0, 255).astype(np.uint8)
    else:
        g8 = np.clip(grau, 0, 255).astype(np.uint8)

    if normieren:
        g8 = untergrund_normieren(g8)

    richtung = cv2.THRESH_BINARY_INV if dunkles_bauteil else cv2.THRESH_BINARY
    schwelle, roh = cv2.threshold(g8, 0, 255, richtung + cv2.THRESH_OTSU)

    kern_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SCHLIESSEN, SCHLIESSEN))
    kern_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OEFFNEN, OEFFNEN))
    bearbeitet = cv2.morphologyEx(roh, cv2.MORPH_CLOSE, kern_s)
    bearbeitet = cv2.morphologyEx(bearbeitet, cv2.MORPH_OPEN, kern_o)

    anzahl, marken, stats, _ = cv2.connectedComponentsWithStats(bearbeitet, connectivity=8)
    if anzahl <= 1:                                   # nur Hintergrund gefunden
        leer = np.zeros(g8.shape, dtype=bool)
        return leer, {"schwelle": float(schwelle), "anteil": 0.0, "komponenten": 0,
                      "randkontakt": 0.0, "rand_verworfen": 0, "plausibel": False}

    kandidaten = list(range(1, anzahl))
    verworfen = 0
    rand_gegriffen = True          # False = Rueckfall, siehe unten
    if rand_ausschliessen:
        b = RAND_BREITE
        am_rand = set(np.unique(np.concatenate([
            marken[:b, :].ravel(), marken[-b:, :].ravel(),
            marken[:, :b].ravel(), marken[:, -b:].ravel()])).tolist())
        frei = [k for k in kandidaten if k not in am_rand]
        verworfen = len(kandidaten) - len(frei)
        # Bleibt nichts uebrig, ist die Annahme "Bauteil liegt frei im Bild" fuer
        # diese Aufnahme falsch. Dann lieber die groesste Komponente ueberhaupt
        # nehmen und die Aufnahme als unplausibel markieren, als eine leere Maske
        # zu liefern und den Flaechenanteil auf null zu setzen.
        if frei:
            kandidaten = frei
        else:
            rand_gegriffen = False

    groesste = max(kandidaten, key=lambda k: stats[k, cv2.CC_STAT_AREA])
    maske = (marken == groesste).astype(np.uint8)

    konturen, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(maske, konturen, -1, color=1, thickness=cv2.FILLED)

    fertig = maske.astype(bool)
    anteil = float(fertig.mean())
    rand = np.concatenate([fertig[0, :], fertig[-1, :], fertig[:, 0], fertig[:, -1]])
    return fertig, {
        "schwelle": float(schwelle),
        "anteil": anteil,
        "komponenten": int(anzahl - 1),
        "randkontakt": float(rand.mean()),
        "rand_verworfen": int(verworfen),
        "plausibel": bool(MIN_ANTEIL <= anteil <= MAX_ANTEIL and rand_gegriffen),
    }


# ---- Konzentration --------------------------------------------------------
def konzentrationsfaktor(karte: np.ndarray, maske: np.ndarray):
    """(Faktor, Massenanteil, Flaechenanteil).

    Faktor = Massenanteil / Flaechenanteil. 1,0 bedeutet, dass die Karte auf dem
    Bauteil nicht mehr Masse traegt als eine gleichverteilte - das Bauteil
    bekommt dann genau seinen Flaechenanteil ab. Werte darueber heissen
    Konzentration.

    Diese Normierung ist der Grund, warum Vollbild und Zuschnitt ueberhaupt
    vergleichbar sind: der Rohanteil haengt schon daran, wie gross das Bauteil
    im Bild ist (Stufe 1: rund 24 %, Stufe 2: rund 14 % der Flaeche).

    NaN, wenn die Karte leer ist - der Regelfall bei Occlusion auf
    i.O.-Aufnahmen, wo kein abgedeckter Bereich die Wahrscheinlichkeit senkt.
    """
    massenanteil, flaechenanteil = anteil_in_maske(karte, maske)
    if flaechenanteil <= 1e-12:
        return float("nan"), massenanteil, flaechenanteil
    if massenanteil != massenanteil:                  # NaN durchreichen
        return float("nan"), massenanteil, flaechenanteil
    return float(massenanteil / flaechenanteil), massenanteil, flaechenanteil


def _verforme(maske: np.ndarray, pixel: int) -> np.ndarray:
    """Maske um `pixel` erodieren (negativ) oder dilatieren (positiv)."""
    if pixel == 0:
        return maske
    k = 2 * abs(pixel) + 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    roh = maske.astype(np.uint8)
    aus = cv2.dilate(roh, kern) if pixel > 0 else cv2.erode(roh, kern)
    return aus.astype(bool)


def maskensensitivitaet(karte: np.ndarray, maske: np.ndarray,
                        stufen=(-16, -8, 0, 8, 16)) -> dict:
    """Konzentrationsfaktor auf erodierten und dilatierten Masken.

    Beantwortet die Frage, die sonst jeder Pruefer stellt: wie stark haengt der
    Wert daran, wie genau das Bauteil maskiert wurde? Statt einer Zusicherung
    steht damit eine Spanne im Ergebnisteil.

    Fuer die Deutung wichtig: der Faktor ist bereits auf den Flaechenanteil
    normiert. Eine groessere Maske erhoeht Zaehler UND Nenner - bliebe der
    Faktor konstant, waere die Karte in der Randzone genauso dicht wie auf dem
    Bauteil. Genau das unterscheidet ein robustes Ergebnis von einem, das nur am
    Maskenrand haengt.
    """
    aus = {}
    for s in stufen:
        faktor, anteil, flaeche = konzentrationsfaktor(karte, _verforme(maske, s))
        aus[str(s)] = {"faktor": faktor, "massenanteil": anteil, "flaechenanteil": flaeche}
    return aus


# ---- Deletion / Insertion (maskenfrei) ------------------------------------
def _reihenfolge(karte: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pixelindizes nach Attribution absteigend, Gleichstaende zufaellig gemischt.

    Das Mischen ist nicht kosmetisch. Die Occlusion-Karte ist stueckweise
    konstant (ein Wert je 64x64-Feld), und eine Karte ohne Attribution ist
    ueberall null. Ohne Tiebreak entfernte `argsort` die Pixel eines Plateaus
    immer von links oben nach rechts unten - die Deletion-Kurve maesse dann die
    Lesereihenfolge des Arrays mit.
    """
    flach = np.abs(karte).ravel()
    return np.lexsort((rng.random(flach.size), -flach))


def _auc(kurve: np.ndarray, anteile: np.ndarray) -> float:
    """Flaeche unter der Kurve ueber dem Anteil ersetzter Pixel (0 bis 1)."""
    return float(np.trapezoid(kurve, anteile))


def _p(model, batch: torch.Tensor, ziel: int) -> np.ndarray:
    return torch.softmax(model(batch), dim=1)[:, ziel].detach().cpu().numpy()


@torch.no_grad()
def deletion_insertion(model, x: torch.Tensor, karte: np.ndarray, ziel: int,
                       schritte: int = 20, batch: int = 8,
                       baseline: torch.Tensor = None,
                       rng: np.random.Generator = None) -> dict:
    """Deletion- und Insertion-Kurve einer Attributionskarte.

    x: (1, C, H, W), bereits normalisiert. karte: (H, W).
    baseline: Ersatzwert; ohne Angabe der Nulltensor - dieselbe Baseline, die
    die Integrated Gradients verwenden (`completeness` vergleicht gegen
    `torch.zeros_like(x)`). Eine andere Baseline waere ein zweiter,
    stillschweigender Bezugspunkt im selben Kapitel.

    Deletion: von x ausgehend werden die hoechstattribuierten Pixel zuerst durch
    die Baseline ersetzt. Traegt die Karte, faellt p(ziel) schnell - eine
    KLEINE Flaeche ist gut.
    Insertion: von der Baseline ausgehend werden dieselben Pixel zuerst
    eingesetzt. Hier ist eine GROSSE Flaeche gut.

    Beide Groessen kommen ohne Bauteilmaske aus und messen am Modell selbst,
    nicht an einer Annahme darueber, wo das Bauteil liegt.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    if baseline is None:
        baseline = torch.zeros_like(x)
    _, _, h, w = x.shape
    if karte.shape[:2] != (h, w):
        raise ValueError(f"Karte {karte.shape[:2]} passt nicht zum Bild {(h, w)}")

    ordnung = _reihenfolge(karte, rng)
    n = ordnung.size
    anteile = np.linspace(0.0, 1.0, schritte + 1)
    grenzen = np.round(anteile * n).astype(int)

    def _kurve(start: torch.Tensor, quelle: torch.Tensor) -> np.ndarray:
        quelle_flach = quelle.reshape(1, -1, h * w)
        werte, puffer = [], []
        for k in grenzen:
            bild = start.clone().reshape(1, -1, h * w)
            if k > 0:
                idx = torch.from_numpy(ordnung[:k].copy()).to(x.device)
                bild[0, :, idx] = quelle_flach[0, :, idx]
            puffer.append(bild.reshape(x.shape))
            if len(puffer) == batch:
                werte.append(_p(model, torch.cat(puffer), ziel))
                puffer = []
        if puffer:
            werte.append(_p(model, torch.cat(puffer), ziel))
        return np.concatenate(werte)

    deletion = _kurve(x, baseline)          # x -> Baseline, wichtigste zuerst
    insertion = _kurve(baseline, x)         # Baseline -> x, wichtigste zuerst
    return {
        "anteile": anteile.tolist(),
        "deletion_kurve": deletion.tolist(),
        "insertion_kurve": insertion.tolist(),
        "deletion_auc": _auc(deletion, anteile),
        "insertion_auc": _auc(insertion, anteile),
    }


# ---- Sanity-Check (Adebayo et al. 2018) -----------------------------------
def sanity_spearman(karte_a: np.ndarray, karte_b: np.ndarray) -> dict:
    """Rangkorrelation zweier Attributionskarten.

    Adebayo et al. zeigen, dass manche Verfahren auch bei zufaelligen Gewichten
    noch dieselbe Karte liefern - sie erklaeren dann nicht das Modell, sondern
    zeigen Bildkanten. Der Test randomisiert die Gewichte schichtweise und
    vergleicht gegen die Karte des trainierten Modells. Bricht die Korrelation
    ein, haengt die Karte am Modell.

    Berichtet werden Rangkorrelation und ihr Betrag: ein Vorzeichenwechsel
    bedeutet fuer eine Relevanzkarte keine Aehnlichkeit, aber auch keine
    Unabhaengigkeit, und wuerde den Mittelwert sonst kuenstlich druecken.
    """
    from scipy.stats import spearmanr

    a = np.abs(karte_a).ravel()
    b = np.abs(karte_b).ravel()
    if a.size != b.size:
        raise ValueError(f"Karten verschiedener Groesse: {a.size} und {b.size}")
    if a.std() < 1e-12 or b.std() < 1e-12:   # konstante Karte: Rang undefiniert
        return {"spearman": float("nan"), "betrag": float("nan")}
    rho = float(spearmanr(a, b).statistic)
    return {"spearman": rho, "betrag": abs(rho)}


def randomisiere_schichten(model, bis_schicht: int, seed: int = 0):
    """Die letzten `bis_schicht` gewichtstragenden Schichten neu initialisieren.

    Kaskadierend von hinten nach vorn, so wie Adebayo et al. es beschreiben:
    Stufe 1 randomisiert nur die letzte Schicht, Stufe 2 die letzten zwei und so
    fort. Arbeitet auf dem uebergebenen Modell - der Aufrufer gibt eine Kopie
    hinein, keinen Verweis auf das Modell der Messung.

    Rueckgabe: (Modell, Gesamtzahl gewichtstragender Schichten). Die zweite
    Zahl braucht der Aufrufer, um die Randomisierungsstufen sinnvoll zu waehlen.
    """
    g = torch.Generator().manual_seed(seed)
    schichten = [m for m in model.modules()
                 if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    for m in schichten[len(schichten) - bis_schicht:]:
        with torch.no_grad():
            neu = torch.empty(m.weight.shape, device="cpu")
            if neu.dim() >= 2:
                torch.nn.init.kaiming_normal_(neu, generator=g)
            else:
                torch.nn.init.normal_(neu, generator=g)
            m.weight.copy_(neu.to(m.weight.device))
            if m.bias is not None:
                m.bias.zero_()
    return model, len(schichten)
