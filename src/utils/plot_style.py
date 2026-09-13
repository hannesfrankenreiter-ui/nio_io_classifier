"""
src/utils/plot_style.py
Einheitlicher Diagramm-Stil für alle Abbildungen der Arbeit, abgeleitet aus
`Grafiken BA/Softmax.py` — dem Skript, aus dem die Funktionsgraphen des
Theorieteils stammen. Damit sehen die Diagramme des Ergebniskapitels aus wie
die des Theoriekapitels.

Merkmale (verbindlich, nicht lokal überschreiben):
Serif-Schrift, Titel in normaler Strichstärke, schwarze Achsen und Ticks,
feines Raster über beide Achsen, Legende ohne Rahmen, kein umlaufender Rahmen,
schlanke Pfeil-Achsen und die gedeckte, druckfreundliche Palette.

Einzige bewusste Abweichung von Softmax.py: dort steht die Legende in einem
grauen Kasten, hier steht sie frei.

Die Serif-Kette beginnt bei CMU Serif; ist keine davon installiert, landet
matplotlib bei Times New Roman — der Schrift des Fließtextes im Dokument.

Verwendung:

    from src.utils.plot_style import BLUE, RED, curve_style, style_axes

    with curve_style():
        fig, ax = plt.subplots()
        ax.plot(x, y, color=BLUE)
        style_axes(ax)
        fig.savefig(...)
"""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- gedeckte Palette (identisch zur HTML-/Softmax-Darstellung) ----
BLUE  = "#0F2DB3"   # Training / Primärkurve        (COLOR_1 in Softmax.py)
RED   = "#D1495B"   # Validierung / Sekundärkurve   (COLOR_2)
GREEN = "#2A9D4E"   # optionale dritte Serie        (COLOR_3)
AMBER = "#C77D14"   # optionale vierte Serie (Backbone-Vergleich, vier Kurven)
GRID  = "#E0E0E0"   # Rasterlinien                  (COLOR_GRID)
INK   = "#25282E"   # dunkler Text auf hellem Grund
INK2  = "#4C525B"   # Nebentext
MUT   = "#8A919B"   # Hilfslinien / Nebenserien
ANNOT = "#6B6B6B"   # Anmerkungen im Diagramm       (COLOR_ANNOT)

# Sequentielle Skala fuer Flaechendarstellungen (Matrizen, Heatmaps): von fast
# weiss bis BLUE. Damit traegt die Primaerfarbe der Arbeit auch die Flaechen und
# nicht das fremde matplotlib-"Blues" mit seinem Cyan-Stich.
#
# BLUE ist das Maximum der Skala - darueber hinaus wird nicht weiter abgedunkelt.
# Alle Stufen darunter sind reine Aufhellungen davon (Mischung mit Weiss), die
# Skala hat also genau einen Farbton. Das kostet Trennschaerfe am oberen Ende:
# dicht beieinanderliegende Spitzenwerte (F1 0,96...0,99) sehen aehnlich aus.
# In den Matrizen der Arbeit steht in jeder Zelle ohnehin die Zahl, die Farbe
# ordnet nur grob ein.
#
# Die Stuetzstellen folgen etwa einer quadratischen Kennlinie (Weiss-Anteil ^1,6)
# statt einer linearen: der helle Bereich zieht sich dadurch weiter nach oben und
# die oberen Stufen liegen nicht alle aufeinander.
#
# Beschriftung auf der Flaeche: ab etwa 70 % der Skala ist Weiss sicher lesbar
# (Kontrast >= 5:1), darunter INK.
BLAU_SKALA = LinearSegmentedColormap.from_list("zeiss_indigo", [
    (0.00, "#FDFDFE"),
    (0.15, "#F3F4FB"),
    (0.30, "#D9DEF3"),
    (0.45, "#BAC3E9"),
    (0.60, "#97A4DE"),
    (0.72, "#7586D3"),
    (0.83, "#5066C8"),
    (0.92, "#2D47BC"),
    (1.00, BLUE),
])

# Neutrale Variante derselben Kennlinie: von fast weiss bis INK2. Sie endet
# bewusst weit vor Schwarz - ein Grauverlauf bis in die Naehe von Schwarz macht
# die Flaeche schwer und zieht den Blick von den Kurvendiagrammen weg.
#
# INK2 ist der dunkelste hier vertretbare Endton: darueber wird die Flaeche zu
# schwer, darunter traegt sie die weisse Beschriftung nicht mehr. Bei einem
# Endton #6E747D kaeme weisse Schrift auf der Stufe 0,91 nur noch auf rund 3:1.
GRAU_SKALA = LinearSegmentedColormap.from_list("neutral_grau", [
    (0.00, "#FEFEFE"),
    (0.15, "#F6F7F7"),
    (0.30, "#E5E6E7"),
    (0.45, "#CDCFD1"),
    (0.60, "#B0B3B7"),
    (0.72, "#95999E"),
    (0.83, "#7A7F85"),
    (0.92, "#62686F"),
    (1.00, INK2),
])

# Achsen, Beschriftungen und Ticks sind schwarz — wie in Softmax.py. Die grauen
# Töne oben bleiben für Nebenserien und Anmerkungen reserviert.
_RC = {
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "normal",
    "axes.titlecolor": "black",
    "axes.labelsize": 12,
    "axes.labelcolor": "black",
    "axes.linewidth": 0.8,
    "text.color": "black",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.pad": 5,
    "ytick.major.pad": 5,
    "legend.frameon": False,   # bewusste Abweichung von Softmax.py
    "legend.fontsize": 10.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


# Serifenlose Alternative. Segoe UI ist auf jedem Windows vorhanden und die
# Schrift, in der die Abbildungen bis zum 11.08.2026 gesetzt waren.
SEGOE = "Segoe UI"


def curve_style(schrift: str | None = None):
    """
    rc_context mit dem einheitlichen Kurven-Stil (als ``with``-Block nutzen).

    schrift: ``None`` = Serifenkette der Arbeit (Standard, passt zum
    Fliesstext des Dokuments). Sonst der Name einer serifenlosen Schrift,
    etwa ``SEGOE``. Mathematik wird dann mitgezogen (``dejavusans`` statt
    Computer Modern) - ein CM-Tau neben serifenloser Beschriftung faellt auf.

    Schachtelbar: ein innerer Block ueberschreibt den aeusseren, eine einzelne
    Abbildung kann also aus dem Stil des Kapitels ausscheren.
    """
    if schrift is None:
        return plt.rc_context(_RC)
    return plt.rc_context({
        **_RC,
        "font.family": "sans-serif",
        "font.sans-serif": [schrift, "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
    })


def style_axes(ax, grid_axis: str = "both"):
    """
    Rahmen ausblenden, feines Raster einblenden und schlanke Pfeil-Achsen entlang
    der linken und unteren Kante zeichnen (Look des Softmax-Funktionsgraphen).

    grid_axis: 'both' (Standard, wie in Softmax.py), 'x' oder 'y'.

    Softmax.py zeichnet die Pfeile in Datenkoordinaten durch den Ursprung. Hier
    liegen sie an der Achsenkante, weil die Diagramme der Arbeit meist keinen
    Ursprung im Bild haben (Verlustachse 0,10-0,90, x beginnt bei 1).
    """
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.4, zorder=0)
    ax.tick_params(direction="in", length=4, width=0.6, colors="black")

    # Pfeil-Achsen in Achsen-Koordinaten (unabhängig vom Datenbereich)
    arrow = dict(arrowstyle="->", color="black", linewidth=1.5)
    ax.annotate("", xy=(1.02, 0.0), xytext=(0.0, 0.0), xycoords="axes fraction",
                arrowprops=arrow, annotation_clip=False, zorder=6)
    ax.annotate("", xy=(0.0, 1.03), xytext=(0.0, 0.0), xycoords="axes fraction",
                arrowprops=arrow, annotation_clip=False, zorder=6)
