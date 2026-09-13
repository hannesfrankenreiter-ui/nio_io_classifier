#!/usr/bin/env python3
"""
scripts/thesis_kap5_abbildungen.py
Erzeugt saemtliche Diagramme des Ergebniskapitels (Kapitel 5) aus den bereits
abgelegten Ergebnisdateien einer Versuchsstufe. Kein Modell, kein torch - die
Daten kommen ueber `kap5_daten.py`, dieselbe Quelle, aus der auch die Tabellen
des Kapitels gebaut werden.

Ausgabe: thesis/abbildungen/kap5/ (Vektorgrafik, SVG)
    verlustkurven.svg         2x2, Verlust auf Trainings- und Validierungsanteil
    trainingsverlaeufe.svg    2x2, je Backbone die drei Freeze-Varianten
    f1_matrix.svg             4x3-Matrix F1 (n.i.O.) @0,5
    guete_aufwand.svg         F1 @0,5 ueber Rechenzeit je Bild
    konfusionsmatrizen.svg    bestes und zweitbestes Modell bei ihrer val-Schwelle
    roc.svg                   alle Laeufe, links vollstaendig, rechts Ausschnitt
    schwellenverlauf.svg      Precision/Recall/F1 ueber tau, bestes Modell

Massgeblich fuer die Groesse: Die Bilder werden im Dokument auf 16,0 cm Breite
eingefuegt. Die Figuren sind mit 19 cm etwas breiter angelegt, damit die
Beschriftungen beim Herunterskalieren bei rund 9 pt landen und damit knapp
unter der Schriftgroesse des Fliesstextes bleiben.

Der Stil kommt vollstaendig aus src/utils/plot_style.py (Look der Funktions-
graphen des Theorieteils, vgl. Grafiken BA/Softmax.py) - hier keine eigenen
Farb- oder Schriftwerte setzen.

Alle Stufen schreiben in denselben Ordner. Ohne `--praefix` ueberschriebe die
zweite Versuchsstufe deshalb die Diagramme der ersten - genau wie bei den
Galerien in `xai_kennzahlen.py`. Der Praefix wird dem Dateinamen vorangestellt,
nicht angehaengt, damit die sieben Diagramme einer Stufe im Ordner beieinander
stehen.

Aufruf:
    python scripts/thesis_kap5_abbildungen.py
    python scripts/thesis_kap5_abbildungen.py --stufe 1
    python scripts/thesis_kap5_abbildungen.py --stufe 2 --praefix stufe2_
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_curve

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from kap5_daten import (BACKBONE_NAME, FREEZE_NAME, FREEZE_ORDER, METHODEN,  # noqa: E402
                        METHODEN_NAME, de, di_kurven_je_klasse, konfusion,
                        lade_inferenzzeit, lade_laeufe, lade_sweep,
                        lade_xai_kennzahlen, rangfolge, werte_je_verfahren)
from src.utils.plot_style import (AMBER, ANNOT, BLUE, GRAU_SKALA, GREEN, INK,  # noqa: E402
                                  INK2, MUT, RED, SEGOE, curve_style, style_axes)

# ---- Darstellungsgroessen -------------------------------------------------
BREITE_ZOLL = 7.48   # 19 cm; im Dokument auf 16 cm skaliert
FORMAT = "svg"       # Vektor - die Einfuegung auf 16 cm skaliert verlustfrei

BACKBONE_FARBE = {
    "resnet18": BLUE,
    "resnet50": RED,
    "efficientnet_b0": GREEN,
    "convnext_tiny": AMBER,
}
# Farbe, Strichart und Symbol je Freeze-Strategie - die Strichart traegt die
# Unterscheidung auch im Schwarzweissdruck.
FREEZE_STIL = {
    "nurkopf": (MUT, ":", "s"),
    "letzterblock": (BLUE, "--", "^"),
    "vollstaendig": (RED, "-", "o"),
}

# Die rc-Schluessel, die allein die Schrift bestimmen. `abb_konfusionsmatrizen`
# setzt den Kapitelstil auf die matplotlib-Voreinstellungen zurueck und hebt
# genau diese Schluessel davon aus - sonst waere sie die einzige Abbildung, die
# der Schriftzuordnung in `main()` nicht folgt.
SCHRIFT_SCHLUESSEL = ("font.family", "font.serif", "font.sans-serif", "mathtext.fontset")

# Untere Grenze der Kennzahlenachsen. Bewusst nicht null: die Unterschiede
# zwischen den brauchbaren Laeufen liegen saemtlich oberhalb 0,80 und waeren
# sonst nicht ablesbar. Kurven, die darunter verlaufen, werden im Text benannt.
Y_UNTEN = 0.40

# Achsenbeschriftung mit deutschem Dezimalkomma - matplotlib setzt sonst Punkte,
# was im deutschsprachigen Fliesstext danebensteht.
_KOMMA = FuncFormatter(lambda w, _: f"{w:g}".replace(".", ","))


def komma_achsen(ax, x: bool = True, y: bool = True):
    if x:
        ax.xaxis.set_major_formatter(_KOMMA)
    if y:
        ax.yaxis.set_major_formatter(_KOMMA)


# Wird in main() aus --praefix gesetzt. Modulweit statt als Argument durch alle
# sieben Zeichenfunktionen gereicht: der Praefix betrifft ausschliesslich den
# Dateinamen, nicht das Bild.
PRAEFIX = ""


def speichern(fig, out_dir, name: str) -> str:
    """Abbildung ablegen, Figur schliessen, Pfad zurueckgeben."""
    pfad = os.path.join(out_dir, f"{PRAEFIX}{name}.{FORMAT}")
    fig.savefig(pfad, format=FORMAT, bbox_inches="tight")
    plt.close(fig)
    return pfad


def abb_trainingsverlaeufe(laeufe, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(BREITE_ZOLL, 5.4), sharey=True)
    unterschritten = []
    for ax, backbone in zip(axes.ravel(), BACKBONE_NAME):
        for freeze in FREEZE_ORDER:
            lauf = next(l for l in laeufe if l["backbone"] == backbone and l["freeze"] == freeze)
            farbe, stil, marker = FREEZE_STIL[freeze]
            ax.plot(lauf["val_ep"], lauf["val_f1"], color=farbe, linestyle=stil,
                    linewidth=1.6, label=FREEZE_NAME[freeze])
            if min(lauf["val_f1"]) < Y_UNTEN:
                unterschritten.append(lauf["run"])
            b_ep = lauf["beste_epoche"]
            b_f1 = lauf["val_f1"][lauf["val_ep"].index(b_ep)]
            if b_f1 >= Y_UNTEN:
                ax.plot([b_ep], [b_f1], marker=marker, color=farbe, markersize=5,
                        markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        ax.set_title(BACKBONE_NAME[backbone])
        ax.set_xlabel("Epoche")
        ax.set_ylim(Y_UNTEN, 1.005)
        ax.set_xlim(left=1)
        style_axes(ax, grid_axis="both")
        komma_achsen(ax, x=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("F1 (n.i.O.), Validierung")

    handles = [Line2D([], [], color=FREEZE_STIL[f][0], linestyle=FREEZE_STIL[f][1],
                      marker=FREEZE_STIL[f][2], markersize=5, linewidth=1.6,
                      label=FREEZE_NAME[f]) for f in FREEZE_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0),
               columnspacing=2.4, handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pfad = speichern(fig, out_dir, "trainingsverlaeufe")
    if unterschritten:
        print(f"  Hinweis: unterhalb der y-Achse verlaufen {', '.join(sorted(set(unterschritten)))}")
    return pfad


def abb_verlustkurven(laeufe, out_dir):
    """
    Verlust auf Trainings- und Validierungsanteil, 2x2 je Backbone.

    Der Validierungsverlust ist die Groesse, die ueber Abbruch und gesicherten
    Modellstand entscheidet (monitor: val_loss) - er gehoert deshalb vor die
    F1-Kurve. Der Trainingsverlust laeuft duenn und blass daneben; sein Abstand
    zur Validierungskurve ist das Einzige, woran sich Ueberanpassung ablesen
    laesst.
    """
    y_unten, y_oben = 0.10, 0.90
    fig, axes = plt.subplots(2, 2, figsize=(BREITE_ZOLL, 5.4), sharey=True)
    for ax, backbone in zip(axes.ravel(), BACKBONE_NAME):
        for freeze in FREEZE_ORDER:
            lauf = next(l for l in laeufe if l["backbone"] == backbone and l["freeze"] == freeze)
            farbe, stil, marker = FREEZE_STIL[freeze]
            ax.plot(lauf["train_ep"], lauf["train_loss"], color=farbe, linestyle=stil,
                    linewidth=0.9, alpha=0.40)
            ax.plot(lauf["val_ep"], lauf["val_loss"], color=farbe, linestyle=stil,
                    linewidth=1.6)
            b_ep = lauf["beste_epoche"]
            ax.plot([b_ep], [lauf["val_loss"][lauf["val_ep"].index(b_ep)]], marker=marker,
                    color=farbe, markersize=5, markeredgecolor="white",
                    markeredgewidth=0.8, zorder=5)
        ax.set_title(BACKBONE_NAME[backbone])
        ax.set_xlabel("Epoche")
        ax.set_ylim(y_unten, y_oben)
        ax.set_xlim(left=1)
        style_axes(ax, grid_axis="both")
        komma_achsen(ax, x=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Verlust")

    handles = [Line2D([], [], color=FREEZE_STIL[f][0], linestyle=FREEZE_STIL[f][1],
                      marker=FREEZE_STIL[f][2], markersize=5, linewidth=1.6,
                      label=FREEZE_NAME[f]) for f in FREEZE_ORDER]
    handles.append(Line2D([], [], color=MUT, linewidth=0.9, alpha=0.5,
                          label="blass: Training"))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0),
               columnspacing=2.0, handlelength=2.4)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return speichern(fig, out_dir, "verlustkurven")


def abb_f1_matrix(laeufe, out_dir):
    backbones = list(BACKBONE_NAME)
    werte = np.array([[next(l["f1_05"] for l in laeufe
                            if l["backbone"] == b and l["freeze"] == f)
                       for f in FREEZE_ORDER] for b in backbones])

    # Hoehe 4,6 Zoll: die Zellen stehen damit bei rund 2:1 statt 3,4:1 wie bei den
    # frueheren 2,9 Zoll - sonst wirken sie als Streifen, nicht als Blockmatrix.
    # Im Dokument (16 cm breit) sind das rund 9,8 cm Hoehe.
    fig, ax = plt.subplots(figsize=(BREITE_ZOLL, 4.6))
    bild = ax.imshow(werte, cmap=GRAU_SKALA, vmin=Y_UNTEN, vmax=1.00, aspect="auto")
    ax.set_xticks(range(len(FREEZE_ORDER)), [FREEZE_NAME[f] for f in FREEZE_ORDER])
    ax.set_yticks(range(len(backbones)), [BACKBONE_NAME[b] for b in backbones])
    ax.tick_params(length=0, colors="black")
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Umschaltpunkt der Schriftfarbe bei 0,90, nicht bei 0,82: die Skala endet bei
    # INK2 statt in einem dunklen Ton, die Zellen bleiben also laenger hell. Bei
    # 0,8428 haette Weiss nur noch rund 3:1 Kontrast, INK dagegen 4,7:1.
    for i in range(werte.shape[0]):
        for j in range(werte.shape[1]):
            wert = werte[i, j]
            ax.text(j, i, de(wert), ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white" if wert > 0.90 else INK)
    # Bewusst ohne Trennlinien zwischen den Zellen: die Farbe traegt die Aussage,
    # weisse Fugen zerschneiden die Flaeche und wirken im Dokument wie ein Rahmen.
    ax.grid(False)

    # Farbskala ohne Beschriftung: welche Groesse sie zeigt und dass sie bei
    # tau = 0,5 gilt, steht in der Bildunterschrift (build_kap5_payload.py,
    # Marke A3) - danebengesetzt waere es dieselbe Angabe zweimal.
    # aspect 27 statt der voreingestellten 20: sonst bleibt der Balken kuerzer als
    # die Matrix und haengt mittig daneben.
    cb = fig.colorbar(bild, ax=ax, fraction=0.025, pad=0.02, aspect=27)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=3, colors="black")
    cb.ax.yaxis.set_major_formatter(_KOMMA)
    fig.tight_layout()
    return speichern(fig, out_dir, "f1_matrix")


def abb_guete_aufwand(laeufe, zeiten, out_dir):
    """
    F1 ueber Rechenzeit, Punktgroesse = trainierbare Parameter.

    Die beiden Legenden stehen ueber dem Diagramm, nicht darin: im Feld selbst
    verdeckten sie den einzigen Punkt der unteren Bildhaelfte (ConvNeXt-Tiny,
    nur Kopf, F1 = 0,4318). Im Feld ist kein Platz, der auch bei anderen Daten
    frei bleibt - oben ist er es immer.
    """
    fig, ax = plt.subplots(figsize=(BREITE_ZOLL, 4.3))
    params = np.array([l["test"]["n_params_trainable"] for l in laeufe], dtype=float)
    groesse = 30 + 320 * (params / params.max()) ** 0.5

    for lauf, gr in zip(laeufe, groesse):
        ax.scatter(zeiten[lauf["backbone"]]["batch_median"], lauf["f1_05"], s=gr,
                   marker=FREEZE_STIL[lauf["freeze"]][2],
                   facecolor=BACKBONE_FARBE[lauf["backbone"]], edgecolor="white",
                   linewidth=1.0, alpha=0.9, zorder=4)

    ax.set_xlabel("Rechenzeit je Bild [ms]")
    ax.set_ylabel("F1 (n.i.O.) bei $\\tau$ = 0,5")
    ax.set_ylim(Y_UNTEN, 1.02)
    ax.set_xlim(0, 105)
    style_axes(ax, grid_axis="both")
    komma_achsen(ax, x=False)

    h_backbone = [Line2D([], [], marker="o", linestyle="none", markersize=7,
                         markerfacecolor=BACKBONE_FARBE[b], markeredgecolor="white",
                         label=BACKBONE_NAME[b]) for b in BACKBONE_NAME]
    h_freeze = [Line2D([], [], marker=FREEZE_STIL[f][2], linestyle="none", markersize=7,
                       markerfacecolor=MUT, markeredgecolor="white",
                       label=FREEZE_NAME[f]) for f in FREEZE_ORDER]
    # Backbone zweispaltig (vier Eintraege waeren in einer Zeile zu breit),
    # Freeze einzeilig - die drei Strategien sind eine Steigerung und sollen
    # in dieser Reihenfolge hintereinander stehen, nicht ueber Eck.
    leg1 = fig.legend(handles=h_backbone, loc="upper left", bbox_to_anchor=(0.02, 1.0),
                      ncol=2, title="Backbone (Farbe)", fontsize=9.5, title_fontsize=9.5,
                      columnspacing=1.2, handletextpad=0.4, labelspacing=0.4,
                      borderaxespad=0.2)
    leg1._legend_box.align = "left"
    leg2 = fig.legend(handles=h_freeze, loc="upper left", bbox_to_anchor=(0.46, 1.0),
                      ncol=3, title="Freeze-Strategie (Symbol)", fontsize=9.5,
                      title_fontsize=9.5, columnspacing=1.2, handletextpad=0.4,
                      borderaxespad=0.2)
    leg2._legend_box.align = "left"

    ax.text(0.015, 0.03, "Punktgröße ~ trainierbare Parameter", transform=ax.transAxes,
            fontsize=9, color=ANNOT)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    return speichern(fig, out_dir, "guete_aufwand")


def abb_konfusionsmatrizen(laeufe_ausgewaehlt, out_dir):
    """
    Konfusionsmatrizen der zwei bestplatzierten Modelle - Darstellung wie die
    je Lauf abgelegte `metrics/confusion_matrix.png`.

    Gezaehlt wird bei der Schwelle, die der Lauf auf dem Validierungsanteil
    bestimmt hat (`metrics/threshold.json`), NICHT bei 0,5 (Ansage Hannes,
    20.08.2026). Das ist der Betriebspunkt, mit dem das Modell tatsaechlich
    entscheidet, und damit derselbe, den die Laufausgabe zeigt - die beiden
    Darstellungen sollen dieselben Zahlen tragen. Die Auswahl der zwei Modelle
    bleibt davon unberuehrt: sie folgt weiter F1 (n.i.O.) bei 0,5 (Abschnitt
    4.5.2). Auswahl und Betriebspunkt sind zwei verschiedene Fragen.

    Die Schwelle steht im Feldtitel. Ohne sie waere die Abbildung nicht mehr
    lesbar: Das Kapitel bezieht alle Werte auf 0,5, sofern nichts anderes
    dabeisteht, und hier steht ausdruecklich etwas anderes.

    Das Rendering ist `_save_confusion_matrix` aus `src/evaluation/evaluator.py`
    nachgebaut, damit Abbildung und Laufausgabe nicht zwei Gestalten derselben
    Sache sind: matplotlib-"Blues" mit der Standard-Normierung (Skalenende auf
    der groessten Zaehlung), Farbbalken daneben, umlaufender Rahmen, keine
    weissen Fugen zwischen den Zellen, Zellentext "Zahl (Anteil)".

    Ein Farbbalken fuer beide Felder, rechts aussen (Ansage Hannes, 13.08.2026).
    Das setzt eine gemeinsame Skala voraus: waeren die Felder wie im Evaluator
    je fuer sich normiert (173 hier, 168 dort), gaebe der eine Balken die Werte
    des linken Feldes falsch wieder. Das Skalenende ist deshalb die groesste
    Zaehlung ueber BEIDE Matrizen - was in einer Vergleichsabbildung ohnehin das
    Richtige ist: gleiche Zahl, gleiche Farbe.

    Deshalb setzt die Funktion den Stil aus `plot_style.py` bewusst ausser
    Kraft (`rc_context` mit den matplotlib-Voreinstellungen) - die einzige
    Abbildung des Kapitels, die das tut. Grund steht im README, Abschnitt 4.
    Ausgenommen davon ist die *Schrift*: sie kommt weiter aus dem umgebenden
    `curve_style(...)`, damit die Zuordnung in `main()` auch hier gilt und die
    Abbildung nicht als einzige in der matplotlib-Hausschrift dasteht.

    Zwei Abweichungen von der Vorlage bleiben, beide sprachlich, nicht
    gestalterisch: die Klassen heissen wie im uebrigen Dokument "i.O."/"n.i.O."
    statt "io"/"nio", und der Anteil traegt ein deutsches Dezimalkomma.
    """
    matrizen = [konfusion(lauf, lauf["schwelle"]) for lauf in laeufe_ausgewaehlt]

    # Voreinstellungen wiederherstellen: `main()` legt curve_style() (Serif,
    # schwarze Achsen, Raster) um alle Abbildungen; hier ist genau das falsch.
    # Die Schriftschluessel des Aufrufers bleiben stehen - zurueckgesetzt wird
    # die Gestaltung, nicht die Schrift.
    voreinstellung = dict(matplotlib.rcParamsDefault)
    voreinstellung.update({s: plt.rcParams[s] for s in SCHRIFT_SCHLUESSEL})
    vmax = max(int(m.max()) for m in matrizen)

    with plt.rc_context(voreinstellung):
        # constrained statt tight_layout: `tight_layout` kann den nachtraeglich
        # angehaengten gemeinsamen Farbbalken nicht einplanen, die Beschriftung
        # des rechten Feldes laeuft dann ins linke.
        fig, axes = plt.subplots(1, 2, figsize=(BREITE_ZOLL, 3.1), layout="constrained")
        for ax, lauf, m in zip(axes, laeufe_ausgewaehlt, matrizen):
            gesamt = m.sum()
            im = ax.imshow(m, cmap="Blues", vmin=0, vmax=vmax)

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["i.O.", "n.i.O."], fontsize=10)
            ax.set_yticklabels(["i.O.", "n.i.O."], fontsize=10)
            ax.set_xlabel("Vorhergesagt", fontsize=10)
            ax.set_ylabel("Tatsächlich", fontsize=10)
            ax.set_title(f"{BACKBONE_NAME[lauf['backbone']]}, "
                         f"{FREEZE_NAME[lauf['freeze']]}\n"
                         f"$\\tau$ = {de(lauf['schwelle'], 3)}", fontsize=11)

            for i in range(2):
                for j in range(2):
                    anteil = m[i, j] / gesamt * 100
                    ax.text(j, i, f"{m[i, j]}\n({de(anteil, 1)} %)",
                            ha="center", va="center", fontsize=10,
                            color="white" if m[i, j] > vmax / 2 else "black")

        # `ax=list(axes)` nimmt beiden Feldern gleich viel Platz, sie bleiben
        # also gleich breit. Der Balken an nur eines gehaengt macht das linke
        # Feld breiter als das rechte.
        fig.colorbar(im, ax=list(axes))
        return speichern(fig, out_dir, "konfusionsmatrizen")


def abb_roc(laeufe, out_dir):
    """
    ROC je Backbone, drei Kurven im Feld.

    Alle zwoelf Kurven in einer Achse waren nicht mehr zu verfolgen: Neun von
    ihnen liegen im linken oberen Eck praktisch uebereinander. Aufgeteilt nach
    Backbone bleiben drei Kurven je Feld, und der Vollbereich kann beibehalten
    werden - ein Ausschnitt haette den schwaechsten Lauf aus dem Bild gedraengt.
    Die Feinunterschiede der dicht beieinanderliegenden Kurven traegt die
    Flaeche unter der Kurve in der Legende.

    Die Felder sind bewusst NICHT quadratisch (kein `set_aspect("equal")`).
    Quadratische Felder zwingen die Abbildung in eine Hoehe von rund 15 cm im
    Dokument und lassen links und rechts der Felder breite weisse Streifen
    stehen, weil die Breite dann der Hoehe folgt statt dem Satzspiegel. Mit
    freiem Seitenverhaeltnis fuellen die Felder die 19 cm Figurbreite wie in
    allen anderen 2x2-Abbildungen des Kapitels, und die Hoehe faellt auf rund
    10,5 cm. Preis dafuer: die Zufallsgerade laeuft nicht mehr unter 45 Grad.
    Sie ist Orientierungsmarke, keine Messgroesse - abgelesen wird die AUC aus
    der Legende.
    """
    fig, axes = plt.subplots(2, 2, figsize=(BREITE_ZOLL, 4.8), sharex=True, sharey=True)
    for ax, backbone in zip(axes.ravel(), BACKBONE_NAME):
        ax.plot([0, 1], [0, 1], color=MUT, linestyle=(0, (2, 3)), linewidth=1.0, zorder=1)
        for freeze in FREEZE_ORDER:
            lauf = next(l for l in laeufe if l["backbone"] == backbone and l["freeze"] == freeze)
            fpr, tpr, _ = roc_curve(lauf["y_true"], lauf["p_nio"])
            farbe, stil, _ = FREEZE_STIL[freeze]
            ax.plot(fpr, tpr, color=farbe, linestyle=stil, linewidth=1.6, zorder=3,
                    label=f"{FREEZE_NAME[freeze]}  (AUC {de(lauf['test']['roc_auc'])})")
        ax.set_title(BACKBONE_NAME[backbone])
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.005)
        style_axes(ax, grid_axis="both")
        komma_achsen(ax)
        ax.legend(loc="lower right", fontsize=8.5, handlelength=2.2,
                  borderpad=0.2, labelspacing=0.35, bbox_to_anchor=(1.02, -0.02))
    for ax in axes[1, :]:
        ax.set_xlabel("Falsch-positiv-Rate")
    for ax in axes[:, 0]:
        ax.set_ylabel("Richtig-positiv-Rate")

    fig.tight_layout()
    return speichern(fig, out_dir, "roc")


def abb_schwellenverlauf(lauf, out_dir):
    sweep = lade_sweep(lauf)
    taus = [z["tau"] for z in sweep]

    fig, ax = plt.subplots(figsize=(BREITE_ZOLL, 3.8))
    ax.plot(taus, [z["precision"] for z in sweep], color=BLUE, linestyle="--",
            linewidth=1.8, label="Precision (n.i.O.)")
    ax.plot(taus, [z["recall"] for z in sweep], color=RED, linestyle=":",
            linewidth=1.8, label="Recall (n.i.O.)")
    ax.plot(taus, [z["f1"] for z in sweep], color=GREEN, linewidth=2.0,
            label="F1 (n.i.O.)")

    tau_val = lauf["schwelle"]
    for x, txt in ((0.5, "$\\tau$ = 0,5"),
                   (tau_val, "val-Schwelle " + de(tau_val, 3))):
        ax.axvline(x, color=MUT, linewidth=1.0, linestyle=(0, (2, 3)), zorder=1)
        ax.text(x - 0.008, 0.615, txt, rotation=90, va="bottom", ha="right",
                fontsize=9, color=ANNOT)

    ax.set_xlabel("Entscheidungsschwelle $\\tau$")
    ax.set_ylabel("Kennzahl auf dem Testanteil")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.60, 1.02)
    style_axes(ax, grid_axis="both")
    komma_achsen(ax)
    # Legende unter die Achse: der freie Bereich im Diagramm wird von den
    # senkrechten Schwellenmarken samt Beschriftung belegt.
    ax.legend(loc="upper center", ncol=3, fontsize=9.5, bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    return speichern(fig, out_dir, "schwellenverlauf")


# Farbe je Erklaerungsverfahren. Eigene Zuordnung, weil in den XAI-Abbildungen
# weder Backbone noch Freeze-Strategie vorkommen - die Kodierung des uebrigen
# Kapitels waere hier ohne Bedeutung. Innerhalb der Abbildung gilt sie
# durchgaengig: dasselbe Verfahren hat links im Kasten und rechts in der Kurve
# dieselbe Farbe.
METHODE_FARBE = {
    "integrated_gradients": BLUE,
    "saliency": GREEN,
    "occlusion": AMBER,
    "zufall": MUT,
}


def abb_xai_kennzahlen(xai, out_dir):
    """Konzentration und Deletion der drei Verfahren, 1x3.

    Links: Verteilung des Konzentrationsfaktors je Verfahren ueber den
    Testanteil, mit der Referenzlinie 1,0 - dem Wert einer gleichverteilten
    Karte. Der Kasten sagt mehr als der Mittelwert allein, weil die Verteilungen
    rechtsschief sind.

    Mitte und rechts: die Deletion-Kurve, getrennt nach Klasse. Sie kommt ohne
    Bauteilmaske aus: entfernt werden die hoechstattribuierten Bildpunkte zuerst,
    und gemessen wird, wie schnell die Wahrscheinlichkeit der vorhergesagten
    Klasse faellt. Je tiefer die Kurve, desto besser trifft die Karte. Die
    zufaellige Reihenfolge steht als Bezug daneben.

    **Warum nach Klasse getrennt.** Beide Faelle in eine Kurve zu mitteln
    verdeckt den eigentlichen Befund: Auf n.i.O.-Aufnahmen gibt es eine oertlich
    begrenzte Evidenz, die eine Karte treffen kann - dort trennen sich die
    Verfahren deutlich vom Zufall. Auf i.O.-Aufnahmen gibt es keine; die
    Abwesenheit eines Fehlers ist keine Stelle im Bild, und entsprechend liegen
    dort alle Reihenfolgen einschliesslich der zufaelligen dicht beieinander.
    Der gemittelte Wert liegt zwischen beiden Regimen und beschreibt keines.

    Bewusst keine Aufnahme und keine Heatmap: der Abschnitt muss ohne
    Bildmaterial des Bauteils auskommen.
    """
    fig, (ax_box, ax_nio, ax_io) = plt.subplots(1, 3, figsize=(BREITE_ZOLL, 3.4))

    # ---- links: Konzentrationsfaktor ------------------------------------
    daten, namen, farben = [], [], []
    for m in METHODEN:
        werte = werte_je_verfahren(xai, "konzentrationsfaktor", m)
        if not werte:
            continue
        daten.append(werte)
        namen.append(METHODEN_NAME[m].replace(" ", chr(10)))
        farben.append(METHODE_FARBE[m])

    kasten = ax_box.boxplot(daten, patch_artist=True, widths=0.55, showfliers=False,
                            medianprops=dict(color="black", linewidth=1.4),
                            whiskerprops=dict(color=INK2, linewidth=0.9),
                            capprops=dict(color=INK2, linewidth=0.9))
    for feld, farbe in zip(kasten["boxes"], farben):
        feld.set(facecolor=farbe, alpha=0.30, edgecolor=farbe, linewidth=1.2)
    # Ausreisser sind abgeschaltet und die Einzelwerte stattdessen als Punktwolke
    # darueber gelegt: bei mehreren hundert Aufnahmen ist die Punktdichte die
    # eigentliche Aussage, waehrend eine Wolke aus Ausreisserkreuzen nur den
    # Kasten verdeckt.
    rng = np.random.default_rng(0)
    for i, (werte, farbe) in enumerate(zip(daten, farben), start=1):
        x = i + (rng.random(len(werte)) - 0.5) * 0.28
        ax_box.scatter(x, werte, s=2.5, color=farbe, alpha=0.16, linewidths=0, zorder=1)

    ax_box.axhline(1.0, color=RED, linewidth=1.1, linestyle=(0, (2, 3)), zorder=2)
    # UNTER die Linie, rechtsbuendig: oberhalb liegt in beiden Stufen ein
    # Kasten, unterhalb ist die Flaeche frei (kein Verfahren faellt im Median
    # unter die Gleichverteilung).
    ax_box.text(len(namen) + 0.45, 0.93, "gleichverteilte Karte", fontsize=7.5,
                color=ANNOT, va="top", ha="right")
    ax_box.set_xticks(range(1, len(namen) + 1))
    ax_box.set_xticklabels(namen, fontsize=8.5)
    ax_box.set_ylabel("Konzentrationsfaktor")
    # Obergrenze aus den Whisker-Enden, nicht aus dem groessten Einzelwert: in
    # Stufe 1 reicht der bis 22 und drueckte alle drei Kaesten in das untere
    # Fuenftel der Achse. Die Punktwolke wird an dieser Grenze abgeschnitten -
    # sie ist Textur, die Aussage tragen Kasten und Whisker.
    obergrenze = max(float(kappe.get_ydata()[0]) for kappe in kasten["caps"])
    ax_box.set_ylim(0, obergrenze * 1.12)
    style_axes(ax_box, grid_axis="y")
    komma_achsen(ax_box, x=False)

    # ---- Mitte und rechts: Deletion je Klasse ---------------------------
    kurven = di_kurven_je_klasse(xai)
    for ax, klasse, titel in ((ax_nio, "nio", "n.i.O."), (ax_io, "io", "i.O.")):
        n = 0
        for m in METHODEN + ["zufall"]:
            if m not in kurven.get(klasse, {}):
                continue
            k = kurven[klasse][m]
            n = k["n"]
            ax.plot(k["anteile"], k["deletion"], color=METHODE_FARBE[m],
                    linewidth=1.8, linestyle=":" if m == "zufall" else "-",
                    label=METHODEN_NAME[m])
        ax.set_title(f"Klasse {titel}  ($n$ = {n})", fontsize=9.5, pad=5)
        ax.set_xlabel("Anteil ersetzter Bildpunkte")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        style_axes(ax, grid_axis="both")
        komma_achsen(ax)
    ax_nio.set_ylabel("$p$ der vorhergesagten Klasse")
    # Legende unter beide Kurvenfelder statt in eines hinein: bei der Klasse
    # n.i.O. faellt jede Kurve durch die rechte obere Ecke, bei i.O. bleibt
    # keine Ecke frei.
    griffe, beschriftungen = ax_nio.get_legend_handles_labels()
    fig.legend(griffe, beschriftungen, loc="lower center", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.62, -0.06))
    fig.tight_layout()
    return speichern(fig, out_dir, "xai_kennzahlen")


def abb_xai_maskensensitivitaet(xai, out_dir):
    """Konzentrationsfaktor ueber der Verformung der Bauteilmaske.

    Die Abbildung beantwortet den naheliegenden Einwand gegen jede maskenbasierte
    Kennzahl: dass ihr Wert daran haengt, wie gut das Bauteil maskiert wurde.
    Aufgetragen ist der Median des Faktors ueber der Erosion beziehungsweise
    Dilatation der Maske in Pixeln. Eine flache Linie heisst, dass das Ergebnis
    nicht an der Maskengrenze haengt.
    """
    sens = xai["zusammenfassung"].get("maskensensitivitaet", {})
    stufen = xai.get("sens_stufen") or [-16, -8, 0, 8, 16]

    fig, ax = plt.subplots(figsize=(BREITE_ZOLL * 0.62, 3.2))
    for m in METHODEN:
        if m not in sens:
            continue
        werte = [sens[m][str(s)]["median"] for s in stufen]
        ax.plot(stufen, werte, color=METHODE_FARBE[m], linewidth=1.9,
                marker="o", markersize=4.5, label=METHODEN_NAME[m])
    ax.axhline(1.0, color=RED, linewidth=1.1, linestyle=(0, (2, 3)), zorder=1)
    ax.axvline(0.0, color=MUT, linewidth=0.9, linestyle=(0, (2, 3)), zorder=1)
    ax.set_xlabel("Verformung der Bauteilmaske [px]   (negativ: erodiert)")
    ax.set_ylabel("Konzentrationsfaktor (Median)")
    ax.set_xticks(stufen)
    ax.set_ylim(bottom=0)
    style_axes(ax, grid_axis="both")
    komma_achsen(ax, x=False)
    ax.legend(loc="best", fontsize=8.5, frameon=False)
    fig.tight_layout()
    return speichern(fig, out_dir, "xai_maskensensitivitaet")


def main():
    ap = argparse.ArgumentParser(description="Diagramme fuer Kapitel 5 erzeugen")
    ap.add_argument("--stufe", type=int, default=1)
    ap.add_argument("--praefix", default="",
                    help="Praefix der Dateinamen, z. B. 'stufe2_'. Ohne Angabe "
                         "werden die Diagramme der Versuchsstufe 1 ueberschrieben.")
    args = ap.parse_args()

    global PRAEFIX
    PRAEFIX = args.praefix

    stufen_dir = os.path.join(PROJECT_ROOT, "outputs", "runs", f"Stufe_{args.stufe}")
    out_dir = os.path.join(PROJECT_ROOT, "thesis", "abbildungen", "kap5")
    os.makedirs(out_dir, exist_ok=True)

    laeufe = lade_laeufe(stufen_dir)
    print(f"Laeufe gefunden: {len(laeufe)}")
    if len(laeufe) != 12:
        print("  WARNUNG: erwartet werden zwoelf Laeufe.")
    zeiten = lade_inferenzzeit(stufen_dir)
    rang = rangfolge(laeufe, zeiten)
    print(f"Bestes Modell:      {rang[0]['run']}  F1@0,5 = {de(rang[0]['f1_05'])}")
    print(f"Zweitbestes Modell: {rang[1]['run']}  F1@0,5 = {de(rang[1]['f1_05'])}")

    # Schrift je Abbildung. Steht bewusst hier und nicht in den Zeichen-
    # funktionen: so ist an einer Stelle ablesbar, welche Abbildung serifenlos
    # gesetzt ist und welche in der Serifenkette des Fliesstextes.
    #   SEGOE = Segoe UI (Ansage Hannes, 13.08.2026)
    #   None  = Serifenkette aus plot_style.py
    # abb_konfusionsmatrizen setzt die uebrige Gestaltung intern zurueck (sie
    # soll wie die Laufausgabe aussehen), folgt hier aber der Schrift.
    abbildungen = [
        (abb_verlustkurven,      (laeufe,),         SEGOE),
        (abb_trainingsverlaeufe, (laeufe,),         SEGOE),
        (abb_f1_matrix,          (laeufe,),         None),
        (abb_guete_aufwand,      (laeufe, zeiten),  None),
        (abb_konfusionsmatrizen, (rang[:2],),       SEGOE),
        (abb_roc,                (laeufe,),         SEGOE),
        (abb_schwellenverlauf,   (rang[0],),        SEGOE),
    ]

    # Die beiden XAI-Abbildungen brauchen den Messlauf von `xai_kennzahlen.py`.
    # Fehlt er, entstehen die uebrigen sieben trotzdem - ein noch nicht
    # gerechneter Messlauf darf die Diagrammerzeugung nicht blockieren.
    xai_json = os.path.join(stufen_dir, "xai_kennzahlen.json")
    if os.path.isfile(xai_json):
        xai = lade_xai_kennzahlen(stufen_dir)
        n = xai["zusammenfassung"].get("n", len(xai.get("bilder", [])))
        print(f"XAI-Kennzahlen:     {xai['lauf']}, {n} Aufnahmen")
        if not xai["zeilen"]:
            print("  WARNUNG: xai_kennzahlen.csv fehlt - Kastendiagramm bleibt leer. "
                  "xai_kennzahlen.py neu laufen lassen.")
        abbildungen += [
            (abb_xai_kennzahlen,          (xai,), SEGOE),
            (abb_xai_maskensensitivitaet, (xai,), SEGOE),
        ]
    else:
        print(f"XAI-Kennzahlen:     fehlen ({os.path.relpath(xai_json, PROJECT_ROOT)}) "
              f"- die beiden XAI-Abbildungen entstehen nicht.")
    pfade = []
    for zeichnen, argumente, schrift in abbildungen:
        with curve_style(schrift=schrift):
            pfade.append(zeichnen(*argumente, out_dir))

    for p in pfade:
        print(f"  geschrieben: {os.path.relpath(p, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
