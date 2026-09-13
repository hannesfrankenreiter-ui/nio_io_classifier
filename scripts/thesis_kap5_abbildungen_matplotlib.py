#!/usr/bin/env python3
"""Kapitel-5-Diagramme im gerahmten Matplotlib-Stil erzeugen.

Die Daten stammen wie bei den bestehenden Abbildungen ausschliesslich aus
``scripts/kap5_daten.py``. Die Ausgabe liegt bewusst in einem separaten Ordner,
damit keine derzeit im Dokument verwendete Abbildung ueberschrieben wird.

Standardausgabe::

    thesis/abbildungen/kap5/matplotlib_stil/stufe_1/*.svg
    thesis/abbildungen/kap5/matplotlib_stil/stufe_2/*.svg
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
from sklearn.metrics import roc_curve

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kap5_daten import (  # noqa: E402
    BACKBONE_NAME,
    FREEZE_NAME,
    FREEZE_ORDER,
    METHODEN,
    METHODEN_NAME,
    de,
    di_kurven_je_klasse,
    lade_inferenzzeit,
    lade_laeufe,
    lade_sweep,
    lade_xai_kennzahlen,
    rangfolge,
    werte_je_verfahren,
)


# Die bestehenden Kapitelabbildungen enden bei rund 528,361 pt. Wegen
# bbox_inches="tight" ist diese Endbreite nicht mit figsize[0] identisch.
BREITE_ZOLL = 7.48
ZIELBREITE_PT = 528.360933
# Hoehe der 2x2-Verlaufsraster: wie dieselben Abbildungen im uebrigen Kapitelstil
# (rund 386 pt). Legendenband und Ankerhoehe folgen daraus, damit der Kopfraum in
# Zoll derselbe bleibt, wenn die Hoehe sich aendert.
HOEHE_RASTER_ZOLL = 5.50
LEGENDE_ANKER = 1.0 - 0.10 / HOEHE_RASTER_ZOLL
LEGENDE_RAND = 1.0 - 0.54 / HOEHE_RASTER_ZOLL
# Das ROC-Raster hat keine Kopflegende und richtet sich nach seinem eigenen
# Gegenstueck im uebrigen Kapitelstil (rund 336 pt).
HOEHE_ROC_ZOLL = 4.88
GITTER = "#E0E0E0"
RAHMEN = "#333333"
TEXT = "#333333"

# Freigegebener Stil: 75 % der Linienstärke des ersten Prototyps.
LINIENFAKTOR = 0.75
LINIE_NEBEN = 1.5 * LINIENFAKTOR
LINIE_HAUPT = 2.35 * LINIENFAKTOR

FREEZE_STIL = {
    "nurkopf": ("#8C8C8C", ":", "s"),
    "letzterblock": ("#4C72B0", "--", "^"),
    "vollstaendig": ("#C44E52", "-", "o"),
}

METHODE_FARBE = {
    "integrated_gradients": "#4C72B0",
    "saliency": "#55A868",
    "occlusion": "#DD8452",
    "zufall": "#8C8C8C",
}

DEZIMALKOMMA = FuncFormatter(lambda wert, _: f"{wert:g}".replace(".", ","))


def diagramm_stil():
    """Gerahmter, serifenloser Stil des freigegebenen Prototyps."""
    return plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.titleweight": "normal",
        "axes.labelsize": 11.5,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": RAHMEN,
        "axes.linewidth": 1.25,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 1.15,
        "ytick.major.width": 1.15,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "path",
    })


def style_axes(ax, grid_axis: str = "both"):
    """Vollrahmen und feines Gitternetz hinter den Daten."""
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, color=GITTER, linewidth=0.4)
    ax.tick_params(top=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(RAHMEN)
        spine.set_linewidth(1.25)


def komma_achsen(ax, x: bool = True, y: bool = True):
    if x:
        ax.xaxis.set_major_formatter(DEZIMALKOMMA)
    if y:
        ax.yaxis.set_major_formatter(DEZIMALKOMMA)


def _svg_breite_pt(pfad: str) -> float:
    with open(pfad, encoding="utf-8") as f:
        kopf = "".join(f.readline() for _ in range(8))
    treffer = re.search(r'<svg[^>]*width="([0-9.]+)pt"', kopf)
    if not treffer:
        raise ValueError(f"SVG-Breite nicht lesbar: {pfad}")
    return float(treffer.group(1))


def speichern(fig, out_dir: str, name: str, format_: str = "svg") -> str:
    """Figur ablegen und SVG-Endbreite auf die Kapitelbreite kalibrieren."""
    os.makedirs(out_dir, exist_ok=True)
    pfad = os.path.join(out_dir, f"{name}.{format_}")
    pad = 0.0632

    if format_ == "svg":
        for _ in range(4):
            fig.savefig(pfad, format="svg", bbox_inches="tight", pad_inches=pad)
            ist = _svg_breite_pt(pfad)
            abweichung = ZIELBREITE_PT - ist
            if abs(abweichung) <= 0.002:
                break
            pad += abweichung / 144.0
            if pad < 0:
                raise ValueError(
                    f"Inhalt von {name} ist breiter als die Zielbreite; "
                    "Schrift oder Layout muss angepasst werden."
                )
        ist = _svg_breite_pt(pfad)
        if abs(ist - ZIELBREITE_PT) > 0.01:
            raise AssertionError(f"SVG-Breite {name}: {ist:.6f} pt")
    else:
        fig.savefig(pfad, format="png", dpi=180, bbox_inches="tight", pad_inches=pad)

    plt.close(fig)
    return pfad


def freeze_legende():
    griffe = [
        Line2D(
            [], [],
            color=FREEZE_STIL[freeze][0],
            linestyle=FREEZE_STIL[freeze][1],
            marker=FREEZE_STIL[freeze][2],
            markeredgecolor="white",
            markeredgewidth=0.55,
            markersize=5.8,
            linewidth=LINIE_HAUPT,
            label=FREEZE_NAME[freeze],
        )
        for freeze in FREEZE_ORDER
    ]
    return griffe


def abb_verlustkurven(laeufe):
    fig, axes = plt.subplots(2, 2, figsize=(BREITE_ZOLL, HOEHE_RASTER_ZOLL), sharey=True)
    for ax, backbone in zip(axes.ravel(), BACKBONE_NAME):
        for freeze in FREEZE_ORDER:
            lauf = next(
                item for item in laeufe
                if item["backbone"] == backbone and item["freeze"] == freeze
            )
            farbe, linie, marker = FREEZE_STIL[freeze]
            ax.plot(
                lauf["train_ep"], lauf["train_loss"],
                color=farbe, linestyle=linie, linewidth=LINIE_NEBEN,
                alpha=0.18, zorder=1,
            )
            ax.plot(
                lauf["val_ep"], lauf["val_loss"],
                color=farbe, linestyle=linie, linewidth=LINIE_HAUPT, zorder=3,
            )
            epoche = lauf["beste_epoche"]
            verlust = lauf["val_loss"][lauf["val_ep"].index(epoche)]
            ax.plot(
                epoche, verlust,
                marker=marker, color=farbe, markersize=7.6,
                markeredgecolor="white", markeredgewidth=1.25, zorder=5,
            )

        ax.set_title(BACKBONE_NAME[backbone], pad=8)
        ax.set_xlabel("Epoche")
        ax.set_xlim(1, max(max(l["val_ep"]) for l in laeufe if l["backbone"] == backbone))
        ax.set_ylim(0.10, 0.90)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=4))
        style_axes(ax)
        komma_achsen(ax, x=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Verlust")

    griffe = freeze_legende()
    griffe.append(Line2D([], [], color="#777777", linewidth=LINIE_NEBEN,
                         alpha=0.28, label="blass: Training"))
    fig.legend(griffe, [g.get_label() for g in griffe], loc="outside upper center",
               ncol=4, bbox_to_anchor=(0.5, LEGENDE_ANKER), columnspacing=1.7,
               handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, LEGENDE_RAND), h_pad=1.7, w_pad=2.1)
    return fig


def abb_trainingsverlaeufe(laeufe):
    fig, axes = plt.subplots(2, 2, figsize=(BREITE_ZOLL, HOEHE_RASTER_ZOLL), sharey=True)
    for ax, backbone in zip(axes.ravel(), BACKBONE_NAME):
        for freeze in FREEZE_ORDER:
            lauf = next(
                item for item in laeufe
                if item["backbone"] == backbone and item["freeze"] == freeze
            )
            farbe, linie, marker = FREEZE_STIL[freeze]
            ax.plot(
                lauf["val_ep"], lauf["val_f1"],
                color=farbe, linestyle=linie, linewidth=LINIE_HAUPT, zorder=3,
            )
            epoche = lauf["beste_epoche"]
            f1 = lauf["val_f1"][lauf["val_ep"].index(epoche)]
            if f1 >= 0.40:
                ax.plot(
                    epoche, f1,
                    marker=marker, color=farbe, markersize=7.6,
                    markeredgecolor="white", markeredgewidth=1.25, zorder=5,
                )

        ax.set_title(BACKBONE_NAME[backbone], pad=8)
        ax.set_xlabel("Epoche")
        ax.set_xlim(1, max(max(l["val_ep"]) for l in laeufe if l["backbone"] == backbone))
        # Sichtbarer Kopfraum, damit Kurven und Marker bei F1 = 1,0 nicht am Rahmen liegen.
        ax.set_ylim(0.40, 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=4))
        style_axes(ax)
        komma_achsen(ax, x=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("F1 (n.i.O.), Validierung")

    griffe = freeze_legende()
    fig.legend(griffe, [g.get_label() for g in griffe], loc="outside upper center",
               ncol=3, bbox_to_anchor=(0.5, LEGENDE_ANKER), columnspacing=2.2,
               handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, LEGENDE_RAND), h_pad=1.7, w_pad=2.1)
    return fig


def abb_roc(laeufe):
    fig, axes = plt.subplots(2, 2, figsize=(BREITE_ZOLL, HOEHE_ROC_ZOLL), sharex=True, sharey=True)
    for ax, backbone in zip(axes.ravel(), BACKBONE_NAME):
        ax.plot([0, 1], [0, 1], color="#8C8C8C", linestyle=":",
                linewidth=LINIE_NEBEN, zorder=1, label="Zufall")
        for freeze in FREEZE_ORDER:
            lauf = next(
                item for item in laeufe
                if item["backbone"] == backbone and item["freeze"] == freeze
            )
            fpr, tpr, _ = roc_curve(lauf["y_true"], lauf["p_nio"])
            farbe, linie, _ = FREEZE_STIL[freeze]
            ax.plot(
                fpr, tpr,
                color=farbe, linestyle=linie, linewidth=LINIE_HAUPT, zorder=3,
                label=f"{FREEZE_NAME[freeze]}  (AUC {de(lauf['test']['roc_auc'])})",
            )
        ax.set_title(BACKBONE_NAME[backbone], pad=8)
        ax.set_xlim(0, 1)
        # Kopfraum für ROC-Kurven, die über längere Abschnitte bei TPR = 1,0 liegen.
        ax.set_ylim(0, 1.05)
        style_axes(ax)
        komma_achsen(ax)
        ax.legend(loc="lower right", fontsize=7.6, handlelength=2.1,
                  borderpad=0.2, labelspacing=0.28)
    for ax in axes[1, :]:
        ax.set_xlabel("Falsch-positiv-Rate")
    for ax in axes[:, 0]:
        ax.set_ylabel("Richtig-positiv-Rate")
    fig.tight_layout(h_pad=1.8, w_pad=2.0)
    return fig


def abb_schwellenverlauf(lauf):
    sweep = lade_sweep(lauf)
    taus = [zeile["tau"] for zeile in sweep]

    fig, ax = plt.subplots(figsize=(BREITE_ZOLL, 4.5))
    ax.plot(taus, [z["precision"] for z in sweep], color="#4C72B0",
            linestyle="--", linewidth=LINIE_HAUPT, label="Precision (n.i.O.)")
    ax.plot(taus, [z["recall"] for z in sweep], color="#C44E52",
            linestyle=":", linewidth=LINIE_HAUPT, label="Recall (n.i.O.)")
    ax.plot(taus, [z["f1"] for z in sweep], color="#55A868",
            linestyle="-", linewidth=LINIE_HAUPT, label="F1 (n.i.O.)")

    tau_val = lauf["schwelle"]
    for x, text in ((0.5, "$\\tau$ = 0,5"),
                    (tau_val, "val-Schwelle " + de(tau_val, 3))):
        ax.axvline(x, color="#8C8C8C", linewidth=0.9, linestyle=":", zorder=2)
        ax.text(x - 0.008, 0.615, text, rotation=90, va="bottom", ha="right",
                fontsize=9, color="#666666")

    ax.set_xlabel("Entscheidungsschwelle $\\tau$")
    ax.set_ylabel("Kennzahl auf dem Testanteil")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.60, 1.02)
    style_axes(ax)
    komma_achsen(ax)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.19),
              columnspacing=2.0, handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return fig


def abb_xai_kennzahlen(xai):
    fig, (ax_box, ax_nio, ax_io) = plt.subplots(
        1, 3, figsize=(BREITE_ZOLL, 3.65), gridspec_kw={"width_ratios": [1.0, 1.12, 1.12]}
    )

    daten, namen, farben = [], [], []
    for methode in METHODEN:
        werte = werte_je_verfahren(xai, "konzentrationsfaktor", methode)
        if not werte:
            continue
        daten.append(werte)
        namen.append(METHODEN_NAME[methode].replace(" ", "\n"))
        farben.append(METHODE_FARBE[methode])

    kasten = ax_box.boxplot(
        daten, patch_artist=True, widths=0.55, showfliers=False,
        medianprops=dict(color=RAHMEN, linewidth=1.05),
        whiskerprops=dict(color="#666666", linewidth=0.8),
        capprops=dict(color="#666666", linewidth=0.8),
    )
    for feld, farbe in zip(kasten["boxes"], farben):
        feld.set(facecolor=farbe, alpha=0.30, edgecolor=farbe, linewidth=0.9)

    rng = np.random.default_rng(0)
    for i, (werte, farbe) in enumerate(zip(daten, farben), start=1):
        x = i + (rng.random(len(werte)) - 0.5) * 0.28
        ax_box.scatter(x, werte, s=2.2, color=farbe, alpha=0.16,
                       linewidths=0, zorder=1)

    ax_box.axhline(1.0, color="#C44E52", linewidth=0.9, linestyle=(0, (2, 3)), zorder=2)
    ax_box.text(len(namen) + 0.45, 0.93, "gleichverteilte Karte", fontsize=7.2,
                color="#666666", va="top", ha="right")
    ax_box.set_xticks(range(1, len(namen) + 1))
    ax_box.set_xticklabels(namen, fontsize=8.1)
    ax_box.set_ylabel("Konzentrationsfaktor")
    obergrenze = max(float(kappe.get_ydata()[0]) for kappe in kasten["caps"])
    ax_box.set_ylim(0, obergrenze * 1.12)
    style_axes(ax_box)
    komma_achsen(ax_box, x=False)

    kurven = di_kurven_je_klasse(xai)
    for ax, klasse, titel in ((ax_nio, "nio", "n.i.O."), (ax_io, "io", "i.O.")):
        n = 0
        for methode in METHODEN + ["zufall"]:
            if methode not in kurven.get(klasse, {}):
                continue
            kurve = kurven[klasse][methode]
            n = kurve["n"]
            ax.plot(
                kurve["anteile"], kurve["deletion"],
                color=METHODE_FARBE[methode],
                linewidth=LINIE_HAUPT if methode != "zufall" else LINIE_NEBEN,
                linestyle=":" if methode == "zufall" else "-",
                label=METHODEN_NAME[methode],
            )
        ax.set_title(f"Klasse {titel}  ($n$ = {n})", fontsize=9.5,
                     fontweight="normal", pad=5)
        ax.set_xlabel("Anteil ersetzter Bildpunkte", fontsize=9.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        style_axes(ax)
        komma_achsen(ax)
    ax_nio.set_ylabel("$p$ der vorhergesagten Klasse", fontsize=9.2)

    griffe, beschriftungen = ax_nio.get_legend_handles_labels()
    fig.legend(griffe, beschriftungen, loc="outside lower center", ncol=4,
               fontsize=8.0, bbox_to_anchor=(0.62, -0.015),
               columnspacing=1.4, handlelength=2.2)
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=1.2)
    return fig


ABBILDUNGEN = (
    ("verlustkurven", abb_verlustkurven),
    ("trainingsverlaeufe", abb_trainingsverlaeufe),
    ("roc", abb_roc),
)


def erzeuge_stufe(stufe: int, basis_out: str, format_: str = "svg"):
    stufen_dir = os.path.join(PROJECT_ROOT, "outputs", "runs", f"Stufe_{stufe}")
    out_dir = os.path.join(basis_out, f"stufe_{stufe}")
    laeufe = lade_laeufe(stufen_dir)
    if len(laeufe) != 12:
        raise RuntimeError(f"Stufe {stufe}: erwartet 12 Laeufe, gefunden {len(laeufe)}")

    zeiten = lade_inferenzzeit(stufen_dir)
    bester_lauf = rangfolge(laeufe, zeiten)[0]
    xai = lade_xai_kennzahlen(stufen_dir)
    if not xai.get("zeilen"):
        raise RuntimeError(f"Stufe {stufe}: xai_kennzahlen.csv fehlt oder ist leer")

    pfade = []
    with diagramm_stil():
        for name, funktion in ABBILDUNGEN:
            pfade.append(speichern(funktion(laeufe), out_dir, name, format_))
        pfade.append(speichern(abb_schwellenverlauf(bester_lauf), out_dir,
                               "schwellenverlauf", format_))
        pfade.append(speichern(abb_xai_kennzahlen(xai), out_dir,
                               "xai_kennzahlen", format_))
    return pfade


def main():
    parser = argparse.ArgumentParser(
        description="Kapitel-5-Diagramme im gerahmten Matplotlib-Stil erzeugen"
    )
    parser.add_argument("--stufe", type=int, choices=(1, 2), action="append",
                        help="Nur diese Stufe erzeugen; ohne Angabe beide Stufen")
    parser.add_argument("--format", choices=("svg", "png"), default="svg")
    parser.add_argument(
        "--ausgabe",
        default=os.path.join(PROJECT_ROOT, "thesis", "abbildungen", "kap5",
                             "matplotlib_stil"),
    )
    args = parser.parse_args()

    stufen = args.stufe or [1, 2]
    alle_pfade = []
    for stufe in stufen:
        alle_pfade.extend(erzeuge_stufe(stufe, args.ausgabe, args.format))

    for pfad in alle_pfade:
        print(os.path.relpath(pfad, PROJECT_ROOT))


if __name__ == "__main__":
    main()
