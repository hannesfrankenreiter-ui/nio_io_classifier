"""
Bilder eines Laufs: Modelleingang, Attributionskarten, Rueckprojektion
=====================================================================
Hier steht die Darstellung, die sowohl `scripts/pipeline_original.py` (auf Zuruf) als auch
`train.py` (automatisch am Ende jedes Trainings) benutzt. Bewusst an einer Stelle: zwei
Fassungen derselben Darstellung wuerden auseinanderdriften, und dann saehen die Bilder aus
einem Lauf anders aus als die nachtraeglich erzeugten.

Ausgabeform der automatischen Rueckprojektion: **ein Ordner je Originalbild** mit vier
Dateien - die Aufnahme und die drei Methoden darauf, jede in voller Exportbreite. Eine
Reihe nebeneinander waere zwar bequemer zu vergleichen, gaebe aber jedem Feld nur ein
Viertel der Anzeigebreite; ein 900-px-Bauteil landete damit bei rund hundert Pixeln.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.dataset import CLASSES, IODataset, build_transforms
from src.inference.lauf import lade_schwelle
from src.preprocessing.korb_zoom import zuschnittplan_aus_config
from src.xai.integrated_gradients import (
    integrated_gradients_signed,
    occlusion_map,
    saliency_map,
)
from src.xai.rueckprojektion import (
    STILE,
    anzeigeskala,
    breite_auf,
    karte_in_anzeige,
    normieren,
    ueberlagern,
)

METHODEN = ("integrated_gradients", "saliency", "occlusion")
KURZ = {"integrated_gradients": "IG", "saliency": "Saliency", "occlusion": "Occlusion"}
LEER_GRENZE = 1e-8          # darunter gilt eine Karte als leer (Occlusion kann das sein)


# ──────────────────────────────────────────────────────────────
#  Vorverarbeitung
# ──────────────────────────────────────────────────────────────

def als_pil(zuschnitt_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(zuschnitt_bgr, cv2.COLOR_BGR2RGB))


def modelleingang(zuschnitt_bgr: np.ndarray, groesse: tuple[int, int]) -> np.ndarray:
    """Das Bild, das das Netz sieht - ohne Normalisierung, als BGR zum Anzeigen.

    Bewusst dieselbe Kette wie der Eval-Transform: PIL, BILINEAR, (Breite, Hoehe).
    """
    hoehe, breite = groesse
    return cv2.cvtColor(
        np.array(als_pil(zuschnitt_bgr).resize((breite, hoehe), Image.BILINEAR)),
        cv2.COLOR_RGB2BGR)


# ──────────────────────────────────────────────────────────────
#  XAI
# ──────────────────────────────────────────────────────────────

def karten_berechnen(model, xai_cfg: dict, tensor: torch.Tensor, ziel: int,
                     methoden) -> dict:
    """Alle gewuenschten Attributionskarten fuer einen Modelleingang.

    Integrated Gradients wird als signierte Rohattribution geholt und erst hier auf [0,1]
    gebracht - integrated_gradients() wuerde min-max je Zuschnitt normieren und die
    Rohwerte verwerfen, die fuer die Completeness-Pruefung gebraucht werden.
    Saliency und Occlusion liefern bereits [0,1] und bleiben, wie das Projekt sie liefert.
    """
    xai_cfg = xai_cfg or {}
    aus = {}
    for m in methoden:
        if m == "integrated_gradients":
            roh = integrated_gradients_signed(
                model, tensor.clone(), ziel,
                int(xai_cfg.get("ig_steps", 50)), int(xai_cfg.get("ig_step_batch", 8)))
            signiert = roh.detach().cpu().numpy().astype(np.float32)
            aus[m] = {"karte": normieren(np.abs(signiert)), "roh": signiert}
        elif m == "saliency":
            aus[m] = {"karte": saliency_map(model, tensor.clone(), ziel), "roh": None}
        elif m == "occlusion":
            aus[m] = {"karte": occlusion_map(model, tensor.clone(), ziel,
                                             int(xai_cfg.get("occ_patch_size", 64))),
                      "roh": None}
        else:
            raise ValueError(f"Unbekannte Methode: {m}")
    for d in aus.values():
        d["leer"] = bool(float(np.max(d["karte"])) <= LEER_GRENZE)
    return aus


# ──────────────────────────────────────────────────────────────
#  Darstellung
# ──────────────────────────────────────────────────────────────

def beschriften(bild: np.ndarray, text: str) -> np.ndarray:
    """Beschriftung oben links, mit dunklem Grund fuer Lesbarkeit auf hellem Foto."""
    aus = bild.copy()
    skalierung = max(0.5, bild.shape[1] / 1400)
    dicke = max(1, round(skalierung * 1.6))
    (tb, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, skalierung, dicke)
    rand = round(6 * skalierung)
    cv2.rectangle(aus, (0, 0), (tb + 2 * rand, th + 2 * rand), (20, 20, 20), cv2.FILLED)
    cv2.putText(aus, text, (rand, th + rand), cv2.FONT_HERSHEY_SIMPLEX, skalierung,
                (245, 245, 245), dicke, cv2.LINE_AA)
    return aus


def tafel(bilder: list, abstand: int = 8) -> np.ndarray:
    """Bilder nebeneinander auf gleiche Hoehe, mit schmalem Steg dazwischen."""
    hoehe = max(b.shape[0] for b in bilder)
    skaliert = [cv2.resize(b, (round(b.shape[1] * hoehe / b.shape[0]), hoehe),
                           interpolation=cv2.INTER_AREA) if b.shape[0] != hoehe else b
                for b in bilder]
    steg = np.full((hoehe, abstand, 3), 255, np.uint8)
    teile = []
    for i, b in enumerate(skaliert):
        if i:
            teile.append(steg)
        teile.append(b)
    return cv2.hconcat(teile)


def uebersicht_zeichnen(bild: np.ndarray, karte: np.ndarray, zoom, stil: str,
                        gamma, kappung, breite: int, titel: str) -> tuple[np.ndarray, float]:
    """Originalaufnahme mit zurueckprojizierter Karte und Beschriftung.

    Kein Fensterrahmen: die Karte selbst zeigt, wo gezoomt wurde, und ein Rechteck legte
    eine zweite, kuenstliche Struktur ueber die Aufnahme.
    """
    H, W = bild.shape[:2]
    klein = breite_auf(bild, breite)
    anzeige_karte = karte_in_anzeige(karte, zoom.fenster, (H, W), breite)
    anzeige, kennwert = anzeigeskala(anzeige_karte, stil, gamma, kappung)
    return beschriften(ueberlagern(klein, anzeige, stil), titel), kennwert


def schreibe_bildordner(bild: np.ndarray, zoom, karten: dict, methoden, ziel_dir: Path,
                        stil: str, breite: int, kopf: str,
                        gamma=None, kappung=None, qualitaet: int = 90) -> int:
    """Ein Ordner je Aufnahme: die Aufnahme und die drei Methoden darauf.

    Getrennte Dateien statt einer Reihe, damit jede in voller Exportbreite vorliegt - in
    einer Vierer-Reihe bekaeme jedes Feld nur ein Viertel der Anzeigebreite.
    Die Zahlenpraefixe halten die Reihenfolge im Dateibrowser.
    """
    ziel_dir.mkdir(parents=True, exist_ok=True)
    ziel_dir.joinpath("0_original.jpg").write_bytes(
        cv2.imencode(".jpg", beschriften(breite_auf(bild, breite), f"{kopf}   Original"),
                     [cv2.IMWRITE_JPEG_QUALITY, qualitaet])[1].tobytes())
    geschrieben = 1
    for i, m in enumerate(methoden, 1):
        titel = f"{kopf}   {KURZ[m]}" + ("   (Karte leer)" if karten[m]["leer"] else "")
        aus, _ = uebersicht_zeichnen(bild, karten[m]["karte"], zoom, stil,
                                     gamma, kappung, breite, titel)
        ziel_dir.joinpath(f"{i}_{m}.jpg").write_bytes(
            cv2.imencode(".jpg", aus, [cv2.IMWRITE_JPEG_QUALITY, qualitaet])[1].tobytes())
        geschrieben += 1
    return geschrieben


# ──────────────────────────────────────────────────────────────
#  Automatischer Aufruf am Ende des Trainings
# ──────────────────────────────────────────────────────────────

def rueckprojektion_nach_training(model, config: dict, run_dir, logger=None) -> int:
    """Rueckprojektion fuer einige Bilder des XAI-Splits, nach `<run>/xai/rueckprojektion/`.

    Scheitert bewusst weich: fehlt der zuschnitt:-Block, der Originalordner oder ein
    einzelnes Bild, gibt es eine Logzeile und sonst nichts. Ein Diagramm darf ein
    fertiges Training nicht nachtraeglich zu Fall bringen.

    Rueckgabe: Anzahl der geschriebenen Ordner.
    """
    def sagen(text: str, warnung: bool = False) -> None:
        if logger is None:
            print(text)
        elif warnung:
            logger.warning(text)
        else:
            logger.info(text)

    xai_cfg = config.get("xai") or {}
    cfg = xai_cfg.get("rueckprojektion") or {}
    if not cfg.get("enabled", False):
        return 0
    anzahl = int(cfg.get("n_samples", 12))
    if anzahl <= 0:
        return 0

    stil = str(cfg.get("stil", "lauf"))
    if stil not in STILE:
        sagen(f"Rueckprojektion: unbekannter Stil {stil!r}, erlaubt sind "
              f"{sorted(STILE)} - uebersprungen", warnung=True)
        return 0
    breite = int(cfg.get("breite", 2000))
    methoden = [m for m in cfg.get("methoden", METHODEN) if m in METHODEN] or list(METHODEN)

    try:
        plan = zuschnittplan_aus_config(config)
    except ValueError as fehler:
        sagen(f"Rueckprojektion uebersprungen: {fehler}", warnung=True)
        return 0
    if plan.original is None or not plan.original.is_dir():
        sagen(f"Rueckprojektion uebersprungen: Originalordner nicht gefunden "
              f"({plan.original})", warnung=True)
        return 0

    split = str(xai_cfg.get("split", "val")).lower()
    data_dir = Path(config.get("data_dir", "data"))
    try:
        ds = IODataset(str(data_dir / split))
    except (RuntimeError, FileNotFoundError) as fehler:
        sagen(f"Rueckprojektion uebersprungen: {fehler}", warnung=True)
        return 0

    eintraege = [(Path(p).relative_to(data_dir), CLASSES[label]) for p, label in ds.samples]
    idx = np.linspace(0, len(eintraege) - 1, min(anzahl, len(eintraege))).round().astype(int)
    auswahl = [eintraege[i] for i in sorted(set(idx.tolist()))]

    transform = build_transforms(config, train=False)
    schwelle, quelle = lade_schwelle(run_dir, config)
    device = next(model.parameters()).device
    war_training = model.training
    model.eval()

    ziel_wurzel = Path(run_dir) / "xai" / "rueckprojektion"
    sagen(f"Rueckprojektion ({plan.verfahren}, Stil {stil}, Schwelle {schwelle:.4f} aus "
          f"{quelle}): {len(auswahl)} Bilder aus {split} -> {ziel_wurzel}")

    t0, geschrieben, uebersprungen = time.time(), 0, []
    for rel, klasse in auswahl:
        try:
            bild = cv2.imread(str(plan.original / rel), cv2.IMREAD_COLOR)
            if bild is None:
                uebersprungen.append(f"{rel}: Original nicht lesbar")
                continue
            zoom = plan.fenster(bild)
            if not zoom.auswertbar:
                uebersprungen.append(f"{rel}: {zoom.hinweis}")
                continue

            x0, y0, x1, y1 = zoom.fenster
            tensor = transform(als_pil(bild[y0:y1, x0:x1])).unsqueeze(0).to(device)
            with torch.no_grad():
                p_nio = float(torch.softmax(model(tensor), dim=1)[0, 1])
            vorhersage = "nio" if p_nio >= schwelle else "io"
            karten = karten_berechnen(model, xai_cfg, tensor,
                                      1 if vorhersage == "nio" else 0, methoden)

            kopf = (f"{rel.name}   wahr {klasse}   -> {vorhersage.upper()}   "
                    f"p(nio)={p_nio:.3f} (tau={schwelle:.3f})")
            schreibe_bildordner(bild, zoom, karten, methoden,
                                ziel_wurzel / rel.parent / rel.stem,
                                stil, breite, kopf)
            geschrieben += 1
        except Exception as fehler:                 # ein Bild darf den Rest nicht kippen
            uebersprungen.append(f"{rel}: {type(fehler).__name__}: {fehler}")

    if war_training:
        model.train()
    sagen(f"Rueckprojektion fertig: {geschrieben} Ordner in "
          f"{(time.time() - t0) / 60:.1f} min"
          + (f", {len(uebersprungen)} uebersprungen" if uebersprungen else ""))
    for m in uebersprungen[:5]:
        sagen(f"  {m}", warnung=True)
    return geschrieben
