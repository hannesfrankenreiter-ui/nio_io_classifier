"""
Einen trainierten Lauf laden
============================
Der Block "config lesen -> best.pt laden -> build_model -> load_state_dict -> aufs Geraet"
stand bisher viermal im Projekt (src/inference/predict.py, scripts/run_xai_split.py,
scripts/xai_kennzahlen.py, train.py). Hier steht er einmal, damit ein neues Werkzeug nicht
die fuenfte Kopie wird.

Warum das ohne Fallunterscheidung fuer jeden Lauf funktioniert:

  Backbone       ist der einzige Architekturschalter. build_model kennt resnet18,
                 resnet50, efficientnet_b0 und convnext_tiny.
  freeze_backbone aendert nur requires_grad, nie die Architektur - fuer Inferenz also
                 bedeutungslos, egal ob false, true oder "layer4".
  image_size     wird ausschliesslich ueber src.dataset.bildgroesse gelesen, das Zahl
                 (quadratisch) und [Hoehe, Breite] beherrscht. Vorsicht: [320, 800] ist
                 [Hoehe, Breite], obwohl der Datensatz "800x320" heisst.
  Transform      kommt aus build_transforms(config, train=False) - nicht nachgebaut.
                 Eine zweite Fassung wuerde irgendwann von der ersten abweichen.
  Schwelle       aus metrics/threshold.json. config["threshold"] steht in JEDER
                 Run-Config auf 0,5 und ist damit wertlos; die echten Betriebspunkte
                 reichen von 0,209 (S1_01) ueber 0,443 (korb_zuschnitt_512) bis 0,721 (S2).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml

from src.dataset import bildgroesse, build_transforms
from src.models.resnet import build_model


@dataclass
class Lauf:
    """Alles, was man braucht, um ein Bild wie dieser Lauf zu verarbeiten."""
    verzeichnis: Path
    checkpoint: Path
    config: dict
    config_yaml: dict = field(repr=False)
    model: nn.Module = field(repr=False)
    transform: object = field(repr=False)
    groesse: tuple[int, int]          # (Hoehe, Breite) des Modelleingangs
    schwelle: float
    schwelle_quelle: str
    device: torch.device
    epoche: Optional[int] = None

    @property
    def name(self) -> str:
        return self.verzeichnis.name

    @property
    def zuschnitt(self) -> dict:
        """Der zuschnitt:-Block.

        config.yaml gewinnt gegen den Checkpoint: der Checkpoint traegt, was beim
        Training galt, waehrend der Block auch nachtraeglich ergaenzt werden kann -
        er beschreibt die Herkunft der Daten, nicht das Training.
        """
        return self.config_yaml.get("zuschnitt") or self.config.get("zuschnitt") or {}

    @property
    def xai(self) -> dict:
        return self.config.get("xai", {}) or {}

    def als_dict(self) -> dict:
        """Kennwerte zum Mitschreiben in parameter.json."""
        return {
            "lauf": str(self.verzeichnis),
            "checkpoint": self.checkpoint.name,
            "epoche": self.epoche,
            "backbone": self.config.get("backbone"),
            "freeze_backbone": self.config.get("freeze_backbone"),
            "image_size": self.config.get("image_size"),
            "eingang_hxb": list(self.groesse),
            "data_dir": self.config.get("data_dir"),
            "schwelle": round(float(self.schwelle), 6),
            "schwelle_quelle": self.schwelle_quelle,
            "device": str(self.device),
        }


def lade_schwelle(run_dir: Path, config: dict) -> tuple[float, str]:
    """Betriebspunkt des Laufs: metrics/threshold.json, sonst die Config.

    threshold.json entstand erst im Laufe des Projekts; aeltere Laeufe haben sie nicht.
    Dann bleibt nur config["threshold"] (praktisch immer 0,5) - das wird gemeldet, damit
    niemand den Rueckfallwert fuer einen gemessenen Betriebspunkt haelt.
    """
    datei = Path(run_dir) / "metrics" / "threshold.json"
    if datei.exists():
        daten = json.loads(datei.read_text(encoding="utf-8"))
        quelle = (f"metrics/threshold.json ({daten.get('criterion', '?')} auf "
                  f"{daten.get('determined_on', '?')})")
        return float(daten["threshold"]), quelle
    return float(config.get("threshold", 0.5)), "config['threshold'] (keine threshold.json)"


def lade_lauf(run_dir, checkpoint: str = "best.pt", device=None,
              schwelle: Optional[float] = None) -> Lauf:
    """Lauf-Verzeichnis -> Modell, Transform, Betriebspunkt.

    `schwelle` ueberschreibt den Wert aus threshold.json (fuer --threshold auf der CLI).
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Lauf-Verzeichnis nicht gefunden: {run_dir}")

    ckpt_pfad = Path(checkpoint)
    if not ckpt_pfad.is_absolute() and not ckpt_pfad.exists():
        ckpt_pfad = run_dir / "checkpoints" / checkpoint
    if not ckpt_pfad.exists():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {ckpt_pfad}")

    yaml_pfad = run_dir / "config.yaml"
    config_yaml = yaml.safe_load(yaml_pfad.read_text(encoding="utf-8")) if yaml_pfad.exists() else {}

    ckpt = torch.load(ckpt_pfad, map_location="cpu")
    # Der Checkpoint traegt die Config, mit der tatsaechlich trainiert wurde. config.yaml
    # ist eine Dateikopie (run_manager.save_config) und kann davon abweichen, etwa wenn
    # train.py die image_size nach dem Speichern noch veraendert hat (--dummy).
    config = dict(ckpt.get("config") or config_yaml)
    if not config:
        raise RuntimeError(f"Weder config.yaml noch ckpt['config'] in {run_dir}")

    for schluessel in ("backbone", "image_size", "data_dir"):
        a, b = config.get(schluessel), config_yaml.get(schluessel)
        if config_yaml and b is not None and a != b:
            print(f"  Hinweis: {schluessel} weicht ab - Checkpoint {a!r}, config.yaml {b!r}. "
                  f"Es gilt der Checkpoint.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # pretrained=False: die ImageNet-Gewichte wuerden gleich darauf ueberschrieben.
    # Das spart einen Download und aendert nichts am Ergebnis (strict=True prueft es).
    model = build_model(dict(config, pretrained=False))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    wert, quelle = lade_schwelle(run_dir, config)
    if schwelle is not None:
        wert, quelle = float(schwelle), "CLI --threshold"

    return Lauf(
        verzeichnis=run_dir,
        checkpoint=ckpt_pfad,
        config=config,
        config_yaml=config_yaml or {},
        model=model,
        transform=build_transforms(config, train=False),
        groesse=bildgroesse(config),
        schwelle=wert,
        schwelle_quelle=quelle,
        device=device,
        epoche=ckpt.get("epoch"),
    )
