"""
src/dataset.py
PyTorch Dataset + DataLoader-Builder für die IO/NIO Bildklassifikation.

Erwartete Ordnerstruktur:
    data/
      train/io/...  train/nio/...
      val/io/...    val/nio/...
      test/io/...   test/nio/...

Alternativ: CSV-Index mit Spalten  path,label  (label = "io" oder "nio").
"""
import csv
import os
import warnings
from typing import List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ---------------------------------------------------------------
# Klassen-Definitionen (Reihenfolge ist WICHTIG für Label-Mapping)
# ---------------------------------------------------------------
CLASSES: List[str] = ["io", "nio"]
CLASS_TO_IDX: dict = {c: i for i, c in enumerate(CLASSES)}
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ================================================================
#  Dataset
# ================================================================
class IODataset(Dataset):
    """
    Binary-Klassifikations-Dataset für IO / NIO Bauteile.

    Args:
        root:       Pfad zum Split-Ordner (z.B. data/train)
        transform:  torchvision Transform-Pipeline
        csv_index:  Optionaler Pfad zu einer CSV-Datei mit Spalten path,label
    """

    def __init__(
        self,
        root: str,
        transform=None,
        csv_index: Optional[str] = None,
    ):
        self.root = root
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        if csv_index:
            self._load_from_csv(csv_index)
        else:
            self._load_from_folder()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Keine Bilder gefunden in '{root}'. "
                "Stelle sicher, dass Unterordner 'io' und 'nio' existieren "
                "oder nutze --dummy für synthetische Testdaten."
            )

    # ----------------------------------------------------------
    def _load_from_folder(self) -> None:
        for cls in CLASSES:
            cls_dir = os.path.join(self.root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if os.path.splitext(fname)[1].lower() in IMG_EXTENSIONS:
                    self.samples.append(
                        (os.path.join(cls_dir, fname), CLASS_TO_IDX[cls])
                    )

    def _load_from_csv(self, csv_path: str) -> None:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path  = row["path"]
                label = CLASS_TO_IDX[row["label"].strip().lower()]
                if not os.path.exists(path):
                    warnings.warn(
                        f"[IODataset] Pfad existiert nicht: {path} – wird trotzdem eingetragen."
                    )
                self.samples.append((path, label))

    # ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            warnings.warn(
                f"[IODataset] Bild konnte nicht geladen werden: {path} ({exc}). "
                "Ersetze durch schwarzes Dummy-Bild."
            )
            image = Image.new("RGB", (224, 224), color=0)

        if self.transform:
            image = self.transform(image)
        return image, label

    # ----------------------------------------------------------
    def get_class_counts(self) -> dict:
        counts = {cls: 0 for cls in CLASSES}
        for _, label in self.samples:
            counts[CLASSES[label]] += 1
        return counts

    def get_class_weights(self) -> torch.Tensor:
        """
        Berechnet inverse-frequency Gewichte:
            weight_c = N_total / (N_classes * N_c)
        Gibt Tensor [weight_io, weight_nio] zurück.
        """
        counts = self.get_class_counts()
        total  = sum(counts.values())
        n_cls  = len(CLASSES)
        weights = torch.tensor(
            [total / (n_cls * max(counts[cls], 1)) for cls in CLASSES],
            dtype=torch.float32,
        )
        return weights

    def get_sample_weights(self) -> List[float]:
        """Gibt pro-Sample Gewicht zurück (für WeightedRandomSampler)."""
        class_weights = self.get_class_weights()
        return [float(class_weights[label]) for _, label in self.samples]


# ================================================================
#  Transform-Builder
# ================================================================
def bildgroesse(config: dict) -> Tuple[int, int]:
    """Zielgroesse aus der Config als (Hoehe, Breite) - die Reihenfolge von T.Resize.

    image_size darf sein:
        512          -> (512, 512)   quadratisch, wie bisher
        [320, 800]   -> (320, 800)   rechteckig, [Hoehe, Breite]

    Rechteckig lohnt sich, wenn die Bilder selbst nicht quadratisch sind: ein
    2999x1200-Zuschnitt auf 512x512 gequetscht verliert horizontal 5,9x, auf
    800x320 dagegen gleichmaessig 3,75x bei gleicher Pixelzahl.
    """
    wert = config.get("image_size", 224)
    if isinstance(wert, (list, tuple)):
        if len(wert) != 2:
            raise ValueError(f"image_size braucht genau zwei Werte [Hoehe, Breite], bekam {wert!r}")
        hoehe, breite = int(wert[0]), int(wert[1])
    else:
        hoehe = breite = int(wert)
    if hoehe <= 0 or breite <= 0:
        raise ValueError(f"image_size muss positiv sein, bekam {wert!r}")
    return hoehe, breite


def build_transforms(config: dict, train: bool = True) -> T.Compose:
    """Erstellt die Augmentation-/Normalisierungs-Pipeline aus der Config."""
    aug        = config.get("augmentation", {})
    norm       = aug.get("normalize", {})
    mean       = norm.get("mean", [0.485, 0.456, 0.406])
    std        = norm.get("std",  [0.229, 0.224, 0.225])
    zielgroesse = bildgroesse(config)

    if train:
        # Skalen-Augmentation: Die Bauteil-Zuschnitte aus korb_referenz.py sind
        # unterschiedlich gross (Seitenlaenge 307-2108 px, Spreizung 4,3x zwischen
        # 5. und 95. Perzentil). Nach dem Skalieren auf image_size erscheint
        # derselbe Defekt daher unterschiedlich breit. RandomResizedCrop laesst
        # das Netz diese Invarianz explizit lernen, statt sie aus der zufaelligen
        # Groessenverteilung ziehen zu muessen.
        # ratio fest auf 1:1 - das Seitenverhaeltnis darf NICHT variieren, sonst
        # ist die Verzerrung wieder drin, die der quadratische Zuschnitt beseitigt.
        # ACHTUNG: RandomResizedCrop zoomt nur hinein. Bei kleinem scale-Minimum
        # kann ein randstaendiger Defekt aus dem Bild fallen - das erzeugt
        # NIO-Bilder ohne sichtbaren Defekt, also Labelrauschen.
        rrc = aug.get("random_resized_crop", {})
        if rrc and rrc.get("enabled", False):
            ops = [T.RandomResizedCrop(
                zielgroesse,
                scale=tuple(rrc.get("scale", [0.6, 1.0])),
                ratio=tuple(rrc.get("ratio", [1.0, 1.0])),
                antialias=True,
            )]
        else:
            ops = [T.Resize(zielgroesse)]

        if aug.get("random_horizontal_flip", True):
            ops.append(T.RandomHorizontalFlip())
        if aug.get("random_vertical_flip", False):
            ops.append(T.RandomVerticalFlip())
        rot = aug.get("random_rotation", 0)
        if rot:
            ops.append(T.RandomRotation(rot))

        cj = aug.get("color_jitter")
        if cj:
            ops.append(T.ColorJitter(
                brightness=cj.get("brightness", 0),
                contrast=cj.get("contrast",   0),
                saturation=cj.get("saturation", 0),
                hue=cj.get("hue", 0),
            ))

        # GaussianBlur – Robustheit gegen Schärfevariationen
        gb = aug.get("gaussian_blur", {})
        if gb and gb.get("enabled", False):
            kernel = gb.get("kernel_size", 5)
            sigma  = gb.get("sigma", [0.1, 1.5])
            p      = gb.get("p", 0.2)
            ops.append(T.RandomApply([T.GaussianBlur(kernel_size=kernel, sigma=sigma)], p=p))

        ops += [T.ToTensor(), T.Normalize(mean, std)]

        # RandomErasing – nach ToTensor, vor Normalize wäre falsch; hier nach Normalize
        re = aug.get("random_erasing", {})
        if re and re.get("enabled", False):
            ops.append(T.RandomErasing(
                p=re.get("p", 0.3),
                scale=tuple(re.get("scale", [0.02, 0.15])),
                ratio=tuple(re.get("ratio", [0.3, 3.3])),
                value=0,
            ))
    else:
        ops = [
            T.Resize(zielgroesse),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]

    return T.Compose(ops)


# ================================================================
#  DataLoader-Builder
# ================================================================
def build_dataloaders(
    config: dict,
    generator: Optional[torch.Generator] = None,
    worker_init_fn=None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Erstellt Train-, Val- und Test-DataLoader aus der Config.
    Unterstützt optionalen WeightedRandomSampler für Klassenimbalance.

    Args:
        config          : Konfigurations-Dict
        generator       : torch.Generator für reproduzierbare Shuffle-Reihenfolge
        worker_init_fn  : Seeding-Funktion für DataLoader-Worker
    """
    data_dir     = config.get("data_dir", "data")
    batch_size   = config.get("batch_size", 16)
    num_workers  = config.get("num_workers", 0)
    use_sampler  = config.get("use_weighted_sampler", False)
    csv_index    = config.get("csv_index", None)
    pin_memory   = torch.cuda.is_available()

    train_tf = build_transforms(config, train=True)
    eval_tf  = build_transforms(config, train=False)

    train_ds = IODataset(os.path.join(data_dir, "train"), train_tf, csv_index=csv_index)
    val_ds   = IODataset(os.path.join(data_dir, "val"),   eval_tf)
    test_ds  = IODataset(os.path.join(data_dir, "test"),  eval_tf)

    # Gemeinsame DataLoader-Optionen. persistent_workers/prefetch_factor sind nur
    # gültig, wenn num_workers > 0 (sonst wirft PyTorch einen Fehler).
    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )
    if num_workers > 0:
        # Worker über Epochen hinweg am Leben halten (spart Neustart-Overhead unter Windows-spawn)
        common_kwargs["persistent_workers"] = True
        common_kwargs["prefetch_factor"] = 4

    # BatchNorm bricht im Training ab, wenn der letzte Batch nur 1 Sample enthält
    # (ValueError "Expected more than 1 value per channel"). drop_last verwirft den
    # unvollständigen letzten Batch – aber nur, wenn genug Daten für mindestens einen
    # vollen Batch da sind, sonst wäre der Loader bei winzigen (Dummy-)Sets leer.
    # Nur der Train-Loader; Val/Test sollen alle Samples auswerten (dort ist BatchNorm
    # ohnehin im eval-Modus und mit Batch-Größe 1 unkritisch).
    drop_last = len(train_ds) >= batch_size

    if use_sampler:
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(train_ds, sampler=sampler, drop_last=drop_last, **common_kwargs)
    else:
        train_loader = DataLoader(
            train_ds, shuffle=True, generator=generator, drop_last=drop_last, **common_kwargs
        )

    val_loader  = DataLoader(val_ds,  shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **common_kwargs)

    return train_loader, val_loader, test_loader
