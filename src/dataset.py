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
                self.samples.append((path, label))

    # ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
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
def build_transforms(config: dict, train: bool = True) -> T.Compose:
    """Erstellt die Augmentation-/Normalisierungs-Pipeline aus der Config."""
    aug        = config.get("augmentation", {})
    norm       = aug.get("normalize", {})
    mean       = norm.get("mean", [0.485, 0.456, 0.406])
    std        = norm.get("std",  [0.229, 0.224, 0.225])
    image_size = config.get("image_size", 224)

    if train:
        ops = [T.Resize((image_size, image_size))]

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

        ops += [T.ToTensor(), T.Normalize(mean, std)]
    else:
        ops = [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]

    return T.Compose(ops)


# ================================================================
#  DataLoader-Builder
# ================================================================
def build_dataloaders(
    config: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Erstellt Train-, Val- und Test-DataLoader aus der Config.
    Unterstützt optionalen WeightedRandomSampler für Klassenimbalance.
    """
    data_dir     = config.get("data_dir", "data")
    batch_size   = config.get("batch_size", 16)
    num_workers  = config.get("num_workers", 0)
    use_sampler  = config.get("use_weighted_sampler", False)
    csv_index    = config.get("csv_index", None)

    train_tf = build_transforms(config, train=True)
    eval_tf  = build_transforms(config, train=False)

    train_ds = IODataset(os.path.join(data_dir, "train"), train_tf, csv_index=csv_index)
    val_ds   = IODataset(os.path.join(data_dir, "val"),   eval_tf)
    test_ds  = IODataset(os.path.join(data_dir, "test"),  eval_tf)

    if use_sampler:
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader