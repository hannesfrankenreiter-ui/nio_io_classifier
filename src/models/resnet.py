"""
src/models/resnet.py
Baut ein ResNet-Modell mit austauschbarem Backbone und konfigurierbarem Freeze.

Unterstuetzte Backbones: resnet18, resnet50
Output: 2 Logits (fuer CrossEntropyLoss / Softmax)
"""
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 2

# Mapping: Backbone-Name -> torchvision Konstruktor + Weights
_BACKBONE_REGISTRY = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
}


def build_model(config: dict) -> nn.Module:
    """
    Erstellt ein ResNet-Modell gemaess Config.

    config-Schluessel:
        backbone        (str): "resnet18" | "resnet50"
        pretrained      (bool): ImageNet-Weights laden
        freeze_backbone (bool|str): False | True | "head" | "layer4"
    """
    backbone_name = config.get("backbone", "resnet18")
    pretrained = config.get("pretrained", True)
    freeze_mode = config.get("freeze_backbone", False)

    if backbone_name not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unbekanntes Backbone: '{backbone_name}'. "
            f"Waehle eines aus: {list(_BACKBONE_REGISTRY.keys())}"
        )

    constructor, weights = _BACKBONE_REGISTRY[backbone_name]
    model = constructor(weights=weights if pretrained else None)

    # Backbone einfrieren (Transfer Learning)
    if freeze_mode is True or freeze_mode == "head":
        for param in model.parameters():
            param.requires_grad = False
    elif freeze_mode == "layer4":
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True

    # Tieferer Klassifikationskopf (immer trainierbar)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES),
    )
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
