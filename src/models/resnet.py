"""
src/models/resnet.py
Baut ein CNN-Modell mit austauschbarem Backbone und konfigurierbarem Freeze.

Unterstuetzte Backbones: resnet18, resnet50, efficientnet_b0, convnext_tiny
Output: 2 Logits (fuer CrossEntropyLoss / Softmax)
"""
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 2

# Mapping: Backbone-Name -> (torchvision Konstruktor, Weights)
# DEFAULT wählt automatisch die besten verfügbaren Weights (z.B. IMAGENET1K_V2 für ResNet50).
_BACKBONE_REGISTRY = {
    "resnet18":        (models.resnet18,       models.ResNet18_Weights.DEFAULT),
    "resnet50":        (models.resnet50,        models.ResNet50_Weights.DEFAULT),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
    "convnext_tiny":   (models.convnext_tiny,   models.ConvNeXt_Tiny_Weights.DEFAULT),
}

# Parameter-Prefix des letzten Backbone-Blocks (für freeze_backbone="layer4")
_LAST_STAGE_PREFIX = {
    "resnet18":        ["layer4"],
    "resnet50":        ["layer4"],
    "efficientnet_b0": ["features.7", "features.8"],
    "convnext_tiny":   ["features.6", "features.7"],
}

# Parameter-Prefix des Klassifikationskopfes (für differenzierte Lernraten)
_HEAD_PREFIX = {
    "resnet18":        ["fc."],
    "resnet50":        ["fc."],
    "efficientnet_b0": ["classifier."],
    "convnext_tiny":   ["classifier."],
}


def head_param_prefixes(backbone_name: str) -> list:
    """Parameter-Prefixe des Klassifikationskopfes.

    Wird von build_optimizer genutzt, um Kopf und Backbone in getrennte
    Parametergruppen mit eigenen Lernraten aufzuteilen (der vortrainierte
    Backbone verträgt meist deutlich weniger als der frisch initialisierte Kopf).
    """
    return _HEAD_PREFIX.get(backbone_name, ["fc."])


def _build_head(in_features: int) -> nn.Sequential:
    """
    Klassifikationskopf mit BatchNorm für stabileres Training.

    Dropout(0.3) → Linear(in→512) → BatchNorm1d(512) → ReLU
                 → Dropout(0.2) → Linear(512→NUM_CLASSES)
    """
    hidden = 512
    return nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden, NUM_CLASSES),
    )


def build_model(config: dict) -> nn.Module:
    """
    Erstellt ein CNN-Modell gemaess Config.

    config-Schluessel:
        backbone        (str): "resnet18" | "resnet50" | "efficientnet_b0" | "convnext_tiny"
        pretrained      (bool): ImageNet-Weights laden
        freeze_backbone (bool|str): False | True | "layer4"
    """
    backbone_name = config.get("backbone", "resnet18")
    pretrained    = config.get("pretrained", True)
    freeze_mode   = config.get("freeze_backbone", False)

    if backbone_name not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unbekanntes Backbone: '{backbone_name}'. "
            f"Waehle eines aus: {list(_BACKBONE_REGISTRY.keys())}"
        )

    constructor, weights = _BACKBONE_REGISTRY[backbone_name]
    model = constructor(weights=weights if pretrained else None)

    # ---- Backbone einfrieren (Transfer Learning) -----------------
    if freeze_mode is True or freeze_mode == "head":
        for param in model.parameters():
            param.requires_grad = False
    elif freeze_mode == "layer4":
        for param in model.parameters():
            param.requires_grad = False
        last_stage_prefixes = _LAST_STAGE_PREFIX.get(backbone_name, ["layer4"])
        for name, param in model.named_parameters():
            if any(name.startswith(p) or f".{p}." in name or name == p
                   for p in last_stage_prefixes):
                param.requires_grad = True

    # ---- Klassifikationskopf anhängen (immer trainierbar) --------
    if backbone_name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc    = _build_head(in_features)
        head_params = model.fc.parameters()

    elif backbone_name.startswith("efficientnet"):
        # EfficientNet: forward() führt torch.flatten vor classifier aus
        in_features      = model.classifier[-1].in_features
        model.classifier = _build_head(in_features)
        head_params      = model.classifier.parameters()

    elif backbone_name.startswith("convnext"):
        # ConvNeXt: classifier enthält LayerNorm2d + Flatten + Linear
        # Nur die letzte Linear-Schicht ersetzen (behält Normierung + Flatten)
        in_features          = model.classifier[-1].in_features
        model.classifier[-1] = _build_head(in_features)
        head_params          = model.classifier[-1].parameters()

    for param in head_params:
        param.requires_grad = True

    return model


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
