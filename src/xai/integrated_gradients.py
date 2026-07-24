"""
src/xai/integrated_gradients.py
Three XAI methods for PyTorch (no Captum required):

  1. Integrated Gradients  - path-based attribution
  2. Saliency Map          - simple input gradients
  3. Occlusion Map         - patch-based sensitivity

Config in config.yaml:
    xai:
      enabled: true
      method: integrated_gradients   # integrated_gradients | saliency | occlusion | all
      n_samples: 10
      overlay: true
      ig_steps: 50         # only for Integrated Gradients
      occ_patch_size: 32   # only for Occlusion Map
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF

matplotlib.use("Agg")

CLASSES = ["io", "nio"]


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1]."""
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _denorm_to_pil(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """Denormalize (C,H,W) tensor and return PIL image."""
    t = tensor * std + mean
    t = t.clamp(0, 1)
    return TF.to_pil_image(t)


def integrated_gradients_signed(
    model,
    input_tensor: torch.Tensor,
    target_class: int,
    steps: int = 50,
    step_batch: int = 8,
) -> torch.Tensor:
    """
    Signed Integrated Gradients attribution (right Riemann sum, Formel 4-1):

        IG_i = (x_i - x'_i) * (1/m) * sum_{k=1..m} dF(x' + (k/m)(x - x')) / dx_i

    Erfüllt das Completeness-Axiom: sum(IG) ~ F(x) - F(x')
    (Sundararajan et al. 2017, Diskretisierungsfehler O(1/m)).

    Die m Stützpunkte werden in Mini-Batches à `step_batch` verarbeitet und die
    Gradienten aufsummiert. Ergebnis ist mathematisch identisch zum Verarbeiten
    aller Steps auf einmal (Summe/m = Mittelwert), aber der Spitzen-VRAM bleibt
    beschränkt: ein einzelner (m, C, H, W)-Backward bei 512px/ResNet-50 sprengt
    die 16 GB und wird vom Windows-Treiber ins System-RAM ausgelagert (~50x
    langsamer). Chunking hält alles auf der GPU.

    Args:
        model        : nn.Module (eval mode)
        input_tensor : (1, C, H, W) normalized image
        target_class : target class index
        steps        : number of Riemann approximation steps (m)
        step_batch   : Anzahl gleichzeitig verarbeiteter Stützpunkte (VRAM-Puffer)

    Returns:
        attribution  : (H, W) torch.Tensor, signiert (über Kanäle summiert)
    """
    model.eval()
    baseline = torch.zeros_like(input_tensor)                                      # Nulltensor als Baseline

    alphas = (
        torch.arange(1, steps + 1, device=input_tensor.device, dtype=input_tensor.dtype)
        / steps
    ).view(-1, 1, 1, 1)                                                             # a_k = k/m, k = 1..m

    grad_sum = torch.zeros_like(input_tensor)                                       # (1, C, H, W), akkumulierte Gradienten
    for start in range(0, steps, max(1, step_batch)):                              # Stützpunkte in Blöcken abarbeiten
        a = alphas[start : start + step_batch]                                      # (b, 1, 1, 1)
        interp = (baseline + a * (input_tensor - baseline)).requires_grad_(True)    # (b, C, H, W)

        model.zero_grad(set_to_none=True)
        logits = model(interp)
        score = logits[:, target_class].sum()                                       # sum: unskalierter Gradient je Stützpunkt
        score.backward()                                                            # Gradienten berechnen
        grad_sum += interp.grad.detach().sum(dim=0, keepdim=True)                    # Blockgradienten aufsummieren

    avg_grads = (grad_sum / steps).squeeze(0)                                       # (1/m) * Summe der Gradienten (Formel 4-1)
    delta = (input_tensor - baseline).squeeze(0)                                    # Differenz zwischen Eingabe und Baseline
    return (delta * avg_grads).sum(dim=0)                                           # signiert, kanalweise summiert -> (H, W)


def integrated_gradients(
    model,
    input_tensor: torch.Tensor,
    target_class: int,
    steps: int = 50,
    step_batch: int = 8,
) -> np.ndarray:
    """
    Integrated-Gradients-Heatmap für die Visualisierung.

    Betrag der signierten Attribution, normiert auf [0, 1].

    Returns:
        attribution  : (H, W) numpy array normalized to [0, 1]
    """
    attribution = integrated_gradients_signed(
        model, input_tensor, target_class, steps, step_batch
    )
    return _normalize(attribution.abs().detach().cpu().numpy())


def saliency_map(
    model,
    input_tensor: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    """
    Saliency map as absolute input gradient.

    Args:
        model        : nn.Module (eval mode)
        input_tensor : (1, C, H, W)
        target_class : target class index

    Returns:
        saliency : (H, W) numpy array normalized to [0, 1]
    """
    model.eval()
    inp = input_tensor.clone().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(inp)
    class_score = logits[0, target_class]
    class_score.backward()

    grads = inp.grad.detach().squeeze(0)
    saliency = grads.abs().max(dim=0).values
    return _normalize(saliency.cpu().numpy())


def occlusion_map(
    model,
    input_tensor: torch.Tensor,
    target_class: int,
    patch_size: int = 32,
    occlusion_value: float = 0.0,
) -> np.ndarray:
    """
    Occlusion sensitivity map.

    Args:
        model           : nn.Module (eval mode)
        input_tensor    : (1, C, H, W)
        target_class    : target class index
        patch_size      : square patch size
        occlusion_value : value written into occluded patch

    Returns:
        occ_map : (H, W) numpy array normalized to [0, 1]
    """
    model.eval()
    _, _, h, w = input_tensor.shape

    with torch.no_grad():
        base_prob = torch.softmax(model(input_tensor), dim=1)[0, target_class].item()

    row_starts = list(range(0, h, patch_size))
    col_starts = list(range(0, w, patch_size))
    sens = np.zeros((len(row_starts), len(col_starts)), dtype=np.float32)

    with torch.no_grad():
        for ri, rs in enumerate(row_starts):
            for ci, cs in enumerate(col_starts):
                perturbed = input_tensor.clone()
                perturbed[0, :, rs : rs + patch_size, cs : cs + patch_size] = occlusion_value
                prob = torch.softmax(model(perturbed), dim=1)[0, target_class].item()
                sens[ri, ci] = max(0.0, base_prob - prob)

    from PIL import Image as PILImage

    occ_pil = PILImage.fromarray((sens * 255).astype(np.uint8)).resize(
        (w, h), PILImage.BILINEAR
    )
    occ_full = np.array(occ_pil, dtype=np.float32) / 255.0
    return _normalize(occ_full)


def visualize_attribution(
    original_image,
    attribution: np.ndarray,
    save_path: str,
    title: str = "",
    overlay: bool = True,
):
    """Save heatmap as PNG."""
    if overlay:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].imshow(original_image)
        axes[0].set_title("Original", fontsize=11)
        axes[0].axis("off")
        axes[1].imshow(original_image, alpha=0.45)
        axes[1].imshow(attribution, cmap="inferno", alpha=0.65, vmin=0, vmax=1)
        axes[1].set_title(title, fontsize=10)
        axes[1].axis("off")
    else:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(attribution, cmap="inferno", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(plt.cm.ScalarMappable(cmap="inferno"), ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def visualize_all_methods(
    original_image,
    ig: np.ndarray,
    sal: np.ndarray,
    occ: np.ndarray,
    save_path: str,
    title: str = "",
):
    """Save all methods in a 2x2 overview."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(title, fontsize=12)

    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    for ax, attr, name in [
        (axes[0, 1], ig, "Integrated Gradients"),
        (axes[1, 0], sal, "Saliency Map"),
        (axes[1, 1], occ, "Occlusion Map"),
    ]:
        ax.imshow(original_image, alpha=0.4)
        ax.imshow(attr, cmap="inferno", alpha=0.7, vmin=0, vmax=1)
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def run_xai(model, loader, config: dict, run_dir: str, logger=None, out_dir: str = None):
    """
    Erzeugt XAI-Heatmaps für die Bilder in `loader`.
    n_samples begrenzt die Anzahl; "all" oder null = ALLE Bilder im Loader.
    Method ist konfigurierbar: integrated_gradients | saliency | occlusion | all
    out_dir: Zielordner; None = <run_dir>/xai/integrated_gradients (Standard).
             Dient dazu, z.B. val- und test-Heatmaps getrennt abzulegen.
    """
    xai_cfg = config.get("xai", {})
    n_samples = xai_cfg.get("n_samples", 10)
    # "all" / null -> keine Begrenzung (alle Bilder im Loader)
    if n_samples is None or (isinstance(n_samples, str) and n_samples.lower() == "all"):
        n_samples = float("inf")
    overlay = xai_cfg.get("overlay", True)
    steps = xai_cfg.get("ig_steps", 50)
    step_batch = xai_cfg.get("ig_step_batch", 8)   # VRAM-Puffer: Stützpunkte pro Backward
    patch_size = xai_cfg.get("occ_patch_size", 32)
    raw_methods = xai_cfg.get("methods", xai_cfg.get("method", "integrated_gradients"))
    if isinstance(raw_methods, str):
        methods = [raw_methods.lower()]
    elif isinstance(raw_methods, (list, tuple)):
        methods = [str(m).lower() for m in raw_methods]
    else:
        raise ValueError("xai.method/xai.methods must be a string or list of strings")

    if "all" in methods:
        methods = ["integrated_gradients", "saliency", "occlusion"]

    valid_methods = {"integrated_gradients", "saliency", "occlusion"}
    invalid_methods = [m for m in methods if m not in valid_methods]
    if invalid_methods:
        raise ValueError(
            "Invalid xai method(s): "
            f"{invalid_methods}. Valid: integrated_gradients | saliency | occlusion | all"
        )
    methods = list(dict.fromkeys(methods))

    xai_dir = out_dir if out_dir else os.path.join(run_dir, "xai", "integrated_gradients")
    os.makedirs(xai_dir, exist_ok=True)

    norm = config.get("augmentation", {}).get("normalize", {})
    mean = torch.tensor(norm.get("mean", [0.485, 0.456, 0.406])).view(3, 1, 1)
    std = torch.tensor(norm.get("std", [0.229, 0.224, 0.225])).view(3, 1, 1)

    model.eval()
    device = next(model.parameters()).device  # Modell kann auf GPU liegen -> Eingaben mitziehen
    count = 0

    for images, labels in loader:
        for i in range(images.size(0)):
            if count >= n_samples:
                return

            inp = images[i].unsqueeze(0).to(device)
            label = labels[i].item()

            with torch.no_grad():
                pred = model(inp).argmax(dim=1).item()

            title = f"True: {CLASSES[label]}  |  Pred: {CLASSES[pred]}"
            img_pil = _denorm_to_pil(images[i], mean, std)
            save_path = os.path.join(xai_dir, f"img_{count:04d}.png")

            if len(methods) > 1:
                attrs = {}
                if "integrated_gradients" in methods:
                    attrs["Integrated Gradients"] = integrated_gradients(
                        model, inp.clone(), pred, steps, step_batch
                    )
                if "saliency" in methods:
                    attrs["Saliency Map"] = saliency_map(model, inp.clone(), pred)
                if "occlusion" in methods:
                    attrs["Occlusion Map"] = occlusion_map(
                        model, inp.clone(), pred, patch_size
                    )

                fig, axes = plt.subplots(1, 1 + len(attrs), figsize=(4 * (1 + len(attrs)), 4))
                if not isinstance(axes, np.ndarray):
                    axes = np.array([axes])
                axes[0].imshow(img_pil)
                axes[0].set_title("Original")
                axes[0].axis("off")

                for ax, (name, attr) in zip(axes[1:], attrs.items()):
                    ax.imshow(img_pil, alpha=0.4)
                    ax.imshow(attr, cmap="inferno", alpha=0.7, vmin=0, vmax=1)
                    ax.set_title(name)
                    ax.axis("off")

                fig.suptitle(title, fontsize=12)
                plt.tight_layout()
                plt.savefig(save_path, dpi=120, bbox_inches="tight")
                plt.close()
            else:
                method = methods[0]
                if method == "saliency":
                    attr = saliency_map(model, inp.clone(), pred)
                    visualize_attribution(
                        img_pil, attr, save_path, f"Saliency Map\n{title}", overlay
                    )
                elif method == "occlusion":
                    attr = occlusion_map(model, inp.clone(), pred, patch_size)
                    visualize_attribution(
                        img_pil, attr, save_path, f"Occlusion Map\n{title}", overlay
                    )
                else:
                    attr = integrated_gradients(model, inp.clone(), pred, steps, step_batch)
                    visualize_attribution(
                        img_pil,
                        attr,
                        save_path,
                        f"Integrated Gradients\n{title}",
                        overlay,
                    )

            if logger:
                logger.info(f"XAI {methods} gespeichert: {save_path}")
            count += 1
