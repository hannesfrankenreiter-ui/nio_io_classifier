"""
Create a synthetic IO/NIO dataset with configurable difficulty.
"""
import argparse
import os
from typing import Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _get_difficulty_params(level: str) -> Dict[str, float]:
    level = level.lower()
    if level not in {"easy", "medium", "hard"}:
        raise ValueError("difficulty must be one of: easy, medium, hard")

    table = {
        "easy": {
            "bg_noise": 6.0,
            "shape_jitter": 0.08,
            "shape_min_frac": 0.40,
            "shape_max_frac": 0.62,
            "blur_max": 0.3,
            "occlusion_prob": 0.05,
            "occlusion_max_frac": 0.10,
            "flip_shape_prob": 0.0,
            "rotation_deg": 8.0,
            "brightness_jitter": 0.07,
            "contrast_jitter": 0.08,
        },
        "medium": {
            "bg_noise": 12.0,
            "shape_jitter": 0.14,
            "shape_min_frac": 0.34,
            "shape_max_frac": 0.64,
            "blur_max": 0.7,
            "occlusion_prob": 0.20,
            "occlusion_max_frac": 0.18,
            "flip_shape_prob": 0.0,
            "rotation_deg": 18.0,
            "brightness_jitter": 0.14,
            "contrast_jitter": 0.16,
        },
        "hard": {
            "bg_noise": 20.0,
            "shape_jitter": 0.20,
            "shape_min_frac": 0.30,
            "shape_max_frac": 0.66,
            "blur_max": 1.4,
            "occlusion_prob": 0.38,
            "occlusion_max_frac": 0.26,
            "flip_shape_prob": 0.0,
            "rotation_deg": 30.0,
            "brightness_jitter": 0.22,
            "contrast_jitter": 0.24,
        },
    }
    return table[level]


def _build_background(image_size: int, rng: np.random.Generator, bg_noise: float) -> np.ndarray:
    x = np.linspace(0.0, 1.0, image_size, dtype=np.float32)
    y = np.linspace(0.0, 1.0, image_size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    base = rng.uniform(85.0, 145.0)
    grad_x = rng.uniform(-30.0, 30.0)
    grad_y = rng.uniform(-30.0, 30.0)
    pattern = 8.0 * np.sin(2.0 * np.pi * (xx * rng.uniform(1.0, 3.0) + yy * rng.uniform(1.0, 3.0)))

    channel_bias = rng.uniform(-10.0, 10.0, size=(1, 1, 3))
    bg = base + grad_x * xx + grad_y * yy + pattern
    bg = np.stack([bg, bg, bg], axis=2) + channel_bias
    bg += rng.normal(0.0, bg_noise, size=bg.shape)
    return np.clip(bg, 0, 255).astype(np.uint8)


def create_dummy_dataset(
    data_dir: str = "data",
    image_size: int = 320,
    n_per_split: int = 20,
    seed: int = 42,
    difficulty: str = "hard",
    label_noise: float = 0.0,
) -> None:
    """
    Create dummy images under data/{train,val,test}/{io,nio}/.
    """
    rng = np.random.default_rng(seed)
    p = _get_difficulty_params(difficulty)
    label_noise = float(np.clip(label_noise, 0.0, 0.4))

    split_counts = {
        "train": n_per_split,
        "val": max(n_per_split // 2, 4),
        "test": max(n_per_split // 2, 4),
    }

    for split, n in split_counts.items():
        for cls in ["io", "nio"]:
            for i in range(n):
                bg = _build_background(image_size, rng, p["bg_noise"])
                img = Image.fromarray(bg, mode="RGB")

                shape_layer = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
                draw = ImageDraw.Draw(shape_layer)

                fill_val = int(rng.integers(185, 245))
                fill = (fill_val, fill_val, fill_val, int(rng.integers(155, 235)))

                shape_is_circle = cls == "io"
                if rng.random() < p["flip_shape_prob"]:
                    shape_is_circle = not shape_is_circle

                frac = rng.uniform(p["shape_min_frac"], p["shape_max_frac"])
                w = int(image_size * frac * rng.uniform(0.85, 1.15))
                h = int(image_size * frac * rng.uniform(0.85, 1.15))
                w = int(np.clip(w, image_size // 8, image_size - 2))
                h = int(np.clip(h, image_size // 8, image_size - 2))

                cx = image_size // 2 + int(rng.normal(0.0, image_size * p["shape_jitter"]))
                cy = image_size // 2 + int(rng.normal(0.0, image_size * p["shape_jitter"]))
                x0 = int(np.clip(cx - w // 2, 0, image_size - 1))
                y0 = int(np.clip(cy - h // 2, 0, image_size - 1))
                x1 = int(np.clip(cx + w // 2, 0, image_size - 1))
                y1 = int(np.clip(cy + h // 2, 0, image_size - 1))

                if shape_is_circle:
                    draw.ellipse([x0, y0, x1, y1], fill=fill)
                else:
                    draw.rectangle([x0, y0, x1, y1], fill=fill)

                angle = float(rng.uniform(-p["rotation_deg"], p["rotation_deg"]))
                shape_layer = shape_layer.rotate(angle, resample=Image.BICUBIC)
                img.paste(shape_layer, (0, 0), shape_layer)

                if rng.random() < p["occlusion_prob"]:
                    occ_draw = ImageDraw.Draw(img)
                    occ_count = int(rng.integers(1, 4))
                    for _ in range(occ_count):
                        ow = int(image_size * rng.uniform(0.05, p["occlusion_max_frac"]))
                        oh = int(image_size * rng.uniform(0.05, p["occlusion_max_frac"]))
                        ox = int(rng.integers(0, max(1, image_size - ow)))
                        oy = int(rng.integers(0, max(1, image_size - oh)))
                        oc = int(rng.integers(55, 185))
                        occ_draw.rectangle([ox, oy, ox + ow, oy + oh], fill=(oc, oc, oc))

                arr = np.array(img, dtype=np.float32)
                arr += rng.normal(0.0, p["bg_noise"] * 1.2, size=arr.shape)

                brightness = rng.uniform(1.0 - p["brightness_jitter"], 1.0 + p["brightness_jitter"])
                contrast = rng.uniform(1.0 - p["contrast_jitter"], 1.0 + p["contrast_jitter"])
                arr = (arr - 127.5) * contrast + 127.5
                arr *= brightness
                arr = np.clip(arr, 0, 255).astype(np.uint8)

                img = Image.fromarray(arr, mode="RGB")
                blur_radius = float(rng.uniform(0.0, p["blur_max"]))
                if blur_radius > 0:
                    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

                save_cls = cls
                if rng.random() < label_noise:
                    save_cls = "nio" if cls == "io" else "io"

                out_dir = os.path.join(data_dir, split, save_cls)
                os.makedirs(out_dir, exist_ok=True)
                img.save(os.path.join(out_dir, f"{cls}_{i:04d}.png"))

    print(f"\nDummy dataset created in '{data_dir}':")
    print(
        f"  image_size={image_size} | difficulty={difficulty} | "
        f"label_noise={label_noise:.2f}"
    )
    for split, n in split_counts.items():
        print(f"  {split:<6} : {n} io + {n} nio = {2 * n} images")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic IO/NIO dataset")
    parser.add_argument("--data_dir", default="data", help="Output folder")
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument(
        "--n_per_split",
        type=int,
        default=20,
        help="Images per class in train split (val/test = half)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="hard",
        help="Visual complexity level",
    )
    parser.add_argument(
        "--label_noise",
        type=float,
        default=0.0,
        help="Fraction of samples saved with flipped class label",
    )
    args = parser.parse_args()

    create_dummy_dataset(
        data_dir=args.data_dir,
        image_size=args.image_size,
        n_per_split=args.n_per_split,
        seed=args.seed,
        difficulty=args.difficulty,
        label_noise=args.label_noise,
    )
