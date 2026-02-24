#!/usr/bin/env python3
"""
train.py  –  Haupteinstiegspunkt für Training + Evaluation + XAI

Schnellstart mit Dummy-Daten:
    python train.py --dummy

Mit echten Daten:
    python train.py --config configs/default.yaml

Run-Namen überschreiben:
    python train.py --dummy --run_name mein_erster_run

Freeze (nur Head trainieren):
    python train.py --dummy --config configs/default.yaml
    # freeze_backbone: true in YAML setzen
"""
import argparse
import json
import os
import sys

import torch
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="IO/NIO Classifier – Training")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Pfad zur Config-YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Überschreibt den automatisch generierten Run-Namen",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Synthetisches Dummy-Dataset erstellen und nutzen (zum Testen)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ----------------------------------------------------------------
    # 1. Config laden
    # ----------------------------------------------------------------
    if not os.path.exists(args.config):
        print(f"[FEHLER] Config nicht gefunden: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ----------------------------------------------------------------
    # 2. Reproduzierbarkeit
    # ----------------------------------------------------------------
    from src.utils.reproducibility import set_seed
    set_seed(config.get("seed", 42))

    # ----------------------------------------------------------------
    # 3. Run-Ordner erstellen
    # ----------------------------------------------------------------
    from src.utils.run_manager import create_run_dir, save_config
    run_dir = create_run_dir(config, run_name=args.run_name)
    save_config(config, run_dir, config_path=args.config)

    # ----------------------------------------------------------------
    # 4. Logger
    # ----------------------------------------------------------------
    from src.utils.logging_utils import get_logger
    logger = get_logger("train", log_file=os.path.join(run_dir, "train.log"))

    logger.info("=" * 65)
    logger.info("IO/NIO Classifier – Training gestartet")
    logger.info(f"Run-Ordner : {run_dir}")
    logger.info(f"Config     : {args.config}")
    logger.info(f"Dummy-Mode : {args.dummy}")
    logger.info("=" * 65)

    # ----------------------------------------------------------------
    # 5. Dummy-Daten erzeugen (optional)
    # ----------------------------------------------------------------
    if args.dummy:
        from scripts.create_dummy_data import create_dummy_dataset

        dummy_size = int(config.get("dummy_image_size", config.get("image_size", 320)))
        dummy_n_per_split = int(config.get("dummy_n_per_split", 16))
        dummy_difficulty = str(config.get("dummy_difficulty", "hard"))
        dummy_label_noise = float(config.get("dummy_label_noise", 0.0))

        logger.info(
            "Erzeuge Dummy-Dataset "
            f"(image_size={dummy_size}, difficulty={dummy_difficulty}, "
            f"label_noise={dummy_label_noise:.2f})..."
        )
        create_dummy_dataset(
            data_dir=config.get("data_dir", "data"),
            image_size=dummy_size,
            n_per_split=dummy_n_per_split,
            seed=int(config.get("seed", 42)),
            difficulty=dummy_difficulty,
            label_noise=dummy_label_noise,
        )
        config["image_size"] = dummy_size

    # ----------------------------------------------------------------
    # 6. DataLoaders + Klassengewichte
    # ----------------------------------------------------------------
    from src.dataset import build_dataloaders

    train_loader, val_loader, test_loader = build_dataloaders(config)

    train_ds      = train_loader.dataset
    class_counts  = train_ds.get_class_counts()
    class_weights = train_ds.get_class_weights()

    logger.info(f"Klassen-Verteilung (Train): {class_counts}")
    logger.info(f"Klassen-Gewichte  [io, nio]: {class_weights.tolist()}")
    logger.info(f"WeightedSampler aktiv: {config.get('use_weighted_sampler', False)}")

    # Klassen-Info in metrics/ ablegen
    class_info = {
        "class_counts":  class_counts,
        "class_weights": {
            "io":  float(class_weights[0]),
            "nio": float(class_weights[1]),
        },
        "use_class_weights":    config.get("use_class_weights", True),
        "use_weighted_sampler": config.get("use_weighted_sampler", False),
    }
    with open(os.path.join(run_dir, "metrics", "class_info.json"), "w") as f:
        json.dump(class_info, f, indent=2)

    # ----------------------------------------------------------------
    # 7. Modell
    # ----------------------------------------------------------------
    from src.models.resnet import build_model, count_trainable_params

    model = build_model(config)
    n_params = count_trainable_params(model)
    logger.info(
        f"Modell: {config.get('backbone', 'resnet18')} | "
        f"pretrained={config.get('pretrained', True)} | "
        f"freeze_backbone={config.get('freeze_backbone', False)} | "
        f"trainierbare Parameter: {n_params:,}"
    )

    # ----------------------------------------------------------------
    # 8. Training
    # ----------------------------------------------------------------
    from src.training.trainer import Trainer

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        run_dir=run_dir,
        class_weights=class_weights if config.get("use_class_weights", True) else None,
        logger=logger,
    )
    trainer.train()

    from src.utils.plot_metrics import plot_training_curves
    plot_training_curves(run_dir=run_dir, logger=logger)

    # ----------------------------------------------------------------
    # 9. Evaluation auf Testset (bestes Modell)
    # ----------------------------------------------------------------
    logger.info("Lade bestes Modell für Test-Evaluation…")
    best_ckpt = os.path.join(run_dir, "checkpoints", "best.pt")

    ckpt = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    from src.evaluation.evaluator import Evaluator

    evaluator = Evaluator(
        model=model,
        loader=test_loader,
        run_dir=run_dir,
        device=torch.device("cpu"),
        logger=logger,
    )
    test_metrics = evaluator.evaluate()

    # ----------------------------------------------------------------
    # 10. XAI – Integrated Gradients (optional)
    # ----------------------------------------------------------------
    if config.get("xai", {}).get("enabled", False):
        logger.info("Starte XAI (Integrated Gradients)…")
        from src.xai.integrated_gradients import run_xai

        run_xai(model, val_loader, config, run_dir, logger=logger)
    else:
        logger.info(
            "XAI deaktiviert. Aktivieren: 'xai: enabled: true' in config.yaml"
        )

    # ----------------------------------------------------------------
    # 11. Abschluss
    # ----------------------------------------------------------------
    logger.info("")
    logger.info("=" * 65)
    logger.info("FERTIG – alle Ergebnisse unter:")
    logger.info(f"  {run_dir}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
