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
    from src.utils.reproducibility import make_worker_init_fn, set_seed

    seed = config.get("seed", 42)
    set_seed(seed, deterministic=config.get("deterministic", True))

    # Seeded Generator für DataLoader-Shuffle
    generator = torch.Generator()
    generator.manual_seed(seed)
    worker_init_fn = make_worker_init_fn(seed) if config.get("num_workers", 0) > 0 else None

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 65)
    logger.info("IO/NIO Classifier – Training gestartet")
    logger.info(f"Run-Ordner : {run_dir}")
    logger.info(f"Config     : {args.config}")
    logger.info(f"Dummy-Mode : {args.dummy}")
    logger.info(f"Device     : {device}")
    logger.info("=" * 65)

    # ----------------------------------------------------------------
    # 5. Dummy-Daten erzeugen (optional)
    # ----------------------------------------------------------------
    if args.dummy:
        from scripts.create_dummy_data import create_dummy_dataset

        dummy_size        = int(config.get("dummy_image_size", config.get("image_size", 320)))
        dummy_n_per_split = int(config.get("dummy_n_per_split", 16))
        dummy_difficulty  = str(config.get("dummy_difficulty", "hard"))
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
    from src.dataset import bildgroesse, build_dataloaders

    train_loader, val_loader, test_loader = build_dataloaders(
        config,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    train_ds      = train_loader.dataset
    class_counts  = train_ds.get_class_counts()
    class_weights = train_ds.get_class_weights()

    logger.info(f"Klassen-Verteilung (Train): {class_counts}")
    logger.info(f"Klassen-Gewichte  [io, nio]: {class_weights.tolist()}")
    logger.info(f"WeightedSampler aktiv: {config.get('use_weighted_sampler', False)}")

    # Tatsaechlich aktive Augmentation protokollieren. Es gibt keine
    # Config-Validierung – ein Tippfehler im Schluessel wuerde still ignoriert,
    # und color_jitter kennt kein 'enabled'. So ist am Run nachweisbar, was lief.
    logger.info("Aktive Train-Transforms:")
    for op in train_loader.dataset.transform.transforms:
        logger.info(f"  - {op}")

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
    # Strg+C bricht das Training sauber ab und springt direkt in die
    # Test-Evaluation mit dem bisher besten Modell (best.pt), statt das
    # ganze Skript zu killen. Ein zweites Strg+C beendet dann hart.
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning(
            "Training per Strg+C unterbrochen — fahre mit der Test-Evaluation "
            "des bisher besten Modells (best.pt) fort."
        )
        if trainer.tb_writer:
            trainer.tb_writer.close()

    from src.utils.plot_metrics import plot_training_curves

    plot_training_curves(run_dir=run_dir, logger=logger)

    # ----------------------------------------------------------------
    # 9. Evaluation auf Testset (bestes Modell)
    # ----------------------------------------------------------------
    logger.info("Lade bestes Modell für Test-Evaluation…")
    best_ckpt = os.path.join(run_dir, "checkpoints", "best.pt")

    # Fallback, falls vor der ersten Validierung abgebrochen wurde (kein best.pt):
    if not os.path.exists(best_ckpt):
        last_ckpt = os.path.join(run_dir, "checkpoints", "last.pt")
        if os.path.exists(last_ckpt):
            logger.warning(f"best.pt fehlt — nutze {last_ckpt} für die Test-Evaluation.")
            best_ckpt = last_ckpt
        else:
            logger.warning("Kein Checkpoint vorhanden — Test-Evaluation übersprungen.")
            return

    ckpt = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    from src.evaluation.evaluator import Evaluator, find_best_threshold

    # ---- Schwellenwert auf VAL bestimmen ---------------------------
    # Wichtig fuer die Methodik: der Betriebspunkt wird ausschliesslich auf dem
    # Validierungsset gesucht und danach unveraendert auf das Testset angewendet.
    # Wuerde man ihn auf test optimieren, waere die Testzahl wertlos.
    val_evaluator = Evaluator(
        model=model,
        loader=val_loader,
        run_dir=run_dir,
        device=device,
        logger=logger,
        split_name="val",
    )
    val_labels, val_probs = val_evaluator.collect()
    best_threshold, best_val_f1 = find_best_threshold(val_labels, val_probs[:, 1])

    logger.info(
        f"Schwellenwert auf VAL bestimmt (F1-Maximum): {best_threshold:.4f} "
        f"→ val_f1={best_val_f1:.4f} (Standard 0.5 als Vergleich in val_metrics.json)"
    )
    logger.info(
        "Hinweis: val traegt damit drei Rollen (Early Stopping, best.pt, "
        "Schwellenwert). Der Wert ist auf val leicht optimistisch; test bleibt "
        "sauber, weil dort nur angewendet und nicht nachoptimiert wird."
    )

    threshold_info = {
        "threshold":       best_threshold,
        "criterion":       "max_f1",
        "determined_on":   "val",
        "val_f1_at_threshold": best_val_f1,
        "note": (
            "Auf dem Validierungsset bestimmt und unveraendert auf das Testset "
            "angewendet. Fuer die Inferenz: predict.py --threshold <wert>."
        ),
    }
    with open(os.path.join(run_dir, "metrics", "threshold.json"), "w") as f:
        json.dump(threshold_info, f, indent=2)

    # ---- VAL final mit dem gefundenen Schwellenwert ----------------
    # collect() ist gecacht: kein zweiter Forward-Pass.
    val_evaluator.threshold = best_threshold
    val_evaluator.evaluate()

    # ---- TEST mit demselben Schwellenwert --------------------------
    evaluator = Evaluator(
        model=model,
        loader=test_loader,
        run_dir=run_dir,
        device=device,
        logger=logger,
        threshold=best_threshold,
        split_name="test",
    )
    test_metrics = evaluator.evaluate()

    # ----------------------------------------------------------------
    # 10. ONNX-Export (optional)
    # ----------------------------------------------------------------
    if config.get("export_onnx", False):
        try:
            hoehe, breite = bildgroesse(config)
            onnx_path   = os.path.join(run_dir, "model.onnx")
            dummy_input = torch.randn(1, 3, hoehe, breite)
            model_cpu   = model.cpu().eval()

            torch.onnx.export(
                model_cpu,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=14,
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={
                    "input":  {0: "batch_size"},
                    "logits": {0: "batch_size"},
                },
            )
            logger.info(f"ONNX-Modell gespeichert → {onnx_path}")
        except Exception as exc:
            logger.warning(f"ONNX-Export fehlgeschlagen: {exc}")
        finally:
            # model.cpu() oben ist IN-PLACE: ohne diese Zeile liefen XAI und
            # Rueckprojektion danach auf der CPU (beide holen ihr Geraet aus
            # next(model.parameters()).device) - bei ~10 s je Bild auf der GPU
            # waere das keine Verlangsamung, sondern ein Abbruch der Geduld.
            model.to(device)

    # ----------------------------------------------------------------
    # 11. XAI – Integrated Gradients (optional)
    # ----------------------------------------------------------------
    if config.get("xai", {}).get("enabled", False):
        from src.xai.integrated_gradients import run_xai

        # Welches Set erklärt wird: xai.split = val | test
        xai_split  = str(config.get("xai", {}).get("split", "val")).lower()
        xai_loader = test_loader if xai_split == "test" else val_loader
        logger.info(f"Starte XAI (Split: {xai_split}, Integrated Gradients)…")

        run_xai(model, xai_loader, config, run_dir, logger=logger)
    else:
        logger.info(
            "XAI deaktiviert. Aktivieren: 'xai: enabled: true' in config.yaml"
        )

    # Heatmap zurueck ins Originalbild, an die Stelle, an der gezoomt wurde.
    # Braucht den zuschnitt:-Block der Config; fehlt er, gibt es nur eine Logzeile.
    #
    # BEWUSST AUSSERHALB von xai.enabled: der Versuchsplan schaltet xai.enabled ab,
    # weil Heatmaps fuer alle 400 Testbilder rund zwei Stunden je Lauf kosten und
    # keine Messgroesse des Vergleichs sind - die zwoelf Rueckprojektionen sollen
    # die Vergleichslaeufe trotzdem bekommen. xai.rueckprojektion.enabled ist der
    # eigene Schalter dafuer, und die Funktion scheitert weich.
    from src.xai.lauf_bilder import rueckprojektion_nach_training

    rueckprojektion_nach_training(model, config, run_dir, logger=logger)

    # ----------------------------------------------------------------
    # 12. Abschluss
    # ----------------------------------------------------------------
    logger.info("")
    logger.info("=" * 65)
    logger.info("FERTIG – alle Ergebnisse unter:")
    logger.info(f"  {run_dir}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
