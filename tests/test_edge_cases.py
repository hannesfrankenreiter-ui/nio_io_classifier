"""
tests/test_edge_cases.py
Edge-Case-Tests für Dataset, Evaluator und weitere Kernkomponenten.

Deckt ab:
  - Korrupte / nicht lesbare Bilder im Dataset
  - Leeres Dataset
  - Evaluator bei All-Same-Prediction
  - Metriken-Keys bei Klassenimbalance im Testset
  - Neue Augmentierungs-Configs (RandomErasing, GaussianBlur)
  - Konfigurierbarer Threshold in predict_single()
  - Neue Backbones (EfficientNet-B0, ConvNeXt-Tiny)

Ausführen:
    pytest tests/test_edge_cases.py -v
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ================================================================
#  Dataset – Fehlerbehandlung
# ================================================================
class TestDatasetEdgeCases:
    def test_corrupted_image_returns_dummy(self, tmp_path):
        """Korruptes Bild darf __getitem__ nicht zum Absturz bringen."""
        from PIL import Image

        from src.dataset import IODataset, build_transforms

        io_dir  = tmp_path / "io"
        nio_dir = tmp_path / "nio"
        io_dir.mkdir()
        nio_dir.mkdir()

        # 1 valides + 1 korruptes IO-Bild
        Image.new("RGB", (64, 64), color=(255, 0, 0)).save(str(io_dir / "valid.png"))
        (io_dir / "corrupt.png").write_bytes(b"das ist kein PNG")
        # 1 valides NIO-Bild
        Image.new("RGB", (64, 64), color=(0, 0, 255)).save(str(nio_dir / "valid2.png"))

        cfg = {"image_size": 64, "augmentation": {}}
        tf  = build_transforms(cfg, train=False)
        ds  = IODataset(str(tmp_path), transform=tf)

        # Dataset enthält alle 3 Einträge (auch das korrupte)
        assert len(ds) == 3

        # Alle Einträge müssen ohne Exception abrufbar sein
        for i in range(len(ds)):
            img, label = ds[i]
            assert img.shape == (3, 64, 64), f"Falsches Shape bei Index {i}: {img.shape}"
            assert label in (0, 1)

    def test_empty_dataset_raises(self, tmp_path):
        """Leeres Dataset muss RuntimeError auslösen."""
        from src.dataset import IODataset, build_transforms

        io_dir  = tmp_path / "io"
        nio_dir = tmp_path / "nio"
        io_dir.mkdir()
        nio_dir.mkdir()

        cfg = {"image_size": 64, "augmentation": {}}
        tf  = build_transforms(cfg, train=False)

        with pytest.raises(RuntimeError, match="Keine Bilder"):
            IODataset(str(tmp_path), transform=tf)

    def test_missing_class_dir_does_not_crash(self, tmp_path):
        """Fehlendes Unterverzeichnis für eine Klasse führt nicht zum Absturz."""
        from PIL import Image

        from src.dataset import IODataset, build_transforms

        # Nur IO-Verzeichnis anlegen (kein NIO)
        io_dir = tmp_path / "io"
        io_dir.mkdir()
        Image.new("RGB", (64, 64)).save(str(io_dir / "img.png"))

        cfg = {"image_size": 64, "augmentation": {}}
        tf  = build_transforms(cfg, train=False)
        ds  = IODataset(str(tmp_path), transform=tf)
        assert len(ds) == 1
        img, label = ds[0]
        assert img.shape == (3, 64, 64)
        assert label == 0  # io = 0


# ================================================================
#  Augmentation – neue Features
# ================================================================
class TestNewAugmentations:
    def test_random_erasing_does_not_crash(self, tmp_path):
        """RandomErasing-Augmentation darf nicht abstürzen."""
        from PIL import Image

        from src.dataset import IODataset, build_transforms

        io_dir  = tmp_path / "io"
        nio_dir = tmp_path / "nio"
        io_dir.mkdir()
        nio_dir.mkdir()
        for i in range(4):
            Image.new("RGB", (64, 64)).save(str(io_dir / f"{i}.png"))
            Image.new("RGB", (64, 64)).save(str(nio_dir / f"{i}.png"))

        cfg = {
            "image_size": 64,
            "augmentation": {
                "random_erasing": {"enabled": True, "p": 1.0, "scale": [0.05, 0.10]},
            },
        }
        tf = build_transforms(cfg, train=True)
        ds = IODataset(str(tmp_path), transform=tf)
        img, _ = ds[0]
        assert img.shape == (3, 64, 64)

    def test_gaussian_blur_does_not_crash(self, tmp_path):
        """GaussianBlur-Augmentation darf nicht abstürzen."""
        from PIL import Image

        from src.dataset import IODataset, build_transforms

        io_dir  = tmp_path / "io"
        nio_dir = tmp_path / "nio"
        io_dir.mkdir()
        nio_dir.mkdir()
        for i in range(4):
            Image.new("RGB", (64, 64)).save(str(io_dir / f"{i}.png"))
            Image.new("RGB", (64, 64)).save(str(nio_dir / f"{i}.png"))

        cfg = {
            "image_size": 64,
            "augmentation": {
                "gaussian_blur": {"enabled": True, "kernel_size": 5, "sigma": [0.1, 1.5], "p": 1.0},
            },
        }
        tf = build_transforms(cfg, train=True)
        ds = IODataset(str(tmp_path), transform=tf)
        img, _ = ds[0]
        assert img.shape == (3, 64, 64)


# ================================================================
#  Evaluator – Edge Cases
# ================================================================
class TestEvaluatorEdgeCases:
    def _make_evaluator(self, tmp_path, labels):
        """Hilfs-Methode: Evaluator mit fixierten Labels/Predictions."""
        from src.evaluation.evaluator import Evaluator
        from src.models.resnet import build_model

        run_dir = str(tmp_path / "eval")
        for sub in ["metrics", "preds"]:
            os.makedirs(os.path.join(run_dir, sub))

        model  = build_model({"backbone": "resnet18", "pretrained": False, "freeze_backbone": False})
        data   = [(torch.randn(3, 64, 64), lbl) for lbl in labels]
        loader = torch.utils.data.DataLoader(data, batch_size=4)
        device = torch.device("cpu")

        return Evaluator(model, loader, run_dir, device)

    def test_all_same_label_no_crash(self, tmp_path):
        """Evaluator mit nur einer Klasse im Testset darf nicht abstürzen."""
        evaluator = self._make_evaluator(tmp_path, [0, 0, 0, 0, 0, 0, 0, 0])
        metrics   = evaluator.evaluate()

        assert "accuracy"  in metrics
        assert "f1"        in metrics
        assert "roc_auc"   in metrics
        # ROC-AUC ist NaN wenn nur eine Klasse vorhanden
        assert np.isnan(metrics["roc_auc"]) or isinstance(metrics["roc_auc"], float)

    def test_balanced_dataset_has_valid_roc_auc(self, tmp_path):
        """Bei balanciertem Testset muss ROC-AUC in [0, 1] liegen."""
        evaluator = self._make_evaluator(
            tmp_path, [0, 0, 0, 0, 1, 1, 1, 1]
        )
        metrics = evaluator.evaluate()
        roc_auc = metrics["roc_auc"]

        assert not np.isnan(roc_auc), "ROC-AUC darf nicht NaN sein bei 2 Klassen."
        assert 0.0 <= roc_auc <= 1.0

    def test_metrics_keys_complete(self, tmp_path):
        """Alle erwarteten Metriken-Keys müssen vorhanden sein."""
        evaluator = self._make_evaluator(
            tmp_path, [0, 0, 1, 1, 0, 1, 0, 1]
        )
        metrics = evaluator.evaluate()

        expected_keys = {
            "accuracy", "precision", "recall", "f1",
            "roc_auc", "average_precision",
            "n_samples", "n_io", "n_nio",
            "inference_ms_per_image", "n_params_trainable",
        }
        for key in expected_keys:
            assert key in metrics, f"Metrik '{key}' fehlt im Ergebnis-Dict."

        # Nebenkriterien des Backbone-Vergleichs muessen belastbar sein
        assert metrics["inference_ms_per_image"] > 0.0
        assert metrics["n_params_trainable"] > 0

    def test_json_file_written(self, tmp_path):
        """test_metrics.json muss nach evaluate() existieren."""
        import json

        evaluator = self._make_evaluator(
            tmp_path, [0, 0, 1, 1, 0, 1, 0, 1]
        )
        evaluator.evaluate()

        json_path = str(tmp_path / "eval" / "metrics" / "test_metrics.json")
        assert os.path.exists(json_path)
        with open(json_path) as f:
            data = json.load(f)
        assert "accuracy" in data


# ================================================================
#  Inference – Threshold & neue Backbones
# ================================================================
class TestInferenceEdgeCases:
    def test_threshold_nio_class(self, tmp_path):
        """Schwellenwert threshold=0.0 → immer NIO-Vorhersage."""
        from PIL import Image

        from src.inference.predict import get_transform, predict_single
        from src.models.resnet import build_model

        img_path = str(tmp_path / "test.png")
        Image.new("RGB", (64, 64)).save(img_path)

        cfg   = {"backbone": "resnet18", "pretrained": False, "image_size": 64, "augmentation": {}}
        model = build_model(cfg).eval()
        tf    = get_transform(cfg)

        result = predict_single(model, tf, img_path, config={"threshold": 0.0})
        assert result["predicted_class"] == "nio"

    def test_threshold_io_class(self, tmp_path):
        """Schwellenwert threshold=1.0 → immer IO-Vorhersage."""
        from PIL import Image

        from src.inference.predict import get_transform, predict_single
        from src.models.resnet import build_model

        img_path = str(tmp_path / "test2.png")
        Image.new("RGB", (64, 64)).save(img_path)

        cfg   = {"backbone": "resnet18", "pretrained": False, "image_size": 64, "augmentation": {}}
        model = build_model(cfg).eval()
        tf    = get_transform(cfg)

        result = predict_single(model, tf, img_path, config={"threshold": 1.0})
        assert result["predicted_class"] == "io"

    def test_predict_single_probs_sum_to_one(self, tmp_path):
        """prob_io + prob_nio muss 1.0 ergeben (bis auf Gleitkomma-Fehler)."""
        from PIL import Image

        from src.inference.predict import get_transform, predict_single
        from src.models.resnet import build_model

        img_path = str(tmp_path / "test3.png")
        Image.new("RGB", (64, 64)).save(img_path)

        cfg   = {"backbone": "resnet18", "pretrained": False, "image_size": 64, "augmentation": {}}
        model = build_model(cfg).eval()
        tf    = get_transform(cfg)

        result = predict_single(model, tf, img_path)
        assert abs(result["prob_io"] + result["prob_nio"] - 1.0) < 1e-4


class TestNewBackbones:
    @pytest.mark.parametrize("backbone", ["efficientnet_b0", "convnext_tiny"])
    def test_forward_pass(self, backbone):
        """Neue Backbones müssen forward pass liefern – Shape (B, 2)."""
        from src.models.resnet import build_model

        model = build_model({"backbone": backbone, "pretrained": False, "freeze_backbone": False})
        x     = torch.randn(2, 3, 64, 64)
        out   = model(x)
        assert out.shape == (2, 2), f"{backbone}: Erwartet (2,2), erhalten {out.shape}"

    @pytest.mark.parametrize("backbone", ["efficientnet_b0", "convnext_tiny"])
    def test_head_trainable(self, backbone):
        """Head-Parameter neuer Backbones müssen trainierbar sein."""
        from src.models.resnet import build_model, count_trainable_params

        model = build_model({"backbone": backbone, "pretrained": False, "freeze_backbone": True})
        n     = count_trainable_params(model)
        assert n > 0, f"{backbone}: Keine trainierbaren Parameter (Head sollte trainierbar sein)."

    @pytest.mark.parametrize("backbone", ["efficientnet_b0", "convnext_tiny"])
    def test_freeze_reduces_trainable_params(self, backbone):
        """freeze_backbone=True reduziert trainierbare Parameter gegenüber False."""
        from src.models.resnet import build_model, count_trainable_params

        frozen   = build_model({"backbone": backbone, "pretrained": False, "freeze_backbone": True})
        unfrozen = build_model({"backbone": backbone, "pretrained": False, "freeze_backbone": False})

        n_frozen   = count_trainable_params(frozen)
        n_unfrozen = count_trainable_params(unfrozen)
        assert n_frozen < n_unfrozen, (
            f"{backbone}: frozen sollte weniger Parameter haben als unfrozen."
        )


# ================================================================
#  Trainer – neue Features
# ================================================================
class TestTrainerNewFeatures:
    def _make_run_dir(self, tmp_path, name="run"):
        run_dir = str(tmp_path / name)
        for sub in ["checkpoints", "metrics", "preds"]:
            os.makedirs(os.path.join(run_dir, sub))
        return run_dir

    def test_f1_in_checkpoint(self, tmp_path):
        """Checkpoint muss checkpoint_monitor-Feld enthalten."""
        from scripts.create_dummy_data import create_dummy_dataset

        from src.dataset import build_dataloaders
        from src.models.resnet import build_model
        from src.training.trainer import Trainer

        data_dir = str(tmp_path / "data")
        create_dummy_dataset(data_dir, image_size=32, n_per_split=8, seed=0)
        run_dir = self._make_run_dir(tmp_path, "run_f1")

        cfg = {
            "backbone": "resnet18", "pretrained": False, "freeze_backbone": False,
            "image_size": 32, "batch_size": 4, "num_workers": 0,
            "augmentation": {}, "use_class_weights": False,
            "optimizer": "adam", "learning_rate": 1e-3, "weight_decay": 0.0,
            "scheduler": "none", "epochs": 2, "early_stopping": False,
            "log_tensorboard": False, "log_csv": True,
            "early_stopping_monitor": "val_f1",
            "data_dir": data_dir,
        }
        tl, vl, _ = build_dataloaders(cfg)
        model = build_model(cfg)
        trainer = Trainer(model, tl, vl, cfg, run_dir)
        trainer.train()

        ckpt = torch.load(os.path.join(run_dir, "checkpoints", "best.pt"), map_location="cpu")
        assert "checkpoint_monitor" in ckpt
        assert ckpt["checkpoint_monitor"] == "val_f1"

    def test_val_loss_monitor(self, tmp_path):
        """early_stopping_monitor=val_loss muss ohne Fehler trainieren."""
        from scripts.create_dummy_data import create_dummy_dataset

        from src.dataset import build_dataloaders
        from src.models.resnet import build_model
        from src.training.trainer import Trainer

        data_dir = str(tmp_path / "data2")
        create_dummy_dataset(data_dir, image_size=32, n_per_split=8, seed=1)
        run_dir = self._make_run_dir(tmp_path, "run_loss")

        cfg = {
            "backbone": "resnet18", "pretrained": False, "freeze_backbone": False,
            "image_size": 32, "batch_size": 4, "num_workers": 0,
            "augmentation": {}, "use_class_weights": False,
            "optimizer": "adam", "learning_rate": 1e-3, "weight_decay": 0.0,
            "scheduler": "none", "epochs": 2, "early_stopping": False,
            "log_tensorboard": False, "log_csv": True,
            "early_stopping_monitor": "val_loss",
            "data_dir": data_dir,
        }
        tl, vl, _ = build_dataloaders(cfg)
        model = build_model(cfg)
        trainer = Trainer(model, tl, vl, cfg, run_dir)
        trainer.train()
        assert os.path.exists(os.path.join(run_dir, "checkpoints", "best.pt"))

    def test_invalid_monitor_raises(self):
        """Ungültiger Monitor-Wert muss ValueError auslösen."""
        from src.models.resnet import build_model
        from src.training.trainer import Trainer

        model = build_model({"backbone": "resnet18", "pretrained": False, "freeze_backbone": False})
        loader = torch.utils.data.DataLoader(
            [(torch.randn(3, 32, 32), 0)] * 4, batch_size=2
        )
        cfg = {
            "epochs": 1, "early_stopping": False, "log_tensorboard": False,
            "log_csv": True, "scheduler": "none",
            "early_stopping_monitor": "val_map",  # ungültig
        }
        with pytest.raises(ValueError):
            Trainer(model, loader, loader, cfg, "/tmp/dummy_run")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
