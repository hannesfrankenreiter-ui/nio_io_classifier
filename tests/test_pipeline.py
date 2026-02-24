"""
tests/test_pipeline.py
Automatisierte Tests für alle Kernkomponenten.

Ausführen:
    pytest tests/ -v
    pytest tests/ -v --tb=short
"""
import os
import sys

import pytest
import torch

# Projektpfad einbinden
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ================================================================
#  Dummy-Daten Fixture
# ================================================================
@pytest.fixture(scope="module")
def dummy_data_dir(tmp_path_factory):
    """Erstellt einmalig ein temporäres Dummy-Dataset für alle Tests."""
    from scripts.create_dummy_data import create_dummy_dataset

    data_dir = str(tmp_path_factory.mktemp("data"))
    create_dummy_dataset(data_dir, image_size=64, n_per_split=8, seed=0)
    return data_dir


@pytest.fixture(scope="module")
def base_config(dummy_data_dir):
    return {
        "backbone":             "resnet18",
        "pretrained":           False,
        "freeze_backbone":      False,
        "image_size":           64,
        "batch_size":           4,
        "num_workers":          0,
        "augmentation":         {},
        "use_class_weights":    True,
        "use_weighted_sampler": False,
        "data_dir":             dummy_data_dir,
        "optimizer":            "adam",
        "learning_rate":        1e-3,
        "weight_decay":         0.0,
        "scheduler":            "none",
        "epochs":               2,
        "early_stopping":       False,
        "log_tensorboard":      False,
        "log_csv":              True,
        "xai":                  {"enabled": False},
    }


# ================================================================
#  Dataset-Tests
# ================================================================
class TestDataset:
    def test_folder_loading(self, dummy_data_dir):
        from src.dataset import IODataset, build_transforms

        cfg = {"image_size": 64, "augmentation": {}}
        tf  = build_transforms(cfg, train=False)
        ds  = IODataset(os.path.join(dummy_data_dir, "train"), transform=tf)

        assert len(ds) == 16, f"Erwartet 16 (8 io + 8 nio), gefunden: {len(ds)}"

    def test_item_shape(self, dummy_data_dir):
        from src.dataset import IODataset, build_transforms

        cfg    = {"image_size": 64, "augmentation": {}}
        tf     = build_transforms(cfg, train=False)
        ds     = IODataset(os.path.join(dummy_data_dir, "train"), transform=tf)
        img, label = ds[0]

        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 64, 64)
        assert label in (0, 1)

    def test_class_counts_balanced(self, dummy_data_dir):
        from src.dataset import IODataset, build_transforms

        cfg    = {"image_size": 64, "augmentation": {}}
        tf     = build_transforms(cfg, train=False)
        ds     = IODataset(os.path.join(dummy_data_dir, "train"), transform=tf)
        counts = ds.get_class_counts()

        assert counts["io"]  == 8
        assert counts["nio"] == 8

    def test_class_weights_balanced(self, dummy_data_dir):
        from src.dataset import IODataset, build_transforms

        cfg    = {"image_size": 64, "augmentation": {}}
        tf     = build_transforms(cfg, train=False)
        ds     = IODataset(os.path.join(dummy_data_dir, "train"), transform=tf)
        w      = ds.get_class_weights()

        assert w.shape == (2,)
        # Balanciertes Dataset → Gewichte = 1.0
        assert torch.allclose(w, torch.ones(2)), f"Gewichte: {w}"

    def test_sample_weights_length(self, dummy_data_dir):
        from src.dataset import IODataset, build_transforms

        cfg = {"image_size": 64, "augmentation": {}}
        tf  = build_transforms(cfg, train=False)
        ds  = IODataset(os.path.join(dummy_data_dir, "train"), transform=tf)
        sw  = ds.get_sample_weights()

        assert len(sw) == len(ds)

    def test_train_augmentation_does_not_crash(self, dummy_data_dir):
        from src.dataset import IODataset, build_transforms

        cfg = {
            "image_size": 64,
            "augmentation": {
                "random_horizontal_flip": True,
                "random_vertical_flip":   True,
                "random_rotation":        10,
                "color_jitter": {"brightness": 0.2, "contrast": 0.2,
                                 "saturation": 0.1, "hue": 0.0},
            },
        }
        tf = build_transforms(cfg, train=True)
        ds = IODataset(os.path.join(dummy_data_dir, "train"), transform=tf)
        img, _ = ds[0]
        assert img.shape == (3, 64, 64)


# ================================================================
#  DataLoader-Tests
# ================================================================
class TestDataLoader:
    def test_build_dataloaders(self, base_config):
        from src.dataset import build_dataloaders

        tl, vl, testl = build_dataloaders(base_config)
        batch_img, batch_lbl = next(iter(tl))

        assert batch_img.shape[0] <= base_config["batch_size"]
        assert batch_img.shape[1:] == (3, 64, 64)
        assert batch_lbl.dtype == torch.int64


# ================================================================
#  Modell-Tests
# ================================================================
class TestModel:
    def test_build_resnet18(self):
        from src.models.resnet import build_model

        model = build_model({"backbone": "resnet18", "pretrained": False, "freeze_backbone": False})
        x     = torch.randn(2, 3, 64, 64)
        out   = model(x)

        assert out.shape == (2, 2), f"Erwartet (2, 2), erhalten: {out.shape}"

    def test_freeze_backbone(self):
        from src.models.resnet import build_model, count_trainable_params

        frozen  = build_model({"backbone": "resnet18", "pretrained": False, "freeze_backbone": True})
        unfrozen = build_model({"backbone": "resnet18", "pretrained": False, "freeze_backbone": False})

        n_frozen   = count_trainable_params(frozen)
        n_unfrozen = count_trainable_params(unfrozen)

        assert n_frozen < n_unfrozen, "Frozen sollte weniger trainierbare Parameter haben."

    def test_unknown_backbone_raises(self):
        from src.models.resnet import build_model

        with pytest.raises(ValueError):
            build_model({"backbone": "vgg99", "pretrained": False})


# ================================================================
#  Early Stopping
# ================================================================
class TestEarlyStopping:
    def test_stops_after_patience(self):
        from src.training.early_stopping import EarlyStopping

        es = EarlyStopping(patience=3, delta=0.01, mode="max")
        for _ in range(3):
            stopped = es.step(0.5)  # keine Verbesserung
        assert stopped

    def test_resets_on_improvement(self):
        from src.training.early_stopping import EarlyStopping

        es = EarlyStopping(patience=3, delta=0.0, mode="max")
        es.step(0.5)
        es.step(0.4)  # keine Verbesserung
        es.step(0.7)  # Verbesserung → Counter zurücksetzen
        assert es.counter == 0


# ================================================================
#  Vollständiger Mini-Trainingslauf
# ================================================================
class TestTrainer:
    def test_mini_training_run(self, base_config, tmp_path):
        from src.dataset import build_dataloaders
        from src.models.resnet import build_model
        from src.training.trainer import Trainer

        run_dir = str(tmp_path / "test_run")
        for sub in ["checkpoints", "metrics", "preds"]:
            os.makedirs(os.path.join(run_dir, sub))

        train_loader, val_loader, _ = build_dataloaders(base_config)
        model = build_model(base_config)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=base_config,
            run_dir=run_dir,
            class_weights=None,
            logger=None,
        )
        trainer.train()

        assert os.path.exists(os.path.join(run_dir, "checkpoints", "last.pt"))
        assert os.path.exists(os.path.join(run_dir, "checkpoints", "best.pt"))
        assert os.path.exists(os.path.join(run_dir, "metrics", "train_metrics.csv"))

    def test_checkpoint_loadable(self, base_config, tmp_path):
        from src.dataset import build_dataloaders
        from src.models.resnet import build_model
        from src.training.trainer import Trainer

        run_dir = str(tmp_path / "test_run_load")
        for sub in ["checkpoints", "metrics", "preds"]:
            os.makedirs(os.path.join(run_dir, sub))

        train_loader, val_loader, _ = build_dataloaders(base_config)
        model = build_model(base_config)

        trainer = Trainer(model, train_loader, val_loader, base_config, run_dir)
        trainer.train()

        ckpt = torch.load(os.path.join(run_dir, "checkpoints", "best.pt"), map_location="cpu")
        model2 = build_model(base_config)
        model2.load_state_dict(ckpt["model_state_dict"])  # darf nicht crashen
        assert "config" in ckpt


# ================================================================
#  Evaluator
# ================================================================
class TestEvaluator:
    def test_evaluator_produces_metrics(self, base_config, tmp_path):
        from src.dataset import build_dataloaders
        from src.evaluation.evaluator import Evaluator
        from src.models.resnet import build_model
        from src.training.trainer import Trainer

        run_dir = str(tmp_path / "eval_run")
        for sub in ["checkpoints", "metrics", "preds"]:
            os.makedirs(os.path.join(run_dir, sub))

        train_loader, val_loader, test_loader = build_dataloaders(base_config)
        model = build_model(base_config)

        trainer = Trainer(model, train_loader, val_loader, base_config, run_dir)
        trainer.train()

        evaluator = Evaluator(model, test_loader, run_dir, torch.device("cpu"))
        metrics   = evaluator.evaluate()

        for key in ["accuracy", "precision", "recall", "f1"]:
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0


# ================================================================
#  Inference
# ================================================================
class TestInference:
    def test_predict_single(self, dummy_data_dir, base_config, tmp_path):
        from src.inference.predict import get_transform, predict_single
        from src.models.resnet import build_model

        model     = build_model(base_config)
        transform = get_transform(base_config)

        # Erstes Testbild finden
        test_io_dir = os.path.join(dummy_data_dir, "test", "io")
        img_path    = os.path.join(test_io_dir, os.listdir(test_io_dir)[0])

        result = predict_single(model, transform, img_path)

        assert result["predicted_class"] in ("io", "nio")
        assert 0.0 <= result["confidence"] <= 1.0
        assert abs(result["prob_io"] + result["prob_nio"] - 1.0) < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])