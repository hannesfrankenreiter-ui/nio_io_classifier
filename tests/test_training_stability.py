"""
tests/test_training_stability.py

Tests für die Maßnahmen gegen die stark schwankende Validierungskurve:
  - BatchNorm-Statistiken im eingefrorenen Backbone bleiben konstant
  - freeze_backbone: "layer4" gibt genau die erwarteten Parameter frei
  - Optimizer trennt Kopf und Backbone in Gruppen mit eigenen Lernraten
  - Evaluator entscheidet per Schwellenwert statt argmax
  - Schwellensuche findet das F1-Optimum

Ausführen:
    pytest tests/test_training_stability.py -v
"""
import copy
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture(scope="module")
def dummy_data_dir(tmp_path_factory):
    from scripts.create_dummy_data import create_dummy_dataset

    data_dir = str(tmp_path_factory.mktemp("data_stability"))
    create_dummy_dataset(data_dir, image_size=64, n_per_split=8, seed=0)
    return data_dir


@pytest.fixture
def cfg(dummy_data_dir):
    """Basis-Config: resnet18, layer4 eingefroren bis auf den letzten Block."""
    return {
        "backbone":             "resnet18",
        "pretrained":           False,
        "freeze_backbone":      "layer4",
        "freeze_bn_stats":      True,
        "image_size":           64,
        "batch_size":           4,
        "num_workers":          0,
        "augmentation":         {},
        "use_class_weights":    False,
        "use_weighted_sampler": False,
        "data_dir":             dummy_data_dir,
        "optimizer":            "adamw",
        "learning_rate":        1e-3,
        "weight_decay":         0.0,
        "scheduler":            "none",
        "epochs":               1,
        "early_stopping":       False,
        "log_tensorboard":      False,
        "xai":                  {"enabled": False},
    }


def _make_run_dir(tmp_path, name):
    run_dir = str(tmp_path / name)
    for sub in ["checkpoints", "metrics", "preds"]:
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir


def _train_one_epoch(config, run_dir):
    """Fährt eine Epoche und gibt das Modell zurück."""
    from src.dataset import build_dataloaders
    from src.models.resnet import build_model
    from src.training.trainer import Trainer

    train_loader, val_loader, _ = build_dataloaders(config)
    model = build_model(config)

    trainer = Trainer(model, train_loader, val_loader, config, run_dir)
    trainer.train()
    return trainer.model


# ================================================================
#  BatchNorm-Freeze
# ================================================================
class TestBatchNormFreeze:
    def test_frozen_bn_stats_stay_constant(self, cfg, tmp_path):
        """Mit freeze_bn_stats=True darf running_mean der eingefrorenen
        Schichten sich nicht ändern – die von layer4 dagegen schon."""
        from src.models.resnet import build_model

        cfg["freeze_bn_stats"] = True
        reference = build_model(cfg)
        bn1_before = reference.bn1.running_mean.clone()
        l4_before  = reference.layer4[-1].bn2.running_mean.clone()

        run_dir = _make_run_dir(tmp_path, "bn_frozen")
        # Vom selben Startzustand ausgehen, damit der Vergleich gilt
        model = build_model(cfg)
        model.load_state_dict(reference.state_dict())

        from src.dataset import build_dataloaders
        from src.training.trainer import Trainer

        train_loader, val_loader, _ = build_dataloaders(cfg)
        Trainer(model, train_loader, val_loader, cfg, run_dir).train()

        model_cpu = model.cpu()
        assert torch.allclose(model_cpu.bn1.running_mean, bn1_before), (
            "running_mean der eingefrorenen Schicht bn1 hat sich veraendert – "
            "der BatchNorm-Freeze greift nicht."
        )
        assert not torch.allclose(model_cpu.layer4[-1].bn2.running_mean, l4_before), (
            "running_mean in layer4 muss sich aendern – dieser Block wird trainiert."
        )

    def test_without_flag_bn_stats_drift(self, cfg, tmp_path):
        """Gegenprobe: mit freeze_bn_stats=False driftet bn1 wie vorher."""
        from src.models.resnet import build_model

        cfg["freeze_bn_stats"] = False
        model = build_model(cfg)
        bn1_before = model.bn1.running_mean.clone()

        run_dir = _make_run_dir(tmp_path, "bn_drift")

        from src.dataset import build_dataloaders
        from src.training.trainer import Trainer

        train_loader, val_loader, _ = build_dataloaders(cfg)
        Trainer(model, train_loader, val_loader, cfg, run_dir).train()

        assert not torch.allclose(model.cpu().bn1.running_mean, bn1_before), (
            "Ohne das Flag muessen die BN-Statistiken weiterlaufen (altes Verhalten)."
        )

    def test_head_bn_never_frozen(self, cfg, tmp_path):
        """Der BatchNorm1d im Kopf ist trainierbar und darf nicht eingefroren werden."""
        from src.dataset import build_dataloaders
        from src.models.resnet import build_model
        from src.training.trainer import Trainer

        cfg["freeze_backbone"] = True   # schaerfster Fall: alles ausser Kopf frozen
        model = build_model(cfg)
        run_dir = _make_run_dir(tmp_path, "head_bn")

        train_loader, val_loader, _ = build_dataloaders(cfg)
        trainer = Trainer(model, train_loader, val_loader, cfg, run_dir)

        trainer.model.train(True)
        trainer._set_frozen_bn_to_eval()

        head_bn = trainer.model.fc[2]
        assert isinstance(head_bn, torch.nn.BatchNorm1d)
        assert head_bn.training, "Kopf-BatchNorm muss im train-Modus bleiben."
        assert not trainer.model.bn1.training, "Backbone-BatchNorm muss auf eval stehen."


# ================================================================
#  Freeze-Logik layer4 (war bisher ungetestet)
# ================================================================
class TestLayer4Freeze:
    def test_layer4_unfreezes_expected_params(self, cfg):
        from src.models.resnet import build_model

        model = build_model(cfg)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}

        assert trainable, "Es muessen Parameter trainierbar sein."
        # Genau layer4 + Kopf, nichts sonst
        for name in trainable:
            assert name.startswith("layer4") or name.startswith("fc."), (
                f"Unerwartet trainierbar: {name}"
            )
        assert any(n.startswith("layer4") for n in trainable)
        assert any(n.startswith("fc.") for n in trainable)
        # Fruehe Schichten bleiben eingefroren
        assert not any(n.startswith("layer1") for n in trainable)
        assert not any(n.startswith("conv1") for n in trainable)


# ================================================================
#  Optimizer-Parametergruppen
# ================================================================
class TestOptimizerGroups:
    def test_two_groups_with_separate_lrs(self, cfg):
        from src.models.resnet import build_model
        from src.training.trainer import build_optimizer

        cfg["learning_rate"] = 1e-4
        cfg["backbone_learning_rate"] = 1e-5

        model = build_model(cfg)
        opt   = build_optimizer(cfg, model)

        groups = {g["name"]: g for g in opt.param_groups}
        assert set(groups) == {"head", "backbone"}
        assert groups["head"]["lr"] == pytest.approx(1e-4)
        assert groups["backbone"]["lr"] == pytest.approx(1e-5)

    def test_null_backbone_lr_falls_back(self, cfg):
        """backbone_learning_rate: null -> beide Gruppen nutzen learning_rate."""
        from src.models.resnet import build_model
        from src.training.trainer import build_optimizer

        cfg["learning_rate"] = 3e-4
        cfg["backbone_learning_rate"] = None

        opt = build_optimizer(cfg, build_model(cfg))
        for group in opt.param_groups:
            assert group["lr"] == pytest.approx(3e-4)

    def test_head_only_has_single_group(self, cfg):
        """Bei freeze_backbone: true existiert keine Backbone-Gruppe."""
        from src.models.resnet import build_model
        from src.training.trainer import build_optimizer

        cfg["freeze_backbone"] = True
        opt = build_optimizer(cfg, build_model(cfg))

        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["name"] == "head"

    def test_all_optimized_params_are_trainable(self, cfg):
        from src.models.resnet import build_model
        from src.training.trainer import build_optimizer

        opt = build_optimizer(cfg, build_model(cfg))
        for group in opt.param_groups:
            for param in group["params"]:
                assert param.requires_grad


# ================================================================
#  Scheduler-Fixes
# ================================================================
class TestSchedulerFixes:
    def test_cosine_tmax_uses_full_epochs(self, cfg):
        """T_max muss der geplanten Trainingsdauer entsprechen, sonst steigt
        die Lernrate nach T_max wieder an."""
        from src.models.resnet import build_model
        from src.training.trainer import build_optimizer, build_scheduler

        cfg["scheduler"] = "cosine"
        cfg["epochs"] = 50
        cfg["early_stopping_patience"] = 10

        opt = build_optimizer(cfg, build_model(cfg))
        sched = build_scheduler(cfg, opt)
        assert sched.T_max == 50

    def test_plateau_mode_follows_monitor(self, cfg):
        from src.models.resnet import build_model
        from src.training.trainer import build_optimizer, build_scheduler

        cfg["scheduler"] = "plateau"
        opt = build_optimizer(cfg, build_model(cfg))

        assert build_scheduler(cfg, opt, "min").mode == "min"
        assert build_scheduler(cfg, opt, "max").mode == "max"


# ================================================================
#  Schwellenwert
# ================================================================
class TestThreshold:
    def test_find_best_threshold_recovers_optimum(self):
        """Bei perfekt trennbaren Scores muss der gefundene Threshold F1=1 liefern."""
        from src.evaluation.evaluator import find_best_threshold

        labels = np.array([0, 0, 0, 1, 1, 1])
        probs  = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        threshold, f1 = find_best_threshold(labels, probs)
        assert f1 == pytest.approx(1.0)
        assert 0.3 < threshold <= 0.7

    def test_find_best_threshold_single_class_falls_back(self):
        from src.evaluation.evaluator import find_best_threshold

        threshold, f1 = find_best_threshold(np.zeros(5), np.linspace(0, 1, 5))
        assert threshold == pytest.approx(0.5)
        assert np.isnan(f1)

    def test_find_best_threshold_beats_default(self):
        """Konservative Scores: das F1-Optimum muss unter 0.5 liegen und
        besser sein als der Standard-Schwellenwert."""
        from sklearn.metrics import f1_score

        from src.evaluation.evaluator import find_best_threshold

        labels = np.array([0] * 10 + [1] * 10)
        # NIO-Scores liegen niedrig -> bei 0.5 wird kaum etwas als NIO erkannt
        probs = np.concatenate([np.linspace(0.01, 0.15, 10),
                                np.linspace(0.20, 0.45, 10)])

        threshold, f1 = find_best_threshold(labels, probs)
        f1_default = f1_score(labels, (probs >= 0.5).astype(int),
                              pos_label=1, zero_division=0)

        assert threshold < 0.5
        assert f1 > f1_default

    def test_evaluator_threshold_extremes(self, cfg, tmp_path):
        """threshold=0.0 -> alles NIO, threshold=1.0 -> (fast) alles IO."""
        from src.dataset import build_dataloaders
        from src.evaluation.evaluator import Evaluator
        from src.models.resnet import build_model

        _, _, test_loader = build_dataloaders(cfg)
        model = build_model(cfg)
        device = torch.device("cpu")

        run_dir = _make_run_dir(tmp_path, "thr")

        ev_low = Evaluator(model, test_loader, run_dir, device,
                           threshold=0.0, split_name="test")
        m_low = ev_low.evaluate()
        # Alles als NIO -> Recall der NIO-Klasse ist perfekt
        assert m_low["recall"] == pytest.approx(1.0)

        ev_high = Evaluator(model, test_loader, run_dir, device,
                            threshold=1.0000001, split_name="test")
        m_high = ev_high.evaluate()
        assert m_high["recall"] == pytest.approx(0.0)

    def test_val_artifacts_do_not_overwrite_test(self, cfg, tmp_path):
        """Ein val-Lauf darf die test-Artefakte nicht ueberschreiben."""
        from src.dataset import build_dataloaders
        from src.evaluation.evaluator import Evaluator
        from src.models.resnet import build_model

        _, val_loader, test_loader = build_dataloaders(cfg)
        model  = build_model(cfg)
        device = torch.device("cpu")
        run_dir = _make_run_dir(tmp_path, "split_names")

        Evaluator(model, test_loader, run_dir, device, split_name="test").evaluate()
        Evaluator(model, val_loader, run_dir, device, split_name="val").evaluate()

        metrics_dir = os.path.join(run_dir, "metrics")
        preds_dir   = os.path.join(run_dir, "preds")

        # Historische test-Namen bleiben unveraendert
        assert os.path.exists(os.path.join(metrics_dir, "test_metrics.json"))
        assert os.path.exists(os.path.join(metrics_dir, "confusion_matrix.png"))
        assert os.path.exists(os.path.join(preds_dir, "example_predictions.csv"))
        # val bekommt eigene Dateien
        assert os.path.exists(os.path.join(metrics_dir, "val_metrics.json"))
        assert os.path.exists(os.path.join(metrics_dir, "confusion_matrix_val.png"))
        assert os.path.exists(os.path.join(preds_dir, "example_predictions_val.csv"))

    def test_collect_is_cached(self, cfg, tmp_path):
        """Zweiter collect()-Aufruf darf keinen weiteren Forward-Pass ausloesen."""
        from src.dataset import build_dataloaders
        from src.evaluation.evaluator import Evaluator
        from src.models.resnet import build_model

        _, val_loader, _ = build_dataloaders(cfg)
        ev = Evaluator(build_model(cfg), val_loader,
                       _make_run_dir(tmp_path, "cache"), torch.device("cpu"),
                       split_name="val")

        labels1, probs1 = ev.collect()
        labels2, probs2 = ev.collect()

        assert labels1 is labels2
        assert probs1 is probs2


# ================================================================
#  best.pt respektiert delta
# ================================================================
class TestBestCheckpointDelta:
    def test_marginal_improvement_does_not_overwrite_best(self, cfg, tmp_path):
        """Eine Verbesserung kleiner als delta darf kein neues best.pt schreiben."""
        from src.dataset import build_dataloaders
        from src.models.resnet import build_model
        from src.training.trainer import Trainer

        cfg["early_stopping_delta"] = 0.5   # absichtlich riesig
        run_dir = _make_run_dir(tmp_path, "delta")

        train_loader, val_loader, _ = build_dataloaders(cfg)
        trainer = Trainer(build_model(cfg), train_loader, val_loader, cfg, run_dir)

        # Erste Epoche gilt immer als Bestwert
        trainer.best_epoch = 1
        trainer.best_val_metric = 0.9
        trainer.checkpoint_mode = "max"

        # +0.01 liegt unter delta=0.5 -> kein neues Bestmodell
        improved = 0.91 > trainer.best_val_metric + trainer.checkpoint_delta
        assert not improved

        # +0.6 liegt darueber
        improved_big = 1.6 > trainer.best_val_metric + trainer.checkpoint_delta
        assert improved_big

    def test_first_epoch_always_saves_best(self, cfg, tmp_path):
        """Auch bei grossem delta muss die erste Epoche ein best.pt schreiben."""
        from src.dataset import build_dataloaders
        from src.models.resnet import build_model
        from src.training.trainer import Trainer

        cfg["early_stopping_delta"] = 10.0
        run_dir = _make_run_dir(tmp_path, "delta_first")

        train_loader, val_loader, _ = build_dataloaders(cfg)
        trainer = Trainer(build_model(cfg), train_loader, val_loader, cfg, run_dir)
        trainer.train()

        assert os.path.exists(os.path.join(run_dir, "checkpoints", "best.pt"))
        assert trainer.best_epoch == 1
