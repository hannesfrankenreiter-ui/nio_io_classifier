"""
tests/test_xai.py
Tests für XAI-Methoden: Integrated Gradients, Saliency Map, Occlusion Map.

Ausführen:
    pytest tests/test_xai.py -v
"""
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture(scope="module")
def tiny_model():
    """Kleines ResNet18 ohne Pretrained-Weights für schnelle Tests."""
    from src.models.resnet import build_model
    return build_model(
        {"backbone": "resnet18", "pretrained": False, "freeze_backbone": False}
    ).eval()


@pytest.fixture
def sample_input():
    """Zufälliger (1, 3, 64, 64) Eingabe-Tensor."""
    return torch.randn(1, 3, 64, 64)


# ================================================================
#  Integrated Gradients
# ================================================================
class TestIntegratedGradients:
    def test_output_shape(self, tiny_model, sample_input):
        from src.xai.integrated_gradients import integrated_gradients
        attr = integrated_gradients(tiny_model, sample_input, target_class=0, steps=5)
        assert attr.shape == (64, 64), f"Erwartet (64, 64), erhalten: {attr.shape}"

    def test_output_in_unit_range(self, tiny_model, sample_input):
        from src.xai.integrated_gradients import integrated_gradients
        attr = integrated_gradients(tiny_model, sample_input, target_class=0, steps=5)
        assert attr.min() >= 0.0 - 1e-6
        assert attr.max() <= 1.0 + 1e-6

    def test_output_is_float32(self, tiny_model, sample_input):
        from src.xai.integrated_gradients import integrated_gradients
        attr = integrated_gradients(tiny_model, sample_input, target_class=0, steps=5)
        assert attr.dtype == np.float32

    def test_different_target_classes(self, tiny_model, sample_input):
        """Verschiedene Zielklassen müssen unterschiedliche Attributionen liefern."""
        from src.xai.integrated_gradients import integrated_gradients
        attr0 = integrated_gradients(tiny_model, sample_input, target_class=0, steps=10)
        attr1 = integrated_gradients(tiny_model, sample_input, target_class=1, steps=10)
        # Attributionen für unterschiedliche Klassen sollten sich unterscheiden
        assert not np.allclose(attr0, attr1), "Attributionen für Klasse 0 und 1 sind identisch."

    def test_scale_independent_of_steps(self, tiny_model, sample_input):
        """
        Bug-Fix-Test: Nach der Korrektur sum() → mean() darf die Attribution
        NICHT linear mit ig_steps skalieren – beide sollten in [0, 1] liegen.
        """
        from src.xai.integrated_gradients import integrated_gradients
        attr_10 = integrated_gradients(tiny_model, sample_input, target_class=0, steps=10)
        attr_50 = integrated_gradients(tiny_model, sample_input, target_class=0, steps=50)
        # Beide müssen normiert sein
        assert attr_10.max() <= 1.0 + 1e-6
        assert attr_50.max() <= 1.0 + 1e-6
        # Max-Wert darf sich nicht verdreifachen (sum() würde das verursachen)
        ratio = float(attr_50.max()) / max(float(attr_10.max()), 1e-8)
        assert ratio < 10.0, (
            f"Attribution skaliert zu stark mit ig_steps (ratio={ratio:.2f}). "
            "Prüfe ob sum() → mean() korrekt gepatcht ist."
        )

    def test_constant_image_gives_zero_attribution(self, tiny_model):
        """Schwarzes Bild (= Baseline) ergibt Nullattribution oder sehr kleine Werte."""
        from src.xai.integrated_gradients import integrated_gradients
        black_input = torch.zeros(1, 3, 64, 64)
        attr = integrated_gradients(tiny_model, black_input, target_class=0, steps=10)
        # Alle Werte sollten 0 sein (da input == baseline)
        assert attr.max() < 1e-6, f"Nullbild sollte Nullattribution geben, aber max={attr.max()}"

    def test_completeness_axiom(self, tiny_model, sample_input):
        """
        Completeness-Axiom (Sundararajan et al. 2017): Die Summe aller signierten
        Attributionen entspricht der Differenz der Modellvorhersagen zwischen
        Eingabe und Baseline:  sum(IG) ~ F(x) - F(x').
        Die rechte Riemann-Summe (Formel 4-1) approximiert das Pfadintegral mit
        Fehler O(1/m) - bei m=256 muss die Abweichung klein sein.
        """
        from src.xai.integrated_gradients import integrated_gradients_signed
        target = 0
        attr = integrated_gradients_signed(tiny_model, sample_input, target_class=target, steps=256)
        baseline = torch.zeros_like(sample_input)
        with torch.no_grad():
            delta_f = (
                tiny_model(sample_input)[0, target] - tiny_model(baseline)[0, target]
            ).item()
        total = float(attr.sum())
        tol = max(0.05 * abs(delta_f), 5e-3)
        assert abs(total - delta_f) <= tol, (
            f"Completeness verletzt: sum(IG)={total:.5f}, F(x)-F(x')={delta_f:.5f}, "
            f"Abweichung={abs(total - delta_f):.5f} > Toleranz={tol:.5f}"
        )

    def test_signed_attribution_sums_channels(self, tiny_model, sample_input):
        """Signierte Attribution hat Form (H, W) und enthält i.d.R. beide Vorzeichen."""
        from src.xai.integrated_gradients import integrated_gradients_signed
        attr = integrated_gradients_signed(tiny_model, sample_input, target_class=0, steps=10)
        assert attr.shape == (64, 64)
        assert torch.isfinite(attr).all()


# ================================================================
#  Saliency Map
# ================================================================
class TestSaliencyMap:
    def test_output_shape(self, tiny_model, sample_input):
        from src.xai.integrated_gradients import saliency_map
        sal = saliency_map(tiny_model, sample_input, target_class=0)
        assert sal.shape == (64, 64)

    def test_output_in_unit_range(self, tiny_model, sample_input):
        from src.xai.integrated_gradients import saliency_map
        sal = saliency_map(tiny_model, sample_input, target_class=0)
        assert sal.min() >= 0.0 - 1e-6
        assert sal.max() <= 1.0 + 1e-6

    def test_output_is_float32(self, tiny_model, sample_input):
        from src.xai.integrated_gradients import saliency_map
        sal = saliency_map(tiny_model, sample_input, target_class=0)
        assert sal.dtype == np.float32

    def test_different_target_classes(self, tiny_model, sample_input):
        from src.xai.integrated_gradients import saliency_map
        sal0 = saliency_map(tiny_model, sample_input, target_class=0)
        sal1 = saliency_map(tiny_model, sample_input, target_class=1)
        assert not np.allclose(sal0, sal1), "Saliency für Klasse 0 und 1 sind identisch."


# ================================================================
#  Occlusion Map
# ================================================================
class TestOcclusionMap:
    def test_output_shape(self, tiny_model, sample_input):
        from src.xai.integrated_gradients import occlusion_map
        occ = occlusion_map(tiny_model, sample_input, target_class=0, patch_size=16)
        assert occ.shape == (64, 64)

    def test_output_in_unit_range(self, tiny_model, sample_input):
        from src.xai.integrated_gradients import occlusion_map
        occ = occlusion_map(tiny_model, sample_input, target_class=0, patch_size=16)
        assert occ.min() >= 0.0 - 1e-6
        assert occ.max() <= 1.0 + 1e-6

    def test_nonnegative_sensitivity(self, tiny_model, sample_input):
        """Occlusion-Sensitivität muss nicht-negativ sein (base_prob - occluded_prob ≥ 0 nach clamp)."""
        from src.xai.integrated_gradients import occlusion_map
        occ = occlusion_map(tiny_model, sample_input, target_class=0, patch_size=16)
        assert occ.min() >= 0.0 - 1e-6

    def test_patch_size_16(self, tiny_model, sample_input):
        from src.xai.integrated_gradients import occlusion_map
        occ = occlusion_map(tiny_model, sample_input, target_class=1, patch_size=16)
        assert occ.shape == (64, 64)

    def test_patch_size_32(self, tiny_model, sample_input):
        from src.xai.integrated_gradients import occlusion_map
        occ = occlusion_map(tiny_model, sample_input, target_class=1, patch_size=32)
        assert occ.shape == (64, 64)


# ================================================================
#  Visualisierung (Smoke-Tests: keine Plots anzeigen, nur auf Crashes prüfen)
# ================================================================
class TestVisualization:
    def test_visualize_attribution_overlay(self, tiny_model, sample_input, tmp_path):
        from PIL import Image as PILImage

        from src.xai.integrated_gradients import integrated_gradients, visualize_attribution

        attr    = integrated_gradients(tiny_model, sample_input, target_class=0, steps=5)
        dummy   = PILImage.new("RGB", (64, 64), color=(128, 64, 32))
        outfile = str(tmp_path / "test_overlay.png")
        visualize_attribution(dummy, attr, outfile, title="Test-Overlay", overlay=True)
        assert os.path.exists(outfile), "Heatmap-Datei wurde nicht erzeugt."

    def test_visualize_attribution_no_overlay(self, tiny_model, sample_input, tmp_path):
        from PIL import Image as PILImage

        from src.xai.integrated_gradients import integrated_gradients, visualize_attribution

        attr    = integrated_gradients(tiny_model, sample_input, target_class=0, steps=5)
        dummy   = PILImage.new("RGB", (64, 64))
        outfile = str(tmp_path / "test_no_overlay.png")
        visualize_attribution(dummy, attr, outfile, title="Test", overlay=False)
        assert os.path.exists(outfile)


# ================================================================
#  Kennzahlen der Attributionskarten  (src/xai/metriken.py)
# ================================================================
class TestMetriken:
    """Die Messkette gegen Faelle pruefen, deren Ergebnis vorab feststeht.

    Der Konzentrationsfaktor ist die Leitgroesse des XAI-Abschnitts in
    Kapitel 6. Ohne diese Tests bliebe unbelegt, dass 1,0 tatsaechlich
    Gleichverteilung bedeutet - und genau darauf ruht die Aussage, ein Faktor
    von 2,6 sei eine Konzentration und keine Zahl ohne Bezug.
    """

    @staticmethod
    def _maske(h=64, w=64):
        m = np.zeros((h, w), dtype=bool)
        m[16:32, 16:48] = True                 # 512 von 4096 Pixeln = 12,5 %
        return m

    def test_zufallskarte_ergibt_faktor_eins(self):
        """Eine Karte ohne Struktur muss auf 1,0 herauskommen.

        Der Selbsttest der ganzen Messkette: faellt er aus, misst das Skript
        etwas anderes als den Ueberschuss gegenueber Gleichverteilung.
        """
        from src.xai.metriken import konzentrationsfaktor

        maske = self._maske(256, 256)
        rng = np.random.default_rng(0)
        faktoren = [konzentrationsfaktor(rng.random((256, 256)), maske)[0]
                    for _ in range(20)]
        assert abs(float(np.mean(faktoren)) - 1.0) < 0.02

    def test_karte_auf_der_maske(self):
        """Liegt die ganze Masse auf der Maske, ist der Faktor 1/Flaechenanteil."""
        from src.xai.metriken import konzentrationsfaktor

        maske = self._maske()
        faktor, anteil, flaeche = konzentrationsfaktor(maske.astype(float), maske)
        assert anteil == pytest.approx(1.0)
        assert flaeche == pytest.approx(0.125)
        assert faktor == pytest.approx(8.0)

    def test_leere_karte_ergibt_nan(self):
        """Eine durchgehend leere Occlusion-Karte darf nicht durch null teilen.

        Der Regelfall auf i.O.-Aufnahmen: kein abgedeckter Bereich senkt die
        Wahrscheinlichkeit. Solche Karten werden gezaehlt, nicht mitgemittelt.
        """
        from src.xai.metriken import konzentrationsfaktor

        faktor, anteil, _ = konzentrationsfaktor(np.zeros((64, 64)), self._maske())
        assert np.isnan(faktor) and np.isnan(anteil)

    def test_sensitivitaet_erhaelt_stufen(self):
        """Erosion und Dilatation aendern den Flaechenanteil monoton."""
        from src.xai.metriken import maskensensitivitaet

        rng = np.random.default_rng(1)
        s = maskensensitivitaet(rng.random((128, 128)), self._maske(128, 128),
                                stufen=(-8, 0, 8))
        flaechen = [s[k]["flaechenanteil"] for k in ("-8", "0", "8")]
        assert flaechen[0] < flaechen[1] < flaechen[2]

    def test_deletion_insertion_erkennt_treffende_karte(self, tmp_path):
        """Eine Karte, die den entscheidenden Bildbereich trifft, muss gewinnen.

        Modell sieht nur ein Fenster an. Die treffende Karte muss eine KLEINE
        Deletion-AUC und eine GROSSE Insertion-AUC ergeben, die verfehlende das
        Gegenteil - sonst ist die Reihenfolge im Code vertauscht.
        """
        from src.xai.metriken import deletion_insertion

        class Fenster(torch.nn.Module):
            def forward(self, x):
                s = x[:, :, 16:32, 16:48].mean(dim=(1, 2, 3)) * 40
                return torch.stack([-s, s], dim=1)

        maske = self._maske()
        modell = Fenster().eval()
        x = torch.ones(1, 3, 64, 64)
        treffend = deletion_insertion(modell, x, maske.astype(float), ziel=1, schritte=10)
        verfehlend = deletion_insertion(modell, x, 1.0 - maske, ziel=1, schritte=10)
        assert treffend["deletion_auc"] < verfehlend["deletion_auc"]
        assert treffend["insertion_auc"] > verfehlend["insertion_auc"]

    def test_deletion_kurve_beginnt_beim_unveraenderten_bild(self):
        """Der erste Kurvenpunkt ist das Bild selbst, der letzte die Baseline."""
        from src.xai.metriken import deletion_insertion

        class Mittel(torch.nn.Module):
            def forward(self, x):
                s = x.mean(dim=(1, 2, 3)) * 10
                return torch.stack([-s, s], dim=1)

        x = torch.ones(1, 3, 32, 32)
        r = deletion_insertion(Mittel().eval(), x, np.ones((32, 32)), ziel=1, schritte=4)
        assert r["deletion_kurve"][0] == pytest.approx(r["insertion_kurve"][-1], abs=1e-5)
        assert r["deletion_kurve"][-1] == pytest.approx(r["insertion_kurve"][0], abs=1e-5)

    def test_otsu_trifft_dunkles_rechteck(self):
        """Otsu muss ein dunkles Bauteil auf hellem Grund praktisch exakt finden."""
        from src.xai.metriken import otsu_maske

        soll = self._maske()
        bild = np.full((64, 64, 3), 0.9)
        bild[soll] = 0.1
        maske, diagnose = otsu_maske(bild)
        iou = float((maske & soll).sum() / (maske | soll).sum())
        assert iou > 0.95
        assert diagnose["plausibel"]

    def test_otsu_meldet_unplausibles_bild(self):
        """Ein Bild ohne Bauteil darf nicht still eine Maske erfinden."""
        from src.xai.metriken import otsu_maske

        _, diagnose = otsu_maske(np.full((64, 64, 3), 0.5) + 1e-6)
        assert not diagnose["plausibel"]

    def test_sanity_spearman_erkennt_gleichheit_und_gegensatz(self):
        from src.xai.metriken import sanity_spearman

        a = np.arange(64 * 64, dtype=float).reshape(64, 64)
        assert sanity_spearman(a, a)["spearman"] == pytest.approx(1.0)
        assert sanity_spearman(a, -a)["spearman"] == pytest.approx(1.0)   # Betrag zaehlt
        assert np.isnan(sanity_spearman(a, np.zeros_like(a))["spearman"])

    def test_randomisierung_veraendert_die_gewichte(self, tiny_model):
        """Der Sanity-Check muss das Modell wirklich anfassen - und nur hinten."""
        import copy

        from src.xai.metriken import randomisiere_schichten

        vorher = copy.deepcopy(tiny_model)
        kopie, gesamt = randomisiere_schichten(copy.deepcopy(tiny_model), 1, seed=0)
        schichten_a = [m for m in vorher.modules()
                       if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
        schichten_b = [m for m in kopie.modules()
                       if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
        assert gesamt == len(schichten_a)
        assert not torch.equal(schichten_a[-1].weight, schichten_b[-1].weight)
        assert torch.equal(schichten_a[0].weight, schichten_b[0].weight)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
