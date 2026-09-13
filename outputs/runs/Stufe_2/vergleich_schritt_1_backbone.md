# Versuchsstufe 2 – Schritt 1: Backbone-Vergleich

Erzeugt: 2026-08-19 21:35  
Alle Backbones mit **identischer Konfiguration** (`configs/stufe_2_basis.yaml`) und der Freeze-Strategie „letzter Block + Kopf". Variiert ist ausschliesslich das Backbone.  
**Auswahlkriterium: F1 (n.i.O.) @0,5** → ROC-AUC → Rechenzeit je Bild.

| Rang (Fehlalarme @R≥0,99) | Rang (ROC-AUC) | Rang (F1@0,5) | Rang (F1@val-Schw.) | Backbone | F1 (n.i.O.) @0,5 | Precision @0,5 | Recall @0,5 | Fehlalarme @R≥0,99 | Fehlalarme @R=1,00 | ROC-AUC | Average Precision | F1 @val-Schwelle | val-Schwelle | val-F1 @0,5 | Trainierbare Parameter | Rechenzeit je Bild [ms] | Quelle Rechenzeit | Epochen | Laufzeit | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 2 | convnext_tiny | 0,9950 | 0,9901 | 1,0000 | 0 von 200 | 0 | 1,0000 | 1,0000 | 0,9899 | 0,803 | 1,0000 | 15.866.370 | 5,50 | benchmark | 15 | 00:36:31 | `S2_04_convnext_tiny_letzterblock` |
| 4 | 4 | 2 | 3 | resnet50 | 0,9779 | 0,9614 | 0,9950 | 3 von 200 | 14 | 0,9985 | 0,9985 | 0,9755 | 0,441 | 0,9950 | 16.015.874 | 3,02 | benchmark | 15 | 00:34:08 | `S2_02_resnet50_letzterblock` |
| 2 | 2 | 3 | 1 | efficientnet_b0 | 0,9756 | 0,9524 | 1,0000 | 2 von 200 | 4 | 0,9997 | 0,9997 | 0,9901 | 0,580 | 1,0000 | 1.787.314 | 1,59 | benchmark | 29 | 00:27:12 | `S2_03_efficientnet_b0_letzterblock` |
| 3 | 3 | 4 | 4 | resnet18 | 0,9756 | 0,9524 | 1,0000 | 2 von 200 | 5 | 0,9992 | 0,9992 | 0,9612 | 0,814 | 0,9975 | 8.658.434 | 1,02 | benchmark | 27 | 00:21:32 | `S2_01_resnet18_letzterblock` |
