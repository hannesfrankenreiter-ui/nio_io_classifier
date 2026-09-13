# Versuchsstufe 1 – Schritt 1: Backbone-Vergleich

Erzeugt: 2026-08-19 21:37  
Alle Backbones mit **identischer Konfiguration** (`configs/default.yaml`) und der Freeze-Strategie „letzter Block + Kopf". Variiert ist ausschliesslich das Backbone.  
**Auswahlkriterium: F1 (n.i.O.) @0,5** → ROC-AUC → Rechenzeit je Bild.

| Rang (Fehlalarme @R≥0,99) | Rang (ROC-AUC) | Rang (F1@0,5) | Rang (F1@val-Schw.) | Backbone | F1 (n.i.O.) @0,5 | Precision @0,5 | Recall @0,5 | Fehlalarme @R≥0,99 | Fehlalarme @R=1,00 | ROC-AUC | Average Precision | F1 @val-Schwelle | val-Schwelle | val-F1 @0,5 | Trainierbare Parameter | Rechenzeit je Bild [ms] | Quelle Rechenzeit | Epochen | Laufzeit | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 2 | 1 | 3 | resnet50 | 0,9783 | 0,9783 | 0,9783 | 9 von 173 | 9 | 0,9985 | 0,9981 | 0,9550 | 0,270 | 0,9592 | 16.015.874 | 93,75 | benchmark | 33 | 01:32:34 | `S1_02_resnet50_letzterblock` |
| 2 | 3 | 2 | 4 | resnet18 | 0,9675 | 0,9640 | 0,9710 | 7 von 173 | 8 | 0,9972 | 0,9965 | 0,8875 | 0,209 | 0,9027 | 8.658.434 | 24,66 | benchmark | 9 | 00:13:33 | `S1_01_resnet18_letzterblock` |
| 1 | 1 | 3 | 1 | convnext_tiny | 0,9663 | 1,0000 | 0,9348 | 3 von 173 | 3 | 0,9996 | 0,9995 | 0,9818 | 0,216 | 0,9742 | 15.866.370 | 88,29 | benchmark | 30 | 01:36:08 | `S1_04_convnext_tiny_letzterblock` |
| 4 | 4 | 4 | 2 | efficientnet_b0 | 0,9591 | 0,9847 | 0,9348 | 11 von 173 | 34 | 0,9962 | 0,9957 | 0,9638 | 0,329 | 0,8992 | 1.787.314 | 45,33 | benchmark | 12 | 00:20:36 | `S1_03_efficientnet_b0_letzterblock` |
