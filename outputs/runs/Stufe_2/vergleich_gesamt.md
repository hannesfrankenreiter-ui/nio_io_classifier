# Versuchsstufe 2 – Gesamtrangliste aller Kombinationen

Erzeugt: 2026-08-19 21:35  
Alle Kombinationen aus Backbone und Freeze-Strategie, sortiert nach dem Auswahlkriterium.  
**Auswahlkriterium: F1 (n.i.O.) @0,5** → ROC-AUC → Rechenzeit je Bild.

| Rang (Fehlalarme @R≥0,99) | Rang (ROC-AUC) | Rang (F1@0,5) | Rang (F1@val-Schw.) | Backbone | Freeze-Strategie | F1 (n.i.O.) @0,5 | Precision @0,5 | Recall @0,5 | Fehlalarme @R≥0,99 | Fehlalarme @R=1,00 | ROC-AUC | Average Precision | F1 @val-Schwelle | val-Schwelle | val-F1 @0,5 | Trainierbare Parameter | Rechenzeit je Bild [ms] | Quelle Rechenzeit | Epochen | Laufzeit | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 3 | 1 | 8 | convnext_tiny | vollstaendiges Fine-Tuning | 1,0000 | 1,0000 | 1,0000 | 0 von 200 | 0 | 1,0000 | 1,0000 | 0,9691 | 0,952 | 1,0000 | 28.215.906 | 5,50 | benchmark | 20 | 00:08:16 | `S2_12_convnext_tiny_vollstaendig` |
| 4 | 5 | 2 | 7 | resnet50 | vollstaendiges Fine-Tuning | 0,9975 | 1,0000 | 0,9950 | 0 von 200 | 14 | 0,9996 | 0,9997 | 0,9744 | 0,808 | 1,0000 | 24.559.170 | 3,02 | benchmark | 17 | 01:13:13 | `S2_08_resnet50_vollstaendig` |
| 2 | 2 | 3 | 2 | convnext_tiny | letzter Block + Kopf | 0,9950 | 0,9901 | 1,0000 | 0 von 200 | 0 | 1,0000 | 1,0000 | 0,9899 | 0,803 | 1,0000 | 15.866.370 | 5,50 | benchmark | 15 | 00:36:31 | `S2_04_convnext_tiny_letzterblock` |
| 1 | 1 | 4 | 5 | resnet18 | vollstaendiges Fine-Tuning | 0,9924 | 1,0000 | 0,9850 | 0 von 200 | 0 | 1,0000 | 1,0000 | 0,9770 | 0,822 | 0,9975 | 11.441.218 | 1,02 | benchmark | 20 | 00:28:28 | `S2_06_resnet18_vollstaendig` |
| 9 | 9 | 5 | 3 | resnet50 | nur Kopf | 0,9802 | 0,9706 | 0,9900 | 6 von 200 | 19 | 0,9962 | 0,9955 | 0,9778 | 0,442 | 0,9925 | 1.051.138 | 3,02 | benchmark | 22 | 00:40:20 | `S2_07_resnet50_nurkopf` |
| 8 | 7 | 6 | 4 | efficientnet_b0 | vollstaendiges Fine-Tuning | 0,9798 | 0,9898 | 0,9700 | 4 von 200 | 15 | 0,9990 | 0,9990 | 0,9771 | 0,555 | 1,0000 | 4.665.470 | 1,59 | benchmark | 20 | 00:51:08 | `S2_10_efficientnet_b0_vollstaendig` |
| 7 | 8 | 7 | 6 | resnet50 | letzter Block + Kopf | 0,9779 | 0,9614 | 0,9950 | 3 von 200 | 14 | 0,9985 | 0,9985 | 0,9755 | 0,441 | 0,9950 | 16.015.874 | 3,02 | benchmark | 15 | 00:34:08 | `S2_02_resnet50_letzterblock` |
| 5 | 4 | 8 | 1 | efficientnet_b0 | letzter Block + Kopf | 0,9756 | 0,9524 | 1,0000 | 2 von 200 | 4 | 0,9997 | 0,9997 | 0,9901 | 0,580 | 1,0000 | 1.787.314 | 1,59 | benchmark | 29 | 00:27:12 | `S2_03_efficientnet_b0_letzterblock` |
| 6 | 6 | 9 | 9 | resnet18 | letzter Block + Kopf | 0,9756 | 0,9524 | 1,0000 | 2 von 200 | 5 | 0,9992 | 0,9992 | 0,9612 | 0,814 | 0,9975 | 8.658.434 | 1,02 | benchmark | 27 | 00:21:32 | `S2_01_resnet18_letzterblock` |
| 11 | 11 | 10 | 11 | efficientnet_b0 | nur Kopf | 0,9406 | 0,8959 | 0,9900 | 20 von 200 | 32 | 0,9936 | 0,9935 | 0,9384 | 0,465 | 0,9925 | 657.922 | 1,59 | benchmark | 13 | 00:15:27 | `S2_09_efficientnet_b0_nurkopf` |
| 10 | 10 | 11 | 10 | convnext_tiny | nur Kopf | 0,9368 | 0,8811 | 1,0000 | 20 von 200 | 27 | 0,9941 | 0,9941 | 0,9471 | 0,601 | 0,9755 | 395.778 | 5,50 | benchmark | 30 | 00:28:00 | `S2_11_convnext_tiny_nurkopf` |
| 12 | 12 | 12 | 12 | resnet18 | nur Kopf | 0,8552 | 0,7810 | 0,9450 | 110 von 200 | 129 | 0,9350 | 0,9368 | 0,8511 | 0,600 | 0,9707 | 264.706 | 1,02 | benchmark | 24 | 00:17:28 | `S2_05_resnet18_nurkopf` |
