# Versuchsstufe 1 – Schritt 2: Freeze-Strategie je Backbone

Erzeugt: 2026-08-19 21:37  
Volles Raster: **jedes** Backbone mit allen drei Freeze-Strategien, nach Backbone gruppiert.  
Die Zeilen „letzter Block + Kopf" sind die Laeufe aus Schritt 1 – identische Konfiguration, deshalb wiederverwendet statt wiederholt.  
Die Rangspalten beziehen sich auf **alle** Laeufe der Tabelle, nicht auf die jeweilige Backbone-Gruppe.

| Rang (Fehlalarme @R≥0,99) | Rang (ROC-AUC) | Rang (F1@0,5) | Rang (F1@val-Schw.) | Backbone | Freeze-Strategie | F1 (n.i.O.) @0,5 | Precision @0,5 | Recall @0,5 | Fehlalarme @R≥0,99 | Fehlalarme @R=1,00 | ROC-AUC | Average Precision | F1 @val-Schwelle | val-Schwelle | val-F1 @0,5 | Trainierbare Parameter | Rechenzeit je Bild [ms] | Quelle Rechenzeit | Epochen | Laufzeit | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 11 | 11 | 10 | 11 | resnet18 | nur Kopf | 0,8428 | 0,7826 | 0,9130 | 122 von 173 | 123 | 0,9519 | 0,9579 | 0,7383 | 0,342 | 0,8442 | 264.706 | 24,66 | benchmark | 11 | 00:14:11 | `S1_05_resnet18_nurkopf` |
| 6 | 7 | 5 | 9 | resnet18 | letzter Block + Kopf | 0,9675 | 0,9640 | 0,9710 | 7 von 173 | 8 | 0,9972 | 0,9965 | 0,8875 | 0,209 | 0,9027 | 8.658.434 | 24,66 | benchmark | 9 | 00:13:33 | `S1_01_resnet18_letzterblock` |
| 1 | 1 | 2 | 1 | resnet18 | vollstaendiges Fine-Tuning | 0,9822 | 0,9650 | 1,0000 | 0 von 173 | 1 | 1,0000 | 0,9999 | 0,9892 | 0,624 | 0,9337 | 11.441.218 | 24,66 | benchmark | 24 | 00:51:32 | `S1_06_resnet18_vollstaendig` |
| 9 | 9 | 8 | 8 | resnet50 | nur Kopf | 0,9231 | 0,9836 | 0,8696 | 42 von 173 | 62 | 0,9874 | 0,9801 | 0,8924 | 0,613 | 0,8879 | 1.051.138 | 93,75 | benchmark | 17 | 00:44:53 | `S1_07_resnet50_nurkopf` |
| 7 | 6 | 3 | 7 | resnet50 | letzter Block + Kopf | 0,9783 | 0,9783 | 0,9783 | 9 von 173 | 9 | 0,9985 | 0,9981 | 0,9550 | 0,270 | 0,9592 | 16.015.874 | 93,75 | benchmark | 33 | 01:32:34 | `S1_02_resnet50_letzterblock` |
| 10 | 10 | 11 | 10 | resnet50 | vollstaendiges Fine-Tuning | 0,8059 | 0,6782 | 0,9928 | 58 von 173 | 78 | 0,9859 | 0,9864 | 0,7582 | 0,145 | 0,9446 | 24.559.170 | 93,75 | benchmark | 18 | 00:42:54 | `S1_08_resnet50_vollstaendig` |
| 5 | 5 | 4 | 5 | efficientnet_b0 | nur Kopf | 0,9704 | 0,9924 | 0,9493 | 6 von 173 | 8 | 0,9986 | 0,9983 | 0,9744 | 0,461 | 0,8822 | 657.922 | 45,33 | benchmark | 13 | 00:08:12 | `S1_09_efficientnet_b0_nurkopf` |
| 8 | 8 | 7 | 6 | efficientnet_b0 | letzter Block + Kopf | 0,9591 | 0,9847 | 0,9348 | 11 von 173 | 34 | 0,9962 | 0,9957 | 0,9638 | 0,329 | 0,8992 | 1.787.314 | 45,33 | benchmark | 12 | 00:20:36 | `S1_03_efficientnet_b0_letzterblock` |
| 2 | 2 | 9 | 4 | efficientnet_b0 | vollstaendiges Fine-Tuning | 0,9139 | 0,8415 | 1,0000 | 0 von 173 | 4 | 0,9998 | 0,9998 | 0,9787 | 0,650 | 0,9825 | 4.665.470 | 45,33 | benchmark | 44 | 00:16:43 | `S1_10_efficientnet_b0_vollstaendig` |
| 12 | 12 | 12 | 12 | convnext_tiny | nur Kopf | 0,4318 | 1,0000 | 0,2754 | 158 von 173 | 170 | 0,7074 | 0,7080 | 0,5659 | 0,295 | 0,1402 | 395.778 | 88,29 | benchmark | 8 | 00:06:51 | `S1_11_convnext_tiny_nurkopf` |
| 4 | 3 | 6 | 3 | convnext_tiny | letzter Block + Kopf | 0,9663 | 1,0000 | 0,9348 | 3 von 173 | 3 | 0,9996 | 0,9995 | 0,9818 | 0,216 | 0,9742 | 15.866.370 | 88,29 | benchmark | 30 | 01:36:08 | `S1_04_convnext_tiny_letzterblock` |
| 3 | 4 | 1 | 2 | convnext_tiny | vollstaendiges Fine-Tuning | 0,9964 | 1,0000 | 0,9928 | 0 von 173 | 20 | 0,9992 | 0,9991 | 0,9892 | 0,328 | 0,9924 | 28.215.906 | 88,29 | benchmark | 21 | 00:12:50 | `S1_12_convnext_tiny_vollstaendig` |
