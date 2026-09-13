# Übertragbarkeit des Betriebspunkts val → test (Stufe 2)

Der Schwellenwert wird auf **val** F1-optimal bestimmt und unverändert auf **test** angewendet (Abschnitt 4.5.3). Die Tabelle zeigt, was diese Übertragung kostet.

Eine negative Differenz bedeutet: der angepasste Betriebspunkt ist auf test **schlechter** als die feste Schwelle 0,5. Ursache ist das Split-Design — val und test tragen je einen exklusiven Fehlertyp und sind damit bewusst nicht austauschbar.

| Lauf | val-Optimum τ | test-Optimum τ | test-F1 @val-Schwelle | test-F1 @0,5 | Differenz | test-F1-Spanne über val-Plateau | Fehlalarme @R≥1,00 | Fehlalarme @R≥0,99 | Fehlalarme @R≥0,98 | Fehlalarme @R≥0,95 |
|---|---|---|---|---|---|---|---|---|---|---|
| S2_01_resnet18_letzterblock | 0,645 | 0,695 | 0,9612 | 0,9756 | -0,0144 | 0,9281 … 0,9900 | 5 | 2 | 2 | 1 |
| S2_02_resnet50_letzterblock | 0,170 | 0,550 | 0,9755 | 0,9779 | -0,0024 | 0,8909 … 0,9755 | 14 | 3 | 2 | 1 |
| S2_03_efficientnet_b0_letzterblock | 0,290 | 0,650 | 0,9901 | 0,9756 | 0,0145 | 0,8889 … 0,9901 | 4 | 2 | 1 | 0 |
| S2_04_convnext_tiny_letzterblock | 0,495 | 0,610 | 0,9899 | 0,9950 | -0,0051 | 0,9852 … 1,0000 | 0 | 0 | 0 | 0 |
| S2_05_resnet18_nurkopf | 0,590 | 0,525 | 0,8511 | 0,8552 | -0,0041 | 0,8491 … 0,8630 | 129 | 110 | 93 | 56 |
| S2_06_resnet18_vollstaendig | 0,575 | 0,140 | 0,9770 | 0,9924 | -0,0155 | 0,9610 … 1,0000 | 0 | 0 | 0 | 0 |
| S2_07_resnet50_nurkopf | 0,435 | 0,480 | 0,9778 | 0,9802 | -0,0024 | 0,9726 … 0,9802 | 19 | 6 | 6 | 3 |
| S2_08_resnet50_vollstaendig | 0,265 | 0,135 | 0,9744 | 0,9975 | -0,0231 | 0,9744 … 0,9975 | 14 | 0 | 0 | 0 |
| S2_09_efficientnet_b0_nurkopf | 0,440 | 0,650 | 0,9384 | 0,9406 | -0,0022 | 0,8850 … 0,9584 | 32 | 20 | 12 | 6 |
| S2_10_efficientnet_b0_vollstaendig | 0,460 | 0,405 | 0,9771 | 0,9798 | -0,0027 | 0,9592 … 0,9851 | 15 | 4 | 4 | 0 |
| S2_11_convnext_tiny_nurkopf | 0,595 | 0,755 | 0,9471 | 0,9368 | 0,0103 | 0,9431 … 0,9538 | 27 | 20 | 15 | 8 |
| S2_12_convnext_tiny_vollstaendig | 0,125 | 0,125 | 0,9691 | 1,0000 | -0,0309 | 0,9610 … 1,0000 | 0 | 0 | 0 | 0 |
| S2_resnet18_vollstaendig_800x320 | 0,390 | 0,345 | 0,7067 | 0,7067 | 0,0000 | 0,7067 … 0,7067 | 166 | 166 | 166 | 166 |
