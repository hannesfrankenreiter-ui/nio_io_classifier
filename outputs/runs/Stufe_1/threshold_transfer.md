# Übertragbarkeit des Betriebspunkts val → test (Stufe 1)

Der Schwellenwert wird auf **val** F1-optimal bestimmt und unverändert auf **test** angewendet (Abschnitt 4.5.3). Die Tabelle zeigt, was diese Übertragung kostet.

Eine negative Differenz bedeutet: der angepasste Betriebspunkt ist auf test **schlechter** als die feste Schwelle 0,5. Ursache ist das Split-Design — val und test tragen je einen exklusiven Fehlertyp und sind damit bewusst nicht austauschbar.

| Lauf | val-Optimum τ | test-Optimum τ | test-F1 @val-Schwelle | test-F1 @0,5 | Differenz | test-F1-Spanne über val-Plateau | Fehlalarme @R≥1,00 | Fehlalarme @R≥0,99 | Fehlalarme @R≥0,98 | Fehlalarme @R≥0,95 |
|---|---|---|---|---|---|---|---|---|---|---|
| S1_01_resnet18_letzterblock | 0,220 | 0,310 | 0,8875 | 0,9675 | -0,0800 | 0,8389 … 0,9324 | 8 | 7 | 6 | 3 |
| S1_02_resnet50_letzterblock | 0,265 | 0,480 | 0,9550 | 0,9783 | -0,0232 | 0,9356 … 0,9650 | 9 | 9 | 7 | 1 |
| S1_03_efficientnet_b0_letzterblock | 0,320 | 0,330 | 0,9638 | 0,9591 | 0,0047 | 0,9559 … 0,9673 | 34 | 11 | 10 | 4 |
| S1_04_convnext_tiny_letzterblock | 0,195 | 0,110 | 0,9818 | 0,9663 | 0,0155 | 0,9783 … 0,9892 | 3 | 3 | 3 | 0 |
