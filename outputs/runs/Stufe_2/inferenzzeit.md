# Versuchsstufe 2 – Inferenzzeit je Architektur

Erzeugt: 2026-08-27 14:11  
Median aus 30 Wiederholungen nach 10 Aufwaermdurchlaeufen, 512x512 px (HxB), fp32.  
**Je Architektur gemessen, nicht je Lauf**: unter `model.eval()` und `torch.no_grad()` hat die Freeze-Strategie keinen Einfluss auf die Inferenzzeit.  
Ersetzt die Spalte „Rechenzeit je Bild" der Vergleichstabellen, die beilaeufig waehrend zwoelf verschiedener Trainingslaeufe erhoben wurde und dadurch ueberwiegend den GPU-Lastzustand misst.

| Backbone | Batch 16 [ms/Bild] | Einzelbild [ms] | Streuung Batch [ms] | Schnellster Lauf [ms] | Parameter gesamt |
|---|---|---|---|---|---|
| resnet18 | 24,66 | 17,00 | 0,03 | 24,64 | 11.441.218 |
| efficientnet_b0 | 45,33 | 21,33 | 0,03 | 45,29 | 4.665.470 |
| convnext_tiny | 88,27 | 70,69 | 0,56 | 88,21 | 28.215.906 |
| resnet50 | 93,76 | 44,47 | 0,04 | 93,69 | 24.559.170 |
